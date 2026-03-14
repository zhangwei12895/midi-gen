"""
app.py  v2  —  8-bit Music Generator Web Server

新增功能
────────
  ① 生成限制      max_bars 参数（16/32/48/64/不限），转换为 max_gen_tokens
  ② CPU 占满      torch.set_num_threads(cpu_count)，推理全核并行
  ③ 心跳超时      用户离开 3 分钟无心跳 → 自动取消生成
  ④ 可选模型      自动扫描 checkpoints/ 下所有 .pt，/api/models 返回列表
  ⑤ 缓存限制      outputs/ 最多保留 MAX_OUTPUT_FILES 个文件，超出删最旧
  ⑥ presets 时长  返回每个 MIDI 的真实时长（秒）
"""

from __future__ import annotations
import os, sys, time, threading, argparse, datetime, signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, send_file, abort

app = Flask(__name__)

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
OUTPUT_DIR       = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR   = os.path.join(BASE_DIR, "checkpoints")
MAX_OUTPUT_FILES = 50          # outputs/ 最多保留文件数
HEARTBEAT_TIMEOUT = 180        # 秒，用户超时自动取消生成
os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────
_models    : dict[str, object] = {}   # ckpt_name → (model, tokenizer, meta)
_cur_model : str = ""
_lock      = threading.Lock()
_busy      = False
_device    = None

# 心跳
_last_heartbeat : float = 0.0
_cancel_flag    = threading.Event()   # 置位时生成线程应尽快退出


# ══════════════════════════════════════════════
# CPU 线程优化
# ══════════════════════════════════════════════
def _set_cpu_threads():
    try:
        import torch
        n = os.cpu_count() or 1
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
        print(f"[server] CPU threads={n}  interop={max(1, n//2)}")
    except Exception as e:
        print(f"[server] set_num_threads failed: {e}")


# ══════════════════════════════════════════════
# 模型管理
# ══════════════════════════════════════════════
def _load_model(ckpt_path: str):
    """加载并缓存模型，返回 (model, tok, meta)"""
    name = os.path.basename(ckpt_path)
    if name in _models:
        return _models[name]

    import torch
    from generate import load_model
    global _device
    if _device is None:
        _device = torch.device("cpu")

    print(f"[server] 加载模型: {name}")
    t0 = time.time()
    m, tok, meta = load_model(ckpt_path, _device)
    print(f"[server] 就绪  params={meta.get('param_str','')}  耗时={time.time()-t0:.1f}s")
    _models[name] = (m, tok, meta)
    return m, tok, meta


def _scan_checkpoints() -> list[str]:
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return sorted(
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".pt")
    )


# ══════════════════════════════════════════════
# 缓存清理
# ══════════════════════════════════════════════
def _trim_outputs():
    """outputs/ 超过 MAX_OUTPUT_FILES 时删除最旧文件"""
    files = sorted(
        [os.path.join(OUTPUT_DIR, f)
         for f in os.listdir(OUTPUT_DIR) if f.endswith(".mid")],
        key=os.path.getmtime,
    )
    while len(files) > MAX_OUTPUT_FILES:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
            print(f"[cache] 删除旧文件: {os.path.basename(oldest)}")
        except Exception:
            pass


# ══════════════════════════════════════════════
# MIDI 时长读取（轻量，无需加载模型）
# ══════════════════════════════════════════════
def _midi_duration(path: str) -> float:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(path)
        return round(pm.get_end_time(), 1)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════
# 心跳监控线程
# ══════════════════════════════════════════════
def _heartbeat_watcher():
    """后台线程：若超过 HEARTBEAT_TIMEOUT 秒无心跳，取消生成"""
    global _busy
    while True:
        time.sleep(10)
        if _busy and _last_heartbeat > 0:
            elapsed = time.time() - _last_heartbeat
            if elapsed > HEARTBEAT_TIMEOUT:
                print(f"[heartbeat] 超时 {elapsed:.0f}s，取消生成")
                _cancel_flag.set()


_hb_thread = threading.Thread(target=_heartbeat_watcher, daemon=True)
_hb_thread.start()


# ══════════════════════════════════════════════
# API
# ══════════════════════════════════════════════

@app.route("/api/heartbeat", methods=["POST"])
def api_heartbeat():
    """前端每 30 秒 POST 一次，告知用户在线"""
    global _last_heartbeat
    _last_heartbeat = time.time()
    return jsonify({"ok": True})


@app.route("/api/models")
def api_models():
    """返回可用模型列表"""
    files = _scan_checkpoints()
    result = []
    for f in files:
        path = os.path.join(CHECKPOINT_DIR, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        # 从已缓存的 meta 中取参数量
        meta = _models.get(f, (None, None, {}))[2] if f in _models else {}
        result.append({
            "name"     : f,
            "size_mb"  : round(size_mb, 1),
            "param_str": meta.get("param_str", ""),
            "step"     : meta.get("step", 0),
            "loaded"   : f in _models,
        })
    return jsonify(result)


@app.route("/api/status")
def api_status():
    meta = _models.get(_cur_model, (None, None, {}))[2] if _cur_model else {}
    return jsonify({
        "ready"      : bool(_cur_model and _cur_model in _models),
        "busy"       : _busy,
        "cur_model"  : _cur_model,
        "model_name" : meta.get("model_name", ""),
        "param_str"  : meta.get("param_str",  ""),
        "train_time" : meta.get("train_time", ""),
        "step"       : meta.get("step", 0),
        "models"     : _scan_checkpoints(),
    })


@app.route("/api/presets")
def api_presets():
    """列出已生成的 MIDI 文件（最近 30 个），包含真实时长"""
    files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mid")],
        key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        reverse=True,
    )[:30]
    result = []
    for f in files:
        fpath = os.path.join(OUTPUT_DIR, f)
        # 从文件名解析 style
        import re
        sm = re.search(r'_s(\d)_', f)
        style = int(sm.group(1)) if sm else 2
        result.append({
            "filename": f,
            "url"     : f"/outputs/{f}",
            "mtime"   : os.path.getmtime(fpath),
            "duration": _midi_duration(fpath),   # ★ 真实时长
            "style"   : style,
        })
    return jsonify(result)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    global _busy, _last_heartbeat

    data = request.get_json(force=True) or {}

    # 选择模型
    model_name = data.get("model", _cur_model)
    if not model_name:
        ckpts = _scan_checkpoints()
        if not ckpts:
            return jsonify({"error": "没有可用模型"}), 503
        model_name = ckpts[0]

    ckpt_path = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(ckpt_path):
        return jsonify({"error": f"模型文件不存在: {model_name}"}), 404

    if _busy:
        return jsonify({"error": "服务器正忙，请稍后再试"}), 429

    # 参数解析
    from config import GEN_CONFIG, INTENSITY_EN
    style       = max(0, min(4, int(data.get("style",       GEN_CONFIG["default_intensity"]))))
    key         = data.get("key",    GEN_CONFIG["default_key"])
    tempo       = float(data.get("tempo",        GEN_CONFIG["default_tempo"]))
    temperature = float(data.get("temperature",  GEN_CONFIG["temperature"]))
    top_k       = int(data.get("top_k",          GEN_CONFIG["top_k"]))
    top_p       = float(data.get("top_p",        GEN_CONFIG["top_p"]))

    # ★ 生成限制：max_bars → max_tokens 换算（每小节约 50 tokens）
    max_bars = data.get("max_bars", None)   # None = 不限制
    if max_bars is not None:
        max_bars = max(8, int(max_bars))
        # 留 20% 余量，最少保证 min_bars
        max_tokens = max_bars * 55 + 100
    else:
        max_tokens = GEN_CONFIG["max_gen_tokens"]

    with _lock:
        _busy = True
        _cancel_flag.clear()
        _last_heartbeat = time.time()   # 开始生成时重置心跳

    try:
        model, tok, meta = _load_model(ckpt_path)

        import torch
        from generate            import generate_song
        from data.midi_processor import MidiProcessor

        t0 = time.time()
        tokens, sections_hit = generate_song(
            model, tok, _device,
            style        = style,
            key_str      = key,
            tempo        = tempo,
            temperature  = temperature,
            top_k        = top_k,
            top_p        = top_p,
            max_tokens   = max_tokens,        # ★ 传入限制
            cancel_flag  = _cancel_flag,      # ★ 传入取消标志
        )
        elapsed = time.time() - t0

        # 检查是否被取消
        if _cancel_flag.is_set() and len(tokens) < 20:
            return jsonify({"error": "生成已取消（用户超时离开）"}), 499

        proc = MidiProcessor(tok)
        model_info = (f"{meta.get('model_name','?')}"
                      f" {meta.get('param_str','')}"
                      f" style={INTENSITY_EN[style]} temp={temperature:.2f}")
        pm   = proc.tokens_to_midi(tokens, tempo=tempo, model_name=model_info)
        dur  = pm.get_end_time()

        bars  = sum(1 for t in tokens if t == tok.bar_id)
        notes = sum(1 for t in tokens
                    if tok.id2token.get(t, "").startswith("NOTE_ON_"))

        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname  = f"gen_{dt_str}_s{style}_{INTENSITY_EN[style]}_t{int(temperature*100):03d}.mid"
        fpath  = os.path.join(OUTPUT_DIR, fname)
        pm.write(fpath)

        # ★ 缓存清理
        _trim_outputs()

        return jsonify({
            "filename": fname,
            "url"     : f"/outputs/{fname}",
            "duration": round(dur, 1),
            "bars"    : bars,
            "notes"   : notes,
            "sections": sections_hit,
            "elapsed" : round(elapsed, 1),
            "style"   : style,
            "key"     : key,
            "tempo"   : tempo,
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        _busy = False



# ══════════════════════════════════════════════
# 全局收藏 — 存储于 favorites.json（最多 6 首）
# ══════════════════════════════════════════════
FAVORITES_FILE = os.path.join(BASE_DIR, "favorites.json")
_fav_lock = threading.Lock()
MAX_FAVORITES = 6

def _load_favs() -> list:
    try:
        if os.path.exists(FAVORITES_FILE):
            import json
            with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_favs(favs: list):
    import json
    with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
        json.dump(favs, f, ensure_ascii=False, indent=2)

@app.route("/api/favorites", methods=["GET"])
def api_favorites_get():
    """返回全局收藏列表"""
    with _fav_lock:
        return jsonify(_load_favs())

@app.route("/api/favorites", methods=["POST"])
def api_favorites_post():
    """切换收藏状态，最多 6 首"""
    import json
    data = request.get_json(force=True) or {}
    filename = (data.get("filename") or "").strip()
    if not filename:
        return jsonify({"error": "filename required"}), 400

    with _fav_lock:
        favs = _load_favs()
        if filename in favs:
            favs.remove(filename)
            action = "removed"
        else:
            if len(favs) >= MAX_FAVORITES:
                return jsonify({"error": f"收藏已满（最多 {MAX_FAVORITES} 首），请先取消其他收藏"}), 400
            favs.append(filename)
            action = "added"
        _save_favs(favs)
        return jsonify({"action": action, "favorites": favs})


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        abort(404)
    # 允许 Range 请求（手机端分段加载 MIDI 更快）
    response = send_file(path, mimetype="audio/midi",
                         as_attachment=False,
                         conditional=True)
    response.headers["Accept-Ranges"]  = "bytes"
    response.headers["Cache-Control"]  = "public, max-age=3600"
    return response


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "web", "index.html"))

@app.route("/web/<path:filename>")
def serve_web(filename):
    return send_file(os.path.join(BASE_DIR, "web", filename))


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  default=5000, type=int)
    p.add_argument("--ckpt",  default="")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    _set_cpu_threads()

    # 预加载指定模型
    ckpts = _scan_checkpoints()
    target = args.ckpt or (os.path.join(CHECKPOINT_DIR, ckpts[0]) if ckpts else "")
    if target and os.path.exists(target):
        _cur_model = os.path.basename(target)
        _load_model(target)
    elif ckpts:
        _cur_model = ckpts[0]
        print(f"[server] 自动选择模型: {_cur_model}")
        _load_model(os.path.join(CHECKPOINT_DIR, _cur_model))
    else:
        print("[server] 未找到模型，以演示模式启动")

    print(f"\n  ✓ 服务启动  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
