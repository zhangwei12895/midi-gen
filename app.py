"""
app.py  —  8-bit Music Generator Web Server
Flask 后端，CPU 推理 + MIDI 生成 + 文件服务

启动：
  python app.py
  python app.py --host 0.0.0.0 --port 5000 --ckpt ./checkpoints/best_model.pt

端点：
  GET  /                    → 主页面 HTML
  POST /api/generate        → 生成 MIDI，返回 {filename, url, duration, sections, bars}
  GET  /outputs/<filename>  → 下载/播放 MIDI 文件
  GET  /api/status          → 服务状态（模型是否已加载）
  GET  /api/presets         → 预设列表（已生成的文件）
"""

from __future__ import annotations
import os, sys, time, uuid, threading, argparse, json, datetime

# ── 将项目根目录加入搜索路径 ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, send_file, abort

app = Flask(__name__)

# ── 全局状态 ──────────────────────────────────────
_model     = None
_tokenizer = None
_meta      = {}
_lock      = threading.Lock()   # 同一时刻只允许一次生成
_busy      = False
_device    = None

OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════
# 模型加载（启动时执行一次）
# ══════════════════════════════════════════════════
def init_model(ckpt_path: str):
    global _model, _tokenizer, _meta, _device
    import torch
    from generate import load_model

    _device = torch.device("cpu")
    print(f"[server] 加载模型: {ckpt_path}")
    t0 = time.time()
    _model, _tokenizer, _meta = load_model(ckpt_path, _device)
    print(f"[server] 模型就绪  参数={_meta.get('param_str','')}  "
          f"耗时={time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════
@app.route("/api/status")
def api_status():
    return jsonify({
        "ready"      : _model is not None,
        "busy"       : _busy,
        "model_name" : _meta.get("model_name", ""),
        "param_str"  : _meta.get("param_str",  ""),
        "train_time" : _meta.get("train_time", ""),
        "step"       : _meta.get("step", 0),
    })


@app.route("/api/presets")
def api_presets():
    """列出已生成的 MIDI 文件（最近 20 个）"""
    files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mid")],
        key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        reverse=True,
    )[:20]
    result = []
    for f in files:
        result.append({
            "filename": f,
            "url"     : f"/outputs/{f}",
            "mtime"   : os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        })
    return jsonify(result)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    global _busy

    if _model is None:
        return jsonify({"error": "模型尚未加载"}), 503

    if _busy:
        return jsonify({"error": "服务器正忙，请稍后再试"}), 429

    data = request.get_json(force=True) or {}

    # 参数解析
    from config import GEN_CONFIG, INTENSITY_EN
    style       = max(0, min(4, int(data.get("style",       GEN_CONFIG["default_intensity"]))))
    key         = data.get("key",         GEN_CONFIG["default_key"])
    tempo       = float(data.get("tempo", GEN_CONFIG["default_tempo"]))
    temperature = float(data.get("temperature", GEN_CONFIG["temperature"]))
    top_k       = int(data.get("top_k",   GEN_CONFIG["top_k"]))
    top_p       = float(data.get("top_p", GEN_CONFIG["top_p"]))

    with _lock:
        _busy = True

    try:
        import torch
        from generate            import generate_song
        from data.midi_processor import MidiProcessor

        t0     = time.time()
        tokens, sections_hit = generate_song(
            _model, _tokenizer, _device,
            style      = style,
            key_str    = key,
            tempo      = tempo,
            temperature= temperature,
            top_k      = top_k,
            top_p      = top_p,
        )
        elapsed = time.time() - t0

        proc  = MidiProcessor(_tokenizer)
        model_info = (f"{_meta.get('model_name','unknown')}"
                      f"  {_meta.get('param_str','')}"
                      f"  style={INTENSITY_EN[style]}"
                      f"  temp={temperature:.2f}")
        pm    = proc.tokens_to_midi(tokens, tempo=tempo, model_name=model_info)
        dur   = pm.get_end_time()

        # 统计
        bars  = sum(1 for t in tokens if t == _tokenizer.bar_id)
        notes = sum(1 for t in tokens
                    if _tokenizer.id2token.get(t, "").startswith("NOTE_ON_"))

        # 文件名
        dt_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname    = f"gen_{dt_str}_s{style}_{INTENSITY_EN[style]}_t{int(temperature*100):03d}.mid"
        fpath    = os.path.join(OUTPUT_DIR, fname)
        pm.write(fpath)

        return jsonify({
            "filename" : fname,
            "url"      : f"/outputs/{fname}",
            "duration" : round(dur, 1),
            "bars"     : bars,
            "notes"    : notes,
            "sections" : sections_hit,
            "elapsed"  : round(elapsed, 1),
            "style"    : style,
            "key"      : key,
            "tempo"    : tempo,
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        _busy = False


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """提供 MIDI 文件下载"""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="audio/midi",
                     as_attachment=False,
                     download_name=filename)


# ══════════════════════════════════════════════════
# 主页 (内联 HTML，无需 templates 目录)
# ══════════════════════════════════════════════════
@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "web", "index.html"))


@app.route("/web/<path:filename>")
def serve_web(filename):
    return send_file(os.path.join(BASE_DIR, "web", filename))


# ══════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  default=5000, type=int)
    p.add_argument("--ckpt",  default="./checkpoints/best_model.pt")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[警告] 找不到模型: {args.ckpt}")
        print("  启动演示模式（无法生成，只能播放已有文件）")
    else:
        init_model(args.ckpt)

    print(f"\n  ✓ 服务启动  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
