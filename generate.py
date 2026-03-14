"""
generate.py  v4  —  完整结构整首歌生成

v4 新特性
──────────
  --style / -s  曲风强度  0=舒缓 1=轻快 2=均衡 3=充沛 4=激情
                也可用名称: calm / light / medium / energetic / intense
                对应生成不同段落复杂度的完整歌曲

  生成终止机制
    ★ 不再由 --bars 决定长度
    ★ 模型自主生成 EOS（已学习完整歌曲结构）
    ★ 安全限制：max_gen_tokens（默认 2048）

  生成的 MIDI 包含
    · text_event: model=<模型名称>  step=<训练步数>  style=<曲风>
    · 三轨：旋律 / 贝斯 / 和声

  输出文件名格式
    gen_YYYYMMDD_HHMMSS_s<style>_t<temp>.mid
    例：gen_20250312_143022_s2_t090.mid  （均衡曲风，temperature=0.90）
"""

from __future__ import annotations
import os, sys, time, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm
from typing import List, Tuple

from config import (
    GEN_CONFIG, MODEL_CONFIG, OUTPUT_DIR,
    INTENSITY_LABELS, INTENSITY_EN, INTENSITY_STRUCTURE,
    SECTION_NAMES,
)
from data.tokenizer      import MusicTokenizer
from data.midi_processor import MidiProcessor
from model.transformer   import MusicTransformer
from model.music_theory  import (
    MusicTheoryConstraints, GenerationContext, sample_with_constraints,
)

# 曲风名称 → bin
STYLE_NAME_MAP: dict[str, int] = {
    "calm": 0, "light": 1, "medium": 2, "energetic": 3, "intense": 4,
    "舒缓": 0, "轻快":  1, "均衡":   2, "充沛":      3, "激情":   4,
}


# ══════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════
def load_model(
    ckpt_path : str,
    device    : torch.device,
) -> Tuple[MusicTransformer, MusicTokenizer, dict]:
    tok = MusicTokenizer()
    ck  = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("model_config", MODEL_CONFIG)
    m   = MusicTransformer(vocab_size=tok.vocab_size, config=cfg).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()

    meta = {
        "step"      : ck.get("step", 0),
        "val_loss"  : ck.get("val_loss", 0.0),
        "n_params"  : ck.get("n_params", 0),
        "param_str" : ck.get("param_str", ""),
        "train_time": ck.get("train_time", ""),
        "model_name": ck.get("model_name", os.path.basename(ckpt_path)),
    }
    print(f"  模型  {meta['model_name']}")
    print(f"  参数  {meta['param_str'] or meta['n_params']:}")
    print(f"  step={meta['step']}  val_loss={meta['val_loss']:.4f}")
    if meta["train_time"]:
        print(f"  训练完成时间  {meta['train_time']}")
    return m, tok, meta


# ══════════════════════════════════════════════
# 生成完整一首歌（无 KV cache，全序列重算）
# ══════════════════════════════════════════════
@torch.no_grad()
def generate_song(
    model      : MusicTransformer,
    tok        : MusicTokenizer,
    device     : torch.device,
    style      : int   = GEN_CONFIG["default_intensity"],
    key_str    : str   = GEN_CONFIG["default_key"],
    tempo      : float = GEN_CONFIG["default_tempo"],
    temperature: float = GEN_CONFIG["temperature"],
    top_k      : int   = GEN_CONFIG["top_k"],
    top_p      : float = GEN_CONFIG["top_p"],
    rep_penalty: float = GEN_CONFIG["repetition_penalty"],
) -> Tuple[List[int], List[str]]:
    """
    自回归生成完整一首歌。

    终止方式
    ────────
    1. 模型输出 EOS（主要方式，已学习段落结构）
    2. 超过 max_gen_tokens 安全限制

    返回
    ────
    tokens    : 完整 token 序列
    sections_hit : 生成过程中遇到的段落列表（用于显示）
    """
    MIN_BARS  = GEN_CONFIG["min_bars_before_eos"]
    MAX_TOKENS= GEN_CONFIG["max_gen_tokens"]
    MAX_CTX   = 1536

    cons = MusicTheoryConstraints(tok)
    ctx  = GenerationContext(tok)

    # ── 生成 prompt（BOS + 调性 + 速度 + 强度 + 默认音高中心）
    # 音高中心默认 = 中档（bin 2），可按需扩展为参数
    pitch_center_id = tok.encode_pitch_center(62.0)
    prompt = [
        tok.bos_id,
        tok.encode_key(key_str),
        tok.encode_tempo(tempo),
        tok.encode_intensity(style),
        pitch_center_id,
    ]
    for t in prompt:
        ctx.update(t)

    all_tokens   : List[int] = list(prompt)
    sections_hit : List[str] = []

    # 进度显示
    expected_struct = INTENSITY_STRUCTURE.get(style, ["INTRO","VERSE","CHORUS","OUTRO"])
    style_label     = INTENSITY_LABELS[style]
    print(f"\n  曲风: {style_label}（{INTENSITY_EN[style]}）"
          f"  预期结构: {' → '.join(expected_struct)}")

    pbar = tqdm(
        total=MAX_TOKENS,
        desc="  生成",
        bar_format="  {desc} |{bar:30}| {n_fmt}tok  [{elapsed}]",
        unit="tok",
    )

    last_bar        = 0
    bar_note_count  = 0
    last_was_bar    = False
    consec_empty    = 0
    cur_section     = None

    for step in range(MAX_TOKENS):
        # 上下文截断
        if len(all_tokens) > MAX_CTX:
            recent  = all_tokens[-MAX_CTX:]
            bar_pos = next((i for i, t in enumerate(recent) if t == tok.bar_id), 0)
            ctx_ids = recent[bar_pos:]
        else:
            ctx_ids = all_tokens

        ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        logits, _, _ = model(ids, use_cache=False)
        logit_1d = logits[0, -1, :].clone()

        # EOS 保护：至少 MIN_BARS 小节
        if ctx.total_bars < MIN_BARS:
            logit_1d[tok.eos_id] = float("-inf")

        # 连续空 BAR 保护
        if last_was_bar and bar_note_count == 0 and consec_empty >= 1:
            logit_1d[tok.bar_id] -= 3.0
            logit_1d[tok.track_m_id] += 2.0

        nxt = sample_with_constraints(
            logit_1d, cons, ctx,
            temperature=temperature, top_k=top_k,
            top_p=top_p, rep_penalty=rep_penalty,
        )

        # 跟踪小节
        if nxt == tok.bar_id:
            if last_was_bar and bar_note_count == 0:
                consec_empty += 1
            else:
                consec_empty = 0
            last_was_bar   = True
            bar_note_count = 0
        else:
            last_was_bar = False
            info = tok.decode_token(nxt)
            if info["type"] == "note_on":
                bar_note_count += 1
            elif info["type"] == "section":
                sec = info["name"]
                if sec != cur_section:
                    cur_section = sec
                    sections_hit.append(sec)
                    pbar.set_description(f"  生成 [{sec}]")

        ctx.update(nxt)
        all_tokens.append(nxt)
        pbar.update(1)

        if nxt == tok.eos_id:
            break

    pbar.n = step + 1; pbar.refresh(); pbar.close()
    return all_tokens, sections_hit


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="8-bit 音乐生成器 v4（完整结构整首歌）")
    p.add_argument("--checkpoint", "-c",
                   default=os.path.join("./checkpoints", "best_model.pt"),
                   help="检查点路径")
    p.add_argument("--output", "-o", default=None,
                   help="输出 MIDI 路径（不填自动命名）")
    p.add_argument("--style", "-s", default=None,
                   help="曲风强度 0~4 或名称: calm/light/medium/energetic/intense"
                        "  中文: 舒缓/轻快/均衡/充沛/激情")
    p.add_argument("--key",    default=GEN_CONFIG["default_key"])
    p.add_argument("--tempo",  default=GEN_CONFIG["default_tempo"], type=float)
    p.add_argument("--temp",   default=GEN_CONFIG["temperature"],   type=float,
                   help="采样温度 (0.3=保守 ~ 1.5=混乱，推荐 0.8~1.0)")
    p.add_argument("--topk",   default=GEN_CONFIG["top_k"],         type=int)
    p.add_argument("--topp",   default=GEN_CONFIG["top_p"],         type=float)
    args = p.parse_args()

    # ── 解析曲风 ────────────────────────────────
    if args.style is None:
        style = GEN_CONFIG["default_intensity"]
    elif args.style.isdigit():
        style = max(0, min(4, int(args.style)))
    else:
        style = STYLE_NAME_MAP.get(args.style.strip(), GEN_CONFIG["default_intensity"])

    if not os.path.exists(args.checkpoint):
        print(f"  ✗ 找不到检查点: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("┌" + "─"*54 + "┐")
    print("│       8-BIT MUSIC GENERATOR  v4               │")
    print("│       完整结构整首歌 · 曲风强度控制             │")
    print("└" + "─"*54 + "┘")
    print(f"  设备    {device}")
    print(f"  曲风    {INTENSITY_LABELS[style]} ({INTENSITY_EN[style]}, bin={style})")
    print(f"  调性    {args.key}  速度 {args.tempo:.0f} BPM  温度 {args.temp:.2f}")
    print()

    model, tok, meta = load_model(args.checkpoint, device)
    proc = MidiProcessor(tok)

    # ── 生成 ────────────────────────────────────
    t0 = time.time()
    tokens, sections_hit = generate_song(
        model, tok, device,
        style      = style,
        key_str    = args.key,
        tempo      = args.tempo,
        temperature= args.temp,
        top_k      = args.topk,
        top_p      = args.topp,
    )
    elapsed = time.time() - t0

    # ── 构建模型信息字符串（写入 MIDI）──────────
    model_info = (
        f"{meta['model_name']}"
        f"  step={meta['step']}"
        f"  {meta['param_str'] or str(meta['n_params'])}"
        f"  style={INTENSITY_EN[style]}"
        f"  temp={args.temp:.2f}"
    )

    pm  = proc.tokens_to_midi(tokens, tempo=args.tempo, model_name=model_info)
    dur = pm.get_end_time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dt_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_str = f"t{args.temp:.2f}".replace(".", "")
    style_str= f"s{style}"
    out = args.output or os.path.join(
        OUTPUT_DIR, f"gen_{dt_str}_{style_str}_{temp_str}.mid"
    )
    pm.write(out)

    # ── 统计 ────────────────────────────────────
    total_bars = sum(1 for t in tokens if t == tok.bar_id)
    total_notes = sum(1 for t in tokens
                      if tok.id2token.get(t, "").startswith("NOTE_ON_"))

    print(f"\n  ✓ 生成完成")
    print(f"  曲风     {INTENSITY_LABELS[style]}（{INTENSITY_EN[style]}）")
    print(f"  结构     {' → '.join(sections_hit) if sections_hit else '未检测到段落'}")
    print(f"  小节数   {total_bars}")
    print(f"  总音符   {total_notes}")
    print(f"  时长     {dur:.1f}s  ({dur/60:.1f}min)")
    print(f"  耗时     {elapsed:.1f}s")
    print(f"  音轨     {len(pm.instruments)} 条")
    print(f"  模型     {meta['model_name']}")
    print(f"  输出     {out}")


if __name__ == "__main__":
    main()