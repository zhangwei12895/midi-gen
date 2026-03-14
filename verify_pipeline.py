"""
verify_pipeline.py  —  训练数据管道验证工具

功能
────
  从训练数据中取出真实的 token 序列，
  直接用 tokens_to_midi() 还原成 MIDI 文件。
  不经过任何模型——100% 还原原始训练数据。

  输出的 MIDI 与真实训练数据完全一致，可用于确认：
    ✓ 三轨分离是否正确（旋律/贝斯/和声）
    ✓ 音高/节奏/速度是否正常
    ✓ 局部密度 token 是否随每小节变化（韵律性）
    ✓ 空白片段是否消除
    ✓ 整体音乐质量是否符合训练预期

用法
────
  python verify_pipeline.py --midi_dir ./midi_data
  python verify_pipeline.py --midi_dir ./midi_data --n 5 --out ./verify_out

输出
────
  verify_out/
    verify_01_raw.mid         第1首：完整 token 序列还原
    verify_02_raw.mid
    ...
    verify_pipeline_report.txt  详细报告（token 统计 + 密度分布）
"""

from __future__ import annotations
import os, sys, argparse, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pretty_midi
from collections import Counter, defaultdict

from data.tokenizer      import MusicTokenizer
from data.midi_processor import MidiProcessor, process_midi_directory

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_out")


# ══════════════════════════════════════════════════════════════
# 分析 token 序列
# ══════════════════════════════════════════════════════════════
def analyze_tokens(tokens: list, tok: MusicTokenizer) -> dict:
    """统计 token 序列的关键属性"""
    stats = {
        "length": len(tokens),
        "bars": 0,
        "notes_M": 0, "notes_B": 0, "notes_H": 0,
        "density_M": [], "density_B": [], "density_H": [],
        "pitches": [],
        "durations": [],
        "token_types": Counter(),
    }
    cur_track = "M"
    for t in tokens:
        info = tok.decode_token(t)
        tp   = info["type"]
        stats["token_types"][tp] += 1

        if tp == "bar":
            stats["bars"] += 1
        elif tp == "track":
            cur_track = info["track"]
        elif tp == "note_on":
            stats[f"notes_{cur_track}"] += 1
            stats["pitches"].append(info["pitch"])
        elif tp == "note_dur":
            stats["durations"].append(info["dur"])
        elif tp == "density":
            track = info["track"]
            stats[f"density_{track}"].append(info["bin"])

    return stats


def density_bar_str(bins: list) -> str:
    """把密度列表渲染成字符可视化"""
    if not bins: return "(无)"
    labels = "░▒▓█▉"
    return "".join(labels[min(4, b)] for b in bins[:64])


def print_report(seqs: list, tok: MusicTokenizer, report_path: str):
    lines = []
    sep = "═" * 62

    lines.append(sep)
    lines.append("  8-BIT MUSIC STUDIO  —  训练数据管道验证报告")
    lines.append(sep)
    lines.append(f"  共验证: {len(seqs)} 首歌曲\n")

    total_notes = 0
    all_density_M, all_density_B, all_density_H = [], [], []

    for i, (fname, tokens) in enumerate(seqs):
        st = analyze_tokens(tokens, tok)
        total_notes += st["notes_M"] + st["notes_B"] + st["notes_H"]
        all_density_M += st["density_M"]
        all_density_B += st["density_B"]
        all_density_H += st["density_H"]

        pitch_range = (min(st["pitches"]), max(st["pitches"])) if st["pitches"] else (0,0)
        dur_avg     = sum(st["durations"]) / len(st["durations"]) if st["durations"] else 0

        lines.append(f"  ── [{i+1:02d}] {fname}")
        lines.append(f"      tokens={st['length']}  bars={st['bars']}"
                     f"  音符=(M:{st['notes_M']} B:{st['notes_B']} H:{st['notes_H']})")
        lines.append(f"      音高范围={pitch_range[0]}-{pitch_range[1]}"
                     f"  平均时值={dur_avg:.1f}ticks")

        # 局部密度可视化（每小节密度）
        lines.append(f"      旋律密度变化: {density_bar_str(st['density_M'])}")
        lines.append(f"      贝斯密度变化: {density_bar_str(st['density_B'])}")
        lines.append(f"      和声密度变化: {density_bar_str(st['density_H'])}")
        lines.append("")

    lines.append(sep)
    lines.append("  汇总统计")
    lines.append(sep)
    lines.append(f"  总音符数: {total_notes}")

    def avg_density(bins):
        return sum(bins) / len(bins) if bins else 0

    def density_hist(bins):
        if not bins: return "无数据"
        c = Counter(bins)
        total = len(bins)
        return "  ".join(f"{k}={'█'*int(c[k]/total*20)}({c[k]/total*100:.0f}%)"
                         for k in sorted(c))

    lines.append(f"\n  旋律密度分布 (0=稀疏→4=极密):")
    lines.append(f"  {density_hist(all_density_M)}")
    lines.append(f"  平均={avg_density(all_density_M):.2f}\n")
    lines.append(f"  贝斯密度分布:")
    lines.append(f"  {density_hist(all_density_B)}")
    lines.append(f"  平均={avg_density(all_density_B):.2f}\n")
    lines.append(f"  和声密度分布:")
    lines.append(f"  {density_hist(all_density_H)}")
    lines.append(f"  平均={avg_density(all_density_H):.2f}\n")

    lines.append(sep)
    lines.append("  密度说明（局部每小节）")
    lines.append("  ─────────────────────────────")
    lines.append("  ░ = 0 稀疏（0-1个音符/小节）  ← 间奏、过渡段")
    lines.append("  ▒ = 1 轻   （2-4个）")
    lines.append("  ▓ = 2 中等 （5-8个）          ← 主歌")
    lines.append("  █ = 3 密   （9-13个）")
    lines.append("  ▉ = 4 极密 （14+个）           ← 副歌、高潮")
    lines.append("")
    lines.append("  密度随小节变化 → 音乐有强弱起伏韵律 ✓")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  报告已保存: {report_path}")


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="训练数据管道验证")
    p.add_argument("--midi_dir", default="./midi_data", help="训练 MIDI 目录")
    p.add_argument("--n",        type=int, default=3,   help="验证几首（默认3）")
    p.add_argument("--out",      default=OUT_DIR,       help="输出目录")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print()
    print("╔" + "═"*56 + "╗")
    print("║   训练数据管道验证  —  tokens → MIDI 往返测试" + " "*8 + "║")
    print("╚" + "═"*56 + "╝\n")

    tok  = MusicTokenizer()
    proc = MidiProcessor(tok)

    # 扫描 MIDI 文件
    import glob
    midi_files = sorted(
        f for f in glob.glob(os.path.join(args.midi_dir, "*.mid")) +
                   glob.glob(os.path.join(args.midi_dir, "*.midi"))
    )
    if not midi_files:
        print(f"  ✗ 未找到 MIDI 文件: {args.midi_dir}")
        return

    print(f"  发现 {len(midi_files)} 个 MIDI 文件，处理前 {args.n} 首...\n")

    results = []
    for midi_path in midi_files[:args.n * 3]:  # 多处理一些，过滤失败的
        fname = os.path.basename(midi_path)
        tokens = proc.process_file(midi_path)
        if tokens is None or len(tokens) < 32:
            print(f"  ✗ 跳过 {fname}（序列太短或解析失败）")
            continue

        # Token → MIDI 还原
        pm = proc.tokens_to_midi(tokens)
        n_inst = len(pm.instruments)
        dur    = pm.get_end_time()

        # 统计密度变化（确认是局部的）
        st = analyze_tokens(tokens, tok)
        density_varies_M = len(set(st["density_M"])) > 1
        density_varies_B = len(set(st["density_B"])) > 1

        out_path = os.path.join(args.out, f"verify_{len(results)+1:02d}_{fname}")
        pm.write(out_path)

        results.append((fname, tokens))
        status_d = "✓密度变化" if density_varies_M else "⚠密度恒定"
        print(f"  ✓ {fname}")
        print(f"     tokens={len(tokens)}  小节={st['bars']}"
              f"  轨道={n_inst}  时长={dur:.1f}s  {status_d}")
        print(f"     音符: M={st['notes_M']} B={st['notes_B']} H={st['notes_H']}")
        print(f"     已保存: {os.path.basename(out_path)}\n")

        if len(results) >= args.n:
            break

    if not results:
        print("  ✗ 没有成功处理任何文件，请检查 midi_data 目录")
        return

    # 报告
    report_path = os.path.join(args.out, "verify_pipeline_report.txt")
    print_report(results, tok, report_path)
    print(f"\n  ✓ 验证完成！MIDI 文件保存在: {args.out}")
    print(f"  用 studio.py 或任何 MIDI 播放器打开验证文件听听看")


if __name__ == "__main__":
    main()
