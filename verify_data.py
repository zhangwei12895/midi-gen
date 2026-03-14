"""
verify_data.py  —  数据验证脚本

功能
────
  1. MIDI 文件扫描：统计文件数、时长分布
  2. 三轨分离验证：显示每首歌检测到的轨道情况
  3. Token 序列统计：长度分布、Token 类型占比
  4. Token → MIDI 往返验证：不经过模型，直接将处理后的 token 还原为 MIDI
  5. Mock 生成验证：用 token 序列的前半段"假装"生成，验证 decode 流程正确

用法
────
  python verify_data.py --midi_dir ./midi_data
  python verify_data.py --midi_dir ./midi_data --out_dir ./verify_output
  python verify_data.py --midi_dir ./midi_data --max_files 5
"""

from __future__ import annotations
import os, sys, argparse, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter, defaultdict
from typing import List

# ── 可选依赖 ────────────────────────────────────────────────
try:
    import pretty_midi
    HAS_PM = True
except ImportError:
    HAS_PM = False
    print("  [警告] pretty_midi 未安装，跳过 MIDI I/O 验证")

from data.tokenizer      import MusicTokenizer
from data.midi_processor import MidiProcessor


# ════════════════════════════════════════════════════════════
# 工具
# ════════════════════════════════════════════════════════════
def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def ok(msg): print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def err(msg): print(f"  ✗  {msg}")


# ════════════════════════════════════════════════════════════
# 1. MIDI 文件扫描
# ════════════════════════════════════════════════════════════
def scan_midi_files(midi_dir: str) -> List[str]:
    section("① MIDI 文件扫描")
    if not os.path.exists(midi_dir):
        err(f"目录不存在: {midi_dir}"); return []

    files = sorted(
        os.path.join(midi_dir, f) for f in os.listdir(midi_dir)
        if f.lower().endswith((".mid", ".midi"))
    )
    print(f"  目录    : {midi_dir}")
    print(f"  文件数  : {len(files)}")

    if not files:
        err("没有找到 MIDI 文件！"); return []

    if HAS_PM:
        durations, track_counts, errors = [], [], []
        for path in files[:50]:  # 只扫描前50个，避免太慢
            try:
                pm = pretty_midi.PrettyMIDI(path)
                durations.append(pm.get_end_time())
                non_drum = [i for i in pm.instruments if not i.is_drum]
                track_counts.append(len(non_drum))
            except Exception as e:
                errors.append(os.path.basename(path))

        print(f"\n  文件统计（前{min(50,len(files))}个）：")
        print(f"    时长   min={min(durations):.1f}s  "
              f"avg={np.mean(durations):.1f}s  "
              f"max={max(durations):.1f}s")
        print(f"    音轨数 min={min(track_counts)}  "
              f"avg={np.mean(track_counts):.1f}  "
              f"max={max(track_counts)}")
        if errors:
            warn(f"解析失败 {len(errors)} 个: {errors[:3]}")

    return files


# ════════════════════════════════════════════════════════════
# 2. 三轨分离验证
# ════════════════════════════════════════════════════════════
def verify_track_separation(files: List[str], max_files: int = 10):
    section("② 三轨分离验证")
    if not HAS_PM: return

    tok  = MusicTokenizer()
    proc = MidiProcessor(tok)

    sample = files[:max_files]
    print(f"  验证前 {len(sample)} 首：")
    print(f"  {'文件名':<30} {'旋律':>6} {'贝斯':>6} {'背景':>6} {'token':>6}")
    print("  " + "─"*58)

    ok_count, skip_count = 0, 0
    for path in sample:
        fname = os.path.basename(path)[:28]
        try:
            pm = pretty_midi.PrettyMIDI(path)
            m, b, h = proc._split_tracks(pm)
            tempo   = proc._get_tempo(pm)
            m_g = proc._quantize(m, tempo) if m else []
            b_g = proc._quantize(b, tempo) if b else []
            h_g = proc._quantize(h, tempo) if h else []
            seq = proc.process_file(path)
            tlen = len(seq) if seq else 0
            status = "✓" if seq and len(seq) >= 32 else "✗"
            print(f"  {status} {fname:<30} {len(m_g):>6} {len(b_g):>6}"
                  f" {len(h_g):>6} {tlen:>6}")
            if seq: ok_count += 1
            else: skip_count += 1
        except Exception as e:
            print(f"  ✗ {fname:<30} ERROR: {e}")
            skip_count += 1

    print(f"\n  有效: {ok_count}  跳过: {skip_count}")


# ════════════════════════════════════════════════════════════
# 3. Token 序列统计
# ════════════════════════════════════════════════════════════
def analyze_token_sequences(files: List[str], max_files: int = 30):
    section("③ Token 序列统计")

    tok  = MusicTokenizer()
    proc = MidiProcessor(tok)

    seqs = []
    for path in files[:max_files]:
        seq = proc.process_file(path)
        if seq and len(seq) >= 16:
            seqs.append(seq)

    if not seqs:
        err("没有有效序列"); return []

    lengths = [len(s) for s in seqs]
    print(f"  有效序列  : {len(seqs)} / {min(max_files, len(files))}")
    print(f"  序列长度  : min={min(lengths)}  "
          f"avg={int(np.mean(lengths))}  "
          f"max={max(lengths)}")

    # Token 类型分布
    all_tokens = [t for s in seqs for t in s]
    type_count = Counter()
    for t in all_tokens:
        info = tok.decode_token(t)
        type_count[info["type"]] += 1

    total = len(all_tokens)
    print(f"\n  Token 类型分布（共 {total:,} 个）：")
    for typ, cnt in type_count.most_common():
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {typ:<12} {cnt:>8,}  {pct:5.1f}%  {bar}")

    # 小节数分布
    bar_counts = []
    for s in seqs:
        bars = sum(1 for t in s if t == tok.bar_id)
        bar_counts.append(bars)
    print(f"\n  每首小节数: min={min(bar_counts)}  "
          f"avg={int(np.mean(bar_counts))}  "
          f"max={max(bar_counts)}")

    # 三轨出现率
    m_rate = sum(1 for s in seqs for t in s if t == tok.track_m_id) / len(seqs)
    b_rate = sum(1 for s in seqs for t in s if t == tok.track_b_id) / len(seqs)
    h_rate = sum(1 for s in seqs for t in s if t == tok.track_h_id) / len(seqs)
    print(f"\n  平均每首轨道 token 数：")
    print(f"    TRACK_M  {m_rate:.1f}  TRACK_B  {b_rate:.1f}  TRACK_H  {h_rate:.1f}")

    return seqs


# ════════════════════════════════════════════════════════════
# 4. Token → MIDI 往返验证
# ════════════════════════════════════════════════════════════
def verify_roundtrip(seqs: List[List[int]], out_dir: str, n: int = 3):
    section("④ Token→MIDI 往返验证（不经过模型）")
    if not HAS_PM or not seqs:
        warn("跳过（无有效序列或 pretty_midi 未安装）"); return

    os.makedirs(out_dir, exist_ok=True)
    tok  = MusicTokenizer()
    proc = MidiProcessor(tok)

    from config import GEN_CONFIG
    default_tempo = GEN_CONFIG["default_tempo"]

    for i, seq in enumerate(seqs[:n]):
        out_path = os.path.join(out_dir, f"roundtrip_{i+1}.mid")
        try:
            pm  = proc.tokens_to_midi(seq, tempo=default_tempo)
            dur = pm.get_end_time()
            n_notes = sum(len(inst.notes) for inst in pm.instruments)
            n_insts = len(pm.instruments)
            pm.write(out_path)
            ok(f"样本 {i+1}  {n_insts} 轨  {n_notes} 音符  "
               f"时长 {dur:.1f}s  → {os.path.basename(out_path)}")
        except Exception as e:
            err(f"样本 {i+1} 失败: {e}")


# ════════════════════════════════════════════════════════════
# 5. Mock 生成验证
#    取 token 序列前 30% 作为 prompt，直接将剩余 token 接上，
#    模拟"已生成"，验证 decode 流程不报错
# ════════════════════════════════════════════════════════════
def verify_mock_generation(seqs: List[List[int]], out_dir: str):
    section("⑤ Mock 生成验证（假装模型生成，验证 decode 流程）")
    if not HAS_PM or not seqs:
        warn("跳过"); return

    os.makedirs(out_dir, exist_ok=True)
    tok  = MusicTokenizer()
    proc = MidiProcessor(tok)

    from config import GEN_CONFIG
    default_tempo = GEN_CONFIG["default_tempo"]

    seq = seqs[0]   # 用第一首
    cut = max(3, len(seq) // 3)

    prompt_part = seq[:cut]
    rest_part   = seq[cut:]

    print(f"  总 tokens   : {len(seq)}")
    print(f"  Prompt 部分 : {len(prompt_part)} tokens")
    print(f"  '生成' 部分 : {len(rest_part)} tokens")

    # 直接拼接（假装是模型生成的）
    mock_generated = prompt_part + rest_part

    # 统计生成内容
    bars    = sum(1 for t in mock_generated if t == tok.bar_id)
    n_notes = sum(1 for t in mock_generated
                  if tok.id2token.get(t, "").startswith("NOTE_ON_"))
    print(f"  拼接后小节数 : {bars}")
    print(f"  拼接后音符数 : {n_notes}")

    # decode
    try:
        pm  = proc.tokens_to_midi(mock_generated, tempo=default_tempo)
        dur = pm.get_end_time()
        n_inst = len(pm.instruments)
        out_path = os.path.join(out_dir, "mock_generation.mid")
        pm.write(out_path)
        ok(f"Mock 生成 → {n_inst} 轨  {dur:.1f}s  → {os.path.basename(out_path)}")

        # 检查第一个音符的时间不是 0（验证 bar offset 修复）
        all_notes = [n for inst in pm.instruments for n in inst.notes]
        if all_notes:
            all_notes.sort(key=lambda n: n.start)
            first_t = all_notes[0].start
            if first_t > 0.5:
                warn(f"第一个音符在 {first_t:.2f}s，可能有 bar offset 问题")
            else:
                ok(f"第一个音符在 {first_t:.3f}s（bar offset 正常）")
    except Exception as e:
        err(f"Mock 生成失败: {e}")
        import traceback; traceback.print_exc()


# ════════════════════════════════════════════════════════════
# 6. Pitch Transpose 验证
# ════════════════════════════════════════════════════════════
def verify_transpose(seqs: List[List[int]]):
    section("⑥ Pitch Transpose 增强验证")
    if not seqs: return

    from data.dataset import _transpose_tokens
    tok = MusicTokenizer()

    seq = seqs[0]
    note_start, note_end = tok.note_on_range()

    orig_pitches = [tok.id2token[t].split("_")[2]
                    for t in seq if note_start <= t < note_end]
    shifted      = _transpose_tokens(seq, 3, tok)
    new_pitches  = [tok.id2token[t].split("_")[2]
                    for t in shifted if note_start <= t < note_end]

    print(f"  原始音高（前5个）: {orig_pitches[:5]}")
    print(f"  移调+3 （前5个）: {new_pitches[:5]}")

    # 验证音高确实偏移了3
    if orig_pitches and new_pitches:
        orig_first = int(orig_pitches[0])
        new_first  = int(new_pitches[0])
        expected   = orig_first + 3
        if new_first == expected:
            ok(f"Transpose 正确：{orig_first} → {new_first}")
        else:
            warn(f"Transpose 可能有误：{orig_first} → {new_first}"
                 f"（期望 {expected}，可能因 pitch clamp 导致差异）")

    # KEY token 同步验证
    orig_keys = [tok.id2token[t] for t in seq
                 if tok.id2token.get(t,"").startswith("KEY_")]
    new_keys  = [tok.id2token[t] for t in shifted
                 if tok.id2token.get(t,"").startswith("KEY_")]
    if orig_keys and new_keys:
        ok(f"KEY: {orig_keys[0]} → {new_keys[0]}")


# ════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="数据验证脚本")
    p.add_argument("--midi_dir",  default="./midi_data")
    p.add_argument("--out_dir",   default="./verify_output",
                   help="验证输出 MIDI 目录")
    p.add_argument("--max_files", type=int, default=20,
                   help="最多验证多少个文件（-1=全部）")
    args = p.parse_args()

    if args.max_files < 0:
        args.max_files = 9999

    t0 = time.time()
    print()
    print("┌" + "─"*58 + "┐")
    print("│      DATA VERIFY  —  数据管道完整性验证" + " "*18 + "│")
    print("└" + "─"*58 + "┘")

    files = scan_midi_files(args.midi_dir)
    if not files: return

    verify_track_separation(files, max_files=min(10, args.max_files))
    seqs = analyze_token_sequences(files, max_files=args.max_files)
    verify_roundtrip(seqs, args.out_dir, n=3)
    verify_mock_generation(seqs, args.out_dir)
    verify_transpose(seqs)

    section("验证完成")
    print(f"  总耗时: {time.time()-t0:.1f}s")
    if seqs:
        ok(f"数据管道正常，可以开始训练")
        print(f"\n  下一步:")
        print(f"    python train_v3.py --midi_dir {args.midi_dir} --rebuild")
    else:
        err("没有有效数据，请检查 MIDI 文件")


if __name__ == "__main__":
    main()
