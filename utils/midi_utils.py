"""
MIDI 工具集
- 8-bit 音色映射 (General MIDI → 8-bit 风格)
- MIDI 信息统计
- 批量 MIDI 文件检查
- 创建测试用 MIDI 文件
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretty_midi
import numpy as np
from typing import List, Dict

# 8-bit 风格 General MIDI 音色 (Program Number)
EIGHTBIT_PROGRAMS = {
    "lead_square"   : 80,   # Lead 1 (square)        - 主旋律
    "lead_sawtooth" : 81,   # Lead 2 (sawtooth)      - 副旋律
    "bass_synth"    : 38,   # Synth Bass 1            - 低音
    "pulse"         : 82,   # Lead 3 (calliope)       - 脉冲音
    "chiptune"      : 83,   # Lead 4 (chiff)          - 芯片音
}


def get_midi_stats(midi_dir: str) -> Dict:
    """统计 MIDI 目录的基本信息"""
    files = [f for f in os.listdir(midi_dir) if f.lower().endswith((".mid", ".midi"))]
    stats = {
        "total_files": len(files),
        "valid_files": 0,
        "total_notes": 0,
        "avg_duration_sec": 0,
        "pitch_min": 127,
        "pitch_max": 0,
        "tempo_list": [],
    }
    durations = []
    for fname in files:
        try:
            pm = pretty_midi.PrettyMIDI(os.path.join(midi_dir, fname))
            notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
            if not notes:
                continue
            stats["valid_files"] += 1
            stats["total_notes"] += len(notes)
            durations.append(pm.get_end_time())
            for n in notes:
                stats["pitch_min"] = min(stats["pitch_min"], n.pitch)
                stats["pitch_max"] = max(stats["pitch_max"], n.pitch)
            tempos = pm.get_tempo_changes()[1]
            if len(tempos) > 0:
                stats["tempo_list"].append(float(np.median(tempos)))
        except Exception:
            continue

    if durations:
        stats["avg_duration_sec"] = np.mean(durations)
    return stats


def create_demo_midi(output_path: str = "demo.mid"):
    """
    创建一个演示用的 8-bit 风格 MIDI 文件
    C 大调, 120 BPM, 使用 I-IV-V-I 和弦进行
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=EIGHTBIT_PROGRAMS["lead_square"], name="8bit Lead")

    # C 大调音阶 (MIDI 60~72)
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    spb = 0.5  # 秒/拍

    # 8 小节旋律
    melody = [
        # 小节1: C 和弦 (I)
        (60, 0.0, 0.5), (64, 0.5, 1.0), (67, 1.0, 1.5), (64, 1.5, 2.0),
        # 小节2
        (65, 2.0, 2.5), (67, 2.5, 3.0), (69, 3.0, 3.5), (67, 3.5, 4.0),
        # 小节3: F 和弦 (IV)
        (65, 4.0, 4.5), (69, 4.5, 5.0), (72, 5.0, 5.5), (69, 5.5, 6.0),
        # 小节4
        (67, 6.0, 6.5), (65, 6.5, 7.0), (64, 7.0, 7.5), (62, 7.5, 8.0),
        # 小节5: G 和弦 (V)
        (67, 8.0, 8.5), (71, 8.5, 9.0), (74, 9.0, 9.5), (71, 9.5, 10.0),
        # 小节6
        (69, 10.0, 10.5), (67, 10.5, 11.0), (65, 11.0, 11.5), (64, 11.5, 12.0),
        # 小节7: 回到 C (I)
        (64, 12.0, 12.5), (65, 12.5, 13.0), (67, 13.0, 13.5), (69, 13.5, 14.0),
        # 小节8: 终止
        (72, 14.0, 14.5), (71, 14.5, 15.0), (69, 15.0, 15.5), (60, 15.5, 16.0),
    ]

    for pitch, start, end in melody:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))

    pm.instruments.append(inst)
    pm.write(output_path)
    print(f"[Demo] 已创建演示 MIDI: {output_path}")
    return output_path


def print_midi_info(midi_path: str):
    """打印 MIDI 文件详细信息"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    tempos = pm.get_tempo_changes()
    notes_all = [n for i in pm.instruments if not i.is_drum for n in i.notes]
    print(f"{'='*50}")
    print(f"文件: {os.path.basename(midi_path)}")
    print(f"时长: {pm.get_end_time():.2f} 秒")
    print(f"轨道数: {len(pm.instruments)}")
    print(f"音符总数: {len(notes_all)}")
    if len(tempos[1]) > 0:
        print(f"速度: {tempos[1][0]:.1f} BPM")
    if notes_all:
        pitches = [n.pitch for n in notes_all]
        print(f"音高范围: {min(pitches)} ~ {max(pitches)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        create_demo_midi("./midi_data/demo.mid")
    elif len(sys.argv) > 1:
        print_midi_info(sys.argv[1])
    else:
        print("用法:")
        print("  python utils/midi_utils.py demo            # 创建演示 MIDI")
        print("  python utils/midi_utils.py <midi_file>     # 显示 MIDI 信息")