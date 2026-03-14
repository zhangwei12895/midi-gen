"""
data/midi_processor.py  v4
三轨 + 全局强度 + 段落结构 + 音高中心/音区/跨度

v4 新增
───────
  全局强度 INTENSITY_0~4
    综合密度 + 速度 → 0(舒缓)~4(激情)，放在 BOS 后

  全局音高中心 PITCH_CENTER_0~4
    整首歌旋律平均音高的高低，放在 INTENSITY 后

  段落结构检测 SECTION_*
    基于密度轮廓 + 时间位置启发式，仅在切换时插入 token

  per-bar 旋律音区 PITCH_REGISTER_0~2
    低/中/高 音区，紧跟三轨密度 token

  per-bar 旋律音高跨度 PITCH_RANGE_0~4
    紧跟 PITCH_REGISTER token

完整 per-bar token 序列
────────────────────────
  [BAR] [DENSITY_M][DENSITY_B][DENSITY_H] [PITCH_REGISTER] [PITCH_RANGE] [CHORD]
    [TRACK_M] notes...
    [TRACK_B] notes...
    [TRACK_H] notes...
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretty_midi
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

from config import (
    PITCH_MIN, PITCH_MAX,
    TICKS_PER_BEAT, TICKS_PER_BAR,
    MAX_DURATION, MIN_DURATION,
    TEMPO_MIN, TEMPO_MAX,
    SECTION_NAMES,
    INTENSITY_BINS,
    PITCH_CENTER_BINS, PITCH_CENTER_THRESH,
    PITCH_REGISTER_BINS, PITCH_REGISTER_THRESH,
    PITCH_RANGE_BINS, PITCH_RANGE_THRESH,
)
from data.tokenizer import MusicTokenizer, DENSITY_BINS, DENSITY_TRACKS

NOTE_NAMES = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']

CHORD_TEMPLATES: Dict[str, List[int]] = {
    "maj" : [0,4,7],   "min" : [0,3,7],
    "dom7": [0,4,7,10],"maj7": [0,4,7,11],
    "min7": [0,3,7,10],"dim" : [0,3,6],
    "aug" : [0,4,8],   "sus2": [0,2,7], "sus4": [0,5,7],
}
KS_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

MAX_H_NOTES_PER_BAR = 8

# 局部密度阈值（每小节音符数量）
DENSITY_THRESHOLDS = {
    "M": [0, 2,  5,  9, 14],
    "B": [0, 1,  3,  6,  9],
    "H": [0, 2,  6, 12, 18],
}

# 全局强度（综合密度得分阈值）
# 综合密度 = M*0.5 + B*0.3 + H*0.2（每小节加权平均音符数）
INTENSITY_THRESHOLDS = [0, 1.5, 3.5, 6.0, 10.0]


# ────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────

def _density_bin(note_count: int, track: str) -> int:
    thr = DENSITY_THRESHOLDS[track]
    for i in range(len(thr) - 1, -1, -1):
        if note_count >= thr[i]:
            return min(i, DENSITY_BINS - 1)
    return 0


def _smooth(values: List[float], w: int = 2) -> List[float]:
    out = []
    for i in range(len(values)):
        sl = values[max(0, i - w): i + w + 1]
        out.append(sum(sl) / len(sl))
    return out


def _compute_intensity(m_grid, b_grid, h_grid, total_bars: int, tempo: float) -> int:
    """全局曲风强度 0=舒缓 ~ 4=激情"""
    if total_bars == 0:
        return 2
    m_npb = len(m_grid) / total_bars
    b_npb = len(b_grid) / max(1, total_bars)
    h_npb = len(h_grid) / max(1, total_bars)
    density_score = m_npb * 0.5 + b_npb * 0.3 + h_npb * 0.2
    # 速度修正（高 BPM → 更激情）
    tempo_factor = max(-0.5, min(0.5, (tempo - 80) / 80.0))
    combined = density_score * (1.0 + tempo_factor * 0.3)
    for i in range(len(INTENSITY_THRESHOLDS) - 1, -1, -1):
        if combined >= INTENSITY_THRESHOLDS[i]:
            return min(i, INTENSITY_BINS - 1)
    return 0


def _compute_pitch_center(m_grid) -> float:
    """旋律轨全局平均音高"""
    if not m_grid:
        return 60.0
    return float(np.mean([n["pitch"] for n in m_grid]))


def _detect_sections(m_grid, b_grid, h_grid, total_bars: int) -> List[str]:
    """
    基于密度轮廓的段落检测。
    返回长度 = total_bars 的段落名称列表。

    策略
    ────
    1. 每小节综合密度得分 = M×0.5 + B×0.3 + H×0.2
    2. 高斯平滑（窗口=2）
    3. 时间位置 + 密度分位数判断：
       前 15%  → INTRO（无论密度）
       后 12%  → OUTRO
       密度≥p70 → CHORUS
       密度≤p25 且位于 35-75% → BRIDGE（可选）
       其余     → VERSE
    4. 合并长度 < 2 的零散段落
    """
    if total_bars < 4:
        return ["VERSE"] * total_bars

    m_bar = defaultdict(list); [m_bar[n["bar"]].append(n) for n in m_grid]
    b_bar = defaultdict(list); [b_bar[n["bar"]].append(n) for n in b_grid]
    h_bar = defaultdict(list); [h_bar[n["bar"]].append(n) for n in h_grid]

    raw = []
    for i in range(total_bars):
        raw.append(len(m_bar[i]) * 0.5 + len(b_bar[i]) * 0.3 + len(h_bar[i]) * 0.2)

    sm  = _smooth(raw, w=2)
    srt = sorted(sm)
    N   = total_bars
    p25 = srt[max(0, int(0.25 * N))]
    p70 = srt[max(0, int(0.70 * N))]

    sections: List[str] = []
    for i, d in enumerate(sm):
        pos = i / N
        if   pos < 0.15:                               sections.append("INTRO")
        elif pos >= 0.88:                              sections.append("OUTRO")
        elif d >= p70:                                 sections.append("CHORUS")
        elif d <= p25 and 0.35 < pos < 0.75:          sections.append("BRIDGE")
        else:                                          sections.append("VERSE")

    # 合并长度 < 2 的孤立段落
    result = list(sections)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            j = i
            while j < len(result) and result[j] == result[i]:
                j += 1
            if (j - i) < 2 and i > 0:
                for k in range(i, j):
                    result[k] = result[i - 1]
                changed = True
            i = j
    return result


# ════════════════════════════════════════════════
class MidiProcessor:

    def __init__(self, tokenizer: MusicTokenizer):
        self.tok = tokenizer

    # ─── 主入口 ─────────────────────────────────
    def process_file(self, path: str) -> Optional[List[int]]:
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception:
            return None

        tempo   = self._get_tempo(pm)
        key_str = self._detect_key_global(pm)
        m_notes, b_notes, h_notes = self._split_tracks(pm)
        if len(m_notes) < 8:
            return None

        m_grid = self._quantize(m_notes, tempo)
        b_grid = self._quantize(b_notes, tempo) if b_notes else []
        h_grid = self._quantize(h_notes, tempo) if h_notes else []
        if not m_grid:
            return None

        # 消除前置静音
        off = m_grid[0]["bar"]
        if off > 0:
            for n in m_grid: n["bar"] -= off
            for n in b_grid: n["bar"] -= off
            for n in h_grid: n["bar"] -= off
        b_grid = [n for n in b_grid if n["bar"] >= 0]
        h_grid = [n for n in h_grid if n["bar"] >= 0]

        chords = self._detect_chords(m_grid + b_grid + h_grid)
        total_bars = max(
            m_grid[-1]["bar"] if m_grid else 0,
            b_grid[-1]["bar"] if b_grid else 0,
            h_grid[-1]["bar"] if h_grid else 0,
        ) + 1

        intensity    = _compute_intensity(m_grid, b_grid, h_grid, total_bars, tempo)
        avg_pitch    = _compute_pitch_center(m_grid)
        sections     = _detect_sections(m_grid, b_grid, h_grid, total_bars)

        return self._to_tokens(
            m_grid, b_grid, h_grid, chords,
            key_str, tempo, intensity, avg_pitch, sections
        )

    # ─── 三轨分离 ───────────────────────────────
    def _split_tracks(self, pm):
        non_drum = [(inst, inst.notes)
                    for inst in pm.instruments
                    if not inst.is_drum and inst.notes]
        if not non_drum:
            return [], [], []

        scored = []
        for inst, notes in non_drum:
            pitches  = [n.pitch for n in notes]
            avg_p    = float(np.mean(pitches))
            dur      = pm.get_end_time()
            density  = len(notes) / max(dur, 1.0)
            p_range  = max(pitches) - min(pitches)
            nl       = inst.name.lower() if inst.name else ""
            is_bass  = (avg_p < 50 or any(w in nl for w in ["bass","bss","low"]))
            is_mel   = avg_p > 55 and not is_bass
            scored.append({
                "inst": inst, "notes": notes,
                "avg_p": avg_p, "density": density,
                "p_range": p_range, "is_bass": is_bass, "is_mel": is_mel,
                "score": density * 0.5 + p_range * 0.5,
            })

        bass_cands = sorted([s for s in scored if s["is_bass"]],
                            key=lambda x: x["density"], reverse=True)
        mel_cands  = sorted([s for s in scored if not s["is_bass"]],
                            key=lambda x: x["score"], reverse=True)
        bass_inst  = bass_cands[0] if bass_cands else None
        mel_inst   = mel_cands[0]  if mel_cands  else None
        if mel_inst is None and bass_inst is None:
            mel_inst = scored[0]

        used = set()
        if mel_inst:  used.add(id(mel_inst["inst"]))
        if bass_inst: used.add(id(bass_inst["inst"]))
        h_all = [n for s in scored if id(s["inst"]) not in used for n in s["notes"]]

        m = sorted(mel_inst["notes"],  key=lambda n: n.start) if mel_inst  else []
        b = sorted(bass_inst["notes"], key=lambda n: n.start) if bass_inst else []
        h = sorted(h_all,              key=lambda n: n.start)
        return m, b, h

    # ─── 辅助 ───────────────────────────────────
    def _get_tempo(self, pm) -> float:
        _, tempos = pm.get_tempo_changes()
        bpm = float(np.median(tempos)) if len(tempos) else 120.0
        return max(TEMPO_MIN, min(TEMPO_MAX, bpm))

    def _detect_key_global(self, pm) -> str:
        notes = [n for inst in pm.instruments
                 if not inst.is_drum for n in inst.notes]
        return self._detect_key(notes)

    def _detect_key(self, notes) -> str:
        pc = np.zeros(12)
        for n in notes: pc[n.pitch % 12] += max(0., n.end - n.start)
        if pc.sum() < 1e-6: return "C_maj"
        best, key = -np.inf, "C_maj"
        for r in range(12):
            for prof, suf in [(KS_MAJ, "_maj"), (KS_MIN, "_min")]:
                s = float(np.corrcoef(pc, np.roll(prof, r))[0, 1])
                if s > best: best, key = s, f"{NOTE_NAMES[r]}{suf}"
        return key

    def _quantize(self, notes, tempo: float) -> List[Dict]:
        spt = 60.0 / (tempo * TICKS_PER_BEAT)
        out = []
        for n in notes:
            st  = round(n.start / spt)
            et  = round(n.end   / spt)
            dur = max(MIN_DURATION, min(MAX_DURATION, et - st))
            p   = max(PITCH_MIN, min(PITCH_MAX, n.pitch))
            out.append({
                "bar": st // TICKS_PER_BAR, "beat": st % TICKS_PER_BAR,
                "pitch": p, "duration": dur, "velocity": n.velocity,
            })
        out.sort(key=lambda x: (x["bar"], x["beat"], x["pitch"]))
        return out

    def _detect_chords(self, grid) -> Dict[int, Tuple[int, str]]:
        bar_pcs = defaultdict(list)
        for n in grid: bar_pcs[n["bar"]].append(n["pitch"] % 12)
        result = {}
        for bar, pcs in bar_pcs.items():
            if pcs: result[bar] = self._best_chord(Counter(pcs))
        return result

    def _best_chord(self, pc_count: Counter) -> Tuple[int, str]:
        total = sum(pc_count.values())
        best, chord = -1.0, (0, "none")
        for root in range(12):
            for name, ivs in CHORD_TEMPLATES.items():
                s = sum(pc_count.get((root + i) % 12, 0) for i in ivs) / total
                if s > best: best, chord = s, (root, name)
        return chord if best > 0.25 else (0, "none")

    # ─── Token 序列生成（v4 核心）───────────────
    def _to_tokens(
        self,
        m_grid   : List[Dict],
        b_grid   : List[Dict],
        h_grid   : List[Dict],
        chords   : Dict[int, Tuple[int, str]],
        key_str  : str,
        tempo    : float,
        intensity: int,
        avg_pitch: float,
        sections : List[str],
    ) -> List[int]:
        tok = self.tok

        # ── 序列头 ───────────────────────────────
        tokens = [
            tok.bos_id,
            tok.encode_key(key_str),
            tok.encode_tempo(tempo),
            tok.encode_intensity(intensity),
            tok.encode_pitch_center(avg_pitch),   # 全局音高中心
        ]

        def by_bar(grid):
            d = defaultdict(list)
            for n in grid: d[n["bar"]].append(n)
            return d

        m_bar = by_bar(m_grid)
        b_bar = by_bar(b_grid)
        h_bar = by_bar(h_grid)

        max_bar = max(
            m_grid[-1]["bar"] if m_grid else 0,
            b_grid[-1]["bar"] if b_grid else 0,
            h_grid[-1]["bar"] if h_grid else 0,
        )

        cur_section = None  # 追踪当前段落，仅在切换时插入 SECTION token

        for bar_idx in range(max_bar + 1):
            m_notes = m_bar.get(bar_idx, [])
            b_notes = b_bar.get(bar_idx, [])
            h_notes = h_bar.get(bar_idx, [])
            if not m_notes and not b_notes and not h_notes:
                continue

            # ── 段落切换时插入 SECTION token ──────
            sec = sections[bar_idx] if bar_idx < len(sections) else "VERSE"
            if sec != cur_section:
                tokens.append(tok.encode_section(sec))
                cur_section = sec

            # ── BAR ───────────────────────────────
            tokens.append(tok.bar_id)

            # ── 三轨密度 ──────────────────────────
            tokens.append(tok.encode_density("M", _density_bin(len(m_notes), "M")))
            tokens.append(tok.encode_density("B", _density_bin(len(b_notes), "B")))
            tokens.append(tok.encode_density("H", _density_bin(len(h_notes), "H")))

            # ── per-bar 旋律音区 + 音高跨度 ────────
            if m_notes:
                pitches  = [n["pitch"] for n in m_notes]
                avg_p    = float(np.mean(pitches))
                span     = max(pitches) - min(pitches)
                tokens.append(tok.encode_pitch_register(avg_p))
                tokens.append(tok.encode_pitch_range(span))
            else:
                tokens.append(tok.encode_pitch_register(60.0))  # 中音区默认
                tokens.append(tok.encode_pitch_range(0))

            # ── 和弦 ──────────────────────────────
            if bar_idx in chords:
                r, ct = chords[bar_idx]
                tokens.append(tok.encode_chord(r, ct))

            # ── 三轨音符 ──────────────────────────
            if m_notes:
                tokens += self._notes_to_tokens(m_notes, tok.track_m_id, tok)
            if b_notes:
                tokens += self._notes_to_tokens(b_notes, tok.track_b_id, tok)
            if h_notes:
                h_top = sorted(h_notes, key=lambda n: -n["velocity"])
                h_top = sorted(h_top[:MAX_H_NOTES_PER_BAR],
                               key=lambda n: (n["beat"], n["pitch"]))
                tokens += self._notes_to_tokens(h_top, tok.track_h_id, tok)

        tokens.append(tok.eos_id)
        return tokens

    def _notes_to_tokens(self, notes, track_id, tok) -> List[int]:
        out = [track_id]
        prev_beat = -1
        for n in notes:
            if n["beat"] != prev_beat:
                out.append(tok.encode_beat(n["beat"]))
                prev_beat = n["beat"]
            out.append(tok.encode_pitch(n["pitch"]))
            out.append(tok.encode_duration(n["duration"]))
            out.append(tok.encode_velocity(n["velocity"]))
        return out

    # ─── Token → MIDI ───────────────────────────
    def tokens_to_midi(
        self,
        token_ids  : List[int],
        tempo      : float = 120.0,
        model_name : str   = "",     # 写入 MIDI 文本事件
    ):
        import pretty_midi as _pm
        tok = self.tok
        pm  = _pm.PrettyMIDI(initial_tempo=tempo)

        inst_m = _pm.Instrument(program=80, name="Melody")
        inst_b = _pm.Instrument(program=33, name="Bass")
        inst_h = _pm.Instrument(program=48, name="Harmony")

        spt             = 60.0 / (tempo * TICKS_PER_BEAT)
        cur_bar         = -1
        cur_beat        = 0
        cur_track       = "M"
        cur_vel         = 80
        pending_p       = None
        cur_inst        = inst_m
        _bar_had_notes  = False
        track_map       = {"M": inst_m, "B": inst_b, "H": inst_h}

        for tid in token_ids:
            info = tok.decode_token(tid)
            t    = info["type"]

            if t == "bar":
                if cur_bar >= 0 and not _bar_had_notes:
                    pass
                else:
                    cur_bar += 1
                cur_beat       = 0
                pending_p      = None
                _bar_had_notes = False
            elif t == "track":
                cur_track = info["track"]
                cur_inst  = track_map.get(cur_track, inst_m)
                pending_p = None
            elif t == "beat":
                cur_beat = info["pos"]
            elif t == "note_on":
                pending_p = info["pitch"]
            elif t == "note_dur" and pending_p is not None and cur_bar >= 0:
                tick0 = cur_bar * TICKS_PER_BAR + cur_beat
                tick1 = tick0 + info["dur"]
                t0 = tick0 * spt
                t1 = max(t0 + 0.04, tick1 * spt)
                cur_inst.notes.append(_pm.Note(
                    velocity=max(1, min(127, cur_vel)),
                    pitch   =max(0, min(127, pending_p)),
                    start=t0, end=t1,
                ))
                pending_p      = None
                _bar_had_notes = True
            elif t == "velocity":
                cur_vel = info["vel"]
            elif t == "tempo":
                tempo = info["bpm"]
                spt   = 60.0 / (tempo * TICKS_PER_BEAT)
            # section / intensity / pitch_* / density → 解码时跳过

        # 写入模型信息到 MIDI 文本事件
        if model_name:
            pm.text_events.append(_pm.Text(time=0.0, text=f"model={model_name}"))

        for inst in [inst_m, inst_b, inst_h]:
            inst.notes.sort(key=lambda n: n.start)
            if inst.notes:
                pm.instruments.append(inst)
        return pm


# ════════════════════════════════════════════════
# 批量处理
# ════════════════════════════════════════════════
def process_midi_directory(
    midi_dir  : str,
    tokenizer : MusicTokenizer,
) -> List[List[int]]:
    proc  = MidiProcessor(tokenizer)
    files = sorted(
        f for f in os.listdir(midi_dir)
        if f.lower().endswith((".mid", ".midi"))
    )
    if not files:
        raise FileNotFoundError(f"未找到 MIDI 文件: {midi_dir}")

    print(f"  发现 {len(files)} 个 MIDI 文件")
    ok, skip = [], 0

    intensity_counts = [0] * 5
    section_counts   = {s: 0 for s in SECTION_NAMES}

    for i, fname in enumerate(files):
        seq = proc.process_file(os.path.join(midi_dir, fname))
        if seq and len(seq) >= 32:
            ok.append(seq)
            for t in seq[:10]:
                info = tokenizer.decode_token(t)
                if info["type"] == "intensity":
                    intensity_counts[info["bin"]] += 1
                    break
            for t in seq:
                info = tokenizer.decode_token(t)
                if info["type"] == "section":
                    name = info["name"]
                    section_counts[name] = section_counts.get(name, 0) + 1
        else:
            skip += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  [{i+1:>3}/{len(files)}] 有效 {len(ok)}  跳过 {skip}")

    total = max(1, len(ok))
    print(f"\n  ✓ {len(ok)} 首有效歌曲")
    print(f"  强度分布(舒缓→激情): {[f'{c/total*100:.0f}%' for c in intensity_counts]}")
    print(f"  段落统计: " + "  ".join(f"{k}={v}" for k, v in section_counts.items()))
    return ok