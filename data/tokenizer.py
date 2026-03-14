"""
data/tokenizer.py  —  REMI-style Tokenizer v4
词汇表：305 tokens  (v3: 282)

新增 token（+23）
─────────────────
  INTENSITY_0~4     全局曲风强度（BOS级，一首歌一个）
                      0=舒缓 1=轻快 2=均衡 3=充沛 4=激情
  SECTION_*         段落标记（段落切换处插入）
                      INTRO / VERSE / CHORUS / BRIDGE / OUTRO
  PITCH_CENTER_0~4  全局旋律音高中心（BOS级，一首歌一个）
                      0=极低 1=低 2=中 3=高 4=极高
  PITCH_REGISTER_0~2 per-bar 旋律音区（每小节 DENSITY 后）
                      0=低音区 1=中音区 2=高音区
  PITCH_RANGE_0~4   per-bar 旋律音高跨度（紧跟 PITCH_REGISTER）
                      0=窄(≤3) 1=小(4-6) 2=中(7-11) 3=宽(12-17) 4=极宽(18+)

完整序列结构
────────────
  [BOS] KEY TEMPO [INTENSITY_x] [PITCH_CENTER_x]
  [SECTION_INTRO]
  [BAR] [DENSITY_M][DENSITY_B][DENSITY_H] [PITCH_REGISTER_x] [PITCH_RANGE_x] [CHORD]
    [TRACK_M] BEAT NOTE_ON NOTE_DUR VEL ...
    [TRACK_B] ...
    [TRACK_H] ...
  ...更多小节...
  [SECTION_VERSE]
  [BAR] ...
  [SECTION_CHORUS]
  [BAR] ...
  [SECTION_OUTRO]
  [BAR] ...
  [EOS]
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SPECIAL_TOKENS, TRACK_TOKENS, CHORD_TYPES, KEY_LIST,
    PITCH_MIN, PITCH_MAX, MAX_DURATION, MIN_DURATION,
    VELOCITY_BINS, TICKS_PER_BAR, TEMPO_MIN, TEMPO_MAX, TEMPO_BINS,
    # v4 新增
    INTENSITY_BINS, INTENSITY_LABELS, INTENSITY_EN,
    SECTION_NAMES,
    PITCH_CENTER_BINS, PITCH_CENTER_THRESH,
    PITCH_REGISTER_BINS, PITCH_REGISTER_THRESH, PITCH_REGISTER_LABELS,
    PITCH_RANGE_BINS, PITCH_RANGE_THRESH,
)

DENSITY_BINS   = 5
DENSITY_TRACKS = ["M", "B", "H"]


class MusicTokenizer:

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        tokens: list[str] = []

        # ── 基础（与 v3 相同，顺序不变以保持兼容）────────────
        tokens += SPECIAL_TOKENS                   # 5
        tokens += TRACK_TOKENS                     # 3
        tokens.append("BAR")                       # 1
        for i in range(TICKS_PER_BAR):
            tokens.append(f"BEAT_{i}")             # 16
        for p in range(PITCH_MIN, PITCH_MAX + 1):
            tokens.append(f"NOTE_ON_{p}")          # 61
        for d in range(MIN_DURATION, MAX_DURATION + 1):
            tokens.append(f"NOTE_DUR_{d}")         # 16
        for v in range(VELOCITY_BINS):
            tokens.append(f"VELOCITY_{v}")         # 8
        for root in range(12):
            for ct in CHORD_TYPES:
                tokens.append(f"CHORD_{root}_{ct}")# 120
        for k in KEY_LIST:
            tokens.append(f"KEY_{k}")              # 24
        for t in range(TEMPO_BINS):
            tokens.append(f"TEMPO_{t}")            # 13
        # 局部密度（v3 已有）
        for track in DENSITY_TRACKS:
            for b in range(DENSITY_BINS):
                tokens.append(f"DENSITY_{track}_{b}")  # 15

        # ── v4 新增 ──────────────────────────────────────────
        # 全局曲风强度
        for b in range(INTENSITY_BINS):
            tokens.append(f"INTENSITY_{b}")         # 5

        # 段落标记
        for name in SECTION_NAMES:
            tokens.append(f"SECTION_{name}")        # 5

        # 全局旋律音高中心
        for b in range(PITCH_CENTER_BINS):
            tokens.append(f"PITCH_CENTER_{b}")      # 5

        # per-bar 旋律音区
        for b in range(PITCH_REGISTER_BINS):
            tokens.append(f"PITCH_REGISTER_{b}")    # 3

        # per-bar 旋律音高跨度
        for b in range(PITCH_RANGE_BINS):
            tokens.append(f"PITCH_RANGE_{b}")       # 5

        for i, tok in enumerate(tokens):
            self.token2id[tok] = i
            self.id2token[i]   = tok

        self.vocab_size = len(tokens)
        v3_base = 282
        print(f"[Tokenizer v4] vocab={self.vocab_size}  "
              f"(v3={v3_base} + 新增{self.vocab_size - v3_base}="
              f"强度×{INTENSITY_BINS}+段落×{len(SECTION_NAMES)}"
              f"+中心×{PITCH_CENTER_BINS}"
              f"+音区×{PITCH_REGISTER_BINS}"
              f"+跨度×{PITCH_RANGE_BINS})")

    # ── 特殊 IDs ──────────────────────────────────────────────
    @property
    def pad_id(self):      return self.token2id["[PAD]"]
    @property
    def bos_id(self):      return self.token2id["[BOS]"]
    @property
    def eos_id(self):      return self.token2id["[EOS]"]
    @property
    def bar_id(self):      return self.token2id["BAR"]
    @property
    def track_m_id(self):  return self.token2id["[TRACK_M]"]
    @property
    def track_b_id(self):  return self.token2id["[TRACK_B]"]
    @property
    def track_h_id(self):  return self.token2id["[TRACK_H]"]

    def section_id(self, name: str) -> int:
        return self.token2id.get(f"SECTION_{name}", self.token2id["SECTION_VERSE"])

    # ── 编码 ─────────────────────────────────────────────────
    def encode_pitch(self, pitch: int) -> int:
        return self.token2id[f"NOTE_ON_{max(PITCH_MIN, min(PITCH_MAX, pitch))}"]

    def encode_duration(self, dur: int) -> int:
        return self.token2id[f"NOTE_DUR_{max(MIN_DURATION, min(MAX_DURATION, dur))}"]

    def encode_velocity(self, vel: int) -> int:
        b = min(VELOCITY_BINS - 1, vel * VELOCITY_BINS // 128)
        return self.token2id[f"VELOCITY_{b}"]

    def encode_beat(self, tick: int) -> int:
        return self.token2id[f"BEAT_{tick % TICKS_PER_BAR}"]

    def encode_chord(self, root: int, ctype: str) -> int:
        ctype = ctype if ctype in CHORD_TYPES else "none"
        return self.token2id[f"CHORD_{root % 12}_{ctype}"]

    def encode_key(self, key: str) -> int:
        tok = f"KEY_{key.replace(' ', '_')}"
        return self.token2id.get(tok, self.token2id["KEY_C_maj"])

    def encode_tempo(self, bpm: float) -> int:
        bpm = max(TEMPO_MIN, min(TEMPO_MAX, bpm))
        b   = int((bpm - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN) * (TEMPO_BINS - 1))
        return self.token2id[f"TEMPO_{max(0, min(TEMPO_BINS - 1, b))}"]

    def encode_track(self, track: str) -> int:
        return self.token2id[f"[TRACK_{track}]"]

    def encode_density(self, track: str, density_bin: int) -> int:
        b = max(0, min(DENSITY_BINS - 1, density_bin))
        return self.token2id[f"DENSITY_{track}_{b}"]

    # ── v4 新增编码 ──────────────────────────────────────────
    def encode_intensity(self, intensity_bin: int) -> int:
        """全局曲风强度 0=舒缓 ~ 4=激情"""
        b = max(0, min(INTENSITY_BINS - 1, intensity_bin))
        return self.token2id[f"INTENSITY_{b}"]

    def encode_section(self, name: str) -> int:
        return self.token2id.get(f"SECTION_{name}", self.token2id["SECTION_VERSE"])

    def encode_pitch_center(self, avg_pitch: float) -> int:
        """全局旋律音高中心（MIDI pitch 平均值 → 0-4 档）"""
        thresh = PITCH_CENTER_THRESH
        for i in range(len(thresh) - 1, -1, -1):
            if avg_pitch >= thresh[i]:
                return self.token2id[f"PITCH_CENTER_{min(i, PITCH_CENTER_BINS-1)}"]
        return self.token2id["PITCH_CENTER_0"]

    def encode_pitch_register(self, avg_pitch: float) -> int:
        """per-bar 旋律音区（低/中/高）"""
        thresh = PITCH_REGISTER_THRESH
        for i in range(len(thresh) - 1, -1, -1):
            if avg_pitch >= thresh[i]:
                return self.token2id[f"PITCH_REGISTER_{min(i, PITCH_REGISTER_BINS-1)}"]
        return self.token2id["PITCH_REGISTER_0"]

    def encode_pitch_range(self, semitones: int) -> int:
        """per-bar 旋律音高跨度（半音数 → 0-4 档）"""
        thresh = PITCH_RANGE_THRESH
        for i in range(len(thresh) - 1, -1, -1):
            if semitones >= thresh[i]:
                return self.token2id[f"PITCH_RANGE_{min(i, PITCH_RANGE_BINS-1)}"]
        return self.token2id["PITCH_RANGE_0"]

    # ── 解码 ─────────────────────────────────────────────────
    def decode_token(self, tid: int) -> dict:
        tok = self.id2token.get(tid, "[PAD]")

        if tok.startswith("NOTE_ON_"):
            return {"type": "note_on",       "pitch": int(tok.split("_")[2])}
        if tok.startswith("NOTE_DUR_"):
            return {"type": "note_dur",      "dur":   int(tok.split("_")[2])}
        if tok.startswith("BEAT_"):
            return {"type": "beat",          "pos":   int(tok.split("_")[1])}
        if tok.startswith("VELOCITY_"):
            b   = int(tok.split("_")[1])
            vel = int((b + 0.5) * 128 / VELOCITY_BINS)
            return {"type": "velocity",      "vel": vel}
        if tok.startswith("CHORD_"):
            p = tok.split("_")
            return {"type": "chord",         "root": int(p[1]), "ctype": p[2]}
        if tok.startswith("KEY_"):
            return {"type": "key",           "key": "_".join(tok.split("_")[1:])}
        if tok.startswith("TEMPO_"):
            b   = int(tok.split("_")[1])
            bpm = TEMPO_MIN + b * (TEMPO_MAX - TEMPO_MIN) / (TEMPO_BINS - 1)
            return {"type": "tempo",         "bpm": bpm}
        if tok.startswith("DENSITY_"):
            p = tok.split("_")
            return {"type": "density",       "track": p[1], "bin": int(p[2])}
        # ── v4 新增 ──
        if tok.startswith("INTENSITY_"):
            b = int(tok.split("_")[1])
            return {"type": "intensity",     "bin": b,
                    "label": INTENSITY_LABELS[b], "en": INTENSITY_EN[b]}
        if tok.startswith("SECTION_"):
            name = tok.split("_", 1)[1]
            return {"type": "section",       "name": name}
        if tok.startswith("PITCH_CENTER_"):
            return {"type": "pitch_center",  "bin": int(tok.split("_")[2])}
        if tok.startswith("PITCH_REGISTER_"):
            b = int(tok.split("_")[2])
            return {"type": "pitch_register","bin": b,
                    "label": PITCH_REGISTER_LABELS[b]}
        if tok.startswith("PITCH_RANGE_"):
            return {"type": "pitch_range",   "bin": int(tok.split("_")[2])}
        if tok == "BAR":
            return {"type": "bar"}
        if tok in ("[TRACK_M]", "[TRACK_B]", "[TRACK_H]"):
            return {"type": "track",         "track": tok[7]}
        return {"type": "special",           "token": tok}

    def __len__(self): return self.vocab_size

    # ── 范围查询（训练/生成约束用）───────────────────────────
    def _range(self, prefix: str):
        start = end = None
        for i, t in self.id2token.items():
            if t.startswith(prefix):
                if start is None: start = i
                end = i
        return (start, end + 1) if start is not None else (0, 0)

    def note_on_range(self):        return self._range("NOTE_ON_")
    def note_dur_range(self):       return self._range("NOTE_DUR_")
    def beat_range(self):           return self._range("BEAT_")
    def chord_range(self):          return self._range("CHORD_")
    def key_range(self):            return self._range("KEY_")
    def tempo_range(self):          return self._range("TEMPO_")
    def density_range(self):        return self._range("DENSITY_")
    def intensity_range(self):      return self._range("INTENSITY_")
    def section_range(self):        return self._range("SECTION_")
    def pitch_center_range(self):   return self._range("PITCH_CENTER_")
    def pitch_register_range(self): return self._range("PITCH_REGISTER_")
    def pitch_range_range(self):    return self._range("PITCH_RANGE_")