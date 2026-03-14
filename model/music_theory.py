"""
乐理约束模块
- 调性音阶约束 (调外音惩罚)
- 和弦和声约束 (和弦外音惩罚)
- 强拍/弱拍规则
- 重复音符惩罚
- 终止检测 (末尾音接近起始音)
- 8-bit 风格和弦进行规则
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

from config import (
    PITCH_MIN, PITCH_MAX, TICKS_PER_BAR, THEORY_CONFIG, GEN_CONFIG
)
from data.tokenizer import MusicTokenizer

# ── 乐理常量 ──────────────────────────────────
MAJOR_SCALE  = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE  = [0, 2, 3, 5, 7, 8, 10]
CHORD_TONES  = {
    "maj" : [0, 4, 7],
    "min" : [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim" : [0, 3, 6],
    "aug" : [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "none": list(range(12)),   # 无约束
}

# 自然大调内的常用和弦进行 (罗马数字 → 音程度数)
# I=0, ii=2, iii=4, IV=5, V=7, vi=9, vii°=11
COMMON_PROGRESSIONS = [
    [0, 5, 7, 0],   # I - IV - V - I
    [0, 9, 5, 7],   # I - vi - IV - V
    [0, 5, 9, 7],   # I - IV - vi - V
    [0, 7, 9, 5],   # I - V - vi - IV
    [0, 9, 7, 5],   # I - vi - V - IV
    [0, 2, 5, 7],   # I - ii - IV - V
]


class MusicTheoryConstraints:
    """
    在生成时对 logit 施加乐理惩罚
    """

    def __init__(self, tokenizer: MusicTokenizer):
        self.tok = tokenizer
        self.note_on_start, self.note_on_end = tokenizer.note_on_range()
        self.note_dur_start, self.note_dur_end = tokenizer.note_dur_range()
        self.beat_start, self.beat_end = tokenizer.beat_range()
        self.chord_start, self.chord_end = tokenizer.chord_range()
        self.key_start, self.key_end = tokenizer.key_range()
        self.vocab_size = tokenizer.vocab_size

    # ── 主接口 ─────────────────────────────────
    def apply(
        self,
        logits          : torch.Tensor,     # [V] 单步 logit
        context         : "GenerationContext",
    ) -> torch.Tensor:
        """
        对 logit 施加乐理约束
        返回修改后的 logit
        """
        logits = logits.clone().float()

        # 1. 调性约束 (调外音 logit 减小)
        if context.current_key is not None:
            logits = self._apply_key_constraint(logits, context.current_key)

        # 2. 和弦约束
        if context.current_chord is not None:
            logits = self._apply_chord_constraint(
                logits,
                context.current_chord,
                context.current_beat,
                context.current_key,
            )

        # 3. 重复音符惩罚
        logits = self._apply_repetition_penalty(logits, context.recent_pitches)

        # 4. 终止约束 (生成末尾时, 偏向起始音)
        if context.approaching_end:
            logits = self._apply_cadence_bias(logits, context.first_pitch, context.current_key)

        return logits

    # ── 调性约束 ───────────────────────────────
    def _apply_key_constraint(self, logits: torch.Tensor, key_str: str) -> torch.Tensor:
        root, mode = self._parse_key(key_str)
        scale = MAJOR_SCALE if mode == "maj" else MINOR_SCALE
        scale_pcs = set((root + s) % 12 for s in scale)

        penalty = THEORY_CONFIG["out_of_key_penalty"]
        for tok_id in range(self.note_on_start, self.note_on_end):
            pitch = PITCH_MIN + (tok_id - self.note_on_start)
            if pitch % 12 not in scale_pcs:
                logits[tok_id] -= penalty

        return logits

    # ── 和弦约束 ───────────────────────────────
    def _apply_chord_constraint(
        self,
        logits : torch.Tensor,
        chord  : Tuple[int, str],    # (root_pc, chord_type)
        beat   : int,                # 当前拍位 (0-15)
        key    : Optional[str],
    ) -> torch.Tensor:
        root, ctype = chord
        tones = CHORD_TONES.get(ctype, list(range(12)))
        chord_pcs = set((root + t) % 12 for t in tones)

        # 强拍 (拍点 0, 4, 8, 12) 要求和弦音
        is_strong_beat = (beat % 4 == 0)
        penalty = THEORY_CONFIG["non_chord_penalty"]

        if is_strong_beat:
            # 强拍: 强烈惩罚非和弦音
            for tok_id in range(self.note_on_start, self.note_on_end):
                pitch = PITCH_MIN + (tok_id - self.note_on_start)
                if pitch % 12 not in chord_pcs:
                    logits[tok_id] -= penalty * 2.5
        else:
            # 弱拍: 轻微惩罚非和弦音 (允许经过音)
            for tok_id in range(self.note_on_start, self.note_on_end):
                pitch = PITCH_MIN + (tok_id - self.note_on_start)
                if pitch % 12 not in chord_pcs:
                    logits[tok_id] -= penalty * 0.8

        return logits

    # ── 重复惩罚 ───────────────────────────────
    def _apply_repetition_penalty(
        self,
        logits        : torch.Tensor,
        recent_pitches: List[int],
    ) -> torch.Tensor:
        if not recent_pitches:
            return logits

        max_repeat = THEORY_CONFIG["max_repeat_note"]
        # 统计近期音符出现次数
        from collections import Counter
        count = Counter(recent_pitches[-max_repeat * 2:])

        for pitch, cnt in count.items():
            if cnt >= max_repeat:
                tok_id = self.tok.encode_pitch(pitch)
                logits[tok_id] -= cnt * 0.8  # 超过重复次数, 逐渐惩罚

        return logits

    # ── 终止偏置 ───────────────────────────────
    def _apply_cadence_bias(
        self,
        logits     : torch.Tensor,
        first_pitch: Optional[int],
        key_str    : Optional[str],
    ) -> torch.Tensor:
        """
        接近结尾时, 偏向主音 (tonic) 和属音 (dominant)
        使旋律有回归感 (8-bit 无限循环的关键)
        """
        if first_pitch is None and key_str is None:
            return logits

        cadence_pcs = set()
        if key_str:
            root, _ = self._parse_key(key_str)
            # 主音 + 属音
            for degree in THEORY_CONFIG["cadence_degrees"]:
                cadence_pcs.add((root + degree) % 12)

        if first_pitch is not None:
            cadence_pcs.add(first_pitch % 12)  # 回到起始音

        bias = 1.5
        for tok_id in range(self.note_on_start, self.note_on_end):
            pitch = PITCH_MIN + (tok_id - self.note_on_start)
            if pitch % 12 in cadence_pcs:
                logits[tok_id] += bias

        return logits

    # ── 工具 ───────────────────────────────────
    def _parse_key(self, key_str: str) -> Tuple[int, str]:
        """解析 "C_maj" → (0, "maj"), "A_min" → (9, "min")"""
        NOTE_PC = {
            'C':0,'Db':1,'D':2,'Eb':3,'E':4,'F':5,
            'Gb':6,'G':7,'Ab':8,'A':9,'Bb':10,'B':11
        }
        parts = key_str.split("_")
        root_name = parts[0]
        mode = parts[1] if len(parts) > 1 else "maj"
        root = NOTE_PC.get(root_name, 0)
        return root, mode

    def get_chord_progression_bias(
        self,
        current_root: int,
        key_str     : str,
    ) -> Dict[int, float]:
        """
        根据常见和弦进行, 对下一个和弦的根音 pc 给出偏置分数
        返回 {root_pc: bias_score}
        """
        key_root, mode = self._parse_key(key_str)
        bias = {}

        # 在大调中找到当前和弦的度数
        rel = (current_root - key_root) % 12
        for prog in COMMON_PROGRESSIONS:
            for i, degree in enumerate(prog[:-1]):
                if degree == rel and i + 1 < len(prog):
                    next_degree = prog[i + 1]
                    next_root = (key_root + next_degree) % 12
                    bias[next_root] = bias.get(next_root, 0) + 1.0

        return bias


# ══════════════════════════════════════════════
# 生成上下文状态机
# ══════════════════════════════════════════════
class GenerationContext:
    """
    维护生成过程中的音乐状态
    - 当前调性、和弦、拍子位置
    - 近期音符历史
    - 是否接近终止
    """

    def __init__(self, tokenizer: MusicTokenizer):
        self.tok = tokenizer
        self.reset()

    def reset(self):
        self.current_key   : Optional[str]          = None
        self.current_chord : Optional[Tuple[int,str]] = None
        self.current_beat  : int                    = 0
        self.current_bar   : int                    = 0
        self.recent_pitches: List[int]              = []
        self.first_pitch   : Optional[int]          = None
        self.approaching_end: bool                  = False
        self.total_bars    : int                    = 0
        self.target_bars   : int                    = GEN_CONFIG.get("default_bars", GEN_CONFIG.get("max_gen_bars", 32))

    def update(self, token_id: int):
        """解析 token 更新状态"""
        info = self.tok.decode_token(token_id)
        t = info["type"]

        if t == "key":
            self.current_key = info["key"]
        elif t == "chord":
            self.current_chord = (info["root"], info["ctype"])
        elif t == "beat":
            self.current_beat = info["pos"]
        elif t == "bar":
            self.current_bar += 1
            self.total_bars  += 1
            # 检查是否接近目标小节数
            remaining = self.target_bars - self.total_bars
            cadence_window = GEN_CONFIG.get("cadence_window", 4)
            self.approaching_end = (0 < remaining <= cadence_window)
        elif t == "note_on":
            pitch = info["pitch"]
            self.recent_pitches.append(pitch)
            if len(self.recent_pitches) > 20:
                self.recent_pitches = self.recent_pitches[-20:]
            # 记录第一个音符
            if self.first_pitch is None:
                self.first_pitch = pitch


# ══════════════════════════════════════════════
# 采样策略
# ══════════════════════════════════════════════
def sample_with_constraints(
    logits     : torch.Tensor,      # [V]
    constraints: MusicTheoryConstraints,
    context    : GenerationContext,
    temperature: float = 0.92,
    top_k      : int   = 50,
    top_p      : float = 0.92,
    rep_penalty: float = 1.15,
) -> int:
    """
    带乐理约束的 Top-K / Top-P 采样
    """
    # 1. 施加乐理约束
    logits = constraints.apply(logits, context)

    # 2. 温度调节
    logits = logits / max(temperature, 1e-8)

    # 3. 重复惩罚
    if rep_penalty != 1.0 and context.recent_pitches:
        note_start, note_end = constraints.note_on_start, constraints.note_on_end
        for p in set(context.recent_pitches):
            tid = constraints.tok.encode_pitch(p)
            if logits[tid] > 0:
                logits[tid] /= rep_penalty
            else:
                logits[tid] *= rep_penalty

    # 4. Top-K 截断
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < topk_vals[-1]] = float("-inf")

    # 5. Top-P (nucleus)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[remove_mask] = float("-inf")
    logits.scatter_(0, sorted_idx, sorted_logits)

    # 6. 采样
    probs   = F.softmax(logits, dim=-1)
    probs   = torch.clamp(probs, min=0)
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()

    token_id = torch.multinomial(probs, num_samples=1).item()
    return int(token_id)