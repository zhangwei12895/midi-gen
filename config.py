"""
config.py  —  全局配置 v4

新增内容（v4）
─────────────
  INTENSITY token    全局曲风强度  0=舒缓 ~ 4=激情
  SECTION  token     段落标记      INTRO/VERSE/CHORUS/BRIDGE/OUTRO
  PITCH_CENTER token 全局旋律音高中心  0=极低 ~ 4=极高
  PITCH_REGISTER token per-bar 旋律音区  0=低 1=中 2=高
  PITCH_RANGE token  per-bar 旋律音高跨度  0=窄 ~ 4=极宽
  词汇表 282 → 305
"""

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
MIDI_DIR       = "./midi_data"
PROCESSED_DIR  = "./processed"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR        = "./logs"
OUTPUT_DIR     = "./outputs"

# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────
PITCH_MIN    = 36       # C2 — 包含贝斯音域
PITCH_MAX    = 96       # C7
PITCH_RANGE  = PITCH_MAX - PITCH_MIN + 1

TICKS_PER_BEAT = 4      # 16分音符粒度
BEATS_PER_BAR  = 4
TICKS_PER_BAR  = TICKS_PER_BEAT * BEATS_PER_BAR   # 16

MAX_DURATION = 16
MIN_DURATION = 1

VELOCITY_BINS = 8
TEMPO_MIN     = 60
TEMPO_MAX     = 180
TEMPO_BINS    = 13

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[SEP]"]
TRACK_TOKENS   = ["[TRACK_M]", "[TRACK_B]", "[TRACK_H]"]

CHORD_TYPES = [
    "maj", "min", "dom7", "maj7", "min7",
    "dim", "aug", "sus2", "sus4", "none"
]

KEY_LIST = [
    "C_maj","Db_maj","D_maj","Eb_maj","E_maj","F_maj",
    "Gb_maj","G_maj","Ab_maj","A_maj","Bb_maj","B_maj",
    "C_min","Db_min","D_min","Eb_min","E_min","F_min",
    "Gb_min","G_min","Ab_min","A_min","Bb_min","B_min",
]

# ─────────────────────────────────────────────
# 新增 Token 定义（v4）
# ─────────────────────────────────────────────

# 全局曲风强度（放在 BOS 后，一首歌一个）
INTENSITY_BINS   = 5
INTENSITY_LABELS = ["舒缓", "轻快", "均衡", "充沛", "激情"]
INTENSITY_EN     = ["calm", "light", "medium", "energetic", "intense"]

# 段落名称
SECTION_NAMES = ["INTRO", "VERSE", "CHORUS", "BRIDGE", "OUTRO"]

# 全局旋律音高中心（BOS 后，一首歌一个）
# 反映整首歌的旋律偏低/偏高
PITCH_CENTER_BINS   = 5          # 0=极低(≤48) 1=低 2=中 3=高 4=极高(≥72)
PITCH_CENTER_THRESH = [0, 48, 55, 62, 69]   # MIDI pitch 分档

# per-bar 旋律音区（低/中/高 音区）
PITCH_REGISTER_BINS   = 3
PITCH_REGISTER_LABELS = ["低音区", "中音区", "高音区"]
PITCH_REGISTER_THRESH = [0, 52, 64]   # ≤51=低, 52-63=中, ≥64=高

# per-bar 旋律音高跨度（半音数）
PITCH_RANGE_BINS   = 5
PITCH_RANGE_THRESH = [0, 4, 7, 12, 18]   # 窄/小/中/宽/极宽

# ─────────────────────────────────────────────
# 模型配置
# ─────────────────────────────────────────────
MODEL_CONFIG = {
    "d_model"        : 384,
    "nhead"          : 8,
    "num_layers"     : 8,
    "dim_feedforward": 1536,
    "dropout"        : 0.10,
    "max_seq_len"    : 2048,
}

# ─────────────────────────────────────────────
# 训练配置
# ─────────────────────────────────────────────
TRAIN_CONFIG = {
    "batch_size"        : 6,
    "grad_accumulation" : 10,
    "learning_rate"     : 2e-4,
    "weight_decay"      : 0.01,
    "warmup_steps"      : 1000,
    "max_steps"         : 200_000,
    "seq_len"           : 512,
    "num_workers"       : 0,
    "save_every"        : 1000,
    "eval_every"        : 500,
    "use_amp"           : True,
    "clip_grad_norm"    : 1.0,
    "label_smoothing"   : 0.05,
    "val_split"         : 0.15,
}

# ─────────────────────────────────────────────
# 数据增强
# ─────────────────────────────────────────────
AUG_CONFIG = {
    "pitch_transpose_range" : 5,
    "enable_transpose"      : True,
    "enable_tempo_stretch"  : False,
}

# ─────────────────────────────────────────────
# 生成配置
# ─────────────────────────────────────────────
GEN_CONFIG = {
    "temperature"        : 0.90,
    "top_k"              : 40,
    "top_p"              : 0.90,
    "repetition_penalty" : 1.10,
    "default_key"        : "C_maj",
    "default_tempo"      : 120.0,
    "min_bars_before_eos": 8,
    # 默认曲风强度（0=舒缓→4=激情，也可在命令行 --style 覆盖）
    "default_intensity"  : 2,
    # 旧密度控制（保留兼容性，不再作为主控）
    "density_M"          : 2,
    "density_B"          : 2,
    "density_H"          : 1,
    # 最大生成 token 数（安全限制，不再由 bars 限定长度）
    "max_gen_tokens"     : 2048,
}

# ─────────────────────────────────────────────
# 乐理约束
# ─────────────────────────────────────────────
THEORY_CONFIG = {
    "out_of_key_penalty"     : 1.5,
    "non_chord_penalty"      : 0.8,
    "passing_note_prob"      : 0.3,
    "cadence_degrees"        : [0, 7],
    "max_repeat_note"        : 4,
    "strong_beat_chord_prob" : 0.8,
}

# ─────────────────────────────────────────────
# 曲风强度 → 段落结构模板
# 仅用于显示预期结构，实际以模型输出的 EOS 为准
# ─────────────────────────────────────────────
INTENSITY_STRUCTURE = {
    0: ["INTRO","VERSE","OUTRO"],
    1: ["INTRO","VERSE","CHORUS","OUTRO"],
    2: ["INTRO","VERSE","CHORUS","VERSE","CHORUS","OUTRO"],
    3: ["INTRO","VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS","OUTRO"],
    4: ["INTRO","VERSE","CHORUS","VERSE","CHORUS","BRIDGE","CHORUS","CHORUS","OUTRO"],
}

# ─────────────────────────────────────────────
# Studio / generate.py 所需常量（向下兼容）
# ─────────────────────────────────────────────

# 段落名 → 默认小节数（仅用于估算，实际由模型决定）
_SEC_BARS = {"INTRO": 4, "VERSE": 8, "CHORUS": 8, "BRIDGE": 4, "OUTRO": 4}

# STYLE_STRUCTURES：每种曲风的段落列表（段落名, 预期小节数）
STYLE_STRUCTURES: dict[int, list[tuple[str, int]]] = {
    k: [(sec, _SEC_BARS[sec]) for sec in secs]
    for k, secs in INTENSITY_STRUCTURE.items()
}

def total_bars(style: int) -> int:
    """估算某曲风的总小节数（仅供显示，不作为生成限制）"""
    return sum(b for _, b in STYLE_STRUCTURES.get(style, []))

STYLE_NAMES    = INTENSITY_LABELS          # ["舒缓","轻快","均衡","充沛","激情"]
STYLE_NAMES_EN = INTENSITY_EN              # ["calm","light","medium","energetic","intense"]

STYLE_DESCRIPTIONS: dict[int, str] = {
    0: "平静舒缓 · 稀疏旋律 · 轻柔音区 · 适合冥想/背景",
    1: "轻松轻快 · 简单结构 · 清晰旋律线",
    2: "均衡适中 · 前奏→主歌→副歌→主歌→副歌→尾奏",
    3: "充沛有力 · 完整 ABABCB 结构 · 高密度副歌",
    4: "激情澎湃 · 高强度 · 双副歌 · 极宽音高跨度",
}