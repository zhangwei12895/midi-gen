"""
data/dataset.py  —  训练数据集（增强版）

新特性
──────
  1. Pitch Transpose 增强：每次取样随机移调 ±5 半音
     → 12x 数据量效果，解决 val_loss 不降问题
  2. 整首歌采样 + 随机窗口（长歌）
  3. 固定验证集分割（seed=42）
  4. CollatePad：可 pickle（Windows 兼容）
"""

from __future__ import annotations
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from config import TRAIN_CONFIG, PROCESSED_DIR, AUG_CONFIG
from data.tokenizer      import MusicTokenizer
from data.midi_processor import process_midi_directory


# ── Pitch transpose 在 token 层面操作 ────────────────────────
def _transpose_tokens(
    tokens     : List[int],
    semitones  : int,
    tokenizer  : MusicTokenizer,
) -> List[int]:
    """
    将 token 序列中所有 NOTE_ON / KEY 移调 semitones 半音。
    在 token ID 层面直接操作，无需重新量化。
    """
    if semitones == 0:
        return tokens

    from config import PITCH_MIN, PITCH_MAX, KEY_LIST

    note_start, note_end = tokenizer.note_on_range()
    n_pitches = note_end - note_start   # = PITCH_MAX - PITCH_MIN + 1

    result = []
    for tid in tokens:
        # NOTE_ON token
        if note_start <= tid < note_end:
            new_idx = (tid - note_start) + semitones
            new_idx = max(0, min(n_pitches - 1, new_idx))
            result.append(note_start + new_idx)
        # KEY token — 同步移调
        elif tokenizer.id2token.get(tid, "").startswith("KEY_"):
            tok_str = tokenizer.id2token[tid]   # e.g. "KEY_C_maj"
            parts   = tok_str.split("_")         # ["KEY","C","maj"]
            note_name, mode = parts[1], parts[2]
            NOTE_PC = {'C':0,'Db':1,'D':2,'Eb':3,'E':4,'F':5,
                       'Gb':6,'G':7,'Ab':8,'A':9,'Bb':10,'B':11}
            PC_NOTE = {v: k for k,v in NOTE_PC.items()}
            PC_NOTE_FLAT = {0:'C',1:'Db',2:'D',3:'Eb',4:'E',5:'F',
                            6:'Gb',7:'G',8:'Ab',9:'A',10:'Bb',11:'B'}
            old_pc  = NOTE_PC.get(note_name, 0)
            new_pc  = (old_pc + semitones) % 12
            new_key = f"KEY_{PC_NOTE_FLAT[new_pc]}_{mode}"
            result.append(tokenizer.token2id.get(new_key, tid))
        else:
            result.append(tid)
    return result


class MusicDataset(Dataset):

    def __init__(
        self,
        sequences : List[List[int]],
        tokenizer : MusicTokenizer,
        seq_len   : int  = TRAIN_CONFIG["seq_len"],
        augment   : bool = True,
    ):
        self.tok     = tokenizer
        self.seq_len = seq_len
        self.augment = augment
        self.data    = [s for s in sequences if len(s) >= 16]
        self.enable_transpose = augment and AUG_CONFIG.get("enable_transpose", True)
        self.max_shift        = AUG_CONFIG.get("pitch_transpose_range", 5)

        mode = "训练" if augment else "验证"
        avg  = sum(len(s) for s in self.data) // max(1, len(self.data))
        print(f"  [{mode}集] {len(self.data)} 首  avg_len={avg}  "
              f"transpose={'±'+str(self.max_shift) if self.enable_transpose else 'off'}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = list(self.data[idx])   # copy
        L   = len(seq)
        T   = self.seq_len

        # ── Pitch transpose 增强 ──────────────────────────────
        if self.enable_transpose:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                seq = _transpose_tokens(seq, shift, self.tok)

        # ── 窗口裁剪 ─────────────────────────────────────────
        if L <= T + 1:
            chunk = seq
        else:
            if self.augment:
                # 随机从 BAR token 处开始，保持结构完整
                bar_id = self.tok.bar_id
                bar_positions = [i for i, t in enumerate(seq) if t == bar_id]
                if len(bar_positions) > 1:
                    start_bar = random.choice(bar_positions[:-1])
                    chunk = seq[start_bar : start_bar + T + 1]
                else:
                    start = random.randint(0, L - T - 1)
                    chunk = seq[start : start + T + 1]
            else:
                chunk = seq[:T+1]

        inp = torch.tensor(chunk[:-1], dtype=torch.long)
        tgt = torch.tensor(chunk[1:],  dtype=torch.long)
        return inp, tgt

    @property
    def n_songs(self) -> int:
        return len(self.data)


class CollatePad:
    """可 pickle 的 collate 函数（Windows 多进程兼容）"""
    def __init__(self, pad_id: int): self.pad_id = pad_id
    def __call__(self, batch):
        inps, tgts = zip(*batch)
        max_len    = max(x.size(0) for x in inps)
        pad        = self.pad_id
        ib = torch.full((len(inps), max_len), pad, dtype=torch.long)
        tb = torch.full((len(tgts), max_len), pad, dtype=torch.long)
        for i, (inp, tgt) in enumerate(zip(inps, tgts)):
            n = inp.size(0)
            ib[i, :n] = inp
            tb[i, :n] = tgt
        return ib, tb


def build_dataloaders(
    midi_dir  : str,
    tokenizer : MusicTokenizer,
    cache_path: str = None,
) -> Tuple[DataLoader, DataLoader]:

    if cache_path is None:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        cache_path = os.path.join(PROCESSED_DIR, "sequences.pkl")

    if os.path.exists(cache_path):
        print(f"  从缓存加载: {cache_path}")
        with open(cache_path, "rb") as f:
            all_seqs = pickle.load(f)
    else:
        all_seqs = process_midi_directory(midi_dir, tokenizer)
        with open(cache_path, "wb") as f:
            pickle.dump(all_seqs, f)
        print(f"  缓存已保存: {cache_path}")

    if not all_seqs:
        raise ValueError(f"没有有效 MIDI: {midi_dir}")

    rng      = random.Random(42)
    shuffled = all_seqs[:]
    rng.shuffle(shuffled)

    val_n      = max(2, int(len(shuffled) * max(TRAIN_CONFIG["val_split"], 0.15)))
    val_seqs   = shuffled[:val_n]
    train_seqs = shuffled[val_n:] or shuffled

    pad_id  = tokenizer.pad_id
    seq_len = TRAIN_CONFIG["seq_len"]

    train_ds = MusicDataset(train_seqs, tokenizer, seq_len=seq_len, augment=True)
    val_ds   = MusicDataset(val_seqs,   tokenizer, seq_len=seq_len, augment=False)
    collate  = CollatePad(pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True, num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True, drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True, drop_last=False, collate_fn=collate,
    )

    total = len(train_ds) + len(val_ds)
    print(f"  总计 {total} 首  train={len(train_ds)}  val={len(val_ds)}")
    return train_loader, val_loader