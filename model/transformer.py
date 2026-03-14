"""
8-bit 音乐生成 Transformer
- Decoder-only 架构 (GPT 风格)
- 相对位置编码 (Rotary Embedding / ALiBi)
- Pre-LayerNorm (训练更稳定)
- 支持 KV Cache 加速推理
"""

from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import MODEL_CONFIG


# ══════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ══════════════════════════════════════════════
class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) - 相对位置感知，外推性更好
    论文: RoFormer (Su et al., 2021)
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)               # [T, dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)           # [T, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # [1,1,T,dim]
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(
        self,
        q      : torch.Tensor,
        k      : torch.Tensor,
        offset : int = 0,        # KV cache 长度偏移，推理时传入
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len    = q.shape[-2]
        k_len    = k.shape[-2]
        need_len = max(offset + q_len, k_len)
        if need_len > self.cos_cached.shape[-2]:
            self._build_cache(need_len + 512)   # 预留 buffer

        # k 的位置从 0 开始（含 cache 部分，cache 已经 apply 过 RoPE）
        # q 的位置从 offset 开始
        cos_q = self.cos_cached[:, :, offset:offset+q_len, :]
        sin_q = self.sin_cached[:, :, offset:offset+q_len, :]
        cos_k = self.cos_cached[:, :, :k_len, :]
        sin_k = self.sin_cached[:, :, :k_len, :]
        return apply_rotary(q, cos_q, sin_q), apply_rotary(k, cos_k, sin_k)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


# ══════════════════════════════════════════════
# Multi-Head Attention with RoPE
# ══════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead   = nhead
        self.d_head  = d_model // nhead
        self.d_model = d_model

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)
        self.rope     = RotaryEmbedding(self.d_head, max_seq_len)

        # 因果掩码 (下三角)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x            : torch.Tensor,              # [B, T, D]
        key_value_cache: Optional[Tuple] = None,  # (k_cache, v_cache) for inference
        use_cache    : bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, T, D = x.shape

        # QKV 投影
        qkv = self.qkv_proj(x)                             # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # 拆分多头: [B, H, T, Dh]
        def split_heads(t):
            return t.view(B, T, self.nhead, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # RoPE (KV cache 时 q 的位置 = cache 已有长度)
        cache_len = key_value_cache[0].shape[2] if key_value_cache is not None else 0
        q, k = self.rope(q, k, offset=cache_len)

        # KV Cache 拼接 (推理阶段)
        if key_value_cache is not None:
            k_cache, v_cache = key_value_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Scaled Dot-Product Attention
        Tk = k.shape[2]
        scale = math.sqrt(self.d_head)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B,H,T,Tk]

        # 因果掩码
        if not use_cache:  # 训练阶段
            mask = self.causal_mask[:, :, :T, :Tk].bool()
            attn_score = attn_score.masked_fill(~mask, float("-inf"))

        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # 输出
        out = torch.matmul(attn_weight, v)                  # [B, H, T, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out, new_cache


# ══════════════════════════════════════════════
# Feed-Forward Network (SwiGLU)
# ══════════════════════════════════════════════
class SwiGLUFFN(nn.Module):
    """SwiGLU 激活函数 - 比 ReLU/GELU 更好的表达能力"""
    def __init__(self, d_model: int, dim_ffn: int, dropout: float):
        super().__init__()
        # SwiGLU 需要两路投影
        self.gate  = nn.Linear(d_model, dim_ffn, bias=False)
        self.up    = nn.Linear(d_model, dim_ffn, bias=False)
        self.down  = nn.Linear(dim_ffn, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


# ══════════════════════════════════════════════
# Transformer Block (Pre-LN)
# ══════════════════════════════════════════════
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ffn: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn  = CausalSelfAttention(d_model, nhead, dropout, max_seq_len)
        self.ffn   = SwiGLUFFN(d_model, dim_ffn, dropout)

    def forward(
        self,
        x          : torch.Tensor,
        kv_cache   : Optional[Tuple] = None,
        use_cache  : bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Pre-LN: LayerNorm → Attention → 残差
        residual = x
        x, new_cache = self.attn(self.norm1(x), kv_cache, use_cache)
        x = residual + x

        # Pre-LN: LayerNorm → FFN → 残差
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# ══════════════════════════════════════════════
# 主模型: MusicTransformer
# ══════════════════════════════════════════════
class MusicTransformer(nn.Module):
    """
    8-bit 音乐生成 Transformer (Decoder-Only)

    参数估算 (d_model=256, layers=6, ffn=1024):
      Embedding  : vocab_size × 256    ≈ 1.1M
      Attn QKV   : 256 × 768 × 6      ≈ 1.2M
      Attn Out   : 256 × 256 × 6      ≈ 0.4M
      FFN gate   : 256 × 1024 × 6     ≈ 1.6M
      FFN up/down: 同上               ≈ 3.1M
      LM Head    : 256 × vocab_size    ≈ 1.1M
      总计: ≈ 15~18M 参数
      VRAM (fp16 + batch 16 + seq 512) ≈ 1.5~2.5GB
    """

    def __init__(self, vocab_size: int, config: dict = None):
        super().__init__()
        cfg = config or MODEL_CONFIG
        d_model    = cfg["d_model"]
        nhead      = cfg["nhead"]
        num_layers = cfg["num_layers"]
        dim_ffn    = cfg["dim_feedforward"]
        dropout    = cfg["dropout"]
        max_seq    = cfg["max_seq_len"]

        self.d_model  = d_model
        self.max_seq  = max_seq

        # Token 嵌入 (权重与 LM Head 共享, 节省参数)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ffn, dropout, max_seq)
            for _ in range(num_layers)
        ])

        # 最终 LayerNorm
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

        # LM Head (与 token_emb 权重绑定, 减少参数)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # 权重共享

        # 初始化
        self._init_weights()
        print(f"[Model] 参数量: {self.count_params():,}")

    def _init_weights(self):
        """GPT 风格初始化"""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)
        # 缩放残差投影 (稳定深层网络)
        num_layers = len(self.blocks)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 / math.sqrt(2 * num_layers))
            nn.init.normal_(block.ffn.down.weight,      std=0.02 / math.sqrt(2 * num_layers))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids  : torch.Tensor,          # [B, T]
        targets    : Optional[torch.Tensor] = None,  # [B, T]
        kv_caches  : Optional[list] = None,  # list of (k,v) per layer
        use_cache  : bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        返回: (logits [B,T,V], loss, new_kv_caches)
        """
        x = self.token_emb(input_ids)                      # [B, T, D]
        x = x * math.sqrt(self.d_model)                    # 缩放嵌入

        new_kv_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            x, new_kv = block(x, kv_cache=kv, use_cache=use_cache)
            if use_cache:
                new_kv_caches.append(new_kv)

        x      = self.final_norm(x)                        # [B, T, D]
        logits = self.lm_head(x)                           # [B, T, V]

        loss = None
        if targets is not None:
            # 忽略 padding token 的 loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,      # PAD_ID = 0
            )

        return logits, loss, new_kv_caches

    def get_num_params(self) -> dict:
        """按模块统计参数量"""
        d = {}
        d["embedding"] = sum(p.numel() for p in self.token_emb.parameters())
        d["transformer"] = sum(p.numel() for p in self.blocks.parameters())
        d["lm_head"]  = 0  # 权重共享, 不额外计算
        d["total"]    = self.count_params()
        return d