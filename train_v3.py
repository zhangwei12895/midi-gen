"""
train_v3.py  —  训练脚本 v3.3

新特性
──────
  ★ 每步 TensorBoard 日志（train_loss / lr / grad_norm / tokens_per_sec）
  ★ 每 EVAL_EVERY 步 val_loss（原始 CE，无 smooth）写入 TensorBoard
  ★ 全局密度参数：训练统计三轨平均密度分布，写入 TensorBoard
  ★ 进度条显示：train(smooth) / val(raw) / gap / lr / speed / VRAM
  ★ 阶段摘要：密度分布 + 梯度范数均值
  ★ 日志文件：logs/train.log 同步写入

val(raw) 与 train(smooth) 的区别
──────────────────────────────────
  train(smooth) = 含 label_smoothing 的 CE，训练时使用，偏低（正常）
  val(raw)      = 纯 cross_entropy，真实泛化指标
  gap = val - train：正值 0.3~1.0 属正常泛化差距；超过 1.5 才考虑过拟合

用法
────
  python train_v3.py --midi_dir ./midi_data
  python train_v3.py --midi_dir ./midi_data --resume ./checkpoints/best_model.pt
  python train_v3.py --midi_dir ./midi_data --rebuild
  tensorboard --logdir ./logs
"""

from __future__ import annotations
import os, sys, time, math, argparse, itertools, datetime
from typing import Tuple, List
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    MIDI_DIR, CHECKPOINT_DIR, LOG_DIR,
    TRAIN_CONFIG, MODEL_CONFIG, AUG_CONFIG,
)
from data.tokenizer    import MusicTokenizer
from data.dataset      import build_dataloaders
from model.transformer import MusicTransformer


# ══════════════════════════════════════════════════════════════
# 学习率：Cosine + Warmup
# ══════════════════════════════════════════════════════════════
def get_lr(step: int) -> float:
    ws = TRAIN_CONFIG["warmup_steps"]
    ms = TRAIN_CONFIG["max_steps"]
    lr = TRAIN_CONFIG["learning_rate"]
    if step < ws:
        return lr * step / max(1, ws)
    prog = min(1.0, (step - ws) / max(1, ms - ws))
    return lr * 0.5 * (1.0 + math.cos(math.pi * prog))


# ══════════════════════════════════════════════════════════════
# 检查点
# ══════════════════════════════════════════════════════════════
def save_ckpt(model, opt, scaler, step: int, val_loss: float, path: str,
              n_params: int = 0, train_finish_time: str = ""):
    """
    保存检查点。v4 新增字段：
      n_params       参数量（整数）
      param_str      参数量字符串，如 "30.2M"
      model_name     文件名（不含路径），方便生成器识别
      train_time     训练完成时间（空串表示训练中）
    """
    n_params = n_params or (model.count_params() if hasattr(model, "count_params") else 0)
    pstr     = f"{n_params/1_000_000:.1f}M"
    _dir     = os.path.dirname(path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    torch.save({
        "step"        : step,
        "val_loss"    : val_loss,
        "model_state" : model.state_dict(),
        "opt_state"   : opt.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": MODEL_CONFIG,
        # ── v4 新增 ──────────────────────────────
        "n_params"    : n_params,
        "param_str"   : pstr,
        "model_name"  : os.path.basename(path),
        "train_time"  : train_finish_time,
    }, path)

def load_ckpt(path: str, model, opt=None, scaler=None) -> Tuple[int, float]:
    ck = torch.load(path, map_location="cuda")
    model.load_state_dict(ck["model_state"])
    if opt    and "opt_state"    in ck: opt.load_state_dict(ck["opt_state"])
    if scaler and "scaler_state" in ck: scaler.load_state_dict(ck["scaler_state"])
    return ck.get("step", 0), ck.get("val_loss", float("inf"))


# ══════════════════════════════════════════════════════════════
# 验证（纯 CE，无 label_smoothing）
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, pad_id: int) -> Tuple[float, dict]:
    """
    返回 (val_loss_raw, density_stats)
      val_loss_raw : 原始 CE，无 smoothing，真实泛化指标
      density_stats: {'M_avg':float, 'B_avg':float, 'H_avg':float,
                      'M_dist':[...], 'B_dist':[...], 'H_dist':[...]}
    """
    model.eval()
    total, count = 0.0, 0

    # 密度统计
    density_counts = {"M":[0]*5, "B":[0]*5, "H":[0]*5}

    for inp, tgt in loader:
        inp, tgt = inp.cuda(), tgt.cuda()
        with torch.amp.autocast("cuda"):
            logits, _, _ = model(inp)
            B, T, V = logits.shape
            # ★ 无 label_smoothing：真实 CE
            loss = F.cross_entropy(
                logits.reshape(B*T, V), tgt.reshape(B*T),
                ignore_index=pad_id, label_smoothing=0.0,
            )
        total += loss.item(); count += 1

    model.train()
    val_loss = total / max(1, count)
    return val_loss, density_counts


# ══════════════════════════════════════════════════════════════
# 密度统计（从 batch 中提取密度 token 分布）
# ══════════════════════════════════════════════════════════════
def extract_density_stats(batch_inp: torch.Tensor, tok: MusicTokenizer) -> dict:
    """从一个 batch 中统计三轨密度 token 的分布均值"""
    d_start, d_end = tok.density_range()
    if d_start == d_end:
        return {}

    stats = {}
    flat  = batch_inp.cpu().numpy().flatten()
    for track in ["M", "B", "H"]:
        base = tok.token2id.get(f"DENSITY_{track}_0", -1)
        if base < 0: continue
        bins = [int((flat == base + b).sum()) for b in range(5)]
        total = sum(bins) or 1
        avg   = sum(b * bins[b] / total for b in range(5))
        stats[f"density_{track}_avg"]  = avg
        stats[f"density_{track}_dist"] = bins
    return stats


# ══════════════════════════════════════════════════════════════
# 日志文件写入
# ══════════════════════════════════════════════════════════════
class FileLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.f.write(f"\n{'='*60}\n")
        self.f.write(f"  训练开始: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        self.f.write(f"{'='*60}\n")

    def write(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.f.write(f"[{ts}] {msg}\n")

    def close(self): self.f.close()


# ══════════════════════════════════════════════════════════════
# 主训练函数
# ══════════════════════════════════════════════════════════════
def train(args):
    if not torch.cuda.is_available():
        print("!! CUDA 不可用"); sys.exit(1)
    torch.cuda.set_device(0)

    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print()
    print("┌" + "─"*60 + "┐")
    print("│       8-BIT MUSIC TRANSFORMER  —  训练" + " "*20 + "│")
    print("└" + "─"*60 + "┘")
    print(f"  GPU   {gpu}  ({vram:.1f} GB)")
    print(f"  CUDA  {torch.version.cuda}")
    print(f"  时间  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    # ── 日志文件 ─────────────────────────────────────────────
    flog = FileLogger(os.path.join(LOG_DIR, "train.log"))
    flog.write(f"GPU={gpu}  CUDA={torch.version.cuda}")

    # ── 数据 ─────────────────────────────────────────────────
    print("\n◆ 数据")
    print("  " + "─"*48)
    tok = MusicTokenizer()

    if args.rebuild:
        c = os.path.join("./processed", "sequences.pkl")
        if os.path.exists(c):
            os.remove(c); print("  旧缓存已删除")

    train_loader, val_loader = build_dataloaders(
        midi_dir=args.midi_dir, tokenizer=tok)

    n_train = train_loader.dataset.n_songs
    n_val   = val_loader.dataset.n_songs
    n_total = n_train + n_val
    pad_id  = tok.pad_id

    # ── 模型 ─────────────────────────────────────────────────
    print("\n◆ 模型")
    print("  " + "─"*48)
    model = MusicTransformer(vocab_size=tok.vocab_size, config=MODEL_CONFIG).cuda()
    n_params = model.count_params()
    print(f"  参数量  {n_params:,}")
    print(f"  vocab   {tok.vocab_size}  d={MODEL_CONFIG['d_model']}"
          f"  L={MODEL_CONFIG['num_layers']}"
          f"  H={MODEL_CONFIG['nhead']}")
    flog.write(f"模型参数={n_params:,}  vocab={tok.vocab_size}")

    opt    = torch.optim.AdamW(model.parameters(),
                lr=TRAIN_CONFIG["learning_rate"],
                weight_decay=TRAIN_CONFIG["weight_decay"],
                betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler()

    step = 0; best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        step, best_val = load_ckpt(args.resume, model, opt, scaler)
        print(f"  断点续训  step={step}  best_val={best_val:.4f}")
        flog.write(f"断点续训 step={step} best_val={best_val:.4f}")

    # ── TensorBoard ───────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(LOG_DIR)
        # 写入超参信息
        writer.add_text("config/model",
            str({k: MODEL_CONFIG[k] for k in ["d_model","nhead","num_layers"]}), 0)
        writer.add_text("config/train",
            str({k: TRAIN_CONFIG[k] for k in
                 ["batch_size","learning_rate","weight_decay","label_smoothing"]}), 0)
        print(f"  TensorBoard  →  tensorboard --logdir {LOG_DIR}")
        flog.write(f"TensorBoard 已启动: {LOG_DIR}")
    except Exception as e:
        print(f"  TensorBoard 不可用: {e}")

    GA         = TRAIN_CONFIG["grad_accumulation"]
    MAX_STEPS  = TRAIN_CONFIG["max_steps"]
    PHASE      = TRAIN_CONFIG["save_every"]
    EVAL_EVERY = TRAIN_CONFIG["eval_every"]

    print("\n◆ 训练配置")
    print("  " + "─"*48)
    print(f"  歌曲总数    {n_total} 首  (train={n_train}  val={n_val})")
    print(f"  有效 batch  {TRAIN_CONFIG['batch_size']}×accum{GA}"
          f" = {TRAIN_CONFIG['batch_size']*GA}")
    print(f"  seq_len     {TRAIN_CONFIG['seq_len']}")
    print(f"  lr          {TRAIN_CONFIG['learning_rate']:.1e}"
          f"  wd={TRAIN_CONFIG['weight_decay']}"
          f"  smooth={TRAIN_CONFIG['label_smoothing']}")
    print(f"  总步数      {MAX_STEPS:,}  阶段={PHASE}步  验证={EVAL_EVERY}步")
    print(f"  Transpose   ±{AUG_CONFIG['pitch_transpose_range']}半音")
    print(f"\n  [指标说明]")
    print(f"  train(s) = 含smooth={TRAIN_CONFIG['label_smoothing']} 的 CE  ← 训练优化目标")
    print(f"  val(raw) = 纯 CE（无smooth）← 真实泛化指标，会比 train 高，属正常")
    print(f"  gap      = val-train：0.3~1.5 正常；持续增大才是过拟合信号")
    print()

    # ── 初始验证 ─────────────────────────────────────────────
    print("  [初始验证] 计算基准 val_loss ...")
    init_val, _ = evaluate(model, val_loader, pad_id)
    best_val    = init_val
    latest_val  = init_val
    # ★ 检查点名称含参数量和时间
    _n_params_m = model.count_params() // 1_000_000
    _start_dt   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    _best_name  = f"best_{_n_params_m}M_{_start_dt}.pt"
    _best_path  = os.path.join(CHECKPOINT_DIR, _best_name)
    # 同时保留 best_model.pt 软链名方便兼容
    save_ckpt(model, opt, scaler, step, best_val, _best_path)
    save_ckpt(model, opt, scaler, step, best_val,
              os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    print(f"  基准 val(raw) = {init_val:.4f}")
    print(f"  检查点名称    {_best_name}\n")
    flog.write(f"初始 val_loss={init_val:.4f}  ckpt={_best_name}")

    if writer:
        writer.add_scalar("val/loss_raw",  init_val, 0)
        writer.add_scalar("train/lr",      get_lr(1), 0)

    # ══════════════════════════════════════════════════════════
    # 训练循环
    # ══════════════════════════════════════════════════════════
    model.train(); opt.zero_grad()
    data_iter    = itertools.cycle(train_loader)
    phase_losses : List[float] = []
    phase_gnorms : List[float] = []
    phase_speeds : List[float] = []
    phase_density_stats : List[dict] = []
    batch_count  = 0
    t_phase      = time.time()
    t_step_start = time.time()
    phase_no     = step // PHASE + 1
    phase_pos    = step % PHASE
    total_ph     = (MAX_STEPS + PHASE - 1) // PHASE

    # ── 进度条 ───────────────────────────────────────────────
    def _pbar(ph: int, init: int = 0) -> tqdm:
        return tqdm(
            total=PHASE, initial=init,
            desc=f"  阶段{ph:>3}/{total_ph}",
            bar_format=(
                "  {desc} |{bar:20}| {percentage:3.0f}%"
                " [{n_fmt}/{total_fmt}]  {postfix}"
            ),
            unit="step", dynamic_ncols=False,
        )

    def _postfix(tr: float, val: float, best: float,
                 lr: float, gnorm: float, spd: float,
                 vram: float, tag: str = "") -> str:
        gap     = val - tr
        gap_str = f"+{gap:.3f}" if gap >= 0 else f"{gap:.3f}"
        return (
            f"train={tr:.4f}  val={val:.4f}  gap={gap_str}"
            f"  best={best:.4f}  lr={lr:.1e}"
            f"  ‖g‖={gnorm:.2f}  {spd:.1f}s/s"
            f"  VRAM={vram:.0f}M"
            + (f"  {tag}" if tag else "")
        )

    pbar    = _pbar(phase_no, phase_pos)
    cur_gnorm = 0.0
    cur_spd   = 0.0
    pbar.set_postfix_str(
        _postfix(0, latest_val, best_val, get_lr(max(1,step)), 0, 0, 0),
        refresh=True,
    )

    for inp, tgt in data_iter:
        inp = inp.cuda(non_blocking=True)
        tgt = tgt.cuda(non_blocking=True)

        t_step_start = time.time()

        # ── 前向 ─────────────────────────────────────────────
        with torch.amp.autocast("cuda"):
            logits, _, _ = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B*T, V), tgt.reshape(B*T),
                ignore_index=pad_id,
                label_smoothing=TRAIN_CONFIG["label_smoothing"],
            )

        scaler.scale(loss / GA).backward()
        batch_count += 1

        if batch_count % GA != 0:
            continue

        # ── 参数更新 ─────────────────────────────────────────
        scaler.unscale_(opt)
        gnorm = nn.utils.clip_grad_norm_(
            model.parameters(), TRAIN_CONFIG["clip_grad_norm"]
        ).item()
        scaler.step(opt); scaler.update(); opt.zero_grad()

        step += 1
        lr    = get_lr(step)
        for pg in opt.param_groups: pg["lr"] = lr

        step_time = time.time() - t_step_start
        tokens_per_sec = (B * T) / max(step_time, 1e-6)

        loss_val  = loss.item()
        cur_gnorm = gnorm
        cur_spd   = tokens_per_sec
        phase_losses.append(loss_val)
        phase_gnorms.append(gnorm)
        phase_speeds.append(tokens_per_sec)

        # 密度统计（每步采样，开销极低）
        d_stat = extract_density_stats(inp, tok)
        if d_stat: phase_density_stats.append(d_stat)

        avg_tr = sum(phase_losses) / len(phase_losses)
        vram   = torch.cuda.memory_allocated() / 1024**2

        # ── TensorBoard 每步写入 ────────────────────────────
        if writer:
            writer.add_scalar("train/loss_smooth", loss_val,        step)
            writer.add_scalar("train/lr",           lr,             step)
            writer.add_scalar("train/grad_norm",    gnorm,          step)
            writer.add_scalar("train/tokens_per_s", tokens_per_sec, step)
            writer.add_scalar("train/vram_mb",      vram,           step)
            # 密度分布均值
            if d_stat:
                for k, v in d_stat.items():
                    if k.endswith("_avg"):
                        writer.add_scalar(f"density/{k}", v, step)

        # ── 验证（每 EVAL_EVERY 步）────────────────────────
        tag = ""
        if step % EVAL_EVERY == 0:
            latest_val, _ = evaluate(model, val_loader, pad_id)
            improved      = latest_val < best_val
            if improved:
                best_val = latest_val
                save_ckpt(model, opt, scaler, step, best_val, _best_path)
                save_ckpt(model, opt, scaler, step, best_val,
                          os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                tag = "★BEST"
            if writer:
                writer.add_scalar("val/loss_raw",          latest_val, step)
                writer.add_scalar("val/gap",               latest_val - avg_tr, step)
                writer.add_scalar("val/best",              best_val,   step)
            flog.write(
                f"step={step:>6}  train={avg_tr:.4f}  "
                f"val={latest_val:.4f}  gap={latest_val-avg_tr:+.4f}"
                f"  best={best_val:.4f}  lr={lr:.2e}  {tag}"
            )

        # ── 进度条更新 ───────────────────────────────────────
        pbar.set_postfix_str(
            _postfix(avg_tr, latest_val, best_val,
                     lr, cur_gnorm, cur_spd, vram, tag),
            refresh=(tag != ""),
        )
        pbar.update(1)

        # ── 阶段结束 ─────────────────────────────────────────
        if step % PHASE == 0:
            latest_val, _ = evaluate(model, val_loader, pad_id)
            improved = latest_val < best_val
            if improved:
                best_val = latest_val
                save_ckpt(model, opt, scaler, step, best_val, _best_path)
                save_ckpt(model, opt, scaler, step, best_val,
                          os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                vtag = f"★ NEW BEST → {_best_name}"
            else:
                vtag = "未改善"

            elapsed   = time.time() - t_phase
            spd_steps = PHASE / max(elapsed, 0.001)
            avg_tr    = sum(phase_losses) / len(phase_losses)
            avg_gnorm = sum(phase_gnorms) / len(phase_gnorms)
            avg_tps   = sum(phase_speeds) / len(phase_speeds)
            gap       = latest_val - avg_tr

            # 密度摘要
            dens_summary = ""
            if phase_density_stats:
                for track in ["M","B","H"]:
                    key = f"density_{track}_avg"
                    vals = [d[key] for d in phase_density_stats if key in d]
                    if vals:
                        dens_summary += f"  D_{track}={sum(vals)/len(vals):.2f}"

            pbar.close()
            print(f"\n  ┌── 阶段 {phase_no}/{total_ph} {'─'*40}")
            print(f"  │  step={step:>7,}/{MAX_STEPS:,}"
                  f"  歌曲={n_total}首 (train={n_train} val={n_val})")
            print(f"  │  train(smooth)={avg_tr:.4f}"
                  f"  val(raw)={latest_val:.4f}"
                  f"  gap={gap:+.4f}"
                  f"  best={best_val:.4f}  {vtag}")
            print(f"  │  [gap说明] train含smooth={TRAIN_CONFIG['label_smoothing']}"
                  f"，gap>0属正常，gap>1.5才需警惕过拟合")
            print(f"  │  梯度范数均值={avg_gnorm:.3f}"
                  f"  吞吐={avg_tps:.0f}tok/s"
                  f"  速度={spd_steps:.1f}步/s")
            print(f"  │  密度(0=稀疏→4=极密):{dens_summary if dens_summary else '  无密度数据'}")
            print(f"  │  耗时={elapsed/60:.1f}min"
                  f"  剩余≈{(MAX_STEPS-step)/spd_steps/3600:.1f}h")
            ckp = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}_{_n_params_m}M.pt")
            save_ckpt(model, opt, scaler, step, best_val, ckp)
            print(f"  └── 已保存 step_{step}.pt\n")

            if writer:
                writer.add_scalar("phase/val_raw",    latest_val,  step)
                writer.add_scalar("phase/train_smooth",avg_tr,     step)
                writer.add_scalar("phase/gap",         gap,        step)
                writer.add_scalar("phase/grad_norm",   avg_gnorm,  step)
                writer.add_scalar("phase/tokens_per_s",avg_tps,    step)
                writer.add_scalar("phase/speed_steps", spd_steps,  step)
                if dens_summary:
                    for track in ["M","B","H"]:
                        key = f"density_{track}_avg"
                        vals = [d[key] for d in phase_density_stats if key in d]
                        if vals:
                            writer.add_scalar(
                                f"phase/density_{track}", sum(vals)/len(vals), step)

            flog.write(
                f"[PHASE {phase_no}] step={step}"
                f"  train={avg_tr:.4f}  val={latest_val:.4f}"
                f"  gap={gap:+.4f}  best={best_val:.4f}"
                f"  gnorm={avg_gnorm:.3f}  tps={avg_tps:.0f}"
                f"  {vtag}{dens_summary}"
            )

            if step >= MAX_STEPS:
                print(f"  ✓ 训练完成！最优 val_loss={best_val:.4f}")
                flog.write(f"训练完成 val_loss={best_val:.4f}")
                if writer: writer.close()
                flog.close()
                return

            phase_no           += 1
            phase_losses        = []
            phase_gnorms        = []
            phase_speeds        = []
            phase_density_stats = []
            t_phase             = time.time()
            pbar = _pbar(phase_no)
            pbar.set_postfix_str(
                _postfix(0, latest_val, best_val, lr, 0, 0, 0),
                refresh=True,
            )

    pbar.close()
    if writer: writer.close()
    flog.close()

    # ── 训练完成：保存含参数量+时间的最终模型 ──────────────
    dt_end    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _n_params = model.count_params()
    _p_str    = f"{_n_params/1_000_000:.1f}M"
    _d_str    = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # ★ 文件名格式：best_30.2M_20250312_1430.pt
    _final_name = f"best_{_p_str}_{_d_str}.pt"
    _final_path = os.path.join(CHECKPOINT_DIR, _final_name)
    save_ckpt(model, opt, scaler, step, best_val, _final_path,
              n_params=_n_params, train_finish_time=dt_end)
    # 保留 best_model.pt 兼容名（generate.py 默认读取此名称）
    save_ckpt(model, opt, scaler, step, best_val,
              os.path.join(CHECKPOINT_DIR, "best_model.pt"),
              n_params=_n_params, train_finish_time=dt_end)

    print(f"\n  ✓ 训练完成！")
    print(f"  完成时间   {dt_end}")
    print(f"  最优loss   {best_val:.4f}")
    print(f"  参数量     {_p_str}")
    print(f"  最终模型   {_final_name}")
    flog.write(f"训练结束 {dt_end}  best_val={best_val:.4f}  params={_p_str}  ckpt={_final_name}")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    p = argparse.ArgumentParser(description="8-bit 音乐 Transformer 训练")
    p.add_argument("--midi_dir", default=MIDI_DIR,
                   help="MIDI 训练数据目录")
    p.add_argument("--resume",   default=None,
                   help="从检查点续训，例: ./checkpoints/best_model.pt")
    p.add_argument("--rebuild",  action="store_true",
                   help="强制重建 MIDI 缓存（修改了处理逻辑后使用）")
    args = p.parse_args()
    train(args)