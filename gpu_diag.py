"""
gpu_diag.py — GPU 训练问题诊断 + 性能基准测试
逐步排查 GPU 占用为 0 的原因
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*55)
print("  GPU 训练诊断工具")
print("="*55)

# ══════════════════════════════════════════════════════════════
# 1. 基础 GPU 检查
# ══════════════════════════════════════════════════════════════
print("\n[1] GPU 基础检查")
import torch

print(f"  PyTorch 版本  : {torch.__version__}")
print(f"  CUDA 可用     : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n  !! CUDA 不可用，模型运行在 CPU 上，速度极慢")
    print("  解决: 确认显卡驱动已安装，重装 cu124 版 PyTorch")
    sys.exit(1)

print(f"  GPU 名称      : {torch.cuda.get_device_name(0)}")
print(f"  VRAM 总量     : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print(f"  CUDA 版本     : {torch.version.cuda}")

# ══════════════════════════════════════════════════════════════
# 2. 纯 GPU 运算速度基准（排除数据加载问题）
# ══════════════════════════════════════════════════════════════
print("\n[2] 纯 GPU 矩阵运算基准")

device = torch.device("cuda")
sizes = [512, 1024, 2048]
for sz in sizes:
    x = torch.randn(sz, sz, device=device, dtype=torch.float16)
    # 预热
    for _ in range(3):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 50 * 1000
    print(f"  {sz}x{sz} fp16 matmul : {ms:.2f}ms")

# ══════════════════════════════════════════════════════════════
# 3. 模型单步前向传播速度
# ══════════════════════════════════════════════════════════════
print("\n[3] 模型单步前向传播速度")
from model.transformer import MusicTransformer
from data.tokenizer import MusicTokenizer
from config import MODEL_CONFIG, TRAIN_CONFIG

tok = MusicTokenizer()
model = MusicTransformer(vocab_size=tok.vocab_size, config=MODEL_CONFIG).to(device)
model.train()

B = TRAIN_CONFIG["batch_size"]
T = TRAIN_CONFIG["seq_len"]
print(f"  batch_size={B}  seq_len={T}")

ids = torch.randint(1, tok.vocab_size, (B, T), device=device)
tgt = torch.randint(1, tok.vocab_size, (B, T), device=device)

# 预热
with torch.autocast(device_type="cuda", dtype=torch.float16):
    _, loss, _ = model(ids, tgt)
loss.backward()
model.zero_grad()
torch.cuda.synchronize()

# 正式计时：前向
times_fwd = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, loss, _ = model(ids, tgt)
    torch.cuda.synchronize()
    times_fwd.append((time.time() - t0) * 1000)

# 前向 + 反向
times_full = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, loss, _ = model(ids, tgt)
    loss.backward()
    torch.cuda.synchronize()
    times_full.append((time.time() - t0) * 1000)
    model.zero_grad()

fwd_ms  = sum(times_fwd)  / len(times_fwd)
full_ms = sum(times_full) / len(times_full)
print(f"  前向传播      : {fwd_ms:.1f}ms")
print(f"  前向+反向     : {full_ms:.1f}ms")
print(f"  纯 GPU 理论速度: ~{1000/full_ms:.1f} steps/s")

# VRAM
peak = torch.cuda.max_memory_allocated() / 1024**2
total = torch.cuda.get_device_properties(0).total_memory / 1024**2
print(f"  VRAM 占用     : {peak:.0f}MB / {total:.0f}MB ({peak/total*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# 4. 数据加载速度测试（找出瓶颈）
# ══════════════════════════════════════════════════════════════
print("\n[4] 数据加载速度测试")
import pickle
from torch.utils.data import DataLoader
from data.dataset import MusicDataset

cache_path = "./processed/sequences.pkl"
if not os.path.exists(cache_path):
    print("  !! 缓存不存在，先运行: python train.py --midi_dir ./midi_data")
    print("     (等报错或出现 [Dataset] 行后 Ctrl+C 停止，缓存就建好了)")
else:
    with open(cache_path, "rb") as f:
        seqs = pickle.load(f)
    print(f"  缓存序列数    : {len(seqs)}")

    ds = MusicDataset(seqs, tok, seq_len=T, augment=False)
    print(f"  数据集样本数  : {len(ds)}")

    for nw in [0]:
        loader = DataLoader(ds, batch_size=B, shuffle=True,
                            num_workers=nw, pin_memory=True, drop_last=True)
        it = iter(loader)
        # 预热
        next(it)
        t0 = time.time()
        for i, (inp, tgt_) in enumerate(it):
            if i >= 19:
                break
        load_ms = (time.time() - t0) / 20 * 1000
        print(f"  num_workers={nw} : {load_ms:.1f}ms/batch")

    # 关键比较
    print()
    if load_ms > full_ms * 3:
        print(f"  !! 瓶颈在数据加载 ({load_ms:.0f}ms) >> GPU计算 ({full_ms:.0f}ms)")
        print(f"     → 数据加载比 GPU 计算慢 {load_ms/full_ms:.1f}x")
        print(f"     建议: 减小 seq_len 或 batch_size")
    else:
        print(f"  数据加载 {load_ms:.0f}ms  GPU计算 {full_ms:.0f}ms  比例正常")

# ══════════════════════════════════════════════════════════════
# 5. 检测 pin_memory / non_blocking 效果
# ══════════════════════════════════════════════════════════════
print("\n[5] GPU 数据传输速度")
dummy = torch.randint(1, tok.vocab_size, (B, T))

t0 = time.time()
for _ in range(100):
    x = dummy.to(device, non_blocking=False)
torch.cuda.synchronize()
ms_blocking = (time.time() - t0) / 100 * 1000

t0 = time.time()
dummy_pin = dummy.pin_memory()
for _ in range(100):
    x = dummy_pin.to(device, non_blocking=True)
torch.cuda.synchronize()
ms_nonblock = (time.time() - t0) / 100 * 1000

print(f"  blocking     : {ms_blocking:.2f}ms")
print(f"  non_blocking : {ms_nonblock:.2f}ms")

# ══════════════════════════════════════════════════════════════
# 6. 推荐优化配置
# ══════════════════════════════════════════════════════════════
print("\n[6] 针对你的硬件推荐配置")

steps_per_sec = 1000 / full_ms
hours_per_200k = 200000 / steps_per_sec / 3600

print(f"  GPU 理论训练速度 : {steps_per_sec:.1f} steps/s")
print(f"  完成 200k 步预计 : {hours_per_200k:.1f} 小时")
print()

# 根据实际速度给出建议
if full_ms > 500:
    print("  !! 单步超过 500ms，seq_len 可能过大")
    print("  建议修改 config.py:")
    print('    "seq_len"    : 256,   # 从512降到256')
    print('    "batch_size" : 8,     # 从16降到8')
elif full_ms > 200:
    print("  单步 200~500ms，正常范围")
    print("  建议修改 config.py:")
    print('    "seq_len"    : 384,   # 可以适当降低')
else:
    print("  单步 < 200ms，GPU 运算速度良好")
    print("  如果显示器上 GPU 占用仍为 0，是任务管理器采样频率问题")
    print("  用 nvidia-smi dmon 实时查看:")
    print("    nvidia-smi dmon -s u -d 1")

print()
print("="*55)
print("  诊断完成")
print("="*55 + "\n")
