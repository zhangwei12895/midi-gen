"""
check_gpu.py - 验证 GPU 是否真的在训练中被使用
用 nvidia-smi 读取真实的 SM 占用率（不是任务管理器）
"""
import subprocess, sys, time, os, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 1. 用 nvidia-smi 查真实利用率 ────────────────────────────
print("\n[1] nvidia-smi 真实 GPU 利用率")
print("    (任务管理器显示的是 3D/Video 引擎，CUDA 跑在 Compute 引擎，必须用 nvidia-smi 看)\n")

result = subprocess.run(
    ["nvidia-smi",
     "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
if result.returncode == 0:
    parts = [p.strip() for p in result.stdout.strip().split(",")]
    print(f"  GPU 名称     : {parts[0]}")
    print(f"  SM 占用率    : {parts[1]}%   ← 这才是真实 CUDA 计算占用")
    print(f"  显存占用率   : {parts[2]}%")
    print(f"  显存用量     : {parts[3]} / {parts[4]} MB")
    print(f"  温度         : {parts[5]} °C")
else:
    print("  !! nvidia-smi 失败:", result.stderr)
    sys.exit(1)

# ── 2. 实时监控 5 秒（在另一个线程跑 GPU 计算）─────────────
print("\n[2] 实时监控（跑一个简单 GPU 任务，用 nvidia-smi 看变化）")
import torch

if not torch.cuda.is_available():
    print("  !! CUDA 不可用")
    sys.exit(1)

stop_flag = [False]
util_log  = []

def monitor():
    """后台每秒查一次真实 GPU 占用"""
    while not stop_flag[0]:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            vals = r.stdout.strip().split(",")
            util_log.append((float(vals[0].strip()), float(vals[1].strip())))
        time.sleep(0.5)

t = threading.Thread(target=monitor, daemon=True)
t.start()

# 跑一段 GPU 计算
print("  正在 GPU 上运行矩阵乘法 5 秒...")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 5.0:
    x = torch.matmul(x, x)
    x = x / x.norm()  # 防止溢出
torch.cuda.synchronize()

stop_flag[0] = True
time.sleep(0.6)

if util_log:
    avg_util = sum(u for u, _ in util_log) / len(util_log)
    max_util = max(u for u, _ in util_log)
    print(f"  nvidia-smi 采样 {len(util_log)} 次")
    print(f"  平均 SM 占用: {avg_util:.1f}%")
    print(f"  峰值 SM 占用: {max_util:.1f}%")
    if max_util > 10:
        print(f"\n  ✓ GPU 确实在工作！任务管理器显示 0% 是正常的误导")
        print(f"    Windows 任务管理器只显示 DirectX/3D 引擎，不显示 CUDA Compute 引擎")
        print(f"\n  正确查看方式:")
        print(f"    方式1: 新开命令行运行  nvidia-smi dmon -s u -d 1")
        print(f"           看 sm% 列（不是 mem% 列）")
        print(f"    方式2: 任务管理器 → GPU → 下拉选择 'Compute' 引擎（不是 3D）")
    else:
        print(f"\n  !! 即使 nvidia-smi 也显示 GPU 占用很低（{max_util:.0f}%）")
        print(f"     真正的瓶颈可能是：")
        print(f"     1. synchronize() 在每步强制等待，导致 GPU 频繁空闲")
        print(f"     2. batch_size 太小，GPU 很快算完在等数据")
        print(f"     3. seq_len 太短，每批运算量不够")

# ── 3. 检查训练脚本中的 synchronize 调用 ────────────────────
print("\n[3] 检查训练脚本中的性能陷阱")
for fname in ["train_v2.py", "train_fast.py", "train.py"]:
    if not os.path.exists(fname):
        continue
    with open(fname) as f:
        lines = f.readlines()
    sync_lines = [(i+1, l.strip()) for i, l in enumerate(lines)
                  if "synchronize" in l and not l.strip().startswith("#")]
    if sync_lines:
        print(f"  {fname}: 发现 {len(sync_lines)} 处 synchronize() 调用")
        for ln, code in sync_lines:
            print(f"    第{ln}行: {code}")
        print(f"  !! 每步调用 synchronize() 会强制 CPU 等 GPU，大幅降低速度")
    else:
        print(f"  {fname}: 无多余 synchronize() ✓")

print("\n" + "="*55)
print("  结论：")
print("  如果上面 max_util > 10%，GPU 正常工作，只是任务管理器看不到")
print("  训练时用以下命令实时监控真实占用:")
print("    nvidia-smi dmon -s u -d 1")
print("="*55 + "\n")
