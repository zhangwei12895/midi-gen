"""
环境验证脚本
运行: python verify_env.py
逐项检查所有依赖是否正确安装, 并测试核心模块是否可以正常导入
"""

import sys
import os
import importlib
import platform
import time

# ── 颜色输出 (终端) ───────────────────────────────────────────
class C:
    OK     = "\033[92m"  # 绿
    WARN   = "\033[93m"  # 黄
    FAIL   = "\033[91m"  # 红
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def ok(msg):   print(f"  {C.OK}✓{C.RESET}  {msg}")
def warn(msg): print(f"  {C.WARN}!{C.RESET}  {msg}")
def fail(msg): print(f"  {C.FAIL}✗{C.RESET}  {msg}")
def title(msg):print(f"\n{C.BOLD}{C.CYAN}{msg}{C.RESET}")

print(f"\n{C.BOLD}{'='*55}")
print(f"   8-bit Music Generator — 环境验证")
print(f"{'='*55}{C.RESET}\n")

issues = []

# ── 1. 系统信息 ───────────────────────────────────────────────
title("[ 1 ] 系统信息")
print(f"  操作系统  : {platform.system()} {platform.release()}")
print(f"  Python    : {sys.version.split()[0]}  ({sys.executable})")
print(f"  虚拟环境  : {os.environ.get('VIRTUAL_ENV', '未激活 !')}")

if not os.environ.get("VIRTUAL_ENV"):
    warn("当前不在虚拟环境中! 建议先激活: source venv_8bit/bin/activate")
    issues.append("未使用虚拟环境")
else:
    ok(f"虚拟环境: {os.environ['VIRTUAL_ENV']}")

# ── 2. Python 版本检查 ────────────────────────────────────────
title("[ 2 ] Python 版本")
major, minor = sys.version_info[:2]
if major == 3 and minor >= 9:
    ok(f"Python {major}.{minor} (需要 ≥ 3.9)")
else:
    fail(f"Python {major}.{minor} 不满足要求 (需要 ≥ 3.9)")
    issues.append(f"Python版本过低: {major}.{minor}")

# ── 3. 第三方包检查 ───────────────────────────────────────────
title("[ 3 ] 依赖包检查")

PACKAGES = [
    ("torch",        "2.0.0",  "PyTorch"),
    ("torchaudio",   "2.0.0",  "TorchAudio"),
    ("pretty_midi",  "0.2.9",  "PrettyMIDI"),
    ("numpy",        "1.24.0", "NumPy"),
    ("tqdm",         "4.65.0", "tqdm"),
    ("tensorboard",  "2.13.0", "TensorBoard"),
    ("scipy",        "1.11.0", "SciPy"),
    ("matplotlib",   "3.7.0",  "Matplotlib"),
    ("midiutil",     "1.2.1",  "MIDIUtil"),
]

def ver_tuple(v_str):
    try:
        return tuple(int(x) for x in v_str.split(".")[:3])
    except:
        return (0,)

for pkg, min_ver, display in PACKAGES:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        if ver != "?" and ver_tuple(ver) < ver_tuple(min_ver):
            warn(f"{display:<15} {ver} (建议 ≥ {min_ver})")
        else:
            ok(f"{display:<15} {ver}")
    except ImportError:
        fail(f"{display:<15} 未安装!")
        issues.append(f"缺少包: {pkg}")

# ── 4. PyTorch & CUDA 详细检查 ────────────────────────────────
title("[ 4 ] PyTorch & CUDA")
try:
    import torch
    ok(f"PyTorch 版本    : {torch.__version__}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            ok(f"GPU {i}: {props.name}")
            ok(f"  VRAM          : {vram_gb:.1f} GB")
            ok(f"  CUDA Compute  : {props.major}.{props.minor}")
            if vram_gb < 3.5:
                warn(f"  VRAM 较低 ({vram_gb:.1f}GB), 建议降低 batch_size 到 4")
            elif vram_gb < 6:
                ok(f"  VRAM 足够 (batch_size=16 可用)")
            else:
                ok(f"  VRAM 充裕 (可尝试更大 batch_size)")
        ok(f"CUDA 版本       : {torch.version.cuda}")

        # 简单 CUDA 运算测试
        t0 = time.time()
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        ok(f"CUDA 矩阵运算测试: {ms:.1f}ms  ✓")
    else:
        warn("CUDA 不可用 — 将使用 CPU 训练 (速度较慢)")
        warn("如有 NVIDIA GPU, 请重新安装对应 CUDA 版本的 PyTorch")

    # fp16 支持
    if torch.cuda.is_available():
        x_fp16 = torch.randn(100, 100, device="cuda", dtype=torch.float16)
        ok("fp16 混合精度   : 支持 ✓")
    else:
        warn("fp16 混合精度   : 仅 GPU 模式支持")

except ImportError:
    fail("PyTorch 未安装!")
    issues.append("PyTorch 未安装")

# ── 5. 项目模块导入测试 ───────────────────────────────────────
title("[ 5 ] 项目模块检查")

# 将项目根目录加入 path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

modules_to_test = [
    ("config",               "配置文件"),
    ("data.tokenizer",       "Tokenizer"),
    ("data.midi_processor",  "MIDI处理器"),
    ("data.dataset",         "Dataset"),
    ("model.transformer",    "Transformer模型"),
    ("model.music_theory",   "乐理约束"),
    ("utils.midi_utils",     "MIDI工具"),
]

for mod_name, display in modules_to_test:
    try:
        mod = importlib.import_module(mod_name)
        ok(f"{display:<15} 导入成功")
    except ImportError as e:
        fail(f"{display:<15} 导入失败: {e}")
        issues.append(f"模块导入失败: {mod_name}")
    except Exception as e:
        warn(f"{display:<15} 警告: {e}")

# ── 6. Tokenizer 功能测试 ─────────────────────────────────────
title("[ 6 ] Tokenizer 功能测试")
try:
    from data.tokenizer import MusicTokenizer
    tok = MusicTokenizer()
    ok(f"词汇表大小      : {tok.vocab_size}")
    ok(f"NOTE_ON 范围    : {tok.note_on_range()}")
    ok(f"NOTE_DUR 范围   : {tok.note_dur_range()}")

    # 编码/解码往返测试
    encoded = tok.encode_pitch(60)   # C4
    decoded = tok.decode_token(encoded)
    assert decoded["pitch"] == 60, f"编解码不一致: {decoded}"
    ok("编码/解码往返   : 通过 ✓")

    encoded_key = tok.encode_key("C_maj")
    decoded_key = tok.decode_token(encoded_key)
    ok(f"调性编码        : C_maj → {encoded_key} → {decoded_key['key']}")

except Exception as e:
    fail(f"Tokenizer 测试失败: {e}")
    issues.append(f"Tokenizer异常: {e}")

# ── 7. 模型实例化测试 ─────────────────────────────────────────
title("[ 7 ] 模型实例化测试")
try:
    import torch
    from data.tokenizer import MusicTokenizer
    from model.transformer import MusicTransformer
    from config import MODEL_CONFIG

    tokenizer = MusicTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicTransformer(vocab_size=tokenizer.vocab_size, config=MODEL_CONFIG)
    model = model.to(device)
    ok(f"模型创建        : {model.count_params():,} 参数")

    # 前向传播测试
    dummy = torch.randint(0, tokenizer.vocab_size, (2, 32), device=device)
    target = torch.randint(0, tokenizer.vocab_size, (2, 32), device=device)
    with torch.no_grad():
        logits, loss, _ = model(dummy, target)
    ok(f"前向传播        : logits {tuple(logits.shape)}, loss={loss.item():.4f}")

    # 推理速度测试
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    ms_per = (time.time() - t0) / 10 * 1000
    ok(f"推理速度        : {ms_per:.1f}ms/批次 (bs=2, seq=32)")

    param_info = model.get_num_params()
    ok(f"参数分布: 嵌入={param_info['embedding']:,} | Transformer={param_info['transformer']:,}")

except Exception as e:
    fail(f"模型测试失败: {e}")
    import traceback; traceback.print_exc()
    issues.append(f"模型异常: {e}")

# ── 8. MIDI 数据检查 ──────────────────────────────────────────
title("[ 8 ] MIDI 数据目录")
midi_dir = os.path.join(PROJECT_ROOT, "midi_data")
if os.path.exists(midi_dir):
    midi_files = [f for f in os.listdir(midi_dir) if f.lower().endswith((".mid", ".midi"))]
    if len(midi_files) == 0:
        warn(f"midi_data/ 目录为空, 请添加 .mid 文件才能训练")
        warn("推荐数据集: NES Music Database (https://github.com/chrisdonahue/nesmdb)")
        issues.append("midi_data/ 为空")
    else:
        ok(f"发现 {len(midi_files)} 个 MIDI 文件")
        if len(midi_files) < 50:
            warn(f"文件数量较少 ({len(midi_files)}), 建议至少 200 首")
        elif len(midi_files) < 200:
            warn(f"文件数量({len(midi_files)})偏少, 500+ 效果更好")
        else:
            ok(f"数据量充足 ({len(midi_files)} 首)")
else:
    warn("midi_data/ 目录不存在, 将自动创建")
    os.makedirs(midi_dir, exist_ok=True)
    issues.append("midi_data/ 不存在(已创建)")

# ── 汇总报告 ──────────────────────────────────────────────────
print(f"\n{C.BOLD}{'='*55}{C.RESET}")
if not [i for i in issues if "MIDI" not in i and "虚拟" not in i]:
    print(f"{C.OK}{C.BOLD}  ✓ 所有检查通过! 环境准备就绪{C.RESET}")
    print(f"\n  下一步: 将 MIDI 文件放入 midi_data/ 目录, 然后运行:")
    print(f"  {C.CYAN}python train.py --midi_dir ./midi_data{C.RESET}")
else:
    print(f"{C.FAIL}{C.BOLD}  发现以下问题需要解决:{C.RESET}")
    for issue in issues:
        print(f"  {C.FAIL}✗{C.RESET} {issue}")
print(f"{C.BOLD}{'='*55}{C.RESET}\n")
