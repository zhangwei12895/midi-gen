"""
setup_env.py  —  8-bit Music Generator 虚拟环境搭建
用 Python 本身完成所有安装步骤，避免 PowerShell/BAT 语法问题

运行方式:
  Windows: 双击 install.bat
  手动:    python setup_env.py
"""

import sys
import os
import subprocess
import platform

# ── 颜色输出 ──────────────────────────────────────────────────
IS_WIN = platform.system() == "Windows"

def cprint(msg, color="white"):
    codes = {"green": "\033[92m", "red": "\033[91m",
             "yellow": "\033[93m", "cyan": "\033[96m", "white": ""}
    reset = "\033[0m"
    if IS_WIN:
        # Windows CMD 默认不支持 ANSI，直接打印
        print(msg)
    else:
        print(f"{codes.get(color,'')}{msg}{reset}")

def ok(msg):   cprint(f"  [OK]    {msg}", "green")
def info(msg): cprint(f"  [INFO]  {msg}", "cyan")
def warn(msg): cprint(f"  [WARN]  {msg}", "yellow")
def fail(msg): cprint(f"  [ERROR] {msg}", "red")

def run(cmd, check=True, capture=False):
    """运行命令，返回 (returncode, stdout)"""
    kw = dict(capture_output=capture, text=True) if capture else {}
    result = subprocess.run(cmd, **kw)
    if check and result.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(str(c) for c in cmd)}")
    return result

# ══════════════════════════════════════════════════════════════
def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_name   = "venv_8bit"
    venv_dir    = os.path.join(project_dir, venv_name)

    print()
    print("=" * 52)
    print("   8-bit Music Generator  --  环境搭建")
    print("=" * 52)
    print()

    # ── Step 1: 检查 Python 版本 ──────────────────────────────
    info("Step 1/6  检查 Python 版本...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        fail(f"当前 Python {major}.{minor}，需要 3.9 以上")
        fail("请从 https://www.python.org/downloads/ 下载新版本")
        fail("安装时勾选  Add Python to PATH")
        input("\n按 Enter 退出...")
        sys.exit(1)
    ok(f"Python {major}.{minor}  ({sys.executable})")

    # ── Step 2: 创建虚拟环境 ──────────────────────────────────
    info("Step 2/6  创建虚拟环境...")

    if IS_WIN:
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        venv_pip    = os.path.join(venv_dir, "Scripts", "pip.exe")
        activate_cmd = os.path.join(venv_dir, "Scripts", "activate.bat")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        venv_pip    = os.path.join(venv_dir, "bin", "pip")
        activate_cmd = os.path.join(venv_dir, "bin", "activate")

    if os.path.exists(venv_python):
        warn(f"虚拟环境已存在，跳过创建  ({venv_dir})")
        warn(f"如需重建请先删除文件夹: {venv_name}")
    else:
        run([sys.executable, "-m", "venv", venv_dir, "--prompt", "8bit"])
        if not os.path.exists(venv_python):
            fail("虚拟环境创建失败")
            input("\n按 Enter 退出...")
            sys.exit(1)
        ok(f"虚拟环境已创建: {venv_dir}")

    # ── Step 3: 升级 pip ──────────────────────────────────────
    info("Step 3/6  升级 pip / setuptools / wheel...")
    run([venv_python, "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel", "-q"])
    ok("pip 升级完成")

    # ── Step 4: 检测 CUDA，安装 PyTorch ───────────────────────
    info("Step 4/6  检测 GPU 并安装 PyTorch...")

    torch_index = "https://download.pytorch.org/whl/cpu"
    torch_tag   = "CPU"

    try:
        r = run(["nvidia-smi"], check=False, capture=True)
        if r.returncode == 0:
            import re
            m = re.search(r"CUDA Version: (\d+)\.(\d+)", r.stdout)
            if m:
                cuda_maj, cuda_min = int(m.group(1)), int(m.group(2))
                ok(f"检测到 CUDA {cuda_maj}.{cuda_min}")
                # PyTorch wheel 与 CUDA 驱动版本对照
                # CUDA 驱动向下兼容：更高版本驱动可以运行更低版本的 CUDA wheel
                # 目前 PyTorch 官方最高支持到 cu124（对应 CUDA 12.4）
                if cuda_maj >= 13 or (cuda_maj == 12 and cuda_min >= 4):
                    torch_index = "https://download.pytorch.org/whl/cu124"
                    torch_tag   = "cu124"
                    if cuda_maj >= 13:
                        warn(f"CUDA {cuda_maj}.{cuda_min} 暂无对应 wheel，自动使用向下兼容的 cu124")
                elif cuda_maj == 12 and cuda_min >= 1:
                    torch_index = "https://download.pytorch.org/whl/cu121"
                    torch_tag   = "cu121"
                elif cuda_maj == 12:
                    torch_index = "https://download.pytorch.org/whl/cu121"
                    torch_tag   = "cu121"
                    warn("CUDA 12.0 使用 cu121 兼容版本")
                elif cuda_maj == 11 and cuda_min >= 8:
                    torch_index = "https://download.pytorch.org/whl/cu118"
                    torch_tag   = "cu118"
                else:
                    torch_index = "https://download.pytorch.org/whl/cu118"
                    torch_tag   = "cu118"
                    warn("CUDA 版本较旧，使用 cu118 版 PyTorch")
            else:
                warn("nvidia-smi 运行成功但未找到 CUDA 版本号，使用 CPU 版")
        else:
            warn("未检测到 NVIDIA GPU，安装 CPU 版 PyTorch（训练速度较慢）")
    except FileNotFoundError:
        warn("未找到 nvidia-smi，安装 CPU 版 PyTorch")

    print()
    info(f"安装 PyTorch ({torch_tag})，约 2~3 GB，请耐心等待...")
    print(f"  源: {torch_index}")
    print()

    r = run(
        [venv_pip, "install", "torch", "torchaudio",
         "--index-url", torch_index],
        check=False
    )
    if r.returncode != 0:
        fail("PyTorch 安装失败，请检查网络连接后重试")
        fail(f"手动命令: pip install torch torchaudio --index-url {torch_index}")
        input("\n按 Enter 退出...")
        sys.exit(1)
    ok(f"PyTorch ({torch_tag}) 安装完成")

    # ── Step 5: 安装项目依赖 ──────────────────────────────────
    info("Step 5/6  安装项目依赖 (requirements.txt)...")
    req_file = os.path.join(project_dir, "requirements.txt")
    if not os.path.exists(req_file):
        fail(f"找不到 requirements.txt: {req_file}")
        input("\n按 Enter 退出...")
        sys.exit(1)

    r = run([venv_pip, "install", "-r", req_file], check=False)
    if r.returncode != 0:
        fail("依赖安装失败，请查看上方报错")
        input("\n按 Enter 退出...")
        sys.exit(1)
    ok("项目依赖安装完成")

    # ── Step 6: 验证 ──────────────────────────────────────────
    info("Step 6/6  验证安装...")
    print()

    verify_code = (
        "import sys, importlib\n"
        "print('  Python :', sys.version.split()[0])\n"
        "pkgs = [\n"
        "    ('torch','PyTorch'), ('pretty_midi','PrettyMIDI'),\n"
        "    ('numpy','NumPy'), ('tqdm','tqdm'),\n"
        "    ('tensorboard','TensorBoard'), ('scipy','SciPy'),\n"
        "]\n"
        "failed = []\n"
        "for pkg, name in pkgs:\n"
        "    try:\n"
        "        m = importlib.import_module(pkg)\n"
        "        v = getattr(m, '__version__', '?')\n"
        "        print(f'  {name:<15} {v}')\n"
        "    except ImportError:\n"
        "        print(f'  {name:<15} *** MISSING ***')\n"
        "        failed.append(pkg)\n"
        "try:\n"
        "    import torch\n"
        "    if torch.cuda.is_available():\n"
        "        gpu  = torch.cuda.get_device_name(0)\n"
        "        vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
        "        print('  CUDA          ', gpu, round(vram,1), 'GB')\n"
        "    else:\n"
        "        print('  CUDA           not available (CPU mode)')\n"
        "except Exception as e:\n"
        "    print('  CUDA check:', e)\n"
        "if failed:\n"
        "    sys.exit(1)\n"
        "print()\n"
        "print('  所有依赖验证通过!')\n"
    )

    r = run([venv_python, "-c", verify_code], check=False)
    if r.returncode != 0:
        fail("验证失败，请查看上方错误信息")
        input("\n按 Enter 退出...")
        sys.exit(1)

    # ── 生成 activate.bat / activate.sh ───────────────────────
    if IS_WIN:
        bat_path = os.path.join(project_dir, "activate.bat")
        with open(bat_path, "w", encoding="gbk") as f:
            f.write("@echo off\n")
            f.write(f'call "{activate_cmd}"\n')
            f.write("echo.\n")
            f.write("echo [8bit] 虚拟环境已激活\n")
            f.write("echo.\n")
            f.write("cmd /k\n")
        ok(f"已生成快捷激活脚本: activate.bat")
    else:
        sh_path = os.path.join(project_dir, "activate.sh")
        with open(sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"source \"{activate_cmd}\"\n")
            f.write("echo '[8bit] 虚拟环境已激活'\n")
        os.chmod(sh_path, 0o755)
        ok(f"已生成快捷激活脚本: activate.sh")

    # ── 完成提示 ──────────────────────────────────────────────
    print()
    print("=" * 52)
    ok("环境搭建完成!")
    print("=" * 52)
    print()
    if IS_WIN:
        print("  每次使用前，双击运行:")
        print("    activate.bat")
        print()
        print("  验证环境:   python verify_env.py")
        print("  开始训练:   python train.py --midi_dir .\\midi_data")
        print("  训练曲线:   tensorboard --logdir .\\logs")
    else:
        print("  每次使用前激活虚拟环境:")
        print("    source activate.sh")
        print()
        print("  验证环境:   python verify_env.py")
        print("  开始训练:   python train.py --midi_dir ./midi_data")
        print("  训练曲线:   tensorboard --logdir ./logs")
    print()
    input("按 Enter 退出...")


if __name__ == "__main__":
    main()
