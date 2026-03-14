#!/usr/bin/env bash
# =============================================================
#  8-bit Music Generator — 虚拟环境搭建脚本 (Linux / macOS)
#  用法: bash setup_env.sh
# =============================================================
set -e   # 任何命令失败立即退出

VENV_NAME="venv_8bit"
PYTHON_MIN="3.9"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 颜色输出 ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════╗"
echo "║     8-bit 无限音乐生成器 — 环境搭建               ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: 检查系统 Python 版本 ─────────────────────────────
info "Step 1/7 — 检查 Python 版本..."

# 优先用 python3, 再试 python
if command -v python3 &>/dev/null; then
    PYTHON_BIN=$(command -v python3)
elif command -v python &>/dev/null; then
    PYTHON_BIN=$(command -v python)
else
    error "未找到 Python, 请先安装 Python ${PYTHON_MIN}+"
fi

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 9 ]]; then
    error "需要 Python 3.9+, 当前版本: ${PY_VERSION}"
fi
success "Python ${PY_VERSION}  (${PYTHON_BIN})"

# ── Step 2: 检查/安装 venv 模块 ──────────────────────────────
info "Step 2/7 — 检查 venv 模块..."
if ! "$PYTHON_BIN" -m venv --help &>/dev/null; then
    warn "venv 模块不可用, 尝试安装..."
    # Ubuntu/Debian
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y "python${PY_MAJOR}.${PY_MINOR}-venv" python3-venv
    # CentOS/RHEL/Fedora
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3-virtualenv
    elif command -v yum &>/dev/null; then
        sudo yum install -y python3-virtualenv
    # macOS (brew)
    elif command -v brew &>/dev/null; then
        brew install python3
    else
        error "无法自动安装 venv, 请手动安装: pip install virtualenv"
    fi
fi
success "venv 模块可用"

# ── Step 3: 创建虚拟环境 ─────────────────────────────────────
VENV_PATH="${PROJECT_DIR}/${VENV_NAME}"
info "Step 3/7 — 创建虚拟环境: ${VENV_PATH}"

if [ -d "${VENV_PATH}" ]; then
    warn "虚拟环境已存在, 跳过创建 (如需重建请删除 ${VENV_NAME}/ 目录)"
else
    "$PYTHON_BIN" -m venv "${VENV_PATH}" --prompt "8bit"
    success "虚拟环境已创建"
fi

# 激活虚拟环境
source "${VENV_PATH}/bin/activate"
VENV_PYTHON="${VENV_PATH}/bin/python"
VENV_PIP="${VENV_PATH}/bin/pip"
success "虚拟环境已激活: $(which python)"

# ── Step 4: 升级基础工具 ─────────────────────────────────────
info "Step 4/7 — 升级 pip / setuptools / wheel..."
"$VENV_PIP" install --upgrade pip setuptools wheel -q
success "pip $(pip --version | awk '{print $2}')"

# ── Step 5: 检测 GPU / CUDA 版本 ─────────────────────────────
info "Step 5/7 — 检测 GPU 和 CUDA 版本..."

CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" 2>/dev/null || echo "")
    success "GPU: ${GPU_NAME}"
    success "CUDA: ${CUDA_VERSION}"
else
    warn "未检测到 NVIDIA GPU, 将安装 CPU 版本 PyTorch (训练会很慢)"
fi

# 根据 CUDA 版本选择合适的 PyTorch
echo ""
info "Step 5/7 — 安装 PyTorch..."
if [ -z "$CUDA_VERSION" ]; then
    # CPU 版本
    echo "  安装 PyTorch CPU 版本..."
    "$VENV_PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    TORCH_EXTRA="(CPU版)"
else
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    if [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 1 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_EXTRA="cu121"
    elif [[ "$CUDA_MAJOR" -ge 11 && "$CUDA_MINOR" -ge 8 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_EXTRA="cu118"
    elif [[ "$CUDA_MAJOR" -ge 11 && "$CUDA_MINOR" -ge 7 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu117"
        TORCH_EXTRA="cu117"
    else
        warn "CUDA ${CUDA_VERSION} 版本较旧, 使用 cu118 版本"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_EXTRA="cu118"
    fi

    echo "  安装 PyTorch (${TORCH_EXTRA})..."
    "$VENV_PIP" install torch torchvision torchaudio --index-url "${TORCH_INDEX}" -q
fi
success "PyTorch 安装完成 ${TORCH_EXTRA}"

# ── Step 6: 安装项目依赖 ─────────────────────────────────────
info "Step 6/7 — 安装项目依赖 (requirements.txt)..."
"$VENV_PIP" install -r "${PROJECT_DIR}/requirements.txt" -q
success "所有依赖安装完成"

# ── Step 7: 验证安装 ─────────────────────────────────────────
info "Step 7/7 — 验证安装..."

"$VENV_PYTHON" - << 'PYEOF'
import sys
print(f"  Python : {sys.version.split()[0]}")

results = []

# PyTorch
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    results.append(("torch",       f"{torch.__version__}  CUDA={'✓ ' + torch.cuda.get_device_name(0) if cuda_ok else '✗ (CPU模式)'}"))
except ImportError as e:
    results.append(("torch",       f"✗ 未安装: {e}"))

# pretty_midi
try:
    import pretty_midi
    results.append(("pretty_midi", f"{pretty_midi.__version__}  ✓"))
except ImportError as e:
    results.append(("pretty_midi", f"✗ 未安装: {e}"))

# numpy
try:
    import numpy as np
    results.append(("numpy",       f"{np.__version__}  ✓"))
except ImportError as e:
    results.append(("numpy",       f"✗ 未安装: {e}"))

# tqdm
try:
    import tqdm
    results.append(("tqdm",        f"{tqdm.__version__}  ✓"))
except ImportError as e:
    results.append(("tqdm",        f"✗ 未安装: {e}"))

# tensorboard
try:
    import tensorboard
    results.append(("tensorboard", f"{tensorboard.__version__}  ✓"))
except ImportError as e:
    results.append(("tensorboard", f"✗ 未安装: {e}"))

# scipy
try:
    import scipy
    results.append(("scipy",       f"{scipy.__version__}  ✓"))
except ImportError as e:
    results.append(("scipy",       f"✗ 未安装: {e}"))

for pkg, status in results:
    print(f"  {pkg:<15} {status}")

# 检查是否有失败项
failed = [p for p, s in results if "✗" in s]
if failed:
    print(f"\n  ⚠ 以下包安装失败: {failed}")
    sys.exit(1)
else:
    print("\n  所有依赖验证通过 ✓")
PYEOF

# ── 生成快捷激活脚本 ─────────────────────────────────────────
cat > "${PROJECT_DIR}/activate.sh" << EOF
#!/usr/bin/env bash
# 快速激活虚拟环境
source "${VENV_PATH}/bin/activate"
echo "✓ 虚拟环境已激活: 8bit"
echo "  Python: \$(python --version)"
echo "  位置:   \$(which python)"
EOF
chmod +x "${PROJECT_DIR}/activate.sh"

# ── 完成 ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════╗"
echo "║           ✓  环境搭建完成!                        ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  虚拟环境目录:  ${CYAN}${VENV_PATH}${NC}"
echo ""
echo -e "  ${BOLD}每次使用前激活虚拟环境:${NC}"
echo -e "  ${YELLOW}source ${VENV_NAME}/bin/activate${NC}"
echo -e "  或: ${YELLOW}source activate.sh${NC}"
echo ""
echo -e "  ${BOLD}退出虚拟环境:${NC}"
echo -e "  ${YELLOW}deactivate${NC}"
echo ""
echo -e "  ${BOLD}准备 MIDI 数据后开始训练:${NC}"
echo -e "  ${YELLOW}python train.py --midi_dir ./midi_data${NC}"
echo ""
echo -e "  ${BOLD}查看训练过程:${NC}"
echo -e "  ${YELLOW}tensorboard --logdir ./logs${NC}"
echo ""
