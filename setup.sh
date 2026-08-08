#!/usr/bin/env bash
# ==============================================================================
# HumanEgo — 一键式环境设置
# ==============================================================================
#
# 用法：
#   git clone https://github.com/TX-Leo/HumanEgo.git
#   cd HumanEgo
#   conda create -n humanego python=3.11 -y
#   conda activate humanego
#   bash setup.sh
#
# 该脚本的作用：
#   [1] 验证 conda 环境是否处于活动状态
#   [2] 安装 numpy + chumpy （特殊构建顺序）
#   [3] 从 requirements.txt 安装核心依赖项
#   [4] 安装基于 git 的软件包（CoTracker、Orient-Anything）
#   [5] 安装手部追踪包（MediaPipe、WiLoR、HaMeR）
#   [6] 修补 chumpy 和 HaMeR 以实现兼容性
#   [7] 验证所有导入
#
# 安装选项（环境变量）。默认面向“下载预计算数据集并训练”的常见流程，
# 因此跳过机器人硬件和替代手部跟踪包，也不预下载模型权重。
# 每次运行可通过以下环境变量覆盖：
#   SKIP_HARDWARE=1  （默认）跳过 pyrealsense2 和 trossen-arm；设为 0 才安装
#   SKIP_HAND=1      （默认）跳过 MediaPipe / WiLoR / HaMeR；设为 0 才安装
#   PREDOWNLOAD=0    （默认）不预下载权重；设为 1 立即下载
#
# 示例：
#   SKIP_HAND=0 bash setup.sh                  # + MediaPipe/WiLoR/HaMeR （仅用于手部跟踪消融；已发布的 aria_mps 管道不需要它们）
#   SKIP_HARDWARE=0 SKIP_HAND=0 bash setup.sh  # 完整安装（机器人+相机+手）
#
# ==============================================================================

set -euo pipefail

# ── 颜色 ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
_SETUP_T0=$(date +%s); _STEP_T0=""
step()  {
    if [ -n "$_STEP_T0" ]; then echo -e "${GREEN}   ✓ previous step done in $(( $(date +%s) - _STEP_T0 ))s${NC}"; fi
    echo -e "\n${BLUE}══════════════════════════════════════════${NC}"; echo -e "${BLUE} $*${NC}"; echo -e "${BLUE}══════════════════════════════════════════${NC}"
    _STEP_T0=$(date +%s)
}

# ── 项目根──
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
info "Project root: $PROJECT_ROOT"

# ── 安装选项（单一事实来源；有关说明，请参阅标题）──
# 默认轻量级：跳过机器人硬件和替代手部跟踪，不预下载模型权重。
# 覆盖每次运行，例如： SKIP_HARDWARE=0 SKIP_HAND=0 bash setup.sh
SKIP_HARDWARE="${SKIP_HARDWARE:-1}"
SKIP_HAND="${SKIP_HAND:-1}"
PREDOWNLOAD="${PREDOWNLOAD:-0}"

# ==============================================================================
# [1/7] 验证 Conda 环境
# ==============================================================================
step "[1/7] Verifying conda environment"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    error "No conda environment is active (or you're in 'base').
Please run:
  conda create -n humanego python=3.11 -y
  conda activate humanego
  bash setup.sh"
fi

PYTHON="$(which python)"
PIP="$(which pip)"
info "Active conda env: $CONDA_DEFAULT_ENV"
info "Python: $($PYTHON --version) at $PYTHON"

# ==============================================================================
# [2/7] 首先安装 numpy + chumpy （特殊构建顺序）
# ==============================================================================
step "[2/7] Installing numpy and chumpy (requires special build order)"

$PIP install numpy
info "NumPy installed: $($PYTHON -c 'import numpy; print(numpy.__version__)')"

# Chumpy 需要已安装 numpy 且 --no-build-isolation
# Chumpy 0.70 — smplx/hamer/wilor 需要 MANO 手模型
$PIP install --no-build-isolation chumpy
# 不要在这里执行 `import chumpy`：chumpy 0.70 使用 Python 3.11 已移除的
# inspect.getargspec，只有完成步骤 [6/7] 的补丁后才能正常导入。此处仅通过
# pip 元数据确认安装，避免输出无害但容易误解的回溯。
info "chumpy installed: $($PIP show chumpy 2>/dev/null | awk '/^Version:/{print $2}') (made importable by the patch in step [6/7])"

# ==============================================================================
# [3/7] 来自 requirements.txt 的核心依赖项
# ==============================================================================
step "[3/7] Installing core dependencies from requirements.txt"

# --- Project Aria 包：强制使用精确版本，忽略依赖冲突 ---
info "Installing projectaria-tools and projectaria-client-sdk (--no-deps to avoid conflicts)..."
$PIP install --no-deps projectaria-tools==1.7.1
$PIP install --no-deps projectaria-client-sdk==1.1.0

# 如果没有硬件（没有 robot/camera），则过滤掉这些包
# 始终过滤掉 projectaria-tools/client-sdk （上面已使用 --no-deps 安装）
FILTER_PATTERN="^projectaria-tools|^projectaria-client-sdk"
if [ "$SKIP_HARDWARE" = "1" ]; then
    info "SKIP_HARDWARE=1 — filtering out pyrealsense2 and trossen-arm"
    FILTER_PATTERN="$FILTER_PATTERN|^pyrealsense2|^trossen-arm"
fi
grep -v -E "$FILTER_PATTERN" requirements.txt > /tmp/requirements_filtered.txt
$PIP install -r /tmp/requirements_filtered.txt
rm /tmp/requirements_filtered.txt

# 确保 numpy 没有降级
NUMPY_VER=$($PYTHON -c "import numpy; print(numpy.__version__)")
info "NumPy after install: $NUMPY_VER"

info "[3/7] Core dependencies installed"

# ==============================================================================
# [4/7] 基于 Git 的软件包（CoTracker、Orient-Anything）
# ==============================================================================
step "[4/7] Installing git-based packages"

# --- CoTracker (Facebook 研究) ---
info "Installing CoTracker (from GitHub)..."
$PIP install "cotracker @ git+https://github.com/facebookresearch/co-tracker.git"

# --- Orient-Anything V2（物体方向估计器）---
info "Installing Orient-Anything V2 (from GitHub)..."
$PIP install "orient-anything @ git+https://github.com/TX-Leo/orient-anything.git"

info "[4/7] Git-based packages installed"

# ==============================================================================
# [5/7] 手部追踪包 (MediaPipe / WiLoR / HaMeR)
# ==============================================================================
if [ "$SKIP_HAND" = "1" ]; then
    step "[5/7] Skipping hand tracking packages (SKIP_HAND=1)"
else
    step "[5/7] Installing hand tracking packages"

    # --- MediaPipe ---
    info "Installing mediapipe..."
    $PIP install mediapipe

    # --- WiLoR-mini (--no-deps 以避免覆盖 torch/ultralytics) ---
    info "Installing wilor-mini (from GitHub, --no-deps)..."
    $PIP install --no-deps "wilor-mini @ git+https://github.com/warmshao/WiLoR-mini.git"

    # --- HaMeR （--no-deps 以避免拉动 detectron2/mmcv） ---
    info "Installing hamer (from GitHub, --no-deps)..."
    $PIP install --no-deps "hamer @ git+https://github.com/geopavlakos/hamer.git"

    # --- easy_ViTPose （ViTPose 手部探测器用于 HaMeR，替换 MediaPipe） ---
    info "Installing easy_ViTPose (from GitHub)..."
    $PIP install "easy_ViTPose @ git+https://github.com/JunkyByte/easy_ViTPose.git"

    # --- WiLoR/HaMeR/ViTPose 所需的间接依赖 ---
    info "Installing indirect dependencies for hand tracking..."
    $PIP install yacs pytorch-lightning gdown xtcocotools webdataset filterpy ffmpeg-python

    # 确保 numpy 版本没有改变
    NUMPY_NOW=$($PYTHON -c "import numpy; print(numpy.__version__)")
    if [ "$NUMPY_NOW" != "$NUMPY_VER" ]; then
        warn "numpy changed from $NUMPY_VER to $NUMPY_NOW, restoring..."
        $PIP install "numpy==$NUMPY_VER"
    fi

    info "[5/7] Hand tracking packages installed"
fi

# ==============================================================================
# [6/7] 补丁
# ==============================================================================
step "[6/7] Applying compatibility patches"

PYVER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITE_PACKAGES="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"

# --- 针对 Python 3.11+ / NumPy 2.x 的补丁 chumpy ---
CHUMPY_DIR="$SITE_PACKAGES/chumpy"
if [ -d "$CHUMPY_DIR" ]; then
    # 修复 getargspec -> getfullargspec （在 Python 3.11 中删除）
    if grep -q 'inspect\.getargspec' "$CHUMPY_DIR/ch.py" 2>/dev/null; then
        info "Patching chumpy/ch.py: getargspec -> getfullargspec"
        sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' "$CHUMPY_DIR/ch.py"
    else
        info "chumpy/ch.py already patched"
    fi

    # 修复已删除的 numpy 类型别名
    if grep -q "from numpy import bool" "$CHUMPY_DIR/__init__.py" 2>/dev/null; then
        info "Patching chumpy/__init__.py: removing deprecated numpy imports"
        sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf$/from numpy import nan, inf/' "$CHUMPY_DIR/__init__.py"
    else
        info "chumpy/__init__.py already patched"
    fi
else
    warn "chumpy not found at $CHUMPY_DIR, skipping patches"
fi

# --- 补丁 HaMeR vitdet_dataset.py （禁止调试打印）---
HAMER_VITDET=$($PYTHON -c "
try:
    import hamer.datasets.vitdet_dataset as m
    print(m.__file__)
except:
    pass
" 2>/dev/null || echo "")

if [ -n "$HAMER_VITDET" ] && [ -f "$HAMER_VITDET" ]; then
    if grep -q "print(f'{downsampling_factor=}')" "$HAMER_VITDET" 2>/dev/null; then
        info "Patching HaMeR vitdet_dataset.py: removing debug print"
        sed -i "/print(f'{downsampling_factor=}')/d" "$HAMER_VITDET"
    else
        info "HaMeR vitdet_dataset.py already patched"
    fi
fi

info "[6/7] Patches applied"

# ==============================================================================
# [7/7] 验证
# ==============================================================================
step "[7/7] Verifying installation"

VERIFY_OK=true

check_import() {
    local name="$1"
    local code="$2"
    if $PYTHON -c "$code" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $name"
    else
        echo -e "  ${RED}✗${NC} $name"
        VERIFY_OK=false
    fi
}

echo ""
info "=== Core Packages ==="
check_import "torch (CUDA)"          "import torch; assert torch.cuda.is_available(), 'no CUDA'"
check_import "torchvision"           "import torchvision"
check_import "numpy"                 "import numpy"
check_import "scipy"                 "import scipy"
check_import "opencv"                "import cv2"
check_import "open3d"                "import open3d"
check_import "PIL"                   "from PIL import Image"

echo ""
info "=== ML Models ==="
check_import "transformers"          "import transformers"
check_import "sam2"                  "import sam2"
check_import "diffusers"             "import diffusers"
check_import "timm"                  "import timm"
check_import "rembg"                 "import rembg"
check_import "onnxruntime"           "import onnxruntime"
check_import "xformers"              "import xformers"
check_import "ultralytics"           "import ultralytics"

echo ""
info "=== Git Packages ==="
check_import "cotracker"             "import cotracker"
check_import "orient-anything"       "import orient_anything; from orient_anything.vision_tower import VGGT_OriAny_Ref"

echo ""
info "=== Project Aria ==="
check_import "projectaria-tools"     "import projectaria_tools"

echo ""
info "=== Config & Utils ==="
check_import "omegaconf"             "import omegaconf"
check_import "hydra"                 "import hydra"
check_import "wandb"                 "import wandb"
check_import "gradio"                "import gradio"
check_import "rich"                  "import rich"
check_import "pydantic"              "import pydantic"

echo ""
info "=== 3D / Pose ==="
check_import "roma"                  "import roma"
check_import "smplx"                 "import smplx"
check_import "chumpy"                "import chumpy"
check_import "pyrender"              "import pyrender"

if [ "$SKIP_HARDWARE" != "1" ]; then
    echo ""
    info "=== Hardware ==="
    check_import "pyrealsense2"      "import pyrealsense2"
    check_import "trossen_arm"       "import trossen_arm"
fi

if [ "$SKIP_HAND" != "1" ]; then
    echo ""
    info "=== Hand Tracking ==="
    check_import "mediapipe"         "import mediapipe"
    check_import "wilor-mini"        "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline"
    check_import "hamer"             "import hamer"
    check_import "easy_ViTPose"      "from easy_ViTPose import VitInference"
fi

echo ""
info "=== HumanEgo Modules ==="
cd "$PROJECT_ROOT"
check_import "preprocess.OrientAnything"  "from preprocess.OrientAnything import estimate_frame_pca1, estimate_frame_vlm, ORIENT_ANYTHING_AVAILABLE; assert ORIENT_ANYTHING_AVAILABLE"
check_import "utils.utils_math"           "from utils.utils_math import rotmat_to_o6d, normalize_o6d"
check_import "utils.utils_io"             "from utils.utils_io import load_cfg"

# ==============================================================================
# Optional: 预下载模型权重
# ==============================================================================
if [ "$PREDOWNLOAD" = "1" ]; then
    step "Pre-downloading model weights..."
    $PYTHON -c "
from huggingface_hub import hf_hub_download

print('  Downloading hand_landmarker.task ...')
hf_hub_download(repo_id='Leo-TX/mediapipe-hand', filename='hand_landmarker.task')

for f in ['hamer.ckpt', 'model_config.yaml', 'dataset_config.yaml', 'mano_mean_params.npz']:
    print(f'  Downloading {f} ...')
    hf_hub_download(repo_id='Leo-TX/hamer', filename=f)

print('  Downloading MANO_RIGHT.pkl ...')
hf_hub_download(repo_id='warmshao/WiLoR-mini', subfolder='pretrained_models', filename='MANO_RIGHT.pkl')

print('  Downloading Orient-Anything V2 checkpoint ...')
hf_hub_download(repo_id='Viglong/OriAnyV2_ckpt', filename='demo_ckpts/rotmod_realrotaug_best.pt', repo_type='model')

print('  Downloading ViTPose-H wholebody checkpoint ...')
hf_hub_download(repo_id='JunkyByte/easy_ViTPose', filename='torch/wholebody/vitpose-h-wholebody.pth')

print('  Downloading YOLOv8s for ViTPose ...')
hf_hub_download(repo_id='JunkyByte/easy_ViTPose', filename='yolov8/yolov8s.pt')

print('Done!')
" || warn "Pre-download failed (models will be auto-downloaded on first run)"
fi

# ==============================================================================
# 概括
# ==============================================================================
echo ""
if [ "$VERIFY_OK" = true ]; then
    step "✓ Setup Complete!"
    echo ""
    echo "  Environment:  $CONDA_DEFAULT_ENV"
    echo "  Total time:   $(( ($(date +%s) - _SETUP_T0) / 60 ))m $(( ($(date +%s) - _SETUP_T0) % 60 ))s"
    echo "  Python:       $($PYTHON --version)"
    echo "  PyTorch:      $($PYTHON -c 'import torch; print(torch.__version__)')"
    echo "  CUDA:         $($PYTHON -c 'import torch; print(torch.version.cuda)')"
    echo "  NumPy:        $($PYTHON -c 'import numpy; print(numpy.__version__)')"
    echo ""
    echo "  All packages verified successfully!"
    echo ""
    echo "  Quick start:"
    echo "    conda activate $CONDA_DEFAULT_ENV"
    echo "    python scripts/download_data.py --task serve_bread --num 2 --input-only   # 下载示例数据"
    echo "    python -m preprocess.Preprocess --mps_path ./data/serve_bread/aria/mps_serve_bread_000_vrs --task serve_bread   # 预处理"
    echo "    python -m training.FlowMatchingTrainer --task serve_bread --use_cfg --job HumanEgo   # 训练"
    echo ""
else
    step "⚠ Setup finished with some warnings"
    echo ""
    echo "  Some packages failed verification. Check the ✗ marks above."
    echo "  Common fixes:"
    echo "    - GPU not available: ensure NVIDIA drivers are installed"
    echo "    - Hardware (pyrealsense2/trossen-arm): skipped by default; SKIP_HARDWARE=0 to install"
    echo "    - Hand tracking (MediaPipe/WiLoR/HaMeR): skipped by default; SKIP_HAND=0 to install"
    echo ""
fi
