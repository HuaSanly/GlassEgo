#!/usr/bin/env bash
set -euo pipefail

# Create or update the single compute environment, then install the pinned
# research packages that are not reliably distributed as Conda/PyPI releases.

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${GLASSEGO_CONDA_ENV:-glassego-compute}"

command -v conda >/dev/null 2>&1 || {
    echo "conda is required; install Miniconda or Miniforge first." >&2
    exit 1
}

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 pip -y
fi
conda activate "$ENV_NAME"

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 is required, found {sys.version.split()[0]}")
PY

python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

# Chumpy 0.70 predates Python 3.11 and NumPy 1.24. Keep this compatibility
# adjustment local to the environment rather than carrying a second patch tree.
python - <<'PY'
from pathlib import Path
import site

site_packages = Path(site.getsitepackages()[0])
chumpy = site_packages / "chumpy"
ch_file = chumpy / "ch.py"
if ch_file.is_file():
    text = ch_file.read_text(encoding="utf-8")
    text = text.replace("inspect.getargspec", "inspect.getfullargspec")
    ch_file.write_text(text, encoding="utf-8")
init_file = chumpy / "__init__.py"
if init_file.is_file():
    text = init_file.read_text(encoding="utf-8")
    text = text.replace(
        "from numpy import bool, int, float, complex, object, unicode, str, nan, inf",
        "from numpy import nan, inf",
    )
    init_file.write_text(text, encoding="utf-8")
PY

install_research_package() {
    local name="$1"
    local source="$2"
    echo "Installing ${name} (${source##*@})"
    python -m pip install --no-deps "${name} @ ${source}"
}

install_research_package "cotracker" \
    "git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
install_research_package "orient-anything" \
    "git+https://github.com/2D3DGen/Orient-Anything.git@b6b8d49a30990358800c2c466279643c9003052b"
install_research_package "hamer" \
    "git+https://github.com/geopavlakos/hamer.git@3a01849f4148352e9260b69bf28b65d1671a4905"
install_research_package "easy_ViTPose" \
    "git+https://github.com/JunkyByte/easy_ViTPose.git@bb9860359e55b099a507c8000e360d48a27cc36d"

python - <<'PY'
import importlib

for module in ("torch", "torchvision", "cv2", "omegaconf", "yaml", "training", "preprocess"):
    importlib.import_module(module)

import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

if command -v basalt_vio >/dev/null 2>&1; then
    if ldd "$(command -v basalt_vio)" | grep -q "not found"; then
        echo "Basalt has missing shared libraries." >&2
        exit 1
    fi
    basalt_vio --help >/dev/null
else
    echo "Warning: basalt_vio is not on PATH; VIO recomputation will not work." >&2
fi

echo "Compute environment ready: $ENV_NAME"
