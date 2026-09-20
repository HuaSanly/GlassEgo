#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/environment/compute-cu121.yml"
LOCK_FILE="$(dirname "$ENV_FILE")/compute-cu121-linux-64.lock"
PIP_FILE="$(dirname "$ENV_FILE")/pip-compute-cu121.txt"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${GLASSEGO_CONDA_ENV:-glassego-compute}"

command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
else
    conda create --name "$ENV_NAME" --file "$LOCK_FILE" -y
fi
conda activate "$ENV_NAME"
"$PROJECT_ROOT/scripts/build_chumpy_wheel.sh" "$PROJECT_ROOT/environment/wheels"
python -m pip install --no-deps -r "$PIP_FILE"
python -m pip install --no-deps "$PROJECT_ROOT"/environment/wheels/chumpy-0.70-*.whl
python "$PROJECT_ROOT/scripts/check_pip_environment.py"

python - <<'PY'
import importlib
import torch

assert torch.__version__.startswith("2.5.1"), torch.__version__
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
for module in (
    "cv2", "omegaconf", "open3d", "onnxruntime", "sam2", "transformers",
    "cotracker", "orient_anything", "hamer", "easy_ViTPose", "mediapipe",
):
    importlib.import_module(module)
    print(f"import_ok={module}")
PY

if command -v basalt_vio >/dev/null; then
    ldd "$(command -v basalt_vio)" | grep -q "not found" && {
        echo "Basalt has missing shared libraries" >&2
        exit 1
    }
    basalt_vio --help >/dev/null
else
    echo "Basalt not found; install it separately before running preprocessing." >&2
fi

echo "compute environment ready: $ENV_NAME"
