#!/usr/bin/env bash
# =============================================================================
# Build the FA3 kernels nanochat's rexp_rmsnorm / softplus_rmsnorm attention run on (SM90 / H100), from
# the sources vendored in third_party/fa3_softplus (FA3's hopper/ with the softplus-family changes,
# the build + test + benchmark tooling in hopper_softplus/, the cutlass headers). The .so files land
# in third_party/fa3_softplus/hopper_softplus/build/<variant>/, where nanochat looks for them
# (nanochat/fn_rmsnorm_attention.py; $NANOCHAT_FA3_FN_DIR overrides).
#
#   bash trains/build_fa3_softplus.sh                    # what training / inference / the switches use
#   bash trains/build_fa3_softplus.sh all                # + every variant in build.py (benchmarks)
#   bash trains/build_fa3_softplus.sh fn_rexp_pre ...    # just these
#   CUDA_HOME=/path/to/cuda-12.8 bash trains/build_fa3_softplus.sh
#
# Needs: CUDA >= 12.3 (sm_90a; the kernels were developed with 12.8), the repo's .venv with torch
# (built against its ABI), setuptools and ninja (installed here if missing). ~5-10 min per variant;
# variants build in parallel, MAX_JOBS nvcc jobs each (default 8).
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${PYTHON_BIN:-}" ] && [ -x .venv/bin/python ]; then PYTHON_BIN=.venv/bin/python; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_PY=third_party/fa3_softplus/hopper_softplus/build.py

# what nanochat uses: training (rexp, exact softplus), NANOCHAT_FA3_DGAIN=1/2, NANOCHAT_FA3_FUSE_PROJ,
# KV-cache inference (split-KV)
DEFAULT_BUILDS="fn_rexp_pre fn_rexp_pre_g fn_rexp_pre_gr fn_rexp_pre_du split_fn_rexp_pre \
bwd_sp_poly3 bwd_sp_poly3_g bwd_sp_poly3_gr split_sp_poly3"
if [ "$#" -eq 0 ]; then BUILDS="$DEFAULT_BUILDS"
elif [ "$1" = "all" ]; then BUILDS=""
else BUILDS="$*"; fi

# CUDA toolkit: $CUDA_HOME, else nvcc on PATH, else /usr/local/cuda; must be >= 12.3
if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null; then CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    else CUDA_HOME=/usr/local/cuda; fi
fi
NVCC="$CUDA_HOME/bin/nvcc"
[ -x "$NVCC" ] || { echo "ERROR: no nvcc at $NVCC; set CUDA_HOME to a CUDA >= 12.3 toolkit" >&2; exit 1; }
CUDA_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if ! "$PYTHON_BIN" -c "import sys; v = tuple(map(int, '$CUDA_VER'.split('.'))); sys.exit(0 if v >= (12, 3) else 1)"; then
    echo "ERROR: $NVCC is CUDA $CUDA_VER; FA3's SM90 kernels need >= 12.3. Set CUDA_HOME, e.g." >&2
    echo "  CUDA_HOME=/path/to/cuda-12.8 bash trains/build_fa3_softplus.sh" >&2
    exit 1
fi
export CUDA_HOME
echo "CUDA $CUDA_VER at $CUDA_HOME | python $("$PYTHON_BIN" -c 'import sys, torch; print(sys.executable, "torch", torch.__version__)')"

# torch.utils.cpp_extension needs setuptools (module) and ninja (executable on PATH)
export PATH="$("$PYTHON_BIN" -c 'import os, sys; print(os.path.dirname(sys.executable))'):$PATH"
need=""
"$PYTHON_BIN" -c "import setuptools" 2>/dev/null || need="$need setuptools"
command -v ninja >/dev/null || need="$need ninja"
if [ -n "$need" ]; then
    echo "installing:$need"
    if command -v uv >/dev/null && uv pip install --python "$PYTHON_BIN" $need; then :
    elif "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || "$PYTHON_BIN" -m ensurepip >/dev/null 2>&1; then
        "$PYTHON_BIN" -m pip install $need
    else
        echo "ERROR: could not install$need into $PYTHON_BIN (no uv, no pip)" >&2; exit 1
    fi
fi
export MAX_JOBS="${MAX_JOBS:-8}"
# shellcheck disable=SC2086
"$PYTHON_BIN" "$BUILD_PY" $BUILDS
echo "loading them back:"
"$PYTHON_BIN" - $BUILDS <<'PY'
import glob, os, sys, torch
root = "third_party/fa3_softplus/hopper_softplus/build"
names = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{root}/*/fa3_*.so"))
for n in names:
    torch.ops.load_library(f"{root}/{n}/fa3_{n}.so")
    getattr(torch.ops, f"fa3_{n}").fwd
    print(f"  ok {n}")
PY
