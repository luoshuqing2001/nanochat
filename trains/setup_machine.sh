#!/usr/bin/env bash
# One-shot setup for a fresh machine. Source it, do not run it:
#
#   source trains/setup_machine.sh          # env only (every new shell)
#   INSTALL=1 source trains/setup_machine.sh # env + pip install + tokenizer
#
# Everything lands next to the repo rather than under $HOME, because $HOME is not
# writable everywhere (on unites9 it is not, which fails Triton compilation and the
# nanochat cache with EACCES). HOME itself is left alone -- moving it breaks ~/.ssh
# and ~/.gitconfig, and git is how the code got here.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${NANOCHAT_CACHE_ROOT:-$REPO/.cache}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$CACHE/nanochat}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$REPO/base_data_climbmix}"
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export CUDA_CACHE_PATH="$CACHE/nv"
export XDG_CACHE_HOME="$CACHE"
export HF_HOME="$CACHE/hf"
export WANDB_DIR="$CACHE/wandb"
export PYTHONUSERBASE="$CACHE/python"
export PYTHONUNBUFFERED=1
mkdir -p "$NANOCHAT_BASE_DIR/tokenizer" "$NANOCHAT_DATA_DIR" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$WANDB_DIR" \
         "$PYTHONUSERBASE"

if [ "${INSTALL:-0}" = "1" ]; then
    # quack / tvm_ffi / torch_c_dlpack_ext are hard dependencies of flash_attn_4: without
    # them the import fails and attention falls back to SDPA, taking every softplus CuTe
    # kernel with it.
    # `python -m pip`, never the pip script: on unites9 its shebang points at a
    # miniconda interpreter that is not on this machine, so `pip` itself cannot start.
    # Falling back to --user with PYTHONUSERBASE under the cache keeps the install out of
    # a possibly shared and possibly read-only site-packages.
    PKGS="tiktoken rustbpe wandb apache-tvm-ffi torch-c-dlpack-ext quack-kernels"
    python -m pip install -q $PKGS \
        || python -m pip install -q --user $PKGS \
        || echo "pip failed; see 'python -m pip install $PKGS'"
    cp -n "$REPO/trains/tokenizer/"* "$NANOCHAT_BASE_DIR/tokenizer/" 2>/dev/null || true
fi

echo "repo   $REPO"
echo "cache  $CACHE"
echo "data   $NANOCHAT_DATA_DIR  ($(ls "$NANOCHAT_DATA_DIR" 2>/dev/null | wc -l) files)"
