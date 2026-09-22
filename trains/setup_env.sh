#!/usr/bin/env bash
# =============================================================================
# Set up a machine to run these experiments, and check it is actually usable.
#
#   bash trains/setup_env.sh              # check only, install nothing
#   INSTALL=1 bash trains/setup_env.sh    # also pip install the missing packages
#
# It never installs torch: that choice depends on the GPU and the CUDA version, and
# getting it wrong is the main way this goes bad. The script checks the torch you
# have and prints the right command if it is unusable.
#
# Verified on 1x NVIDIA GB10 (sm121). For Blackwell datacenter parts (B200/B300,
# sm100/sm103) the same steps apply and FA4 uses its native SM100 kernels instead of
# the SM80-derived sm120 ones; see "Other GPUs" in trains/README.md.
# =============================================================================
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL="${INSTALL:-0}"
ok=0; warn=0; bad=0
say()  { printf '  %-28s %s\n' "$1" "$2"; }
good() { say "$1" "OK    $2"; ok=$((ok+1)); }
soft() { say "$1" "WARN  $2"; warn=$((warn+1)); }
fail() { say "$1" "FAIL  $2"; bad=$((bad+1)); }

echo "=== GPU and driver ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | sed 's/^/  /'
else
    fail "nvidia-smi" "not found"
fi

echo
echo "=== PyTorch ==="
"$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"  {'torch':<28} FAIL  not importable: {e}")
    sys.exit(0)
print(f"  {'torch':<28} OK    {torch.__version__} (cuda {torch.version.cuda})")
if not torch.cuda.is_available():
    print(f"  {'cuda':<28} FAIL  torch.cuda.is_available() is False")
    sys.exit(0)
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}"
archs = torch.cuda.get_arch_list()
family = [a for a in archs if a.startswith(f"sm_{cap[0]}")]
print(f"  {'device':<28} OK    {torch.cuda.get_device_name(0)}, compute capability {cap[0]}.{cap[1]} ({arch})")
print(f"  {'torch arch list':<28} {'OK   ' if family else 'FAIL '} {archs}")
if not family:
    print(f"  {'':<28}       this build has no sm_{cap[0]}x kernels; install a newer wheel, e.g.")
    print(f"  {'':<28}       pip install --index-url https://download.pytorch.org/whl/cu130 torch")
elif arch not in archs:
    print(f"  {'':<28}       {arch} is not built explicitly but {family} is; same-family")
    print(f"  {'':<28}       binaries normally run (this is how GB10/sm121 uses sm_120)")
bf16 = cap >= (8, 0)
print(f"  {'bf16 tensor cores':<28} {'OK   ' if bf16 else 'WARN '} capability {'>=' if bf16 else '<'} 8.0")
PY

echo
echo "=== Python packages ==="
NANOCHAT_PKGS="pyarrow tiktoken rustbpe regex psutil numpy filelock requests wandb yaml"
FA4_PKGS="cutlass einops tvm_ffi torch_c_dlpack_ext quack typing_extensions"
PIP_NAME_cutlass="nvidia-cutlass-dsl"
PIP_NAME_tvm_ffi="apache-tvm-ffi"
PIP_NAME_torch_c_dlpack_ext="torch-c-dlpack-ext"
PIP_NAME_quack="quack-kernels"
PIP_NAME_yaml="pyyaml"
missing=""
for mod in $NANOCHAT_PKGS $FA4_PKGS; do
    if "$PYTHON_BIN" -c "import $mod" >/dev/null 2>&1; then
        good "$mod" ""
    else
        var="PIP_NAME_$mod"
        pipname="${!var:-$mod}"
        fail "$mod" "missing (pip install $pipname)"
        missing="$missing $pipname"
    fi
done
if [ -n "$missing" ] && [ "$INSTALL" = "1" ]; then
    echo
    echo "installing:$missing"
    "$PYTHON_BIN" -m pip install $missing && echo "re-run this script to verify"
fi

echo
echo "=== nanochat assets ==="
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
DATA_DIR="${NANOCHAT_DATA_DIR:-$REPO_DIR/base_data_climbmix}"
if [ -f "$BASE_DIR/tokenizer/tokenizer.pkl" ] && [ -f "$BASE_DIR/tokenizer/token_bytes.pt" ]; then
    good "tokenizer" "$BASE_DIR/tokenizer"
elif [ -f "$REPO_DIR/trains/tokenizer/tokenizer.pkl" ]; then
    if [ "$INSTALL" = "1" ]; then
        mkdir -p "$BASE_DIR/tokenizer"
        cp "$REPO_DIR/trains/tokenizer/tokenizer.pkl" "$REPO_DIR/trains/tokenizer/token_bytes.pt" "$BASE_DIR/tokenizer/"
        good "tokenizer" "installed from trains/tokenizer -> $BASE_DIR/tokenizer"
    else
        soft "tokenizer" "not in $BASE_DIR/tokenizer, but trains/tokenizer has it -- re-run with INSTALL=1, or: cp trains/tokenizer/* \"$BASE_DIR/tokenizer/\""
    fi
else
    fail "tokenizer" "missing in $BASE_DIR/tokenizer and in trains/tokenizer -- run: python -m scripts.tok_train (note: a re-trained tokenizer is not the same one, see trains/tokenizer/README.md)"
fi
shards=$(ls "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
if [ "$shards" -gt 1 ]; then
    good "data shards" "$shards in $DATA_DIR"
else
    fail "data shards" "none in $DATA_DIR -- set NANOCHAT_DATA_DIR, or: python -m nanochat.dataset -n 200"
fi

echo
echo "=== Triton toolchain (needed by ATTN_KIND=softplus) ==="
"$PYTHON_BIN" - <<'PY'
import subprocess, sys
try:
    import torch, triton, os
except Exception as e:
    print(f"  {'triton':<28} FAIL  not importable: {e}")
    raise SystemExit(0)
print(f"  {'triton':<28} OK    {triton.__version__}")
if not torch.cuda.is_available():
    raise SystemExit(0)
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}a"
ptxas = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas")
if not os.path.exists(ptxas):
    print(f"  {'ptxas':<28} WARN  not at {ptxas}")
else:
    ver = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout
    rel = [l for l in ver.splitlines() if "release" in l]
    helptext = subprocess.run([ptxas, "--help"], capture_output=True, text=True).stdout
    ok = arch in helptext
    print(f"  {'ptxas':<28} {'OK   ' if ok else 'FAIL '} {rel[0].strip() if rel else '?'}")
    print(f"  {'ptxas knows ' + arch:<28} {'OK' if ok else 'FAIL  this is the sm_103a failure; upgrade torch (triton 3.8 ships CUDA 12.9 ptxas)'}")
# the real test: compile and run a kernel on this device
try:
    import triton.language as tl
    src = ("import torch, triton, triton.language as tl\n"
           "@triton.jit\n"
           "def _k(X, Y, N: tl.constexpr):\n"
           "    i = tl.arange(0, N)\n"
           "    tl.store(Y + i, tl.log(1.0 + tl.exp(-tl.abs(tl.load(X + i)))))\n"
           "x = torch.randn(64, device='cuda'); y = torch.empty_like(x)\n"
           "_k[(1,)](x, y, N=64); torch.cuda.synchronize(); print('ok')\n")
    open("/tmp/_triton_probe.py", "w").write(src)
    r = subprocess.run([sys.executable, "/tmp/_triton_probe.py"], capture_output=True, text=True)
    good = r.stdout.strip().endswith("ok")
    print(f"  {'compiles a kernel':<28} {'OK' if good else 'FAIL  ' + r.stderr.strip().splitlines()[-1][:90] if r.stderr else 'FAIL'}")
except Exception as e:
    print(f"  {'compiles a kernel':<28} FAIL  {e}")
# and the real attention kernel end to end
try:
    from nanochat.softplus_attention import softplus_attn_func
    q, k, v = (torch.randn(2, 512, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o = softplus_attn_func(q, k, v, window_size=(255, 0))
    print(f"  {'softplus attention':<28} OK    forward {tuple(o.shape)}")
except Exception as e:
    print(f"  {'softplus attention':<28} FAIL  {type(e).__name__}: {str(e)[:80]}")
PY

echo
echo "=== attention backend ==="
"$PYTHON_BIN" - <<'PY'
import torch
try:
    import nanochat.flash_attention as fa
    import flash_attn_4.fa3_compat as c4
except Exception as e:
    print(f"  {'import':<28} FAIL  {e}")
    raise SystemExit(0)
print(f"  {'resolved backend':<28} {fa.impl_name()}")
print(f"  {'HAS_FA3 / HAS_FA4':<28} {fa.HAS_FA3} / {fa.HAS_FA4}")
print(f"  {'FA4 custom ops':<28} {c4.has_custom_op()}")
if c4.load_error() is not None:
    print(f"  {'FA4 import error':<28} {c4.load_error()}")
if torch.cuda.is_available():
    major = torch.cuda.get_device_capability()[0]
    kernels = {8: "SM80", 9: "SM90", 10: "SM100 (native Blackwell)",
               11: "SM100 family", 12: "SM120 (SM80-derived)"}.get(major, "unsupported")
    print(f"  {'FA4 kernel family':<28} {kernels}; FP8 available: {major == 10}")
    from nanochat.common import get_peak_flops
    pf = get_peak_flops(torch.cuda.get_device_name(0))
    print(f"  {'peak FLOPS entry':<28} " + (f"{pf:.2e}" if pf != float('inf')
          else "missing for this GPU -> MFU will print 0.00 (add it to nanochat/common.py)"))
PY

echo
echo "=== summary: $ok ok, $warn warn, $bad fail ==="
if [ "$bad" -eq 0 ]; then
    echo "Next: DRY_RUN=1 bash trains/train_d12_hybrid_swa_muon.sh"
    echo "      SMOKE=1   bash trains/train_d12_hybrid_swa_muon.sh"
    echo "      python trains/bench_attention.py     # pick DEVICE_BATCH_SIZE / backend"
fi
exit 0
