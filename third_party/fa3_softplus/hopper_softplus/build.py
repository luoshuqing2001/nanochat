"""
Build FA3 (hopper/) twice from the same sources: stock softmax and softplus+RMSNorm variants.

Only hdim 128, bf16, forward, SM90 is compiled -- that is the nanochat training shape, and it
keeps a build to one kernel TU. Every variant uses identical nvcc flags (copied from
hopper/setup.py), so a timing difference between stock and softplus is the attention function,
not the toolchain.

    python hopper_softplus/build.py                 # build all variants (in parallel)
    python hopper_softplus/build.py stock poly3     # just these

Each variant is a separate .so registering its ops under its own namespace
(torch.ops.fa3_stock.fwd, torch.ops.fa3_sp_poly3.fwd, ...), so all can be loaded at once.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOPPER = os.path.join(REPO, "hopper")
def _cuda_home():
    """$CUDA_HOME, else the toolkit of the nvcc on PATH, else /usr/local/cuda. Needs CUDA >= 12.3 (sm_90a)."""
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    import shutil
    nvcc = shutil.which("nvcc")
    return os.path.dirname(os.path.dirname(os.path.realpath(nvcc))) if nvcc else "/usr/local/cuda"


CUDA_HOME = _cuda_home()
BUILD_ROOT = os.path.join(HERE, "build")

# name -> softplus impl (None = stock softmax). See hopper/softplus.h for what each one is.
VARIANTS = {
    "stock": None,
    "sp_mufu2": 0,
    "sp_poly3": 1,
    "sp_poly4": 2,
    "sp_softexp": 3,
    "sp_mix": 4,
    "sp_naive": 5,
    "sp_poly3x": 6,
    "diag_identity": 7,
    "diag_exp": 8,
    "pre_naive": 9,
    "pre_poly3": 10,
    "pre_poly2": 11,
    "pre_mix": 12,
    # the same softplus kernels plus the atomic split-KV path (hopper/epilogue_fwd.hpp)
    "split_pre_poly3": (10, "split"),
    "split_sp_poly3x": (6, "split"),
    "split_diag_exp": (8, "split"),
    "split_fn_rexp_pre": (20, "split", "prescaled"),  # rexp forward + atomic split-KV (decode / long prefill)
    "split_sp_poly3": (1, "split"),  # bwd_sp_poly3's forward (exact softplus, kSpPoly3) + atomic split-KV
    # GQA / MQA decode: PackGQA on (a group's query heads share the CTA and one read of their KV head)
    "stock_pgqa": (None, "pgqa"),
    "fn_rexp_pre_pgqa": (20, "prescaled", "pgqa"),
    "split_fn_rexp_pre_pgqa": (20, "split", "prescaled", "pgqa"),
    # forward + backward (hdim 128 bf16); softplus backward recomputes A with kSpPoly3
    "bwd_stock": (None, "bwd"),
    "bwd_pre_poly3": (10, "bwd", "prescaled", "bwdimpl=10"),
    "bwd_pre_naive": (10, "bwd", "prescaled", "bwdimpl=9"),   # fwd poly3, bwd lg2(1+ex2): MUFU has room there
    "bwd_sp_poly3": (1, "bwd"),
    "bwd_diag_exp": (8, "bwd", "prescaled", "bwdimpl=8"),  # timing floor only: not a softplus gradient
    # forward tile search on pre_poly3: tile=M,N_causal,N_noncausal,MmaPV_is_RS,IntraWGOverlap
    "tile_n96": (10, "tile=128,96,128,1,1"),
    "tile_n112": (10, "tile=128,112,144,1,1"),
    "tile_n160": (10, "tile=128,160,160,1,1"),
    "tile_nc128": (10, "tile=128,128,128,1,1"),
    "tile_m192": (10, "tile=192,128,128,0,1"),
    "tile_noovl": (10, "tile=128,128,176,1,0"),
    "tile_ss": (10, "tile=128,128,176,0,1"),
    # softplus-shaped replacement (hopper/softplus.h kFnRexp), forward + backward
    "fn_rexp": (20, "bwd", "bwdimpl=20"),
    # rexp with scale*log2e folded into Q upstream: call with q' = q*scale*log2e, softmax_scale = ln2
    "fn_rexp_pre": (20, "bwd", "bwdimpl=20", "prescaled", "bwdcfg=2"),
    # backward takes dU directly (RMSNorm backward fused into nanochat's c_proj grad GEMM)
    "fn_rexp_pre_du": (20, "bwd", "bwdimpl=20", "prescaled", "bwdcfg=2", "dudirect"),
    # dgain: the backward takes dZ = dL/d(O * gain) and the fp32 gain instead of dO, and rebuilds dO in
    # the preprocess exactly as torch.compile's kernel would have (bf16(fp32(dZ) * gain), RNE): same bits,
    # one fewer elementwise pass over the activations (read dZ + write dO) in the caller's backward
    "fn_rexp_pre_g": (20, "bwd", "bwdimpl=20", "prescaled", "bwdcfg=2", "dgain"),
    "bwd_sp_poly3_g": (1, "bwd", "dgain"),
    # dgainred: also dgain = sum over tokens of dZ * O, per CTA in that preprocess (it reads both), so the
    # caller drops its own reduction pass over dZ and O; fp32 sums in a different order: not bit-exact
    "fn_rexp_pre_gr": (20, "bwd", "bwdimpl=20", "prescaled", "bwdcfg=2", "dgainred"),
    "bwd_sp_poly3_gr": (1, "bwd", "dgainred"),
    # dqfuse: main CTAs count their dQ contributions per row block (atomic release adds); converter CTAs
    # appended to the grid convert each completed row block to bf16 and zero it; dQaccum persists
    # all-zero across calls: no dQaccum memset, no postprocess kernel (it overlaps the main kernel's tail)
    "fn_rexp_pre_dqf": (20, "bwd", "bwdimpl=20", "prescaled", "bwdcfg=2", "dqfuse"),
    "fn_rexp_pre_du_dqf": (20, "bwd", "bwdimpl=20", "prescaled", "bwdcfg=2", "dudirect", "dqfuse"),
    "bwd_stock_dqf": (None, "bwd", "dqfuse"),
}

DISABLED = ["PAGEDKV", "APPENDKV", "SOFTCAP", "PACKGQA", "FP16", "FP8", "VARLEN",
            "CLUSTER", "HDIM64", "HDIM96", "HDIM192", "HDIM256", "SM8x"]

NVCC_FLAGS = [  # hopper/setup.py, verbatim except --resource-usage kept for register counts
    "-O3", "-std=c++17", "--ftemplate-backtrace-limit=0", "--use_fast_math", "--resource-usage",
    "-lineinfo", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG", "-gencode", "arch=compute_90a,code=sm_90a",
    "--threads", "4",
]


def lib_path(name):
    return os.path.join(BUILD_ROOT, name, f"fa3_{name}.so")


def build(name):
    os.environ["CUDA_HOME"] = CUDA_HOME
    os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ["PATH"]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("MAX_JOBS", "8")
    from torch.utils.cpp_extension import load
    impl = VARIANTS[name]
    opts = impl[1:] if isinstance(impl, tuple) else ()
    split, bwd = "split" in opts, "bwd" in opts
    bwd_impl = [o.split("=")[1] for o in opts if o.startswith("bwdimpl=")]
    impl = impl[0] if isinstance(impl, tuple) else impl
    pgqa = "pgqa" in opts  # PackGQA on: GQA/MQA query heads of a group packed into the M tile
    feature = [f"-DFLASHATTENTION_DISABLE_{f}" for f in DISABLED if not (pgqa and f == "PACKGQA")]
    feature += ["-DFLASH_ATTENTION_DISABLE_HDIMDIFF64", "-DFLASH_ATTENTION_DISABLE_HDIMDIFF192"]
    feature += [f"-DFA3_LIB_NS=fa3_{name}"]
    if not bwd:
        feature += ["-DFLASHATTENTION_DISABLE_BACKWARD"]
    if "dgain" in opts:  # dout = dZ for Z = O * gain (fp32 gain): dO rebuilt bit-exactly in the preprocess
        feature += ["-DFLASHATTN_SOFTPLUS_DOUT_GAIN"]
    if "dgainred" in opts:  # ... and dgain = sum(dZ * O) reduced in the same preprocess pass (not bit-exact)
        feature += ["-DFLASHATTN_SOFTPLUS_DOUT_GAIN", "-DFLASHATTN_SOFTPLUS_DGAIN_REDUCE"]
    if "dqfuse" in opts:  # any function, stock softmax too: dQ memset + postprocess folded into the bwd kernel
        feature += ["-DFLASHATTN_SP_DQFUSE"]
    sources = [os.path.join(HOPPER, "flash_api_stable.cpp"),
               os.path.join(HOPPER, "instantiations/flash_fwd_hdim128_bf16_sm90.cu"),
               os.path.join(HOPPER, "flash_prepare_scheduler.cu")]
    if bwd:
        sources += [os.path.join(HOPPER, "instantiations/flash_bwd_hdim128_bf16_sm90.cu")]
    if pgqa:  # FA3 packs whenever it splits, so a split build needs the packed split instantiation
        sources += [os.path.join(HOPPER, "instantiations/flash_fwd_hdim128_bf16_packgqa_sm90.cu")]
        if split:
            sources += [os.path.join(HOPPER, "instantiations/flash_fwd_hdim128_bf16_split_sm90.cu")]
    if impl is None:  # stock keeps split-KV + combine, for the decode comparison
        sources += [os.path.join(HOPPER, "instantiations/flash_fwd_hdim128_bf16_split_sm90.cu"),
                    os.path.join(HOPPER, "flash_fwd_combine.cu")]
    else:
        feature += ["-DFLASHATTN_SOFTPLUS", f"-DFLASHATTN_SOFTPLUS_IMPL={impl}"]
        if bwd_impl:
            feature += [f"-DFLASHATTN_SOFTPLUS_IMPL_BWD={bwd_impl[0]}"]
        tile = [o.split("=")[1] for o in opts if o.startswith("tile=")]
        if tile:
            m, nc, n, rs, ov = tile[0].split(",")
            feature += [f"-DFLASHATTN_SP_TILE_M={m}", f"-DFLASHATTN_SP_TILE_NC={nc}", f"-DFLASHATTN_SP_TILE_N={n}",
                        f"-DFLASHATTN_SP_TILE_RS={rs}", f"-DFLASHATTN_SP_TILE_OV={ov}"]
        bwd_tile = [o.split("=")[1] for o in opts if o.startswith("bwdtile=")]
        if bwd_tile:
            t = bwd_tile[0].split(",")
            # one macro per value: nvcc splits a -D option at commas
            feature += [f"-DFLASHATTN_SP_BWD_M={t[0]}", f"-DFLASHATTN_SP_BWD_N={t[1]}"]
            feature += [f"-DFLASHATTN_SP_BWD_P{i}={x}" for i, x in enumerate(t[2:])]
        bwd_cfg = [o.split("=")[1] for o in opts if o.startswith("bwdcfg=")]
        if bwd_cfg:
            feature += [f"-DFLASHATTN_SP_BWD_CFG={bwd_cfg[0]}"]
        if "dudirect" in opts:
            feature += ["-DFLASHATTN_SOFTPLUS_DOUT_IS_DU"]
        if "prescaled" in opts:
            feature += ["-DFLASHATTN_SOFTPLUS_PRESCALED"]
        if split:  # combine.cu only to satisfy the linker: run_mha_fwd_combine is never called
            sources += [os.path.join(HERE, "inst_split_nopackgqa_sm90.cu"),
                        os.path.join(HOPPER, "flash_fwd_combine.cu")]
        else:
            feature += ["-DFLASHATTENTION_DISABLE_SPLIT"]
    bdir = os.path.join(BUILD_ROOT, name)
    os.makedirs(bdir, exist_ok=True)
    load(
        name=f"fa3_{name}",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17", "-DTORCH_TARGET_VERSION=0x0209000000000000"] + feature,
        extra_cuda_cflags=NVCC_FLAGS + feature,
        extra_include_paths=[HOPPER, os.path.join(REPO, "csrc/cutlass/include"),
                             os.path.join(CUDA_HOME, "targets/x86_64-linux/include")],
        build_directory=bdir,
        is_python_module=False,
        is_standalone=False,
        verbose=True,
    )
    return name


def load_variant(name):
    """Load a built variant into this process; its ops are torch.ops.fa3_<name>.*"""
    import torch
    torch.ops.load_library(lib_path(name))
    return getattr(torch.ops, f"fa3_{name}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    with ProcessPoolExecutor(max_workers=len(names)) as ex:
        for n in ex.map(build, names):
            print(f"built {n}: {lib_path(n)}", flush=True)
