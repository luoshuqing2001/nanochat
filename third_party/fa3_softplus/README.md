# fa3_softplus: FlashAttention-3 (Hopper) with softplus-family attention + a fused output RMSNorm

The kernels behind nanochat's `rexp_rmsnorm` and (on SM90) `softplus_rmsnorm` attention kinds,
vendored so that this repo alone is enough to train with them.

| path | what |
|---|---|
| `hopper/` | FlashAttention-3's `hopper/` from Dao-AILab/flash-attention at `e9cf2c1`, with the changes of the `lsq/softplus-fa3` branch (commits `df9191f`, `74b2fe5` there): `softplus.h` and edits to the forward / backward mainloops, epilogues, pre/postprocess kernels and `flash_api_stable.cpp`, all behind `FLASHATTN_SOFTPLUS*` / `FLASHATTN_SP_*` macros (without them it is stock FA3) |
| `hopper_softplus/` | `build.py` (one extension per variant, loadable side by side), `sp_attn.py` (wrappers + float64 references), tests, benchmarks, and the measurements (`RESULTS.md`, `bench_*_h100*.md`) |
| `csrc/cutlass/include/` | the CUTLASS headers FA3 compiles against (NVIDIA/cutlass `7127592`, v4.3.0-48), `csrc/cutlass/LICENSE.txt` |
| `LICENSE` | FlashAttention's license (BSD-3-Clause) |

## Build

    bash trains/build_fa3_softplus.sh            # the variants nanochat uses (~10 min, in parallel)
    bash trains/build_fa3_softplus.sh all        # every variant in build.py (benchmarks)

Needs an SM90 GPU to run them, CUDA >= 12.3 to build (set `CUDA_HOME`; developed with 12.8), and the
repo's `.venv` with torch (the script adds setuptools and ninja). The `.so` files go to
`hopper_softplus/build/<variant>/fa3_<variant>.so` (git-ignored), which is where
`nanochat/fn_rmsnorm_attention.py` looks first (`$NANOCHAT_FA3_FN_DIR` overrides; a
`../flash-attention/hopper_softplus/build` checkout is the fallback).

Default variants and who uses them:

| variant | used for |
|---|---|
| `fn_rexp_pre` | rexp forward (+ the plain backward) |
| `fn_rexp_pre_g`, `fn_rexp_pre_gr` | rexp backward with NANOCHAT_FA3_DGAIN=1 (bit-identical, default) / 2 (+ fused dgamma) |
| `fn_rexp_pre_du` | rexp backward with NANOCHAT_FA3_FUSE_PROJ=1 |
| `split_fn_rexp_pre` | rexp KV-cache inference (split-KV, atomic reduction) |
| `bwd_sp_poly3`, `bwd_sp_poly3_g`, `bwd_sp_poly3_gr`, `split_sp_poly3` | the same four roles for exact softplus |

## Tests (on an H100)

    cd third_party/fa3_softplus && ../../.venv/bin/python hopper_softplus/test_fn.py   # etc.
    python -m pytest tests/test_fn_rmsnorm_attention.py tests/test_fn_dgain.py          # from the repo root

`hopper_softplus/test_*.py` import from their own directory; run them from `third_party/fa3_softplus`.
