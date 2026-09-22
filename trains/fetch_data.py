"""
Download only as much ClimbMix as the planned runs actually need.

`python -m nanochat.dataset -n N` has two problems for this repo: N is a raw shard
count with no connection to the model you intend to train, and it always also pulls
shard_06542 (upstream's MAX_SHARD), which then sorts last and silently becomes your
validation set -- invalidating comparisons against every earlier run.

This script sizes the download from the model instead, and pins the validation shard.

    python trains/fetch_data.py --depth 24              # compute-optimal d24, 25% margin
    python trains/fetch_data.py --depth 24 --dry-run    # just the plan
    python trains/fetch_data.py --tokens 100e9          # explicit token budget
    python trains/fetch_data.py --tokens 100e9 --epochs 4   # allow 4 passes over the data
    python trains/fetch_data.py --depth 24 --prune --yes    # also delete surplus shards

How the sizing works:
  tokens needed = 12 x scaling_params(depth)   (the data:param ratio base_train targets)
  unique tokens = tokens / epochs              (see --epochs)
  shards        = ceil(unique tokens x margin / 44.8e6)

--depth sizes a compute-optimal run. Architecture ablations at ~1B usually want far
more than that: the standard setup in the linear/efficient-attention literature is a
1.3B model on 100B tokens (GLA, DeltaNet, Gated Slot Attention, Gated DeltaNet-2,
Physics of LMs 4.1), which is ~77 tokens/param rather than 12. Pass --tokens for that.

--epochs trades storage for data reuse. Muennighoff et al., "Scaling Data-Constrained
Language Models" (NeurIPS 2023), found up to ~4 epochs of repeated data costs
negligible loss compared with fresh tokens, so a 100B-token run can be fed from 25B
unique tokens at a quarter of the disk. nanochat's dataloader cycles automatically and
reports the epoch in the training log.

44.8M tokens/shard is measured, not assumed: the finished d12 run consumed
2,520 x 524,288 = 1.321B tokens and ended at parquet 29, row group 43 of 84, i.e.
29.51 shards. One shard is ~88 MB on disk, so roughly 521M tokens per GB.

Validation set: nanochat uses the *last* shard in sorted order (dataset.py), so the
highest-numbered file you keep defines it. --val-shard defaults to 2499, which is what
the runs in logs/ used; changing it makes past bpb numbers incomparable.
"""

import argparse
import math
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKENS_PER_SHARD = 44.8e6   # measured, see the docstring
SHARD_MB = 88               # measured average
DATA_PARAM_RATIO = 12       # base_train's --target-param-data-ratio default


def scaling_params(depth, aspect_ratio=64, head_dim=128, seq_len=2048, vocab_size=32768):
    """transformer matrices + lm_head, the parameter count base_train sizes data from."""
    import torch
    from nanochat.gpt import GPT, GPTConfig
    dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    cfg = GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=depth,
                    n_head=dim // head_dim, n_kv_head=dim // head_dim, n_embd=dim)
    with torch.device("meta"):
        model = GPT(cfg)
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"], dim, counts["total"]


def verify_yield(tokens, n_train, n_shards):
    """Measure what the dataloader actually delivers, instead of trusting TOKENS_PER_SHARD.

    The raw token count of a shard is not what training consumes: the BOS best-fit
    packer crops documents to fill rows exactly and discards the remainder. Measured
    here: 54.1M raw -> 44.9M delivered, i.e. 17% lost (the docstring in dataloader.py
    says ~35%, which does not match this corpus at seq_len 2048).
    """
    import time
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit as loader
    from nanochat.tokenizer import get_tokenizer

    print(f"\nverifying: consuming {n_shards} shards through the real dataloader (CPU, ~{25*n_shards}s)...")
    B, T = 16, 2048
    delivered, marks = 0, {}
    for _x, _y, st in loader(get_tokenizer(), B, T, split="train", device="cpu"):
        delivered += B * T
        pq = st["pq_idx"]
        marks.setdefault(pq, delivered)
        if pq >= n_shards + 1:
            break
    # skip shard 0: the document buffer is still filling, which biases it
    per_shard = (marks[n_shards + 1] - marks[1]) / n_shards
    total = n_train * per_shard
    print(f"  delivered: {per_shard/1e6:.1f}M tokens per shard "
          f"(the estimate used for sizing was {TOKENS_PER_SHARD/1e6:.1f}M)")
    print(f"  {n_train} train shards -> {total/1e9:.1f}B tokens for a {tokens/1e9:.0f}B budget"
          f"  ({100*(total-tokens)/tokens:+.0f}%)")
    if total < tokens:
        print(f"  SHORT by {(tokens-total)/1e9:.1f}B: training would wrap around and repeat data. "
              f"Re-run with a larger --margin.", file=sys.stderr)
        return 1
    print("  OK: enough for a single pass")
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--depth", type=int, help="size the download for a compute-optimal run at this depth")
    g.add_argument("--tokens", type=float, help="explicit token budget, e.g. 20e9")
    ap.add_argument("--margin", type=float, default=1.25, help="safety factor over the computed need (default 1.25)")
    ap.add_argument("--epochs", type=int, default=1,
                    help="how many passes over the data the run may take (default 1). Up to ~4 costs "
                         "almost nothing in loss (Muennighoff et al. 2023) and divides the disk need.")
    ap.add_argument("--val-shard", type=int, default=2499,
                    help="shard index reserved as the validation set; must stay the highest kept index (default 2499)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="after checking the plan, run the real dataloader over a few shards and report "
                         "how many tokens it actually delivers per shard (~90 s, CPU only)")
    ap.add_argument("--verify-shards", type=int, default=3, help="shards to consume when verifying (default 3)")
    ap.add_argument("--prune", action="store_true", help="also delete shards outside the keep set")
    ap.add_argument("--yes", action="store_true", help="actually delete when --prune is given")
    args = ap.parse_args()

    # Default to the same directory the launcher reads from. Without this, the shards
    # land in $NANOCHAT_BASE_DIR/base_data_climbmix (nanochat.dataset's default) while
    # _engine_hybrid_swa_muon.sh looks in <repo>/base_data_climbmix, and the preflight
    # fails after a 200 GB download. Must be set before nanochat.dataset is imported,
    # since it resolves DATA_DIR at import time.
    if not os.environ.get("NANOCHAT_DATA_DIR"):
        repo_default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "base_data_climbmix")
        os.environ["NANOCHAT_DATA_DIR"] = repo_default
        print(f"NANOCHAT_DATA_DIR not set, defaulting to the launcher's location: {repo_default}")

    import nanochat.dataset as ds  # honors NANOCHAT_DATA_DIR

    if args.depth:
        sp, dim, total = scaling_params(args.depth)
        tokens = DATA_PARAM_RATIO * sp
        print(f"d{args.depth}: dim {dim}, {total/1e9:.2f}B params, {sp/1e6:.0f}M scaling params")
        print(f"compute-optimal horizon: {DATA_PARAM_RATIO} x scaling params = {tokens/1e9:.2f}B tokens")
    else:
        tokens = args.tokens
        print(f"requested budget: {tokens/1e9:.2f}B tokens")

    if args.epochs < 1:
        print("ERROR: --epochs must be >= 1", file=sys.stderr)
        return 1
    unique = tokens / args.epochs
    need = unique * args.margin
    n_train = math.ceil(need / TOKENS_PER_SHARD)
    if args.epochs > 1:
        print(f"over {args.epochs} epochs: {unique/1e9:.2f}B unique tokens needed"
              + ("  (>4 epochs starts to cost measurable loss)" if args.epochs > 4 else ""))
    print(f"with a {args.margin}x margin: {need/1e9:.2f}B unique tokens -> {n_train} train shards "
          f"(~{n_train * SHARD_MB / 1024:.0f} GB) + 1 validation shard")

    if n_train > args.val_shard:
        budget_arg = f"--depth {args.depth}" if args.depth else f"--tokens {args.tokens:g}"
        # floor to 2 decimals: rounding up would push the shard count back past the val shard
        fits_margin = math.floor(args.val_shard * TOKENS_PER_SHARD / tokens * 100) / 100
        min_epochs = math.ceil(need / (args.val_shard * TOKENS_PER_SHARD))
        print(f"\nERROR: {n_train} train shards would run past --val-shard {args.val_shard}.", file=sys.stderr)
        print(f"nanochat validates on the highest-numbered shard, so going past it silently "
              f"redefines the validation set and makes past runs incomparable.\n", file=sys.stderr)
        print("Pick one:", file=sys.stderr)
        if fits_margin > 1.0:
            n_tight = math.ceil(tokens * min(fits_margin, args.margin) / TOKENS_PER_SHARD)
            print(f"  a) keep the budget, trim the margin to {fits_margin:.2f} (the budget still fits "
                  f"below the val shard):\n     python trains/fetch_data.py {budget_arg} "
                  f"--margin {fits_margin:.2f}   -> {n_tight} shards, ~{n_tight*SHARD_MB/1024:.0f} GB",
                  file=sys.stderr)
        print(f"  b) reuse data instead of storing it ({min_epochs} epochs; up to ~4 is nearly free, "
              f"Muennighoff et al. 2023):\n     python trains/fetch_data.py {budget_arg} "
              f"--epochs {min_epochs}", file=sys.stderr)
        print(f"  c) move the validation shard, accepting that past bpb numbers no longer compare:\n"
              f"     python trains/fetch_data.py {budget_arg} --val-shard {n_train + 1}", file=sys.stderr)
        return 1

    data_dir = ds.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    keep = set(range(n_train)) | {args.val_shard}
    present = {int(f.split("_")[1].split(".")[0])
               for f in os.listdir(data_dir) if f.endswith(".parquet")}
    missing = sorted(keep - present)
    surplus = sorted(present - keep)

    print(f"\nsource:   {ds.BASE_URL}")
    print(f"          shards 0..{ds.MAX_SHARD} exist upstream (~{(ds.MAX_SHARD + 1) * TOKENS_PER_SHARD / 1e9:.0f}B "
          f"tokens with this tokenizer; the '400b' in the name counts them with another)")
    print(f"data dir: {data_dir}")
    print(f"  present: {len(present)} shards ({len(present) * SHARD_MB / 1024:.0f} GB)")
    print(f"  need:    {len(keep)} shards (train 0..{n_train - 1}, val {args.val_shard})")
    print(f"  missing: {len(missing)} shards to download ({len(missing) * SHARD_MB / 1024:.0f} GB)")
    print(f"  surplus: {len(surplus)} shards not needed ({len(surplus) * SHARD_MB / 1024:.0f} GB)")

    if missing:
        print(f"  would download: {', '.join(ds.index_to_filename(i) for i in missing[:4])}"
              + (f" ... {ds.index_to_filename(missing[-1])}" if len(missing) > 4 else ""))
    if surplus and not args.prune:
        print(f"\n  to reclaim the surplus: python trains/fetch_data.py "
              + (f"--depth {args.depth}" if args.depth else f"--tokens {args.tokens:g}")
              + f" --val-shard {args.val_shard} --prune --yes")

    if args.verify:
        rc = verify_yield(tokens, n_train, args.verify_shards)
        if args.dry_run:
            return rc

    if args.dry_run:
        print("\n--dry-run: nothing downloaded, nothing deleted")
        return 0

    if missing:
        print(f"\ndownloading {len(missing)} shards with {args.workers} workers...")
        with Pool(processes=args.workers) as pool:
            results = pool.map(ds.download_single_file, missing)
        ok = sum(1 for r in results if r)
        print(f"downloaded {ok}/{len(missing)}")
        if ok != len(missing):
            print("some downloads failed; re-run to retry", file=sys.stderr)
            return 1

    if args.prune and surplus:
        if not args.yes:
            print(f"\n--prune given without --yes: {len(surplus)} shards left in place")
        else:
            freed = 0
            for i in surplus:
                path = os.path.join(data_dir, ds.index_to_filename(i))
                freed += os.path.getsize(path)
                os.remove(path)
            print(f"\ndeleted {len(surplus)} surplus shards, freed {freed / 2**30:.0f} GB")

    final = len([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
    print(f"\n{data_dir}: {final} shards, "
          f"~{(final - 1) * TOKENS_PER_SHARD / 1e9:.1f}B train tokens, "
          f"val = {ds.index_to_filename(max(int(f.split('_')[1].split('.')[0]) for f in os.listdir(data_dir) if f.endswith('.parquet')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
