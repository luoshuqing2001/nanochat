"""
Download only as much ClimbMix as the planned runs actually need.

`python -m nanochat.dataset -n N` has two problems for this repo: N is a raw shard
count with no connection to the model you intend to train, and it always also pulls
shard_06542 (upstream's MAX_SHARD), which then sorts last and silently becomes your
validation set -- invalidating comparisons against every earlier run.

This script sizes the download from the model instead, and pins the validation shard.

    python trains/fetch_data.py --depth 24              # compute-optimal d24, 25% margin
    python trains/fetch_data.py --depth 24 --dry-run    # just the plan
    python trains/fetch_data.py --tokens 20e9           # explicit token budget
    python trains/fetch_data.py --depth 24 --prune --yes   # also delete surplus shards

How the sizing works:
  tokens needed = 12 x scaling_params(depth)   (the data:param ratio base_train targets)
  shards        = ceil(tokens x margin / 44.8e6)

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


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--depth", type=int, help="size the download for a compute-optimal run at this depth")
    g.add_argument("--tokens", type=float, help="explicit token budget, e.g. 20e9")
    ap.add_argument("--margin", type=float, default=1.25, help="safety factor over the computed need (default 1.25)")
    ap.add_argument("--val-shard", type=int, default=2499,
                    help="shard index reserved as the validation set; must stay the highest kept index (default 2499)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prune", action="store_true", help="also delete shards outside the keep set")
    ap.add_argument("--yes", action="store_true", help="actually delete when --prune is given")
    args = ap.parse_args()

    import nanochat.dataset as ds  # honors NANOCHAT_DATA_DIR

    if args.depth:
        sp, dim, total = scaling_params(args.depth)
        tokens = DATA_PARAM_RATIO * sp
        print(f"d{args.depth}: dim {dim}, {total/1e9:.2f}B params, {sp/1e6:.0f}M scaling params")
        print(f"compute-optimal horizon: {DATA_PARAM_RATIO} x scaling params = {tokens/1e9:.2f}B tokens")
    else:
        tokens = args.tokens
        print(f"requested budget: {tokens/1e9:.2f}B tokens")

    need = tokens * args.margin
    n_train = math.ceil(need / TOKENS_PER_SHARD)
    print(f"with a {args.margin}x margin: {need/1e9:.2f}B tokens -> {n_train} train shards "
          f"(~{n_train * SHARD_MB / 1024:.0f} GB) + 1 validation shard")

    if n_train > args.val_shard:
        print(f"\nERROR: {n_train} train shards would run past --val-shard {args.val_shard}, and nanochat "
              f"takes the highest-numbered shard as validation.\nRaise --val-shard (and accept that val "
              f"changes, making past runs incomparable), or lower the budget.", file=sys.stderr)
        return 1

    data_dir = ds.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    keep = set(range(n_train)) | {args.val_shard}
    present = {int(f.split("_")[1].split(".")[0])
               for f in os.listdir(data_dir) if f.endswith(".parquet")}
    missing = sorted(keep - present)
    surplus = sorted(present - keep)

    print(f"\ndata dir: {data_dir}")
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
