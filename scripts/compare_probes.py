"""
Side-by-side of the high-LR probe runs (trains/exp_d12_12b_hlr_probe.sh): train loss and the diagnostics
that matter for the divergence -- residual-stream and MLP activation RMS, attention output RMS, the
RMSNorm gain gamma and the attention c_proj x gamma RMS -- at matching steps.

    python -m scripts.compare_probes [--glob 'logs/d12_12b_probe_lr0.032_*'] [--every 250]
"""
import argparse
import glob
import json
import os
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="logs/d12_12b_probe_lr0.032_*")
ap.add_argument("--every", type=int, default=250)
args = ap.parse_args()

runs = {}
for d in sorted(glob.glob(args.glob)):
    name = re.sub(r"^d12_12b_probe_|_moonlight_\d+_\d+$", "", os.path.basename(d))  # e.g. lr0.032_rexp
    loss, diag = {}, {}
    for line in open(os.path.join(d, "train.log")):
        m = re.match(r"step (\d+)/\d+ .*?loss: ([\d.naninf]+)", line)
        if m:
            try:
                loss[int(m[1])] = float(m[2])
            except ValueError:
                pass
        elif line.startswith("diag_json "):
            j = json.loads(line[len("diag_json "):])
            diag[j["step"]] = j
    runs[name] = (loss, diag)

last = max((max(l) for l, _ in runs.values() if l), default=0)
cols = [("loss", None), ("resid", "act_rms/block/max"), ("mlp", "act_rms/mlp/max"), ("attn", "act_rms/attn/max"),
        ("gamma", "attn_gamma/rms/max"), ("W*g", "weight_rms/attn_c_proj_x_gamma/max"), ("W", "weight_rms/attn_c_proj/max")]
for step in range(0, last + 1, args.every):
    print(f"--- step {step}")
    for name, (loss, diag) in runs.items():
        window = [loss[i] for i in range(step, step + 50) if i in loss]
        cells = [f"loss {statistics.mean(window):6.3f}" if window else "loss    -  "]
        dg = diag.get(step)
        for label, key in cols[1:]:
            v = dg.get(key) if dg else None
            cells.append(f"{label} {v:9.3g}" if isinstance(v, (int, float)) else f"{label}       -")
        print(f"  {name:22s} " + "  ".join(cells))
