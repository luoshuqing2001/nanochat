#!/usr/bin/env bash
# =============================================================================
# The LR stability sweep: softmax vs rexp_rmsnorm (gain lr tied to MATRIX_LR) at MATRIX_LR = LR_MULT x 0.004
# for each LR_MULT in MULTS, via trains/exp_d12_12b_hlr_probe.sh (first STOP_AT_STEP steps of the full
# 12B schedule, diagnostics every 50). Runs are queued over the GPU groups in PAIRS: a group takes the
# next run as soon as its previous one exits. Logs: probe_<arm>_x<mult>.out and logs/d12_12b_probe_*.
#
#   PAIRS="0,1 4,5 6,7" setsid nohup bash trains/run_lr_sweep.sh > lr_sweep.out 2>&1 &
#   MULTS="1 2 4 8 16" ARMS="softmax rexp" STOP_AT_STEP=3000 (the defaults)
#   python -m scripts.compare_probes --glob 'logs/d12_12b_probe_lr*'     # results so far
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIRS="${PAIRS:-0,1 2,3 4,5 6,7}"
MULTS="${MULTS:-1 2 4 8 16}"
ARMS="${ARMS:-softmax rexp}"
export STOP_AT_STEP="${STOP_AT_STEP:-3000}"
jobs_list=()
for m in $MULTS; do for a in $ARMS; do jobs_list+=("$a $m"); done; done
declare -A pid_of
next=0
echo "[$(date '+%F %T')] ${#jobs_list[@]} runs over GPU groups: $PAIRS"
while :; do
    for g in $PAIRS; do
        p="${pid_of[$g]:-}"
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then continue; fi
        if [ "$next" -ge "${#jobs_list[@]}" ]; then continue; fi
        set -- ${jobs_list[$next]}
        echo "[$(date '+%F %T')] start ARM=$1 LR_MULT=$2 on GPUs $g"
        ARM="$1" LR_MULT="$2" GPUS="$g" bash trains/exp_d12_12b_hlr_probe.sh > "probe_$1_x$2.out" 2>&1 &
        pid_of[$g]=$!
        next=$((next + 1))
        sleep 20   # stagger the compiles
    done
    running=0
    for g in $PAIRS; do p="${pid_of[$g]:-}"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && running=$((running + 1)); done
    if [ "$next" -ge "${#jobs_list[@]}" ] && [ "$running" -eq 0 ]; then break; fi
    sleep 30
done
echo "[$(date '+%F %T')] sweep done"
