#!/usr/bin/env bash
# =============================================================================
# Resume a run from its newest checkpoint.
#
# Needed because a multi-day run does not fit in one container: Modal caps a
# function call at 24 h, and the documented pattern for longer training is to make
# the job resumable and restart it. base_train restores the model, the optimizer and
# the dataloader position from a checkpoint, so each window continues where the last
# one stopped.
#
#   bash trains/resume_latest.sh d24_hybridswa_muon_20260921_120000
#   bash trains/resume_latest.sh <run_id> --keep 2      # drop older checkpoints first
#   bash trains/resume_latest.sh <run_id> --dry-run
#
# Everything else is passed through to the size wrapper, and the environment the
# original run used (MUON_VARIANT, TARGET_PARAM_DATA_RATIO, ...) must be set the same
# way -- those are not stored in the checkpoint. The run's config.json records them:
#   cat logs/<run_id>/config.json
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

[ $# -ge 1 ] || { echo "usage: bash trains/resume_latest.sh <run_id> [--keep N] [--dry-run] [extra args]" >&2; exit 2; }
RUN_ID="$1"; shift

KEEP=0
DRY=0
PASSTHRU=()
while [ $# -gt 0 ]; do
    case "$1" in
        --keep) KEEP="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$RUN_ID"
[ -d "$CKPT_DIR" ] || { echo "ERROR: no checkpoint dir $CKPT_DIR" >&2; exit 1; }

# steps present, newest last
mapfile -t STEPS < <(ls "$CKPT_DIR"/model_*.pt 2>/dev/null | sed 's/.*model_0*\([0-9]*\)\.pt/\1/' | sort -n)
[ "${#STEPS[@]}" -gt 0 ] || { echo "ERROR: no model_*.pt in $CKPT_DIR" >&2; exit 1; }
LAST="${STEPS[-1]}"

echo "run:        $RUN_ID"
echo "checkpoints: ${STEPS[*]}"
echo "resuming at: step $LAST  ($(du -sh "$CKPT_DIR" | cut -f1) on disk)"

# The optimizer shard must exist too, or base_train cannot restore the optimizer.
[ -f "$CKPT_DIR/optim_$(printf '%06d' "$LAST")_rank0.pt" ] || {
    echo "ERROR: step $LAST has no optimizer shard; the run was killed mid-save. "\
         "Delete that step's files and re-run to fall back to the previous checkpoint." >&2
    exit 1
}

if [ "$KEEP" -gt 0 ] && [ "${#STEPS[@]}" -gt "$KEEP" ]; then
    DROP=$(( ${#STEPS[@]} - KEEP ))
    echo "pruning:     ${DROP} older checkpoint(s), keeping the newest $KEEP"
    for i in $(seq 0 $((DROP - 1))); do
        s=$(printf '%06d' "${STEPS[$i]}")
        [ "$DRY" = "1" ] && echo "  would remove $CKPT_DIR/*_$s*" \
                         || rm -f "$CKPT_DIR"/model_$s.pt "$CKPT_DIR"/optim_${s}_rank*.pt "$CKPT_DIR"/meta_$s.json
    done
fi

# depth comes from the run id (d24_...) so the right wrapper and its knobs are used
DEPTH_FROM_ID="$(echo "$RUN_ID" | sed -n 's/^d\([0-9]\+\)_.*/\1/p')"
WRAPPER="$REPO_DIR/trains/train_d${DEPTH_FROM_ID}_hybrid_swa_muon.sh"
[ -f "$WRAPPER" ] || { echo "ERROR: cannot infer the size wrapper from '$RUN_ID' (looked for $WRAPPER)" >&2; exit 1; }

echo "wrapper:     $(basename "$WRAPPER")"
echo "command:     bash $(basename "$WRAPPER") --model-tag=$RUN_ID --resume-from-step=$LAST ${PASSTHRU[*]-}"
if [ "$DRY" = "1" ]; then
    echo "--dry-run: not starting"
    exit 0
fi

# --model-tag last wins in argparse, so this overrides the fresh id the wrapper builds
exec bash "$WRAPPER" --model-tag="$RUN_ID" --resume-from-step="$LAST" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
