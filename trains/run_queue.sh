#!/usr/bin/env bash
# =============================================================================
# A job queue over GPU slots: each line of JOBS is "<script under trains/> VAR=value ...", run as
# `VAR=value ... GPUS=<slot> bash trains/<script>`; a slot takes the next job as soon as its last one
# exits, in file order. Blank lines and #-comments are skipped. One log per job: queue_logs/<n>_<name>.out.
#
#   JOBS=trains/overnight_d12.jobs SLOTS="0 1 2 3 4 5 6 7" setsid nohup bash trains/run_queue.sh > queue.out 2>&1 &
#   (a slot may be a GPU list: SLOTS="0,1 2,3")
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:?set JOBS=<jobs file>}"
SLOTS="${SLOTS:-0 1 2 3 4 5 6 7}"
mkdir -p queue_logs
mapfile -t jobs_list < <(grep -v '^\s*#' "$JOBS" | grep -v '^\s*$')
declare -A pid_of job_of
next=0
echo "[$(date '+%F %T')] ${#jobs_list[@]} jobs from $JOBS over slots: $SLOTS"
while :; do
    for s in $SLOTS; do
        p="${pid_of[$s]:-}"
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then continue; fi
        if [ -n "$p" ]; then wait "$p"; echo "[$(date '+%F %T')] done  slot $s (exit $?): ${job_of[$s]}"; pid_of[$s]=""; fi
        if [ "$next" -ge "${#jobs_list[@]}" ]; then continue; fi
        line="${jobs_list[$next]}"
        read -r script vars <<< "$line"
        name="$(printf '%02d' "$next")_$(echo "$script $vars" | sed 's/\.sh//; s/[= ]/_/g; s/[^A-Za-z0-9_.-]//g')"
        echo "[$(date '+%F %T')] start slot $s: $line  -> queue_logs/$name.out"
        env $vars GPUS="$s" bash "trains/$script" > "queue_logs/$name.out" 2>&1 &
        pid_of[$s]=$!
        job_of[$s]="$line"
        next=$((next + 1))
        sleep 20   # stagger the startups (compile, data loader)
    done
    running=0
    for s in $SLOTS; do p="${pid_of[$s]:-}"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && running=$((running + 1)); done
    if [ "$next" -ge "${#jobs_list[@]}" ] && [ "$running" -eq 0 ]; then break; fi
    sleep 30
done
echo "[$(date '+%F %T')] queue done"
