#!/usr/bin/env bash
# Orchestrator for the 5-preset canon run on Haiku 4.5.
# Launches each preset as its own python3 run.py process in parallel.
# Per-run cost-abort is $250. Logs per preset to /tmp/lcvb_<preset>.log.
set -u

cd "$(dirname "$0")"
MODEL="claude-haiku-4-5-20251001"
RUNS_DIR="data/runs"
MASTER_LOG="/tmp/lcvb_full_run.log"

PRESETS=(canon_direct canon_no_distractor canon_uniform_short canon_uniform_medium canon_uniform_long)

echo "FULL RUN LAUNCH @ $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"

PIDS=()
for preset in "${PRESETS[@]}"; do
    LOG="/tmp/lcvb_${preset}.log"
    echo "  launching $preset -> $LOG" | tee -a "$MASTER_LOG"
    python3 pipeline/run.py \
        --prompts-dir "generated/${preset}" \
        --condition "${preset}" \
        --model "${MODEL}" \
        --concurrency 50 --run \
        --cost-abort 250 \
        --runs-dir "${RUNS_DIR}" > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}" | tee -a "$MASTER_LOG"
echo "waiting..." | tee -a "$MASTER_LOG"
wait
echo "FULL RUN COMPLETE @ $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$MASTER_LOG"

# Dump final status of each
for preset in "${PRESETS[@]}"; do
    echo "=== ${preset} (tail) ===" | tee -a "$MASTER_LOG"
    tail -5 "/tmp/lcvb_${preset}.log" | tee -a "$MASTER_LOG"
done
