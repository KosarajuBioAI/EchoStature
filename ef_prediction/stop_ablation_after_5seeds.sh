#!/usr/bin/env bash
# Let the running no-HCL ablation finish seed 57 (5th seed: 53–57),
# then stop the resume script before it starts seed 58/59, and summarize.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:-ablation_ep100_n7}"
LOGDIR="ef_prediction/logs/${TAG}"
SYNTH_LOG="${LOGDIR}/synth_no_hcl_seed57.log"
PARENT_MATCH='resume_ablation_from_run14.sh'
STATUS="ef_prediction/logs/${TAG}/stop_after_5seeds.status"

echo "[$(date)] watcher: stop after 5 seeds (through seed 57)" | tee "$STATUS"

done_seed57() {
  [[ -f "$SYNTH_LOG" ]] || return 1
  # train finished and eval metrics written
  rg -q 'Done\. Best val MSE' "$SYNTH_LOG" || return 1
  ls ef_prediction/eval_results/real_synth_no_hcl_seed57_*"${TAG}"*_metrics.json >/dev/null 2>&1 \
    || ls ef_prediction/eval_results/*synth_no_hcl_seed57*"${TAG}"*metrics.json >/dev/null 2>&1
}

while ! done_seed57; do
  ep="?"
  if [[ -f "${LOGDIR}/real_no_hcl_seed57.log" ]]; then
    ep=$(rg -o '===== Epoch ([0-9]+) =====' -r '$1' "${LOGDIR}/real_no_hcl_seed57.log" 2>/dev/null | tail -1 || true)
  fi
  if [[ -f "$SYNTH_LOG" ]]; then
    sep=$(rg -o '===== Epoch ([0-9]+) =====' -r '$1' "$SYNTH_LOG" 2>/dev/null | tail -1 || true)
    echo "[$(date)] waiting… real57_ep=${ep:-?} synth57_ep=${sep:-?} " | tee -a "$STATUS"
  else
    echo "[$(date)] waiting… real57_ep=${ep:-?} (synth57 not started)" | tee -a "$STATUS"
  fi
  sleep 120
done

echo "[$(date)] seed 57 complete — stopping ablation parent before seed 58" | tee -a "$STATUS"

# Kill only the resume ablation bash (not unrelated jobs)
pids=$(pgrep -f "$PARENT_MATCH" || true)
if [[ -n "${pids}" ]]; then
  echo "[$(date)] killing: ${pids}" | tee -a "$STATUS"
  kill ${pids} 2>/dev/null || true
  sleep 5
  # if still alive, escalate
  pids=$(pgrep -f "$PARENT_MATCH" || true)
  if [[ -n "${pids}" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
else
  echo "[$(date)] no resume_ablation parent found (already exited?)" | tee -a "$STATUS"
fi

# Ensure no leftover train for seed 58+
for pat in 'train_fused --run 16' 'train_real --seed 58' 'train_real_synthetic_only --seed 58' \
           'train_fused --run 17' 'train_real --seed 59' 'train_real_synthetic_only --seed 59'; do
  pgrep -f "$pat" >/dev/null 2>&1 && kill $(pgrep -f "$pat") 2>/dev/null || true
done

echo "[$(date)] summarizing 5-seed ablation" | tee -a "$STATUS"
python ef_prediction/summarize_ablation_7seeds.py --tag "${TAG}" | tee -a "$STATUS"
echo "[$(date)] DONE — report n=5 (seeds 53–57). Do not start seeds 58–59." | tee -a "$STATUS"
