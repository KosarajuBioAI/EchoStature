#!/usr/bin/env bash
# Demographic ablation (normal / zero / shuffle) on the finished ep100_n7
# checkpoints. Inference only — no training.
#
#   bash ef_prediction/run_demo_ablation_7seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="ef_prediction/group_results/demo_ablation_ep100_n7"
mkdir -p "${OUT}"

echo "=== demographic ablation on ep100_n7 checkpoints ==="

for run in 11 12 13 14 15 16 17; do
  seed=$((42 + run))
  echo "----- fused run ${run} -----"
  python ef_prediction/ablate_demographic_effect_overall.py \
    --model fused \
    --checkpoint "ef_prediction/checkpoints/fused/run_${run}_best.pth" \
    --output-dir "${OUT}/fused_run${run}"

  echo "----- real seed ${seed} -----"
  python ef_prediction/ablate_demographic_effect_overall.py \
    --model real \
    --checkpoint "ef_prediction/checkpoints/real/seed_${seed}_best.pth" \
    --output-dir "${OUT}/real_seed${seed}"
done

python ef_prediction/summarize_demo_ablation_7seeds.py --input-dir "${OUT}"
echo "=== demo ablation finished ==="
