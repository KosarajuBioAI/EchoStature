#!/usr/bin/env bash
# Continue the no-HCL 7-seed ablation from fused run 14.
# Runs 11-13 already finished. Run 14 was stopped at epoch 72 and restarts
# from epoch 0. The paused weights are kept beside the live checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-100}"
TAG="${TAG:-ablation_ep${EPOCHS}_n7}"
LOGDIR="ef_prediction/logs/${TAG}"
mkdir -p "${LOGDIR}"

# Stop after 5 seeds total (runs 11–15 / seeds 53–57). i=3..4 => runs 14–15.
# (Runs 11–13 already finished earlier.)
for i in $(seq 3 4); do
  run=$((11 + i))
  seed=$((42 + run))

  echo "===== fused no-HCL run ${run} seed ${seed} $(date) ====="
  python -m ef_prediction.train_fused \
    --run "${run}" --epochs "${EPOCHS}" --hcl-weight 0 \
    --checkpoint-dir ef_prediction/checkpoints/fused_no_hcl \
    2>&1 | tee "${LOGDIR}/fused_no_hcl_run${run}.log"
  python -m ef_prediction.evaluate_ef_fused \
    --run "${run}" --tag "${TAG}" \
    --checkpoint "ef_prediction/checkpoints/fused_no_hcl/run_${run}_best.pth" \
    2>&1 | tee -a "${LOGDIR}/fused_no_hcl_run${run}.log"

  echo "===== real no-HCL seed ${seed} $(date) ====="
  python -m ef_prediction.train_real \
    --seed "${seed}" --epochs "${EPOCHS}" --hcl-weight 0 \
    --checkpoint-dir ef_prediction/checkpoints/real_no_hcl \
    2>&1 | tee "${LOGDIR}/real_no_hcl_seed${seed}.log"
  python -m ef_prediction.evaluate_ef_real \
    --checkpoint "ef_prediction/checkpoints/real_no_hcl/seed_${seed}_best.pth" \
    --tag "real_no_hcl_seed${seed}_${TAG}" \
    2>&1 | tee -a "${LOGDIR}/real_no_hcl_seed${seed}.log"

  echo "===== synth-only no-HCL seed ${seed} $(date) ====="
  python -m ef_prediction.train_real_synthetic_only \
    --seed "${seed}" --epochs "${EPOCHS}" --hcl-weight 0 \
    --checkpoint-dir ef_prediction/checkpoints/synth_only_no_hcl \
    2>&1 | tee "${LOGDIR}/synth_no_hcl_seed${seed}.log"
  python -m ef_prediction.evaluate_ef_real \
    --checkpoint "ef_prediction/checkpoints/synth_only_no_hcl/seed_${seed}_best.pth" \
    --manifest ef_prediction/synthetic_only_manifests/val_synthetic_only.csv \
    --video-root perfect_synthetic_copies \
    --tag "synth_no_hcl_seed${seed}_${TAG}" \
    2>&1 | tee -a "${LOGDIR}/synth_no_hcl_seed${seed}.log"
done

python ef_prediction/summarize_ablation_7seeds.py --tag "${TAG}"
echo "=== ablation resume finished $(date) ==="
