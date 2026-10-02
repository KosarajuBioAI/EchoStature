#!/usr/bin/env bash
# Matched 100-epoch × 7-seed input ablations (Table 3 style), all with HCL off.
# Does NOT overwrite ep100_n7 main-model checkpoints (separate dirs).
#
#   fused no-HCL:   real + rendered, gated EchoStature
#   real no-HCL:    video-only on real clips
#   synth no-HCL:   video-only on synthetic clips
#
# Same seeds as ep100_n7: run 11..17 → seed 53..59
#
# Usage (from repo root, ideally in tmux/byobu):
#   EPOCHS=100 NRUNS=7 bash ef_prediction/run_ablation_7seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-100}"
NRUNS="${NRUNS:-7}"
TAG="${TAG:-ablation_ep${EPOCHS}_n${NRUNS}}"
LOGDIR="ef_prediction/logs/${TAG}"
mkdir -p "${LOGDIR}"
mkdir -p ef_prediction/checkpoints/fused_no_hcl
mkdir -p ef_prediction/checkpoints/real_no_hcl
mkdir -p ef_prediction/checkpoints/synth_only_no_hcl
mkdir -p ef_prediction/multi_run_results
mkdir -p ef_prediction/eval_results

echo "=== ablation ${NRUNS} seeds × ${EPOCHS} epochs, tag ${TAG} ==="
echo "Checkpoints go under fused_no_hcl / real_no_hcl / synth_only_no_hcl"
echo "Will NOT touch checkpoints/fused/run_11..17 or real/seed_53..59"

for i in $(seq 0 $((NRUNS - 1))); do
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
echo "=== ablation finished $(date) ==="
