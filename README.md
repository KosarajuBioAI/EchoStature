# EchoStature

Pediatric ejection fraction (EF) regression with **spatial demographic conditioning**.

EchoStature encodes a real echocardiogram clip and a paired demographically conditioned synthetic reconstruction (from C3DGAN), fuses the two streams with a learned gate, and concatenates a low-dimensional demographic embedding after temporal pooling. A video-only reference model (real clip only) is included for comparison.

**Repository:** https://github.com/KosarajuBioAI/EchoStature

---

## Repository layout

```
ef_prediction/          # training, evaluation, ablations, interpretability
  models/               # video-only and fused (gated) networks
  config.yaml           # hyperparameters
  train_fused.py        # EchoStature (real + rendered + demographics)
  train_real.py         # video-only reference
  evaluate_*.py         # validation metrics
  run_5seeds.sh         # multi-seed training + eval
  run_ablation_7seeds.sh
  generate_fused_counterfactual_grid.py
  regenerate_umap_true_labels.py
  ...
images/                 # architecture / interpretability figures
requirements.txt
```

Model weights (`.pth`), raw videos, and logs are **not** included (see `.gitignore`).

---

## Setup

```bash
git clone https://github.com/KosarajuBioAI/EchoStature.git
cd EchoStature
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision   # install the CUDA build that matches your GPU
```

Configure data paths and training options in `ef_prediction/config.yaml` (clip length, resolution, backbone, fusion mode, HCL weights, etc.).

You need:

- EchoNet-Pediatric (or equivalent) clips + manifests for train/validation  
- Paired C3DGAN reconstructions for the fused model (one synthetic clip per real clip)

---

## Training

From the repo root:

**EchoStature (fused):**
```bash
python -m ef_prediction.train_fused --run 1 --epochs 100
```

**Video-only reference:**
```bash
python -m ef_prediction.train_real --seed 53 --epochs 100
```

**Multi-seed protocol** (fused + real, then summarize):
```bash
EPOCHS=100 NRUNS=7 bash ef_prediction/run_5seeds.sh
```

Checkpoints write under `ef_prediction/checkpoints/` (local only; not committed).

---

## Evaluation

```bash
python -m ef_prediction.evaluate_ef_fused --run 1 --tag ep100_n7
python -m ef_prediction.evaluate_ef_real \
  --checkpoint ef_prediction/checkpoints/real/seed_53_best.pth \
  --tag seed53_ep100_n7
```

Summaries:

```bash
python ef_prediction/summarize_5seeds.py --tag ep100_n7
python ef_prediction/summarize_ablation_7seeds.py
python ef_prediction/summarize_subgroup_7seeds.py
```

---

## Ablations and interpretability

| Script | Purpose |
|--------|---------|
| `run_ablation_7seeds.sh` | Input ablation (real / rendered / fused), no contrastive term |
| `run_demo_ablation_7seeds.sh` | Demographic zero / shuffle at inference |
| `generate_fused_counterfactual_grid.py` | Counterfactual Grad-CAM grid |
| `create_real_vs_fused_gradcam_comparison.py` | Real vs fused attribution maps |
| `regenerate_umap_true_labels.py` | UMAP of regressor input by demographics |
| `c3dgan_paired_anatomy_check.py` | Dark-cavity chamber check on real–rendered pairs |

Example figures are under `images/` (UMAP, Grad-CAM comparison, counterfactual grid, model overview, chamber structure).

---

## Models

- **`pt_efnet_real`** — ResNet backbone + temporal aggregation; optional demographic embedding  
- **`pt_efnet_fused`** — Shared encoder on real and synthetic clips, per-frame gated fusion \(g_t \odot r_t + (1-g_t)\odot \tilde{r}_t\), then demographic embedding  

Default backbone in `config.yaml` is ResNet-34; clips are \(T=32\) frames at \(128\times 128\).

---

## Citation

If you use this code, please cite the EchoStature paper (Pacific Symposium on Biocomputing) and the EchoNet-Pediatric dataset.

```
Reddy et al., Video-based deep learning for automated assessment of
left ventricular ejection fraction in pediatric patients,
JASE, 2023.
```

---

## License

See repository license / organization policy. Do not redistribute patient video data with this code.
