"""Subgroup MAE/RMSE/R2 mean ± SD across ep100_n7 prediction CSVs (CPU only)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ef_prediction.demographics_utils import compute_bmi_category

ROOT = Path(__file__).resolve().parents[1]
TAG = "ep100_n7"
OUT = ROOT / "ef_prediction" / "multi_run_results" / f"subgroup_{TAG}_summary.json"

SEX_LABELS = {"M": "sex_male", "F": "sex_female", "O": "sex_other"}
AGE_BINS = ["0-1", "2-5", "6-10", "11-15", "16-18"]
BMI_BINS = ["underweight", "normal", "overweight", "obese"]


def compute_metrics(errors: np.ndarray, labels: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((labels - labels.mean()) ** 2))
    r2 = 0.0 if denom == 0 else float(1.0 - np.sum(errors ** 2) / denom)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "count": int(len(errors))}


def load_manifest(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["data"]["val_manifest"]
    df = pd.read_csv(path)
    if "bmi_category" not in df.columns:
        df["bmi_category"] = df.apply(
            lambda r: compute_bmi_category(r.get("weight"), r.get("height")),
            axis=1,
        )
    return df.reset_index(drop=True)


def attach(manifest: pd.DataFrame, pred_path: Path, pred_col: str) -> pd.DataFrame:
    preds = pd.read_csv(pred_path)
    if len(manifest) != len(preds):
        raise ValueError(f"len mismatch {pred_path}: {len(manifest)} vs {len(preds)}")
    out = manifest.copy()
    out["True_EF"] = preds["True_EF"].astype(float).values
    out["Predicted_EF"] = preds[pred_col].astype(float).values
    out["Error"] = out["Predicted_EF"] - out["True_EF"]
    return out


def group_key(row: pd.Series) -> list[tuple[str, str]]:
    keys = []
    sex = str(row.get("sex", "")).strip().upper()
    if sex in SEX_LABELS:
        keys.append(("sex", SEX_LABELS[sex]))
    age = str(row.get("age_bin", "")).strip()
    if age in AGE_BINS:
        keys.append(("age", f"age_{age}"))
    bmi = str(row.get("bmi_category", "")).strip().lower()
    if bmi in BMI_BINS:
        keys.append(("bmi", f"bmi_{bmi}"))
    return keys


def metrics_for_df(df: pd.DataFrame) -> dict:
    labels = df["True_EF"].to_numpy(float)
    errors = df["Error"].to_numpy(float)
    out = {"OVERALL": compute_metrics(errors, labels)}
    # collect group membership
    buckets: dict[str, list[int]] = {}
    for i, row in df.iterrows():
        for _, key in group_key(row):
            buckets.setdefault(key, []).append(i)
    for key, idxs in buckets.items():
        part = df.loc[idxs]
        out[key] = compute_metrics(
            part["Error"].to_numpy(float),
            part["True_EF"].to_numpy(float),
        )
    return out


def summarize_runs(run_metrics: list[dict]) -> dict:
    # union of group keys
    keys = sorted({k for m in run_metrics for k in m})
    summary = {}
    for key in keys:
        rows = [m[key] for m in run_metrics if key in m]
        if not rows:
            continue
        entry = {"n": len(rows), "count": rows[0]["count"]}
        for metric in ("MAE", "RMSE", "R2"):
            vals = np.array([r[metric] for r in rows], dtype=float)
            entry[metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
        summary[key] = entry
    return summary


def main() -> None:
    with open(ROOT / "ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)
    manifest = load_manifest(cfg)

    fused_runs = []
    for run in range(11, 18):
        path = ROOT / f"ef_prediction/multi_run_results/fused_run_{run}_{TAG}.csv"
        df = attach(manifest, path, "Predicted_EF_Fused")
        fused_runs.append(metrics_for_df(df))

    real_runs = []
    for seed in range(53, 60):
        path = ROOT / f"ef_prediction/eval_results/real_seed{seed}_{TAG}_results.csv"
        df = attach(manifest, path, "Predicted_EF")
        real_runs.append(metrics_for_df(df))

    payload = {
        "tag": TAG,
        "n_seeds": 7,
        "fused": summarize_runs(fused_runs),
        "real": summarize_runs(real_runs),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")

    # compact print for paper tables
    order = [
        "OVERALL",
        "sex_female",
        "sex_male",
        "age_0-1",
        "age_2-5",
        "age_6-10",
        "age_11-15",
        "age_16-18",
        "bmi_underweight",
        "bmi_normal",
        "bmi_overweight",
        "bmi_obese",
    ]
    print("\nGroup                          | Fused MAE        | Real MAE")
    print("-" * 70)
    for key in order:
        f = payload["fused"].get(key)
        r = payload["real"].get(key)
        if not f or not r:
            continue
        print(
            f"{key:30s} | {f['MAE']['mean']:.2f} ± {f['MAE']['std']:.2f}  "
            f"| {r['MAE']['mean']:.2f} ± {r['MAE']['std']:.2f}  (n={f['count']})"
        )


if __name__ == "__main__":
    main()
