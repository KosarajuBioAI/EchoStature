"""Aggregate normal/zero/shuffle demographic ablation across ep100_n7 seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize_mode(rows, mode: str):
    vals = {k: [] for k in ("MAE", "RMSE", "R2")}
    for r in rows:
        m = r["modes"].get(mode)
        if not m:
            continue
        for k in vals:
            vals[k].append(float(m[k]))
    out = {"n": len(vals["MAE"])}
    for k, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        out[k] = {
            "mean": float(a.mean()) if len(a) else None,
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "values": [float(x) for x in a],
        }
    return out


def load_side(input_dir: Path, pattern: str):
    rows = []
    for d in sorted(input_dir.glob(pattern)):
        cands = list(d.glob("*_metrics.json"))
        if not cands:
            continue
        data = json.loads(cands[0].read_text())
        modes = data.get("metrics") or {}
        if modes:
            rows.append({"dir": d.name, "modes": modes})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="ef_prediction/group_results/demo_ablation_ep100_n7",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir

    fused = load_side(input_dir, "fused_run*")
    real = load_side(input_dir, "real_seed*")

    summary = {
        "fused": {m: summarize_mode(fused, m) for m in ("normal", "zero", "shuffle")},
        "real": {m: summarize_mode(real, m) for m in ("normal", "zero", "shuffle")},
        "n_fused_dirs": len(fused),
        "n_real_dirs": len(real),
    }
    out = input_dir / "demo_ablation_ep100_n7_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
