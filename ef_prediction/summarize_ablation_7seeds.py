"""Mean ± SD for the three no-HCL input ablations (fused / real / synth-only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    if not rows:
        return None
    out = {"n": len(rows), "runs": rows}
    for key in ("MAE", "RMSE", "R2"):
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
        }
    return out


def load_fused(tag: str):
    rows = []
    for path in sorted((ROOT / "ef_prediction/multi_run_results").glob(f"fused_run_*_{tag}_metrics.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "run": data.get("run"),
                "MAE": float(data["MAE"]),
                "RMSE": float(data["RMSE"]),
                "R2": float(data["R2"]),
            }
        )
    return rows


def load_tagged(prefix: str, tag: str):
    rows = []
    for path in sorted((ROOT / "ef_prediction/eval_results").glob(f"{prefix}*_{tag}_metrics.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "MAE": float(data["MAE"]),
                "RMSE": float(data["RMSE"]),
                "R2": float(data["R2"]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    tag = args.tag

    summary = {
        "tag": tag,
        "fused_no_hcl": summarize(load_fused(tag)),
        "real_no_hcl": summarize(load_tagged("real_no_hcl_seed", tag)),
        "synth_no_hcl": summarize(load_tagged("synth_no_hcl_seed", tag)),
    }
    out = ROOT / "ef_prediction" / "multi_run_results" / f"{tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
