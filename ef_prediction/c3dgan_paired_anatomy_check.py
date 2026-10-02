#!/usr/bin/env python3
"""Chamber-structure check on C3DGAN paired clips used by EchoStature EF training.

Distinct from the HICSS ACGAN anatomy report:
  - Data: perfect_synthetic_copies paired with EchoNet-Pediatric sources
    (the rendered stream EchoStature consumes), not ACGAN novel samples.
  - Sampling: 100 random validation pairs (same original_id / demographics).
  - Claim: whether the rendered stream preserves a detectable dark cavity and
    cyclic area change relative to its paired real clip — not generator novelty.
  - Classical dark-cavity connected component (not a trained LV segmenter).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats as spstats

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION")
MAN = ROOT / "perfect_synthetic_copies/perfect_copies_val.csv"
OUT = ROOT / "ef_prediction/anatomy_c3dgan_paired"
FIG = ROOT / "images"
T, S, N = 32, 128, 100
SEED = 42

# Bright, high-contrast palette (distinct from HICSS muted blue/red)
C_REAL = "#0077FF"   # bright blue
C_REN = "#FF3B30"    # bright red-orange

plt.rcParams.update({
    "font.family": "Noto Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def decode_mp4(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.ndim == 3:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if fr.shape != (S, S):
            fr = cv2.resize(fr, (S, S), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    v = np.stack(frames)
    if len(v) != T:
        v = v[np.linspace(0, len(v) - 1, T).astype(int)]
    return v.astype(np.uint8)


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": round(float(x.mean()), 4),
        "sd": round(float(x.std(ddof=1)), 4) if x.size > 1 else None,
        "median": round(float(np.median(x)), 4),
        "p05": round(float(np.percentile(x, 5)), 4),
        "p95": round(float(np.percentile(x, 95)), 4),
    }


def sector_mask(clip01):
    return clip01.mean(0) > 0.02


def cavity_mask(frame01, sector):
    if int(sector.sum()) < 50:
        return np.zeros_like(frame01, dtype=bool)
    thr = float(np.percentile(frame01[sector], 30))
    raw = (frame01 < thr) & sector
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
    if n <= 1:
        return raw
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = lab == k
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return m


def cavity_series(clip01):
    sector = sector_mask(clip01)
    masks, areas = [], []
    for t in range(len(clip01)):
        m = cavity_mask(clip01[t], sector)
        masks.append(m)
        areas.append(float(m.sum()))
    return np.asarray(areas), masks, sector


def area_change(areas):
    a = np.asarray(areas, dtype=np.float64)
    mx = float(a.max())
    if mx < 1:
        return 0.0
    return float((a.max() - a.min()) / mx)


def curve_metrics(areas):
    a = np.asarray(areas, dtype=np.float64)
    ch = area_change(a)
    if a.max() < 80:
        return ch, np.nan, np.nan
    x = a - a.mean()
    sd = float(x.std())
    if sd < 1e-6:
        return ch, 0.0, 0.0
    x = x / sd
    rough = float(np.sqrt(np.mean(np.diff(x, n=2) ** 2)))
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(x) - 1 :]
    ac = ac / max(ac[0], 1e-6)
    peak_ac = float(ac[6:17].max()) if len(ac) > 16 else 0.0
    return ch, rough, peak_ac


def label_pattern(ch, rough, ch_lo, rough_hi):
    if not np.isfinite(rough):
        return "no_cavity"
    if ch < ch_lo:
        return "flat"
    if rough > rough_hi:
        return "erratic"
    return "periodic"


def wall_profile(clip01, masks, n_rays=36, n_bins=24):
    acc = np.zeros(n_bins, dtype=np.float64)
    w = np.zeros(n_bins, dtype=np.float64)
    for t in range(len(clip01)):
        m = masks[t]
        ys, xs = np.where(m)
        if len(xs) < 30:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        fr = clip01[t]
        h, ww = fr.shape
        for k in range(n_rays):
            ang = 2 * np.pi * k / n_rays
            dy, dx = np.sin(ang), np.cos(ang)
            samples = []
            for r in range(0, max(h, ww)):
                y = int(round(cy + r * dy))
                x = int(round(cx + r * dx))
                if y < 0 or x < 0 or y >= h or x >= ww:
                    break
                samples.append(float(fr[y, x]))
            if len(samples) < 8:
                continue
            idx = np.linspace(0, len(samples) - 1, n_bins)
            prof = np.interp(idx, np.arange(len(samples)), samples)
            acc += prof
            w += 1.0
    if w.max() < 1:
        return np.full(n_bins, np.nan)
    return acc / np.maximum(w, 1.0)


def wall_contrast(profile):
    p = np.asarray(profile, dtype=np.float64)
    if not np.isfinite(p).all():
        return np.nan
    d = np.diff(p)
    i_wall = int(np.argmax(d))
    inner = float(p[: max(i_wall, 1)].mean())
    return float(p.max() - inner)


def shape_circularity(mask):
    a = int(mask.sum())
    if a < 30:
        return np.nan
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.nan
    c = max(cnts, key=cv2.contourArea)
    peri = float(cv2.arcLength(c, True))
    return float(4 * np.pi * a / (peri ** 2)) if peri > 1 else np.nan


def analyze_clip(clip_u8):
    v = clip_u8.astype(np.float32) / 255.0
    areas, masks, sector = cavity_series(v)
    ch, rough, peak_ac = curve_metrics(areas)
    found_n = sum(1 for m in masks if m.sum() >= 30)
    circs = [shape_circularity(m) for m in masks]
    circs = [c for c in circs if np.isfinite(c)]
    prof = wall_profile(v, masks)
    # mean cavity area as fraction of sector (for reporting, not HICSS copy)
    sec_n = max(float(sector.sum()), 1.0)
    mean_area_frac = float(np.mean(areas) / sec_n)
    return {
        "areas": areas.tolist(),
        "area_change": round(ch, 4),
        "roughness": None if not np.isfinite(rough) else round(float(rough), 4),
        "ac_peak": None if not np.isfinite(peak_ac) else round(float(peak_ac), 4),
        "frames_with_cavity": int(found_n),
        "cavity_frac_frames": round(found_n / len(masks), 4),
        "mean_circularity": round(float(np.mean(circs)), 4) if circs else 0.0,
        "mean_area_frac": round(mean_area_frac, 4),
        "wall_contrast": wall_contrast(prof),
        "wall_profile": [None if not np.isfinite(x) else round(float(x), 4) for x in prof],
        "mid_mask": masks[len(masks) // 2],
        "mid_frame": v[len(v) // 2],
    }


def build_composite(real_an, ren_an, path_png, path_pdf):
    """2x2 composite — layout/labels differ from HICSS Fig. 2."""
    fig = plt.figure(figsize=(9.2, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # (a,b) trajectories
    rng = np.random.default_rng(0)
    for col_i, rows, title, col in (
        (0, real_an, "(a) Real paired clip", C_REAL),
        (1, ren_an, "(b) C3DGAN rendered pair", C_REN),
    ):
        ax = fig.add_subplot(gs[0, col_i])
        pick = rng.choice(len(rows), size=min(18, len(rows)), replace=False)
        for i in pick:
            ax.plot(rows[i]["areas"], color=col, alpha=0.55, lw=1.35)
        mean = np.mean([r["areas"] for r in rows], axis=0)
        ax.plot(mean, color="black", lw=2.4, label="mean")
        ax.set_title(title)
        ax.set_xlabel("Frame")
        if col_i == 0:
            ax.set_ylabel("Dark-cavity area (pixels)")
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    # (c) wall profile
    ax = fig.add_subplot(gs[1, 0])
    real_p = np.vstack([r["wall_profile"] for r in real_an]).astype(float)
    ren_p = np.vstack([r["wall_profile"] for r in ren_an]).astype(float)
    x = np.arange(real_p.shape[1])
    for arr, lab, c in ((real_p, "Real", C_REAL), (ren_p, "Rendered", C_REN)):
        m = np.nanmean(arr, 0)
        s = np.nanstd(arr, 0)
        ax.plot(x, m, color=c, lw=2.6, label=lab)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.22)
    ax.set_xlabel("Distance from cavity centre (bins)")
    ax.set_ylabel("Mean brightness")
    ax.set_title("(c) Radial brightness profile")
    ax.legend(frameon=False, fontsize=8)

    # (d) area-change vs circularity scatter (not HICSS dual-hist)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(
        [r["area_change"] for r in real_an],
        [r["mean_circularity"] for r in real_an],
        s=36, c=C_REAL, alpha=0.85, label="Real",
        edgecolors="white", linewidths=0.4, zorder=2,
    )
    ax.scatter(
        [r["area_change"] for r in ren_an],
        [r["mean_circularity"] for r in ren_an],
        s=36, c=C_REN, alpha=0.85, label="Rendered",
        edgecolors="white", linewidths=0.4, zorder=3,
    )
    ax.set_xlabel(r"Relative area change $(A_{\max}-A_{\min})/A_{\max}$")
    ax.set_ylabel("Mean cavity circularity")
    ax.set_title("(d) Motion vs shape")
    ax.legend(frameon=False, fontsize=8)

    # Caption lives in the paper; keep panels only.
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MAN)
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(df), size=N, replace=False)
    sample = df.iloc[idxs].reset_index(drop=True)

    real_an, ren_an = [], []
    pairs = []
    log(f"analyze {N} validation pairs from {MAN.name}")
    for i, row in sample.iterrows():
        rpath = ROOT / str(row["original_path"])
        spath = ROOT / str(row["synthetic_path"])
        rc = decode_mp4(rpath)
        sc = decode_mp4(spath)
        if rc is None or sc is None:
            log(f"skip missing {row['original_path']} / {row['synthetic_path']}")
            continue
        ra = analyze_clip(rc)
        sa = analyze_clip(sc)
        # drop heavy arrays from stored mid masks before JSON
        mid_r, mid_s = ra.pop("mid_mask"), sa.pop("mid_mask")
        fr_r, fr_s = ra.pop("mid_frame"), sa.pop("mid_frame")
        real_an.append(ra)
        ren_an.append(sa)
        pairs.append({
            "original_id": int(row["original_id"]) if "original_id" in row else i,
            "original_path": str(row["original_path"]),
            "synthetic_path": str(row["synthetic_path"]),
            "EF": float(row["EF"]),
            "SSIM": float(row["SSIM"]) if "SSIM" in row and pd.notna(row["SSIM"]) else None,
        })
        if (len(real_an) % 20) == 0:
            log(f"  {len(real_an)}/{N}")

    n = len(real_an)
    assert n >= 50, f"too few pairs decoded: {n}"

    def col(rows, key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    real_ch, ren_ch = col(real_an, "area_change"), col(ren_an, "area_change")
    real_rg, ren_rg = col(real_an, "roughness"), col(ren_an, "roughness")
    real_c, ren_c = col(real_an, "mean_circularity"), col(ren_an, "mean_circularity")
    real_f, ren_f = col(real_an, "cavity_frac_frames"), col(ren_an, "cavity_frac_frames")
    real_af, ren_af = col(real_an, "mean_area_frac"), col(ren_an, "mean_area_frac")
    real_w, ren_w = col(real_an, "wall_contrast"), col(ren_an, "wall_contrast")

    ch_lo = float(np.nanpercentile(real_ch, 5))
    rg_hi = float(np.nanpercentile(real_rg, 95))
    for rows, chs, rgs in ((real_an, real_ch, real_rg), (ren_an, ren_ch, ren_rg)):
        for i, r in enumerate(rows):
            r["pattern"] = label_pattern(float(chs[i]), float(rgs[i]), ch_lo, rg_hi)

    def pat_counts(rows):
        s = pd.Series([r["pattern"] for r in rows]).value_counts().to_dict()
        return {k: int(s.get(k, 0)) for k in ("periodic", "flat", "erratic", "no_cavity")}

    def ks(a, b):
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            return None
        st = spstats.ks_2samp(a, b)
        return {"statistic": round(float(st.statistic), 4), "pvalue": float(st.pvalue)}

    # paired agreement: Spearman of area_change and mean abs traj corr
    from scipy.stats import spearmanr
    sp_ch = spearmanr(real_ch, ren_ch)
    traj_corrs = []
    for ra, sa in zip(real_an, ren_an):
        a = np.asarray(ra["areas"], float)
        b = np.asarray(sa["areas"], float)
        if a.std() < 1e-6 or b.std() < 1e-6:
            traj_corrs.append(np.nan)
        else:
            traj_corrs.append(float(np.corrcoef(a, b)[0, 1]))
    traj_corrs = np.asarray(traj_corrs, float)

    # gate fails for rendered vs real 5-95
    lo, hi = np.percentile(real_ch[np.isfinite(real_ch)], [5, 95])
    fail_ch = [i for i, v in enumerate(ren_ch) if v < lo or v > hi]
    fail_flat = [i for i, r in enumerate(ren_an) if r["pattern"] == "flat"]
    fail_any_pat = [i for i, r in enumerate(ren_an) if r["pattern"] in ("flat", "erratic", "no_cavity")]

    # union fail on four measures
    measures = {
        "area_change": (ren_ch, real_ch),
        "roughness": (ren_rg, real_rg),
        "mean_circularity": (ren_c, real_c),
        "wall_contrast": (ren_w, real_w),
    }
    fail_union = set()
    for name, (sv, rv) in measures.items():
        rvf = rv[np.isfinite(rv)]
        lo_i, hi_i = np.percentile(rvf, [5, 95])
        for i, v in enumerate(sv):
            if np.isfinite(v) and (v < lo_i or v > hi_i):
                fail_union.add(i)

    png = FIG / "c3dgan_paired_chamber_structure.png"
    pdf = FIG / "c3dgan_paired_chamber_structure.pdf"
    build_composite(real_an, ren_an, png, pdf)

    # strip wall profiles from JSON (keep summary)
    for rows in (real_an, ren_an):
        for r in rows:
            r.pop("wall_profile", None)
            r.pop("areas", None)

    report = {
        "protocol": {
            "generator": "C3DGAN paired reconstruction used by EchoStature",
            "manifest": str(MAN),
            "n_pairs": n,
            "seed": SEED,
            "frames": T,
            "resolution": S,
            "segmentation": (
                "classical dark-cavity connected component (30th percentile in "
                "ultrasound sector + largest component); not a trained LV segmenter"
            ),
            "note": (
                "Paired real/rendered clips from the EF validation fused manifest. "
                "Not the HICSS ACGAN novel-sample anatomy protocol."
            ),
        },
        "detection": {
            "real_all_frames_cavity": int((real_f == 1.0).sum()),
            "rendered_all_frames_cavity": int((ren_f == 1.0).sum()),
            "real_cavity_frac": summarize(real_f),
            "rendered_cavity_frac": summarize(ren_f),
        },
        "motion": {
            "area_change_real": summarize(real_ch),
            "area_change_rendered": summarize(ren_ch),
            "ks_area_change": ks(real_ch, ren_ch),
            "roughness_real": summarize(real_rg),
            "roughness_rendered": summarize(ren_rg),
            "ks_roughness": ks(real_rg, ren_rg),
            "pattern_real": pat_counts(real_an),
            "pattern_rendered": pat_counts(ren_an),
            "gates": {"area_change_p05": round(ch_lo, 4), "roughness_p95": round(rg_hi, 4)},
            "n_rendered_flat": len(fail_flat),
            "n_rendered_nonperiodic": len(fail_any_pat),
        },
        "geometry": {
            "mean_area_frac_real": summarize(real_af),
            "mean_area_frac_rendered": summarize(ren_af),
            "ks_area_frac": ks(real_af, ren_af),
            "circularity_real": summarize(real_c),
            "circularity_rendered": summarize(ren_c),
            "ks_circularity": ks(real_c, ren_c),
            "wall_contrast_real": summarize(real_w),
            "wall_contrast_rendered": summarize(ren_w),
            "ks_wall_contrast": ks(real_w, ren_w),
        },
        "paired_agreement": {
            "spearman_area_change": {
                "rho": round(float(sp_ch.correlation), 4),
                "pvalue": float(sp_ch.pvalue),
            },
            "trajectory_pearson": summarize(traj_corrs),
        },
        "fails": {
            "n_outside_real_5_95_any_of_four": len(fail_union),
            "n_area_change_outside": len(fail_ch),
            "chance_note": (
                "With four correlated 5–95 gates on n=100, roughly ten chance "
                "fails are expected even under matched distributions."
            ),
        },
        "pairs_meta": pairs,
        "figure": {"png": str(png), "pdf": str(pdf)},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{
        **pairs[i],
        "real_area_change": real_ch[i],
        "ren_area_change": ren_ch[i],
        "real_pattern": real_an[i]["pattern"],
        "ren_pattern": ren_an[i]["pattern"],
        "real_circularity": real_c[i],
        "ren_circularity": ren_c[i],
        "traj_corr": traj_corrs[i],
    } for i in range(n)]).to_csv(OUT / "per_pair.csv", index=False)

    log(json.dumps({
        "n": n,
        "pattern_real": report["motion"]["pattern_real"],
        "pattern_rendered": report["motion"]["pattern_rendered"],
        "area_change": {
            "real": report["motion"]["area_change_real"],
            "rendered": report["motion"]["area_change_rendered"],
            "ks_p": report["motion"]["ks_area_change"]["pvalue"] if report["motion"]["ks_area_change"] else None,
        },
        "traj_corr_mean": report["paired_agreement"]["trajectory_pearson"]["mean"],
        "n_fail_any": report["fails"]["n_outside_real_5_95_any_of_four"],
        "figure": str(png),
    }, indent=2))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
