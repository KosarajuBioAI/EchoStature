"""
Regenerate UMAP (sex / age / BMI) colored by true val-manifest demographics,
not by the fused demographics one-hot (which has constant BMI).

Runs on CPU by default so it does not contend with GPU training.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ef_prediction" / "embedding_plots"
IMG_DIR = ROOT / "images"

# Match paper figures (Times New Roman)
for _ttf in (
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
):
    if Path(_ttf).is_file():
        fm.fontManager.addfont(_ttf)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

AGE_ORDER = ["0-1", "2-5", "6-10", "11-15", "16-18"]
BMI_ORDER = ["underweight", "normal", "overweight", "obese"]
SEX_ORDER = ["F", "M"]

SEX_CMAP = ListedColormap(["#D62728", "#17BECF"])
AGE_CMAP = ListedColormap(
    ["#440154", "#3B528B", "#21918C", "#5DC863", "#FDE725"]
)  # discrete viridis-like, one color per age bin
BMI_CMAP = ListedColormap(["#4C78A8", "#54A24B", "#F58518", "#E45756"])


def stem(path: str) -> str:
    return Path(str(path)).stem


def load_true_labels(cfg: dict, fused_manifest: Path):
    import pandas as pd
    from ef_prediction.demographics_utils import compute_bmi_category

    val = pd.read_csv(ROOT / cfg["data"]["val_manifest"])
    if "bmi_category" not in val.columns:
        val["bmi_category"] = val.apply(
            lambda r: compute_bmi_category(r.get("weight"), r.get("height")),
            axis=1,
        )
    val["_id"] = val["file_name"].map(stem)

    fused = pd.read_csv(fused_manifest)
    fused["_id"] = fused["original_path"].map(stem)
    # Preserve fused row order (same order as DataLoader)
    fused = fused.reset_index(drop=True)
    merged = fused[["_id"]].merge(
        val[["_id", "sex", "age_bin", "bmi_category"]],
        on="_id",
        how="left",
        validate="one_to_one",
    )
    if merged["sex"].isna().any():
        raise RuntimeError("Could not align all fused clips to val_manifest demographics.")
    return merged


def plot_categorical(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    out_path: Path,
    categories: list[str],
    cmap,
    legend_title: str,
):
    """labels are integer codes 0..K-1 matching categories order."""
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # No in-figure title (LaTeX subcaptions already name each panel).
    # Oversized tick/colorbar fonts for ~0.32\textwidth panels.
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=150)
    k = len(categories)
    if isinstance(cmap, ListedColormap) and cmap.N == k:
        listed = cmap
    else:
        listed = ListedColormap([cmap(i / max(k - 1, 1)) for i in range(k)])

    boundaries = np.arange(k + 1) - 0.5
    norm = BoundaryNorm(boundaries, listed.N)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap=listed,
        norm=norm,
        s=6,
        alpha=0.75,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.28, linewidth=1.0)
    ax.tick_params(axis="both", which="major", length=6, width=1.2, labelsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    # Tight color bar attached to the axes (no floating legend / big white gap).
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cb = fig.colorbar(sc, cax=cax, ticks=np.arange(k))
    cb.ax.set_yticklabels(categories, fontsize=18, linespacing=0.92)
    cb.ax.tick_params(length=0, pad=5, labelsize=18)
    # Multiline BMI labels need a bit more room beside the bar.
    if any("\n" in str(c) for c in categories):
        cb.set_label(legend_title, fontsize=20, labelpad=20)
    else:
        cb.set_label(legend_title, fontsize=20, labelpad=12)
    cb.outline.set_linewidth(1.1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="ef_prediction/checkpoints/fused/run_1_best.pth",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Use cpu to avoid GPU training contention.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Reload umap_true_labels_cache.npz and only redraw PNGs.",
    )
    args = parser.parse_args()

    with open(ROOT / "ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "umap_true_labels_cache.npz"

    if args.replot_only:
        if not cache.is_file():
            raise FileNotFoundError(cache)
        data = np.load(cache)
        z = data["z"]
        sex_codes = data["sex"]
        age_codes = data["age"]
        bmi_codes = data["bmi"]
        print("Reloaded cache", cache, "z", z.shape)
    else:
        import torch
        import umap
        from torch.utils.data import DataLoader

        from ef_prediction.dataset_demographics import DualVideoEFDataset
        from ef_prediction.demographics_utils import AGE_MAP, BMI_MAP, SEX_MAP
        from ef_prediction.models.pt_efnet_fused import PTEFNetFused

        device = torch.device(args.device)
        fused_manifest = ROOT / cfg["data"]["val_manifest_fused"]
        true = load_true_labels(cfg, fused_manifest)

        sex_codes = np.array(
            [SEX_MAP.get(str(s).strip().upper(), 0) for s in true["sex"]], dtype=int
        )
        sex_codes = np.where(sex_codes > 1, 0, sex_codes)

        age_codes = np.array(
            [AGE_MAP.get(str(a).strip(), -1) for a in true["age_bin"]], dtype=int
        )
        bmi_codes = np.array(
            [BMI_MAP.get(str(b).strip().lower(), -1) for b in true["bmi_category"]], dtype=int
        )
        if (age_codes < 0).any() or (bmi_codes < 0).any():
            raise RuntimeError("Unknown age_bin or bmi_category in aligned labels.")

        print("Label counts (true manifest):")
        print("  sex", {SEX_ORDER[i]: int((sex_codes == i).sum()) for i in range(2)})
        print("  age", {AGE_ORDER[i]: int((age_codes == i).sum()) for i in range(5)})
        print("  bmi", {BMI_ORDER[i]: int((bmi_codes == i).sum()) for i in range(4)})

        dataset = DualVideoEFDataset(
            manifest_path=str(fused_manifest),
            video_root_dir=cfg["data"]["original_video_dir"],
            synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
            video_length=cfg["model"]["video_length"],
            video_size=cfg["model"]["video_size"],
            fused=True,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                real_video, syn_video, ef, sex, age, bmi, demo_vec = batch
                real_video = real_video.to(device)
                syn_video = syn_video.to(device)
                demo_vec = demo_vec.to(device).float()
                _, emb = model(
                    real_video,
                    syn_video,
                    demo_vec,
                    return_embedding="head_input",
                )
                embeddings.append(emb.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        if len(embeddings) != len(true):
            raise RuntimeError(
                f"Embedding/label length mismatch: {len(embeddings)} vs {len(true)}"
            )

        print("Embeddings", embeddings.shape)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        z = reducer.fit_transform(embeddings)
        np.savez(
            cache,
            z=z,
            sex=sex_codes,
            age=age_codes,
            bmi=bmi_codes,
            embeddings=embeddings,
        )
        print(f"Cached {cache}")

    # Filenames kept for PSB_v1.tex includes; plot titles are short/readable.
    targets = [
        ("UMAP_-_SEX_head_input.png", "Sex", sex_codes, SEX_ORDER, SEX_CMAP, "Sex"),
        ("UMAP_-_AGE_head_input.png", "Age bin", age_codes, AGE_ORDER, AGE_CMAP, "Age bin"),
        (
            "UMAP_-_BMI_head_input.png",
            "BMI category",
            bmi_codes,
            # Two-line labels keep the colorbar compact in the paper panel.
            ["Under\nweight", "Normal", "Over\nweight", "Obese"],
            BMI_CMAP,
            "BMI",
        ),
    ]
    for fname, title, labels, cats, cmap, legend in targets:
        out = OUT_DIR / fname
        plot_categorical(
            z,
            labels,
            title=title,
            out_path=out,
            categories=cats,
            cmap=cmap,
            legend_title=legend,
        )
        dest = IMG_DIR / fname
        dest.write_bytes(out.read_bytes())
        print(f"Copied -> {dest}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
