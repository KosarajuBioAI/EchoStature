"""Figure 1 in the original two-row layout. Times, 300 dpi, notes only in the gap."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon

fm.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/times.ttf")
fm.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

OUT = Path(__file__).resolve().parent


def rbox(ax, x, y, w, h, fc, ec, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, transform=ax.transAxes, clip_on=False,
    ))


def T(ax, x, y, s, size=8, bold=False, color="#222", ha="center", va="center"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, color=color,
            fontsize=size, fontweight="bold" if bold else "regular", linespacing=1.05)


def arr(ax, a, b):
    ax.add_patch(FancyArrowPatch(
        a, b, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color="#444", clip_on=False,
    ))


def video_icon(ax, x, y):
    for i, dx in enumerate((0.0, 0.012, 0.024)):
        ax.add_patch(FancyBboxPatch(
            (x + dx, y + i * 0.012), 0.055, 0.055,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor="#1c1c1c", edgecolor="#888", linewidth=0.4,
            transform=ax.transAxes, clip_on=False,
        ))


def bars(ax, x, y):
    for i, (bw, bh, c) in enumerate((
        (0.012, 0.07, "#9ec9ef"), (0.014, 0.09, "#6aa6db"), (0.016, 0.11, "#3d7ab8"),
    )):
        ax.add_patch(Rectangle(
            (x + i * 0.02, y), bw, bh, facecolor=c, edgecolor="#2C5F8A",
            linewidth=0.4, transform=ax.transAxes, clip_on=False,
        ))


def cells(ax, x, y, color="#8FCB9B"):
    for i in range(3):
        ax.add_patch(FancyBboxPatch(
            (x + i * 0.028, y), 0.022, 0.028,
            boxstyle="round,pad=0.001,rounding_size=0.003",
            facecolor=color, edgecolor="#3d7a4a", linewidth=0.5,
            transform=ax.transAxes, clip_on=False,
        ))
    T(ax, x + 0.09, y + 0.014, "...", size=8)


def dots(ax, x, y, color):
    for i in range(4):
        ax.add_patch(Circle((x, y - i * 0.022), 0.008, facecolor=color,
                            edgecolor="#555", linewidth=0.3, transform=ax.transAxes))


def heart(ax, x, y):
    ax.add_patch(Polygon(
        [(x, y + 0.02), (x + 0.03, y + 0.05), (x + 0.06, y + 0.02), (x + 0.03, y - 0.03)],
        closed=True, facecolor="#f4a6b5", edgecolor="#a33", linewidth=0.6,
        transform=ax.transAxes,
    ))


def main():
    fig = plt.figure(figsize=(10.6, 5.05), dpi=300)
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    T(ax, 0.34, 0.97, "REAL MODEL", size=11, bold=True, color="#2C5F8A")
    T(ax, 0.30, 0.03, "FUSED MODEL", size=11, bold=True, color="#7B4B9A")

    # ---- top row ----
    rbox(ax, 0.015, 0.62, 0.145, 0.30, "#EAF3FC", "#7EAFD4")
    T(ax, 0.087, 0.86, "Real Echocardiography\nVideo", size=7.6, bold=True)
    video_icon(ax, 0.05, 0.70)
    T(ax, 0.087, 0.655, "32 frames\n128×128", size=7)

    rbox(ax, 0.185, 0.62, 0.145, 0.30, "#EAF3FC", "#7EAFD4")
    T(ax, 0.257, 0.86, "Spatial Feature\nExtraction (ResNet)", size=7.4, bold=True)
    bars(ax, 0.21, 0.66)

    rbox(ax, 0.355, 0.62, 0.17, 0.30, "#E7F6EA", "#8FCB9B")
    T(ax, 0.44, 0.86, "Temporal Modeling\n(LSTM)", size=7.4, bold=True)
    cells(ax, 0.375, 0.70)

    rbox(ax, 0.55, 0.62, 0.12, 0.30, "#FFF6E8", "#E0B15A")
    T(ax, 0.61, 0.86, "Feature\nRepresentation", size=7.4, bold=True)
    dots(ax, 0.61, 0.76, "#b07bdb")

    rbox(ax, 0.78, 0.64, 0.145, 0.26, "#F2F2F2", "#888")
    T(ax, 0.852, 0.84, "Demographic\nFeatures", size=7.6, bold=True)
    T(ax, 0.852, 0.72, "Sex\nAge\nBMI", size=7.4)

    arr(ax, (0.16, 0.77), (0.185, 0.77))
    arr(ax, (0.33, 0.77), (0.355, 0.77))
    arr(ax, (0.525, 0.77), (0.55, 0.77))

    # ---- bottom row ----
    rbox(ax, 0.015, 0.12, 0.145, 0.30, "#F3E9FA", "#C4A6DE")
    T(ax, 0.087, 0.36, "Synthetic Echocardiography\nVideo", size=7.2, bold=True)
    video_icon(ax, 0.05, 0.20)
    T(ax, 0.087, 0.155, "32 frames\n128×128", size=7)

    rbox(ax, 0.185, 0.12, 0.145, 0.30, "#EAF3FC", "#7EAFD4")
    T(ax, 0.257, 0.36, "Spatial Feature\nExtraction (ResNet)", size=7.4, bold=True)
    bars(ax, 0.21, 0.16)

    rbox(ax, 0.355, 0.12, 0.17, 0.30, "#E7F6EA", "#8FCB9B")
    T(ax, 0.44, 0.36, "Temporal Modeling\n(LSTM)", size=7.4, bold=True)
    cells(ax, 0.375, 0.20)

    rbox(ax, 0.55, 0.08, 0.12, 0.22, "#FFF6E8", "#E0B15A")
    T(ax, 0.61, 0.26, "Feature\nRepresentation", size=7.2, bold=True)
    dots(ax, 0.61, 0.16, "#b07bdb")

    rbox(ax, 0.55, 0.40, 0.16, 0.18, "#FDECEC", "#E09A9A")
    T(ax, 0.63, 0.54, "Gated Fusion", size=7.6, bold=True)
    T(ax, 0.63, 0.46, r"$\sigma$", size=11)

    rbox(ax, 0.73, 0.40, 0.11, 0.18, "#FFF6E8", "#E0B15A")
    T(ax, 0.785, 0.52, "Fused Feature\nRepresentation", size=6.8, bold=True)
    dots(ax, 0.785, 0.44, "#e0a040")

    rbox(ax, 0.855, 0.40, 0.09, 0.18, "#E7F6EA", "#8FCB9B")
    T(ax, 0.90, 0.52, "Concatenation", size=6.8, bold=True)
    T(ax, 0.90, 0.45, "+", size=12, bold=True)

    rbox(ax, 0.78, 0.12, 0.10, 0.22, "#EAF3FC", "#7EAFD4")
    T(ax, 0.83, 0.30, "Fully Connected\nLayers (MLP)", size=6.6, bold=True)

    rbox(ax, 0.90, 0.12, 0.085, 0.22, "#F3E9FA", "#C4A6DE")
    T(ax, 0.942, 0.30, "EF Prediction\n(%)", size=6.8, bold=True)
    heart(ax, 0.912, 0.16)

    arr(ax, (0.16, 0.27), (0.185, 0.27))
    arr(ax, (0.33, 0.27), (0.355, 0.27))
    arr(ax, (0.525, 0.27), (0.55, 0.19))
    arr(ax, (0.67, 0.77), (0.67, 0.58))
    arr(ax, (0.71, 0.49), (0.73, 0.49))
    arr(ax, (0.84, 0.49), (0.855, 0.49))
    arr(ax, (0.925, 0.77), (0.90, 0.58))
    arr(ax, (0.90, 0.40), (0.88, 0.34))
    arr(ax, (0.88, 0.23), (0.90, 0.23))

    # Information only in the white band between the rows.
    T(ax, 0.78, 0.575,
      r"Shared ResNet-34.   $d$: 2 sex + 5 age + 4 BMI, mapped to a 32-D embedding.",
      size=7.6, color="#333")
    T(ax, 0.78, 0.545,
      r"The synthetic clip is a stored reconstruction $G(x_{\mathrm{real}}, d)$, used only by the fused model.",
      size=7.6, color="#333")

    fig.savefig(OUT / "model_overview.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "model_overview.pdf", facecolor="white")
    print(OUT / "model_overview.png")


if __name__ == "__main__":
    main()
