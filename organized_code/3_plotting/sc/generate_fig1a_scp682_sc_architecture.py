from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D


MM_TO_IN = 1 / 25.4
WIDTH_MM = 116.675
HEIGHT_MM = 58.535


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "04_figure_source_data" / "fig1_model_architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)


COL = {
    "rna": "#7EA6C8",
    "foundation": "#D9E7F2",
    "pathway": "#E8C46C",
    "site": "#D889A2",
    "graph": "#C6C7E8",
    "fusion": "#BFD7C2",
    "head": "#EFE6D1",
    "line": "#374151",
    "muted": "#6B7280",
    "light": "#F7F7F7",
}


def add_box(ax, xy, w, h, text, fc, ec="#374151", fontsize=6.0, lw=0.9, r=0.025):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        linespacing=1.05,
    )
    return patch


def arrow(ax, p1, p2, rad=0.0, lw=1.0, color=None, ms=8):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or COL["line"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def draw_matrix(ax, x, y, w, h):
    rows, cols = 7, 6
    for i in range(rows):
        for j in range(cols):
            c = COL["rna"] if (i + j) % 3 else "#A9C5DD"
            if i in (1, 5):
                c = "#E8A4B8"
            ax.add_patch(
                Rectangle(
                    (x + j * w / cols, y + i * h / rows),
                    w / cols,
                    h / rows,
                    facecolor=c,
                    edgecolor="#334155",
                    linewidth=0.35,
                )
            )
    ax.text(x + w / 2, y + h + 0.025, "scRNA\ncells × genes", ha="center", va="bottom", fontsize=5.7)
    ax.text(x - 0.018, y + h / 2, "input", ha="right", va="center", fontsize=5.5, color=COL["muted"], rotation=90)


def draw_pathway_tokens(ax, x, y, w, h):
    labels = ["BCR/BTK", "JAK/STAT", "MAPK", "mTOR/S6", "NFκB"]
    gap = h / (len(labels) * 5.6)
    token_h = (h - gap * (len(labels) - 1)) / len(labels)
    for i, lab in enumerate(labels):
        yy = y + h - (i + 1) * token_h - i * gap
        add_box(ax, (x, yy), w, token_h, lab, COL["pathway"], fontsize=4.9, lw=0.55, r=0.015)
    ax.text(x + w / 2, y + h + 0.025, "pathway tokens", ha="center", va="bottom", fontsize=5.7)


def draw_site_queries(ax, x, y, w, h):
    n = 7
    gap = h / 42
    qh = (h - gap * (n - 1)) / n
    for i in range(n):
        yy = y + h - (i + 1) * qh - i * gap
        col = COL["site"] if i < 5 else "#E7B8C6"
        ax.add_patch(
            FancyBboxPatch(
                (x, yy),
                w,
                qh,
                boxstyle="round,pad=0.003,rounding_size=0.010",
                facecolor=col,
                edgecolor="#7F1D1D",
                linewidth=0.35,
            )
        )
    ax.text(x + w / 2, y + h + 0.025, "site queries\n56 readouts", ha="center", va="bottom", fontsize=5.5)


def draw_graph(ax, cx, cy, r):
    pts = []
    for i in range(15):
        ang = 2 * math.pi * i / 15
        rr = r * (0.42 + 0.58 * ((i * 7) % 11) / 10)
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    for i, p in enumerate(pts):
        for j in range(i + 1, len(pts)):
            if (i * 3 + j * 5) % 9 in (0, 1):
                ax.add_line(Line2D([p[0], pts[j][0]], [p[1], pts[j][1]], color="#9CA3AF", lw=0.35, alpha=0.85))
    for i, (px, py) in enumerate(pts):
        color = COL["site"] if i in (0, 4, 8, 11) else COL["graph"]
        size = 0.010 if i in (0, 4, 8, 11) else 0.007
        ax.add_patch(Circle((px, py), size, facecolor=color, edgecolor="#374151", linewidth=0.35))
    ax.text(cx, cy + r + 0.035, "ScNET site graph", ha="center", va="bottom", fontsize=5.7)
    ax.text(cx, cy - r - 0.022, "56 supervised + 7,313 auxiliary\n882,959 edges", ha="center", va="top", fontsize=4.7, color=COL["muted"], linespacing=1.0)


def draw_prediction_bars(ax, x, y, w, h):
    labels = ["pSTAT3", "pRPS6", "pMAPK", "pBTK", "pNFKB"]
    vals = [0.76, 0.62, 0.54, 0.42, 0.30]
    for i, (lab, val) in enumerate(zip(labels, vals)):
        yy = y + h - (i + 1) * h / 5 + 0.006
        bh = h / 7.5
        ax.add_patch(Rectangle((x, yy), w, bh, facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.35))
        ax.add_patch(Rectangle((x, yy), w * val, bh, facecolor=COL["rna"], edgecolor="none"))
        ax.text(x - 0.007, yy + bh / 2, lab, ha="right", va="center", fontsize=4.2)
    ax.text(x + w / 2, y + h + 0.025, "predicted\nphospho profile", ha="center", va="bottom", fontsize=5.5)


def main():
    plt.rcParams.update({
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    })
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.018, 0.948, "a", ha="left", va="top", fontsize=9.5, fontweight="bold")
    ax.text(0.055, 0.946, "SCP682-SC architecture", ha="left", va="top", fontsize=8.2, fontweight="bold")

    y_top = 0.58
    y_bot = 0.17
    draw_matrix(ax, 0.045, y_top, 0.105, 0.245)
    add_box(ax, (0.205, y_top + 0.050), 0.122, 0.142, "scFoundation\nencoder", COL["foundation"], fontsize=5.8)
    add_box(ax, (0.375, y_top + 0.052), 0.130, 0.138, "gene-level\nattention pooling", "#F3E6B0", fontsize=5.5)
    draw_pathway_tokens(ax, 0.558, y_top - 0.002, 0.107, 0.250)

    arrow(ax, (0.152, y_top + 0.123), (0.205, y_top + 0.123))
    arrow(ax, (0.327, y_top + 0.123), (0.375, y_top + 0.123))
    arrow(ax, (0.505, y_top + 0.123), (0.558, y_top + 0.123))

    draw_site_queries(ax, 0.064, y_bot + 0.005, 0.062, 0.250)
    add_box(ax, (0.190, y_bot + 0.052), 0.125, 0.140, "cross-attention\nsite × pathway", "#F2D8E1", fontsize=5.5)
    draw_graph(ax, 0.418, y_bot + 0.127, 0.078)
    add_box(ax, (0.555, y_bot + 0.060), 0.132, 0.125, "fusion\npathway state +\nGNN correction", COL["fusion"], fontsize=5.0)
    add_box(ax, (0.735, y_bot + 0.065), 0.102, 0.115, "masked\nphospho head", COL["head"], fontsize=5.5)
    draw_prediction_bars(ax, 0.910, y_bot + 0.030, 0.060, 0.205)

    arrow(ax, (0.612, y_top - 0.006), (0.253, y_bot + 0.203), rad=-0.10, lw=0.75, color="#6B7280", ms=7)
    arrow(ax, (0.126, y_bot + 0.130), (0.190, y_bot + 0.130))
    arrow(ax, (0.315, y_bot + 0.130), (0.340, y_bot + 0.130))
    arrow(ax, (0.496, y_bot + 0.130), (0.555, y_bot + 0.130))
    arrow(ax, (0.687, y_bot + 0.130), (0.735, y_bot + 0.130))
    arrow(ax, (0.837, y_bot + 0.130), (0.908, y_bot + 0.222), lw=0.85, ms=7)

    ax.text(0.207, 0.514, "RNA representation", ha="center", va="center", fontsize=4.7, color=COL["muted"])
    ax.text(0.618, 0.514, "biological routing", ha="center", va="center", fontsize=4.7, color=COL["muted"])
    ax.text(0.412, 0.118, "bulk-derived phosphosite graph prior", ha="center", va="center", fontsize=4.7, color=COL["muted"])
    ax.text(0.786, 0.118, "cell-level phospho prediction", ha="center", va="center", fontsize=4.7, color=COL["muted"])

    for ext in ("svg", "pdf", "png"):
        out = OUT_DIR / f"fig1a_scp682_sc_architecture_116p675x58p535mm.{ext}"
        if ext == "png":
            fig.savefig(out, dpi=600, facecolor="white")
        else:
            fig.savefig(out, facecolor="white")
    plt.close(fig)

    print(OUT_DIR)


if __name__ == "__main__":
    main()
