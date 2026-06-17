from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as _Circle
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_figures" / "fig1" / "rebuild"
OUT.mkdir(parents=True, exist_ok=True)

FIG_W = 18
FIG_H = 11
X_CORR = FIG_H / FIG_W


def Circle(xy, radius, **kwargs):
    return Ellipse(xy, width=2 * radius * X_CORR, height=2 * radius, **kwargs)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    }
)


BLUE = "#3B82C4"
BLUE_DARK = "#24659D"
BLUE_LIGHT = "#EAF3FB"
ORANGE = "#E8912B"
ORANGE_LIGHT = "#FFF1DE"
GREEN = "#4AA66A"
GREEN_LIGHT = "#EAF7EF"
GRAY = "#5B6670"
GRAY_LIGHT = "#F6F7F8"
GRAY_LINE = "#9AA6B2"
YELLOW = "#E7B84A"
YELLOW_LIGHT = "#FFF6D8"
PURPLE = "#8B78C8"
PURPLE_LIGHT = "#F0ECFA"
PINK = "#D6668A"
PINK_LIGHT = "#FCEBF1"
BLACK = "#222222"


def rounded(ax, x, y, w, h, edge, face="white", lw=1.2, r=0.012, ls="-", z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=8, weight="normal", color=BLACK, ha="center", va="center", linespacing=1.0):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
    )


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.2, style="-", ms=8, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=style,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
        )
    )


def step_badge(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 0.012, facecolor=color, edgecolor="white", linewidth=0.9, zorder=5))
    label(ax, x, y - 0.0005, str(n), size=7.3, weight="bold", color="white")


def lane(ax, x, y, w, h, color, title):
    rounded(ax, x, y, w, h, edge=color, face="white", lw=1.4, r=0.010)
    rounded(ax, x + 0.006, y + h - 0.034, 0.086, 0.026, edge=color, face="white", lw=1.0, r=0.006)
    label(ax, x + 0.049, y + h - 0.021, title, size=9.6, weight="bold", color=color)


def heatmap(ax, x, y, w, h, rows=7, cols=10, seed=0, cmap=None, grid="#FFFFFF"):
    rng = np.random.default_rng(seed)
    if cmap is None:
        cmap = ["#D9E8F5", "#A9CBE3", "#6EA6CF", "#2F75B5", "#EA8AA7"]
    data = rng.integers(0, len(cmap), size=(rows, cols))
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x + c * cw, y + (rows - r - 1) * ch),
                    cw,
                    ch,
                    facecolor=cmap[data[r, c]],
                    edgecolor=grid,
                    linewidth=0.45,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#7B8794", linewidth=0.7))


def small_bars(ax, x, y, w, h, color=BLUE):
    vals = [0.35, 0.72, 0.52, 0.88, 0.45]
    gap = w / 15
    bw = (w - 6 * gap) / 5
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x + gap + i * (bw + gap), y), bw, h * v, facecolor=color, edgecolor=color, alpha=0.75))


def draw_encoder(ax, x, y, w, h, color=BLUE):
    rounded(ax, x, y, w, h, edge=color, face="#F7FBFF", lw=0.9, r=0.007)
    n = 5
    xs = np.linspace(x + w * 0.18, x + w * 0.82, n)
    y1, y2 = y + h * 0.32, y + h * 0.70
    for a in xs:
        for b in xs:
            ax.plot([a, b], [y1, y2], color=GRAY_LINE, lw=0.45, alpha=0.65)
    for a in xs:
        ax.add_patch(Circle((a, y1), h * 0.070, facecolor="#BFD7ED", edgecolor=color, linewidth=0.6))
        ax.add_patch(Circle((a, y2), h * 0.070, facecolor="#BFD7ED", edgecolor=color, linewidth=0.6))


def draw_network(ax, x, y, w, h, seed=1, node_color=BLUE, aux_color=None, square=False, faint=False):
    rng = np.random.default_rng(seed)
    n = 15
    pts = rng.random((n, 2))
    pts[:, 0] = x + w * (0.08 + 0.84 * pts[:, 0])
    pts[:, 1] = y + h * (0.10 + 0.80 * pts[:, 1])
    center = np.array([x + w / 2, y + h / 2])
    for i in range(n):
        d = np.sum((pts - pts[i]) ** 2, axis=1)
        for j in np.argsort(d)[1:4]:
            if j > i:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#A6ADB5", lw=0.65 if not faint else 0.45, alpha=0.8)
    for i, p in enumerate(pts):
        color = node_color
        if aux_color and i % 4 == 0:
            color = aux_color
        if square and i % 5 == 0:
            ax.add_patch(Rectangle((p[0] - 0.006, p[1] - 0.006), 0.012, 0.012, facecolor=color, edgecolor="white", linewidth=0.5, zorder=3))
        else:
            ax.add_patch(Circle(p, 0.0065, facecolor=color, edgecolor="white", linewidth=0.45, zorder=3))
    ax.add_patch(Circle(center, 0.001, alpha=0))


def vector_dots(ax, x, y, w, h, color=ORANGE, n=18, fill_every=3):
    xs = np.linspace(x, x + w, n)
    for i, xx in enumerate(xs):
        fill = color if i % fill_every != 0 else "white"
        ax.add_patch(Circle((xx, y + h / 2), h * 0.26, facecolor=fill, edgecolor=color, linewidth=0.8))


def pathway_tokens(ax, x, y, w, h, color=YELLOW):
    names = ["BCR/BTK", "MAPK/ERK", "AKT-mTOR-S6", "NF-kB", "Cell cycle", "Stress/IFN"]
    gap = h * 0.055
    bh = (h - gap * (len(names) - 1)) / len(names)
    for i, name in enumerate(names):
        yy = y + h - (i + 1) * bh - i * gap
        rounded(ax, x, yy, w, bh, edge="#D3A334", face="#FFF6D8", lw=0.8, r=0.004)
        label(ax, x + w / 2, yy + bh / 2, name, size=6.8, weight="bold", color="#537E4F" if "Cell" in name or "Stress" in name else "#6F4F00")


def bar_profile(ax, x, y, w, h, color=BLUE):
    names = ["pSTAT3", "pRPS6", "pMAPK", "pBTK", "pNFKB"]
    vals = [0.84, 0.62, 0.70, 0.38, 0.26]
    gap = h * 0.10
    bh = (h - gap * (len(names) - 1)) / len(names)
    for i, (name, val) in enumerate(zip(names, vals)):
        yy = y + h - (i + 1) * bh - i * gap
        label(ax, x, yy + bh / 2, name, size=6.7, ha="right")
        ax.add_patch(Rectangle((x + 0.012, yy), w * 0.76, bh, facecolor="white", edgecolor="#8EA1B4", linewidth=0.75))
        ax.add_patch(Rectangle((x + 0.012, yy), w * 0.76 * val, bh, facecolor=color, edgecolor=color, linewidth=0))
    ax.plot([x + 0.012, x + w * 0.772], [y - 0.008, y - 0.008], color="#777777", lw=0.7)
    label(ax, x + 0.012, y - 0.022, "0", size=6, ha="center")
    label(ax, x + w * 0.772, y - 0.022, "1", size=6, ha="center")


def delta_components(ax, x, y, w, h):
    names = ["core", "residual", "common"]
    cols = ["#F1B15E", "#E99442", "#F7D29C"]
    for i, (name, col) in enumerate(zip(names, cols)):
        yy = y + h - (i + 1) * (h * 0.24) - i * h * 0.09
        ax.add_patch(Rectangle((x, yy), w * 0.72, h * 0.22, facecolor=col, edgecolor=col))
        label(ax, x + w * 0.80, yy + h * 0.11, f"delta_{name}", size=6.5, ha="left")


def validation_card(ax, x, y, w, h, title, lines, color):
    rounded(ax, x, y, w, h, edge="#CCD2D8", face="white", lw=0.8, r=0.008)
    label(ax, x + 0.012, y + h - 0.025, title, size=8.4, weight="bold", color=color, ha="left")
    for i, line in enumerate(lines):
        label(ax, x + 0.012, y + h - 0.050 - i * 0.020, line, size=6.4, color=BLACK, ha="left")


def draw_bulk_lane(ax):
    x0, y0, w, h = 0.015, 0.675, 0.805, 0.255
    lane(ax, x0, y0, w, h, BLUE, "SCP682 bulk")

    card_y = y0 + 0.047
    card_h = 0.152
    x_input, w_input = x0 + 0.022, 0.130
    x_base, w_base = x0 + 0.192, 0.190
    x_graph, w_graph = x0 + 0.435, 0.210
    x_out, w_out = x0 + 0.690, 0.105

    step_badge(ax, x_input - 0.004, y0 + h - 0.028, 1, BLUE)
    label(ax, x_input + w_input / 2, y0 + h - 0.030, "INPUT", size=7.0, weight="bold")
    heatmap(ax, x_input + 0.012, card_y + 0.035, 0.075, 0.082, rows=8, cols=9, seed=4)
    ax.add_patch(Rectangle((x_input + 0.093, card_y + 0.035), 0.009, 0.082, facecolor="#F17CA0", edgecolor="white", linewidth=0.4))
    ax.add_patch(Rectangle((x_input + 0.093, card_y + 0.015), 0.009, 0.016, facecolor="#8DD1C4", edgecolor="white", linewidth=0.4))
    label(ax, x_input + 0.048, card_y + 0.128, "Bulk RNA expression", size=7.0, weight="bold")
    label(ax, x_input + 0.048, card_y + 0.010, "2,048 genes x n samples", size=6.1)
    label(ax, x_input + 0.113, card_y + 0.077, "17 cancer\ncontexts", size=5.8)

    arrow(ax, x_input + w_input + 0.012, card_y + 0.088, x_base - 0.010, card_y + 0.088, BLUE_DARK, lw=1.1)

    step_badge(ax, x_base - 0.006, y0 + h - 0.028, 2, BLUE)
    rounded(ax, x_base, card_y, w_base, card_h, edge=BLUE, face=BLUE_LIGHT, lw=1.0, r=0.008)
    label(ax, x_base + w_base / 2, card_y + card_h + 0.023, "CALIBRATED BASELINE B_phi", size=7.2, weight="bold")
    rows = [
        ("Cognate RNA-to-site", "strip"),
        ("Parent-protein constrained", "bars"),
        ("Latent cross-pathway", "network"),
    ]
    for i, (txt, icon) in enumerate(rows):
        yy = card_y + card_h - 0.045 - i * 0.044
        rounded(ax, x_base + 0.020, yy, w_base - 0.040, 0.034, edge=BLUE, face="white", lw=0.7, r=0.004)
        if icon == "strip":
            for k, col in enumerate(["#DF5B7F", "#5EA7D5", "#8DD1C4", "#E9C24D", "#A984D6"]):
                ax.add_patch(Rectangle((x_base + 0.027 + k * 0.009, yy + 0.006), 0.006, 0.022, facecolor=col, edgecolor="none"))
        elif icon == "bars":
            small_bars(ax, x_base + 0.025, yy + 0.008, 0.045, 0.020, BLUE)
        else:
            draw_network(ax, x_base + 0.023, yy + 0.004, 0.045, 0.026, seed=12, node_color=BLUE, faint=True)
        label(ax, x_base + 0.104, yy + 0.017, txt, size=5.9)
    heatmap(ax, x_base + w_base - 0.060, card_y + 0.010, 0.046, 0.034, rows=3, cols=5, seed=8)
    label(ax, x_base + w_base - 0.037, card_y + 0.050, "Initial all-site\nprediction", size=5.6)

    arrow(ax, x_base + w_base + 0.012, card_y + 0.088, x_graph - 0.012, card_y + 0.088, BLUE_DARK, lw=1.1)

    step_badge(ax, x_graph - 0.006, y0 + h - 0.028, 3, BLUE)
    rounded(ax, x_graph, card_y, w_graph, card_h, edge=BLUE, face="white", lw=1.0, r=0.008)
    label(ax, x_graph + w_graph / 2, card_y + card_h + 0.023, "DUAL-AXIS GRAPH RESIDUAL", size=7.2, weight="bold")
    draw_network(ax, x_graph + 0.015, card_y + 0.058, 0.062, 0.070, seed=20, node_color=BLUE)
    label(ax, x_graph + 0.046, card_y + 0.040, "G_site\n420,102 edges", size=5.8)
    draw_network(ax, x_graph + 0.103, card_y + 0.058, 0.062, 0.070, seed=23, node_color=BLUE)
    label(ax, x_graph + 0.134, card_y + 0.040, "G_sample\nk = 25", size=5.8)
    rounded(ax, x_graph + 0.078, card_y + 0.060, 0.026, 0.056, edge=BLUE, face=BLUE_LIGHT, lw=0.8, r=0.004)
    label(ax, x_graph + 0.091, card_y + 0.088, "G_theta", size=6.0, weight="bold", color=BLUE_DARK)
    arrow(ax, x_graph + 0.072, card_y + 0.092, x_graph + 0.082, card_y + 0.092, BLUE_DARK, lw=0.9, ms=6)
    arrow(ax, x_graph + 0.103, card_y + 0.092, x_graph + 0.116, card_y + 0.092, BLUE_DARK, lw=0.9, ms=6)
    rounded(ax, x_graph + 0.170, card_y + 0.062, 0.026, 0.053, edge=BLUE, face="#DCECF9", lw=0.7, r=0.004)
    label(ax, x_graph + 0.183, card_y + 0.088, "Delta", size=6.2, weight="bold", color=BLUE_DARK)

    arrow(ax, x_graph + w_graph + 0.012, card_y + 0.088, x_out - 0.010, card_y + 0.088, BLUE_DARK, lw=1.1)

    step_badge(ax, x_out - 0.006, y0 + h - 0.028, 4, BLUE)
    rounded(ax, x_out, card_y, w_out, card_h, edge=BLUE, face="white", lw=1.0, r=0.008)
    label(ax, x_out + w_out / 2, card_y + card_h + 0.023, "OUTPUT", size=7.2, weight="bold")
    label(ax, x_out + w_out / 2, card_y + 0.126, "y_hat = B_phi + alpha Delta", size=7.0, weight="bold", color=BLUE_DARK)
    heatmap(ax, x_out + 0.030, card_y + 0.055, 0.050, 0.047, rows=4, cols=6, seed=33)
    label(ax, x_out + w_out / 2, card_y + 0.030, "n x 18,592 phosphosites", size=5.9)
    label(ax, x_out + w_out / 2, card_y + 0.012, "CPTAC rho=0.55", size=5.9, color=BLUE_DARK)


def draw_ppko_lane(ax):
    x0, y0, w, h = 0.015, 0.420, 0.805, 0.218
    lane(ax, x0, y0, w, h, ORANGE, "SCP682-PPKO")

    card_y = y0 + 0.040
    card_h = 0.122
    xs = [x0 + 0.025, x0 + 0.210, x0 + 0.355, x0 + 0.585]
    ws = [0.145, 0.100, 0.185, 0.205]

    step_badge(ax, xs[0] - 0.004, y0 + h - 0.028, 1, ORANGE)
    label(ax, xs[0] + ws[0] / 2, y0 + h - 0.030, "THREE INPUTS", size=7.0, weight="bold")
    vector_dots(ax, xs[0] + 0.010, card_y + 0.094, ws[0] - 0.025, 0.020, ORANGE, 17, 4)
    vector_dots(ax, xs[0] + 0.010, card_y + 0.057, ws[0] - 0.025, 0.020, "#C7893B", 17, 2)
    vector_dots(ax, xs[0] + 0.010, card_y + 0.020, ws[0] - 0.025, 0.020, "#E3B373", 17, 3)
    label(ax, xs[0] + 0.006, card_y + 0.121, "Baseline phospho b", size=6.1, ha="left")
    label(ax, xs[0] + 0.006, card_y + 0.084, "Observation mask m", size=6.1, ha="left")
    label(ax, xs[0] + 0.006, card_y + 0.047, "Signed drug target t", size=6.1, ha="left")
    rounded(ax, xs[0] + 0.006, card_y - 0.006, ws[0] - 0.012, 0.018, edge=ORANGE, face="white", lw=0.8, r=0.004)
    label(ax, xs[0] + ws[0] / 2, card_y + 0.003, "decryptM: 60 drug-cell comparisons", size=5.6)

    arrow(ax, xs[0] + ws[0] + 0.012, card_y + 0.070, xs[1] - 0.010, card_y + 0.070, ORANGE, lw=1.1)

    step_badge(ax, xs[1] - 0.004, y0 + h - 0.028, 2, ORANGE)
    rounded(ax, xs[1], card_y, ws[1], card_h, edge=ORANGE, face=ORANGE_LIGHT, lw=1.0, r=0.008)
    label(ax, xs[1] + ws[1] / 2, y0 + h - 0.030, "ENCODERS", size=7.0, weight="bold")
    draw_encoder(ax, xs[1] + 0.020, card_y + 0.035, 0.060, 0.062, ORANGE)
    label(ax, xs[1] + ws[1] / 2, card_y + 0.016, "h_base + h_target", size=5.9)

    arrow(ax, xs[1] + ws[1] + 0.014, card_y + 0.070, xs[2] - 0.010, card_y + 0.070, ORANGE, lw=1.1)

    step_badge(ax, xs[2] - 0.004, y0 + h - 0.028, 3, ORANGE)
    rounded(ax, xs[2], card_y, ws[2], card_h, edge=ORANGE, face="white", lw=1.0, r=0.008)
    label(ax, xs[2] + ws[2] / 2, y0 + h - 0.030, "REGULATORY GRAPH + GATE", size=7.0, weight="bold")
    draw_network(ax, xs[2] + 0.015, card_y + 0.026, 0.090, 0.077, seed=43, node_color=BLUE, aux_color=ORANGE, square=True)
    rounded(ax, xs[2] + 0.118, card_y + 0.045, 0.052, 0.050, edge=ORANGE, face="#FAD9AF", lw=0.7, r=0.004)
    label(ax, xs[2] + 0.144, card_y + 0.070, "Gate(b, t)", size=6.3, weight="bold", color="#70460C")
    label(ax, xs[2] + ws[2] / 2, card_y + 0.013, "5 edge types, target-driven propagation", size=5.8)

    arrow(ax, xs[2] + ws[2] + 0.014, card_y + 0.070, xs[3] - 0.010, card_y + 0.070, ORANGE, lw=1.1)

    step_badge(ax, xs[3] - 0.004, y0 + h - 0.028, 4, ORANGE)
    rounded(ax, xs[3], card_y, ws[3], card_h, edge=ORANGE, face="white", lw=1.0, r=0.008)
    label(ax, xs[3] + ws[3] / 2, y0 + h - 0.030, "DELTA p AND VALIDATION", size=7.0, weight="bold")
    delta_components(ax, xs[3] + 0.015, card_y + 0.032, 0.090, 0.085)
    rounded(ax, xs[3] + 0.122, card_y + 0.060, 0.060, 0.045, edge=ORANGE, face=ORANGE_LIGHT, lw=0.8, r=0.005)
    label(ax, xs[3] + 0.152, card_y + 0.082, "Delta p\nsum", size=6.4, weight="bold")
    label(ax, xs[3] + 0.152, card_y + 0.028, "P100 dir-acc 0.864\nTCGA AUC 0.717", size=5.9, color="#70460C")


def draw_sc_lane(ax):
    x0, y0, w, h = 0.015, 0.205, 0.805, 0.180
    lane(ax, x0, y0, w, h, GREEN, "SCP682-SC")

    card_y = y0 + 0.034
    card_h = 0.095
    xs = [x0 + 0.026, x0 + 0.205, x0 + 0.345, x0 + 0.590]
    ws = [0.135, 0.100, 0.190, 0.205]

    step_badge(ax, xs[0] - 0.004, y0 + h - 0.028, 1, GREEN)
    label(ax, xs[0] + ws[0] / 2, y0 + h - 0.030, "TRAINING CELLS", size=7.0, weight="bold")
    rng = np.random.default_rng(101)
    centers = [(xs[0] + 0.035, card_y + 0.077, "#58B26B"), (xs[0] + 0.088, card_y + 0.060, "#4596D9"), (xs[0] + 0.055, card_y + 0.027, "#EF5F72")]
    for cx, cy, col in centers:
        pts = rng.normal(0, 0.010, size=(26, 2))
        for px, py in pts:
            ax.add_patch(Circle((cx + px, cy + py), 0.0028, facecolor=col, edgecolor="none", alpha=0.85))
    label(ax, xs[0] + ws[0] / 2, card_y + 0.010, "121,847 cells, 56 readouts", size=5.9)

    arrow(ax, xs[0] + ws[0] + 0.014, card_y + 0.060, xs[1] - 0.010, card_y + 0.060, GREEN, lw=1.1)

    step_badge(ax, xs[1] - 0.004, y0 + h - 0.028, 2, GREEN)
    rounded(ax, xs[1], card_y, ws[1], card_h, edge=GREEN, face=GREEN_LIGHT, lw=1.0, r=0.008)
    label(ax, xs[1] + ws[1] / 2, y0 + h - 0.030, "FROZEN ENCODER", size=7.0, weight="bold")
    draw_encoder(ax, xs[1] + 0.020, card_y + 0.034, 0.060, 0.052, GREEN)
    label(ax, xs[1] + ws[1] / 2, card_y + 0.016, "scFoundation", size=6.0, weight="bold", color="#2F7D4E")

    arrow(ax, xs[1] + ws[1] + 0.014, card_y + 0.060, xs[2] - 0.010, card_y + 0.060, GREEN, lw=1.1)

    step_badge(ax, xs[2] - 0.004, y0 + h - 0.028, 3, GREEN)
    rounded(ax, xs[2], card_y, ws[2], card_h, edge=GREEN, face="white", lw=1.0, r=0.008)
    label(ax, xs[2] + ws[2] / 2, y0 + h - 0.030, "PATHWAY-SITE ROUTING", size=7.0, weight="bold")
    pathway_tokens(ax, xs[2] + 0.012, card_y + 0.019, 0.068, 0.076, GREEN)
    ax.add_patch(Circle((xs[2] + 0.106, card_y + 0.060), 0.024, facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=0.9))
    label(ax, xs[2] + 0.106, card_y + 0.060, "x\nsoftmax", size=5.7)
    for i in range(5):
        ax.add_patch(Rectangle((xs[2] + 0.148, card_y + 0.027 + i * 0.014), 0.008, 0.010, facecolor=GREEN, edgecolor=GREEN, alpha=0.82))
    label(ax, xs[2] + 0.154, card_y + 0.013, "56 site queries", size=5.5)
    arrow(ax, xs[2] + 0.082, card_y + 0.060, xs[2] + 0.096, card_y + 0.060, GREEN, lw=0.8, ms=6)
    arrow(ax, xs[2] + 0.130, card_y + 0.060, xs[2] + 0.148, card_y + 0.060, GREEN, lw=0.8, ms=6)

    arrow(ax, xs[2] + ws[2] + 0.014, card_y + 0.060, xs[3] - 0.010, card_y + 0.060, GREEN, lw=1.1)

    step_badge(ax, xs[3] - 0.004, y0 + h - 0.028, 4, GREEN)
    rounded(ax, xs[3], card_y, ws[3], card_h, edge=GREEN, face="white", lw=1.0, r=0.008)
    label(ax, xs[3] + ws[3] / 2, y0 + h - 0.030, "ScNET GRAPH FUSION", size=7.0, weight="bold")
    draw_network(ax, xs[3] + 0.015, card_y + 0.020, 0.088, 0.070, seed=61, node_color=GREEN, aux_color="#B7DCC2")
    ax.add_patch(Circle((xs[3] + 0.124, card_y + 0.060), 0.017, facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=0.9))
    label(ax, xs[3] + 0.124, card_y + 0.060, "+", size=12, weight="bold", color=GREEN)
    for r in range(4):
        for c in range(6):
            ax.add_patch(Circle((xs[3] + 0.155 + c * 0.011, card_y + 0.036 + r * 0.014), 0.004, facecolor="#7DC99A", edgecolor="none", alpha=0.8))
    label(ax, xs[3] + 0.062, card_y + 0.009, "7,369 nodes, 882,959 edges", size=5.7, color="#2F7D4E")
    label(ax, xs[3] + 0.190, card_y + 0.009, "per-cell readouts", size=5.7, color="#2F7D4E")


def draw_shared_prior(ax):
    x, y, w, h = 0.835, 0.205, 0.150, 0.725
    rounded(ax, x, y, w, h, edge="#C5CBD2", face=GRAY_LIGHT, lw=1.0, r=0.010)
    label(ax, x + w / 2, y + h - 0.030, "Shared phosphosite\ngraph priors", size=9.0, weight="bold")
    label(ax, x + w / 2, y + h - 0.066, "420,102 edges", size=6.7, color=GRAY)
    rounded(ax, x + 0.018, y + h - 0.145, w - 0.036, 0.052, edge="#D7DCE1", face="white", lw=0.8, r=0.007)
    label(ax, x + w / 2, y + h - 0.113, "CoPheeMap\n386,224", size=7.0, weight="bold")
    rounded(ax, x + 0.018, y + h - 0.215, w - 0.036, 0.052, edge="#D7DCE1", face="white", lw=0.8, r=0.007)
    label(ax, x + w / 2, y + h - 0.183, "CoPheeKSA\n6,520", size=7.0, weight="bold")
    rounded(ax, x + 0.018, y + h - 0.285, w - 0.036, 0.052, edge="#D7DCE1", face="white", lw=0.8, r=0.007)
    label(ax, x + w / 2, y + h - 0.253, "KSTAR\n45,600", size=7.0, weight="bold")
    draw_network(ax, x + 0.025, y + 0.235, w - 0.050, 0.210, seed=77, node_color="#BFC5CC", aux_color="#E0B464")
    label(ax, x + w / 2, y + 0.205, "common structure,\nindependent parameters", size=6.5, color=GRAY)
    for yy in [0.800, 0.540, 0.285]:
        arrow(ax, x, yy, 0.815, yy, color="#9EA7B0", lw=0.9, style=(0, (3, 3)), ms=6)


def draw_validation(ax):
    x, y, w, h = 0.015, 0.030, 0.970, 0.120
    rounded(ax, x, y, w, h, edge="#C8CDD2", face="white", lw=1.0, r=0.010)
    rounded(ax, x + 0.405, y + h - 0.025, 0.160, 0.026, edge="#D8DDE2", face=GRAY_LIGHT, lw=0.8, r=0.012)
    label(ax, x + 0.485, y + h - 0.012, "Independent external validation", size=7.8, weight="bold")
    validation_card(
        ax,
        x + 0.030,
        y + 0.018,
        0.270,
        0.078,
        "Bulk",
        ["FU-iCCA n=208, TU-SCLC n=107", "CHCC-HBV FPKM/RSEM n=159 each", "external rho=0.32-0.37"],
        BLUE_DARK,
    )
    validation_card(
        ax,
        x + 0.355,
        y + 0.018,
        0.260,
        0.078,
        "Perturbation",
        ["LINCS P100: 125 comparisons", "15 compounds, 7 mechanism classes", "TCGA RPPA: n=64, 11 responders"],
        ORANGE,
    )
    validation_card(
        ax,
        x + 0.670,
        y + 0.018,
        0.270,
        0.078,
        "Single cell",
        ["SIGNAL-seq, Phospho-seq, GSE300551", "Vivo-seq Th17", "cross-platform readout ranking"],
        GREEN,
    )


def draw_legend(ax):
    x, y, w, h = 0.050, 0.006, 0.670, 0.016
    items = [(BLUE, "bulk flow"), (ORANGE, "perturbation flow"), (GREEN, "single-cell flow"), ("#BFC5CC", "graph prior"), (PINK, "supervised site")]
    xx = x
    for color, name in items:
        ax.add_patch(Circle((xx, y + h / 2), 0.005, facecolor=color, edgecolor="none"))
        label(ax, xx + 0.012, y + h / 2, name, size=5.8, ha="left", color=GRAY)
        xx += 0.120


def main():
    fig = plt.figure(figsize=(18, 11), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    label(ax, 0.015, 0.985, "a", size=15, weight="bold", ha="left")
    label(
        ax,
        0.050,
        0.985,
        "Architecture of SCP682 for virtual phospho-signaling measurement, perturbation and single-cell projection",
        size=10.8,
        weight="bold",
        ha="left",
    )
    label(
        ax,
        0.050,
        0.965,
        "Three independently trained modules share a phosphosite regulatory graph while keeping separate parameters and task-specific inputs.",
        size=7.6,
        color=GRAY,
        ha="left",
    )

    draw_bulk_lane(ax)
    draw_ppko_lane(ax)
    draw_sc_lane(ax)
    draw_shared_prior(ax)
    draw_validation(ax)
    draw_legend(ax)

    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"scp682_fig1_rebuild_v1.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
