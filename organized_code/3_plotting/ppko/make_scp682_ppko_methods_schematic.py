from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
OUT = ROOT / "04_figures" / "schematics"
OUT.mkdir(parents=True, exist_ok=True)

for font_path in [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\Arial.ttf"),
    Path(r"C:\Windows\Fonts\NotoSans-Regular.ttf"),
]:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


BLACK = "#000000"
GRID = "#BFD0DA"
GRID_DARK = "#8FA7B5"
BALL = "#B7C7F0"
BASELINE = "#F5B97A"
DOWN1 = "#A9C9E2"
DOWN2 = "#5A89B3"
MASK = "#D9D9D9"
OPERATOR = "#333333"
PALE_YELLOW = "#F8E7A8"
PALE_CYAN = "#CDE9E7"
PALE_PINK = "#F4D4DA"
PALE_GRAY = "#F5F5F5"
FIG_W = 12.8
FIG_H = 6.4
Y_VISUAL_SCALE = FIG_W / FIG_H


def add_text(ax, x, y, s, size=8.5, weight="normal", ha="center", va="center", color=BLACK):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va=va,
        color=color,
        linespacing=1.05,
    )


def screen_circle(ax, x, y, r, **kwargs):
    ax.add_patch(Ellipse((x, y), width=2 * r, height=2 * r * Y_VISUAL_SCALE, **kwargs))


def arrow(ax, start, end, lw=0.75, ms=8, color=BLACK, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def projection(u, v):
    cx, cy = 0.245, 0.455
    sx, sy, skew, depth = 0.135, 0.078, 0.052, 0.205
    r2 = u * u + 0.86 * v * v
    z = -0.39 * math.exp(-r2 / 0.23)
    return cx + sx * u + skew * v, cy + sy * v + depth * z


def draw_left_panel(ax):
    add_text(ax, 0.035, 0.89, "a", size=13, weight="bold", ha="left")

    us = np.linspace(-1.65, 1.65, 180)
    vs = np.linspace(-1.40, 1.40, 180)

    for v in np.linspace(-1.35, 1.35, 12):
        pts = np.array([projection(float(u), float(v)) for u in us])
        ax.plot(pts[:, 0], pts[:, 1], color=GRID, lw=0.62, solid_capstyle="round")

    for u in np.linspace(-1.60, 1.60, 13):
        pts = np.array([projection(float(u), float(v)) for v in vs])
        ax.plot(pts[:, 0], pts[:, 1], color=GRID, lw=0.62, solid_capstyle="round")

    theta = np.linspace(0, 2 * np.pi, 220)
    for i, r in enumerate([0.18, 0.27, 0.36, 0.46, 0.58, 0.72, 0.88, 1.08]):
        pts = []
        for t in theta:
            u = r * math.cos(float(t))
            v = 0.78 * r * math.sin(float(t))
            pts.append(projection(u, v))
        pts = np.array(pts)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=GRID_DARK if i < 4 else GRID,
            lw=0.72 if i < 4 else 0.58,
            alpha=0.95 if i < 4 else 0.80,
        )

    center_x, center_y = projection(0.0, 0.0)
    ball_r = 0.030
    ball_yr = ball_r * Y_VISUAL_SCALE
    ball_center = (center_x, center_y + 0.049)
    screen_circle(ax, ball_center[0], ball_center[1], ball_r, facecolor=BALL, edgecolor=BLACK, linewidth=0.85)
    add_text(ax, ball_center[0], ball_center[1] + 0.060, "Gated operator", size=8.8)

    arrow(ax, (ball_center[0], ball_center[1] - ball_yr - 0.004), (ball_center[0], ball_center[1] - 0.083), lw=0.75, ms=7)
    add_text(ax, ball_center[0] + 0.052, ball_center[1] - 0.055, "Operator", size=8.2, ha="left")

    legend_x, legend_y = 0.075, 0.145
    screen_circle(ax, legend_x, legend_y + 0.034, 0.008, facecolor=BALL, edgecolor=BLACK, linewidth=0.7)
    add_text(ax, legend_x + 0.025, legend_y + 0.034, "PPKO target", size=8.2, ha="left")
    ax.plot(
        [legend_x - 0.010, legend_x + 0.010],
        [legend_y - 0.006, legend_y - 0.006],
        color=GRID_DARK,
        lw=1.1,
    )
    ax.plot(
        [legend_x - 0.010, legend_x, legend_x + 0.010],
        [legend_y - 0.010, legend_y - 0.002, legend_y - 0.010],
        color=GRID_DARK,
        lw=0.8,
    )
    add_text(ax, legend_x + 0.025, legend_y - 0.006, "Local response", size=8.2, ha="left")


def draw_heatmap(ax, x, y, w, h, rows, cols, palette, edge="#FFFFFF", outer=True):
    cell_w = w / cols
    cell_h = h / rows
    for r in range(rows):
        for c in range(cols):
            color = palette[(r * 2 + c + (r + c) // 2) % len(palette)]
            ax.add_patch(
                Rectangle(
                    (x + c * cell_w, y + (rows - 1 - r) * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=0.55,
                )
            )
    if outer:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=BLACK, linewidth=0.65))


def draw_mask_strip(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=BLACK, linewidth=0.55))
    xs = np.linspace(x + 0.006, x + w - 0.006, 3)
    ys = np.linspace(y + 0.010, y + h - 0.010, 6)
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            if (i + j) % 3 != 1:
                screen_circle(ax, float(xx), float(yy), 0.0020, facecolor=MASK, edgecolor="none")
            else:
                screen_circle(ax, float(xx), float(yy), 0.0020, facecolor="#FFFFFF", edgecolor=MASK, linewidth=0.35)


def draw_encoder(ax, x, y, w, h):
    pts = [
        (x + 0.12 * w, y),
        (x + 0.88 * w, y),
        (x + w, y + 0.50 * h),
        (x + 0.88 * w, y + h),
        (x + 0.12 * w, y + h),
        (x, y + 0.50 * h),
    ]
    ax.add_patch(Polygon(pts, closed=True, facecolor=PALE_GRAY, edgecolor=BLACK, linewidth=0.75))
    for yy in np.linspace(y + 0.025, y + h - 0.025, 3):
        ax.plot([x + 0.026, x + w - 0.026], [yy, yy], color=MASK, lw=0.55)
    add_text(ax, x + w / 2, y + h / 2, "Graph prior encoder", size=7.8)


def draw_noise_surface(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="none"))
    xs = np.linspace(0, 1, 80)
    for k in range(5):
        yy = y + h * (0.18 + 0.14 * k)
        wave = yy + h * 0.05 * np.sin(2 * np.pi * (xs * 1.15 + k * 0.15))
        ax.plot(x + xs * w, wave, color="#B8C4CB", lw=0.55)
    for k in range(5):
        xx = x + w * (0.10 + 0.18 * k)
        ys = np.linspace(0.15, 0.82, 60)
        curve = xx + w * 0.025 * np.sin(2 * np.pi * ys + k)
        ax.plot(curve, y + ys * h, color="#B8C4CB", lw=0.55)
    ridge_x = x + xs * w
    ridge_y = y + h * (0.50 + 0.16 * np.exp(-((xs - 0.55) ** 2) / 0.035) - 0.08 * np.exp(-((xs - 0.26) ** 2) / 0.025))
    ax.plot(ridge_x, ridge_y, color=DOWN2, lw=0.95)
    screen_circle(ax, x + 0.60 * w, y + 0.64 * h, 0.0037, facecolor=DOWN2, edgecolor=BLACK, linewidth=0.35)


def draw_decomposition(ax, x, y):
    bars = [
        ("Δcore", DOWN2, y + 0.046),
        ("Δresidual", DOWN1, y + 0.023),
        ("Δcommon", "#C7D7E3", y),
    ]
    for label, color, yy in bars:
        ax.add_patch(Rectangle((x, yy), 0.090, 0.0125, facecolor=color, edgecolor=BLACK, linewidth=0.45))
        add_text(ax, x + 0.106, yy + 0.0062, label, size=7.5, ha="left")
    add_text(ax, x + 0.162, y + 0.029, "+", size=11.0)
    return (x + 0.162, y + 0.029)


def draw_pathway(ax, x, y):
    node_positions = [(x, y + 0.080), (x, y + 0.046), (x, y + 0.012)]
    for idx, pos in enumerate(node_positions):
        screen_circle(ax, pos[0], pos[1], 0.0078, facecolor=DOWN1 if idx < 2 else DOWN2, edgecolor=BLACK, linewidth=0.55)
    arrow(ax, (x, y + 0.069), (x, y + 0.056), lw=0.62, ms=6.2, color=DOWN2)
    arrow(ax, (x, y + 0.035), (x, y + 0.022), lw=0.62, ms=6.2, color=DOWN2)
    arrow(ax, (x + 0.026, y + 0.081), (x + 0.026, y + 0.026), lw=0.70, ms=7.2, color=DOWN2)
    add_text(ax, x, y - 0.018, "Pathway down", size=7.4)


def draw_right_panel(ax):
    add_text(ax, 0.505, 0.89, "b", size=13, weight="bold", ha="left")

    input_x, input_y, input_w, input_h = 0.535, 0.600, 0.105, 0.162
    draw_heatmap(ax, input_x, input_y, input_w, input_h, 7, 6, [BASELINE, PALE_YELLOW, PALE_CYAN, PALE_PINK])
    add_text(ax, input_x + input_w / 2, input_y + input_h + 0.030, "Phosphosites", size=8.2)
    add_text(ax, input_x - 0.036, input_y + input_h / 2, "Samples", size=8.0)
    add_text(ax, input_x + input_w / 2, input_y - 0.030, "Baseline phospho", size=8.2)

    mask_x, mask_y, mask_w, mask_h = 0.652, 0.615, 0.026, 0.132
    draw_mask_strip(ax, mask_x, mask_y, mask_w, mask_h)
    add_text(ax, mask_x + mask_w / 2, mask_y + mask_h + 0.024, "Observed mask", size=7.4)

    enc_x, enc_y, enc_w, enc_h = 0.705, 0.622, 0.135, 0.095
    draw_encoder(ax, enc_x, enc_y, enc_w, enc_h)

    z_x, z_y, z_w, z_h = 0.716, 0.802, 0.113, 0.052
    draw_heatmap(ax, z_x, z_y, z_w, z_h, 3, 8, [MASK, PALE_CYAN, DOWN1, BASELINE], outer=True)
    add_text(ax, z_x + z_w / 2, z_y + z_h + 0.029, "Target variable z", size=8.0)

    op_x, op_y, op_w, op_h = 0.675, 0.426, 0.205, 0.098
    ax.add_patch(Rectangle((op_x, op_y), op_w, op_h, facecolor=OPERATOR, edgecolor=BLACK, linewidth=0.78))
    add_text(ax, op_x + op_w / 2, op_y + op_h / 2, "SCP682-PPKO operator", size=8.3, color="#FFFFFF")

    noise_x, noise_y, noise_w, noise_h = 0.540, 0.326, 0.130, 0.078
    draw_noise_surface(ax, noise_x, noise_y, noise_w, noise_h)
    add_text(ax, noise_x + noise_w / 2, noise_y + noise_h + 0.020, "xT ~ N(0, I)", size=7.7)
    add_text(ax, noise_x + noise_w / 2, noise_y - 0.027, "Signed target", size=8.0)

    plus_pos = draw_decomposition(ax, 0.705, 0.287)

    out_x, out_y, out_w, out_h = 0.878, 0.458, 0.076, 0.174
    draw_heatmap(ax, out_x, out_y, out_w, out_h, 7, 5, [DOWN1, DOWN2, "#D8E8F2", PALE_CYAN])
    add_text(ax, out_x + out_w / 2, out_y + out_h + 0.030, "Predicted Δp", size=8.3)
    add_text(ax, out_x + out_w + 0.027, out_y + out_h / 2, "Sites", size=8.0)
    draw_pathway(ax, 0.968, 0.352)

    arrow(ax, (input_x + input_w + 0.008, input_y + input_h * 0.53), (enc_x - 0.008, enc_y + enc_h * 0.53), lw=0.70, ms=7.3)
    arrow(ax, (enc_x + enc_w * 0.42, enc_y + enc_h + 0.002), (z_x + z_w * 0.34, z_y - 0.006), lw=0.70, ms=7.3)
    arrow(ax, (z_x + z_w * 0.90, z_y - 0.006), (op_x + op_w * 0.90, op_y + op_h + 0.008), lw=0.70, ms=7.3)
    arrow(ax, (noise_x + noise_w + 0.006, noise_y + noise_h * 0.56), (op_x - 0.010, op_y + op_h * 0.38), lw=0.70, ms=7.3)
    arrow(ax, (op_x + op_w * 0.50, op_y - 0.004), (0.750, 0.347), lw=0.70, ms=7.3)
    arrow(ax, (plus_pos[0] + 0.020, plus_pos[1]), (out_x - 0.012, out_y + out_h * 0.40), lw=0.70, ms=7.3)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_left_panel(ax)
    draw_right_panel(ax)

    ax.plot([0.492, 0.492], [0.115, 0.875], color="#E5E5E5", lw=0.60)

    svg_path = OUT / "scp682_ppko_nature_methods_mechanism.svg"
    png_path = OUT / "scp682_ppko_nature_methods_mechanism.png"
    pdf_path = OUT / "scp682_ppko_nature_methods_mechanism.pdf"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(svg_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
