"""
模型: SCP682-SC
作用: 生成 SCP682-SC 单细胞磷酸化预测结果图 3 的图源表和 SVG。
输入: paper_materials_SCP682_SC11 下的外部验证、消融、跨平台和 QuRIE delta 表。
输出: 04_figure_source_data/fig3 下的 panel 数据表、单 panel SVG 和组合 SVG。
依赖: Python, pandas, numpy, matplotlib, scipy。
原始路径: E:\data\gongke\TCGA-TCPA\paper_materials_SCP682_SC11\03_code\generate_fig3.py
原始版本: 2026-05-27 论文素材图源脚本。
"""
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon

# Antibody clone variant detector: e.g. MAPK14_pSitePending_D3F9, RELA_pSitePending_93H1
CLONE_SUFFIX_PATTERN = re.compile(r'_pSitePending_[A-Z0-9]+$')

def is_clone_variant(target_id):
    return bool(CLONE_SUFFIX_PATTERN.search(str(target_id)))

# ---------- Constants ----------
COLORS = {
    'base_pred_green': '#6CBFB5',
    'graph_residual_gold': '#D4A56B',
    'external_blue': '#1F3A5F',
    'sc_purple': '#9C8FC4',
    'aux_grey': '#A8A8A8',
    'mismatch_red': '#C97064',
}

SRC_DIR = str(Path(__file__).resolve().parents[1])
OUT_DIR = os.path.join(SRC_DIR, '04_figure_source_data', 'fig3')

PDO_CAF = 'signal_seq_gse256404_pdo_caf_2024'
VIVO = 'vivo_seq_th17_2025'
VIVO_KEEP_TARGET = 'STAT3_Y705'  # Vivo-seq Th17: 只保留 STAT3_Y705 作为锚点位点
COHORT_LABELS = {
    'signal_seq_gse256403_hela_2024': 'HeLa',
    'phospho_seq_blair_2025_phospho_multi': 'Blair',
    'gse300551_iccite_plex_kinase_2025': 'GSE300551',
    'vivo_seq_th17_2025': 'Vivo-seq Th17',
}
COHORT_ORDER = ['HeLa', 'Blair', 'GSE300551', 'Vivo-seq Th17']

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'svg.fonttype': 'none',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})


# ---------- Data loaders ----------
def load_data():
    ext_dir = os.path.join(SRC_DIR, '01_key_results', 'external_validation')
    per_targets = {
        'HeLa': pd.read_csv(os.path.join(ext_dir, 'per_target_signal_seq_hela.tsv'), sep='\t'),
        'Blair': pd.read_csv(os.path.join(ext_dir, 'per_target_blair.tsv'), sep='\t'),
        'GSE300551': pd.read_csv(os.path.join(ext_dir, 'per_target_gse300551.tsv'), sep='\t'),
        'Vivo-seq Th17': pd.read_csv(os.path.join(ext_dir, 'per_target_vivo_seq_th17.tsv'), sep='\t'),
    }
    ablation_per_target = pd.read_csv(os.path.join(SRC_DIR, '02_data_tables', 'bulk_site_graph_matched_ablation_per_target.tsv'), sep='\t')
    qurie_per_target = pd.read_csv(os.path.join(SRC_DIR, '02_data_tables', 'qurie_ibrutinib_delta_per_target.tsv'), sep='\t')
    stat3 = pd.read_csv(os.path.join(SRC_DIR, '02_data_tables', 'stat3_y705_cross_platform.tsv'), sep='\t')
    ctnnd1 = pd.read_csv(os.path.join(OUT_DIR, 'panel_a_hela_ctnnd1_t310_scatter.tsv'), sep='\t')
    return per_targets, ablation_per_target, qurie_per_target, stat3, ctnnd1


def apply_site_filters(df, cohort_display_name, target_col='target_id'):
    """Apply uniform site filters: drop clone variants; Vivo: keep only STAT3_Y705."""
    df = df.copy()
    # Drop clone variants
    df = df[~df[target_col].apply(is_clone_variant)].copy()
    # Vivo-seq Th17: anchor only
    if cohort_display_name == 'Vivo-seq Th17':
        df = df[df[target_col] == VIVO_KEEP_TARGET].copy()
    return df


# ---------- Panel B ----------
def panel_b_data(per_targets):
    """Build panel-b cohort summary directly from per_target tables, applying clone and Vivo filters."""
    rows = []
    for display_name in COHORT_ORDER:
        df = per_targets[display_name]
        filtered = apply_site_filters(df, display_name)
        valid = filtered.dropna(subset=['per_target_spearman'])
        if len(valid) == 0:
            continue
        vals = valid['per_target_spearman'].astype(float)
        site_count = len(vals)
        sites_kept = ';'.join(valid['target_id'].astype(str).tolist())
        sites_dropped_clone = ';'.join(
            df[df['target_id'].apply(is_clone_variant)]['target_id'].astype(str).tolist()
        )
        notes_parts = []
        if sites_dropped_clone:
            notes_parts.append(f'dropped_clone_variants={sites_dropped_clone}')
        if display_name == 'Vivo-seq Th17':
            notes_parts.append(f'restricted_to_{VIVO_KEEP_TARGET}')
        rows.append({
            'display_cohort': display_name,
            'site_count': site_count,
            'median_spearman': float(vals.median()),
            'mean_spearman': float(vals.mean()),
            'min_spearman': float(vals.min()),
            'max_spearman': float(vals.max()),
            'sites_kept': sites_kept,
            'notes': ';'.join(notes_parts),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, 'fig3_panel_b_data.tsv'), sep='\t', index=False, na_rep='NA')
    return df_out


def panel_b_plot(df, ax):
    x = np.arange(len(df))
    bars = ax.bar(x, df['median_spearman'], color=COLORS['sc_purple'],
                  edgecolor='black', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, df['median_spearman']):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f'{val:.3f}',
                ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(df['display_cohort'], rotation=20, ha='right')
    ax.set_ylabel('Median Spearman')
    ax.set_ylim(0, 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', length=2)


# ---------- Panel C ----------
def panel_c_left_data(ctnnd1):
    df = ctnnd1[['predicted', 'observed', 'cell_id', 'target_id', 'cohort_id']].copy()
    df.to_csv(os.path.join(OUT_DIR, 'fig3_panel_c_left_data.tsv'), sep='\t', index=False, na_rep='NA')
    return df


def panel_c_right_data(stat3):
    df = stat3[stat3['test_dataset'] != 'all'].copy()
    label_map = {
        'gse300551_iccite_plex_kinase_2025': 'GSE300551',
        'vivo_seq_th17_2025': 'Vivo-seq Th17',
    }
    df['display_platform'] = df['test_dataset'].map(label_map)
    df_out = df[['display_platform', 'test_dataset', 'target_id', 'n', 'spearman', 'pearson']]
    df_out.to_csv(os.path.join(OUT_DIR, 'fig3_panel_c_right_data.tsv'), sep='\t', index=False, na_rep='NA')
    return df_out


def panel_c_plot_left(ctnnd1_df, ax):
    ax.scatter(ctnnd1_df['observed'], ctnnd1_df['predicted'],
               s=3, alpha=0.45, c=COLORS['external_blue'], linewidths=0)
    lo = min(ctnnd1_df['observed'].min(), ctnnd1_df['predicted'].min()) - 0.2
    hi = max(ctnnd1_df['observed'].max(), ctnnd1_df['predicted'].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], ls='--', color='grey', linewidth=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_aspect('equal')
    ax.text(0.05, 0.95, 'ρ = 0.631\nn = 1,143',
            transform=ax.transAxes, va='top', ha='left', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', length=2)


def panel_c_plot_right(stat3_df, ax):
    x = np.arange(len(stat3_df))
    bars = ax.bar(x, stat3_df['spearman'], color=COLORS['sc_purple'],
                  edgecolor='black', linewidth=0.5, width=0.5)
    for bar, val, n in zip(bars, stat3_df['spearman'], stat3_df['n']):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                f'{val:.3f}\n(n={n:,})', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(stat3_df['display_platform'], rotation=15, ha='right')
    ax.set_ylabel('Spearman')
    ax.set_ylim(0, 0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', length=2)


# ---------- Panel D ----------
def panel_d_data(_unused, ablation_per_target):
    """Compute cohort-level ablation deltas directly from per_target, applying clone and Vivo filters."""
    # Strip PDO-CAF, pooled rows, NA rows
    per_target = ablation_per_target[
        (ablation_per_target['test_dataset'] != PDO_CAF) &
        (ablation_per_target['test_dataset'] != 'all')
    ].copy()
    per_target = per_target.dropna(
        subset=['spearman_expanded_site_graph', 'spearman_matched_no_site_graph']
    ).copy()
    # Drop clone variants uniformly
    per_target = per_target[~per_target['target_id'].apply(is_clone_variant)].copy()
    # Vivo-seq Th17: keep only STAT3_Y705
    cohort_back = {v: k for k, v in COHORT_LABELS.items()}
    vivo_id = cohort_back['Vivo-seq Th17']
    vmask = per_target['test_dataset'] == vivo_id
    per_target = per_target[~vmask | (per_target['target_id'] == VIVO_KEEP_TARGET)].reset_index(drop=True)

    # Rebuild cohort summaries from filtered per_target.
    # Delta is (median no_graph) − (median with_graph), NOT median of per-site deltas.
    # Wilcoxon below still uses per-site paired observations.
    by_cohort_rows = []
    for display_name in COHORT_ORDER:
        cid = cohort_back[display_name]
        sub = per_target[per_target['test_dataset'] == cid]
        if len(sub) == 0:
            continue
        med_with = float(sub['spearman_expanded_site_graph'].median())
        med_no = float(sub['spearman_matched_no_site_graph'].median())
        by_cohort_rows.append({
            'display_cohort': display_name,
            'cohort_id': cid,
            'median_spearman_expanded_site_graph': med_with,
            'median_spearman_matched_no_site_graph': med_no,
            'delta_no_graph_minus_graph': med_no - med_with,
            'site_count_expanded_site_graph': int(len(sub)),
            'site_count_matched_no_site_graph': int(len(sub)),
        })
    by_cohort = pd.DataFrame(by_cohort_rows)

    # Wilcoxon: paired test on 30 (target × cohort) pairs across 4 non-PDO-CAF cohorts
    w_stat, w_p = wilcoxon(per_target['spearman_expanded_site_graph'],
                            per_target['spearman_matched_no_site_graph'],
                            alternative='greater')
    site_level_median_delta = per_target['delta_no_graph_minus_graph'].median()
    site_level_mean_delta = per_target['delta_no_graph_minus_graph'].mean()

    # "Combined (macro)" bar: median across the 4 cohort-level summaries
    macro_expanded = by_cohort['median_spearman_expanded_site_graph'].median()
    macro_no_graph = by_cohort['median_spearman_matched_no_site_graph'].median()
    macro_delta = by_cohort['delta_no_graph_minus_graph'].median()

    rows = []
    for _, r in by_cohort.iterrows():
        rows.append({
            'group': r['display_cohort'],
            'expanded_median_spearman': r['median_spearman_expanded_site_graph'],
            'no_graph_median_spearman': r['median_spearman_matched_no_site_graph'],
            'delta_no_graph_minus_graph': r['delta_no_graph_minus_graph'],
            'site_count': int(r['site_count_expanded_site_graph']),
        })
    rows.append({
        'group': 'Combined (macro)',
        'expanded_median_spearman': macro_expanded,
        'no_graph_median_spearman': macro_no_graph,
        'delta_no_graph_minus_graph': macro_delta,
        'site_count': len(per_target),
    })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'fig3_panel_d_data.tsv'), sep='\t', index=False, na_rep='NA')

    wdf = pd.DataFrame([{
        'test': 'wilcoxon_signed_rank_one_sided_greater',
        'comparison': 'spearman_expanded_site_graph > spearman_matched_no_site_graph',
        'n_pairs': len(per_target),
        'statistic': w_stat,
        'p_value': w_p,
        'site_level_median_delta_no_graph_minus_graph': site_level_median_delta,
        'site_level_mean_delta_no_graph_minus_graph': site_level_mean_delta,
        'cohort_level_median_delta_no_graph_minus_graph': macro_delta,
        'cohorts_included': 'HeLa, Blair, GSE300551, Vivo-seq Th17',
        'cohorts_excluded': 'PDO-CAF (signal_seq_gse256404_pdo_caf_2024)',
    }])
    wdf.to_csv(os.path.join(OUT_DIR, 'fig3_panel_d_wilcoxon.tsv'), sep='\t', index=False)
    return df, w_stat, w_p, len(per_target)


def panel_d_plot(df, ax):
    n = len(df)
    x = np.arange(n)
    w = 0.36
    b1 = ax.bar(x - w / 2, df['expanded_median_spearman'], w,
                label='With expanded GNN', color=COLORS['graph_residual_gold'],
                edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w / 2, df['no_graph_median_spearman'], w,
                label='No expanded GNN', color=COLORS['aux_grey'],
                edgecolor='black', linewidth=0.5)
    for i, (a, b, d) in enumerate(zip(df['expanded_median_spearman'],
                                       df['no_graph_median_spearman'],
                                       df['delta_no_graph_minus_graph'])):
        my = max(a, b)
        ax.text(i, my + 0.02, f'{d:+.3f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(df['group'], rotation=20, ha='right')
    ax.set_ylabel('Median Spearman')
    ymax = max(df['expanded_median_spearman'].max(),
               df['no_graph_median_spearman'].max()) + 0.12
    ax.set_ylim(0, ymax)
    ax.legend(loc='upper right', frameon=False, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', length=2)


# ---------- Panel E ----------
def panel_e_data(qurie_per_target):
    df = qurie_per_target[qurie_per_target['context'] == 'time180'].copy()
    df_out = df[['target_id', 'target_index', 'real_delta', 'pred_delta',
                 'direction_match', 'abs_ratio', 'n_control', 'n_drug']]
    df_out.to_csv(os.path.join(OUT_DIR, 'fig3_panel_e_data.tsv'), sep='\t', index=False, na_rep='NA')
    return df_out


def panel_e_plot(df, ax):
    match = df[df['direction_match'] == True]
    mis = df[df['direction_match'] == False]
    ax.scatter(match['real_delta'], match['pred_delta'], s=22, alpha=0.85,
               c=COLORS['base_pred_green'], edgecolors='black', linewidths=0.4,
               label=f'Sign match (n={len(match)})')
    ax.scatter(mis['real_delta'], mis['pred_delta'], s=22, alpha=0.85,
               c=COLORS['mismatch_red'], edgecolors='black', linewidths=0.4,
               label=f'Sign mismatch (n={len(mis)})')
    lo = min(df['real_delta'].min(), df['pred_delta'].min()) - 0.08
    hi = max(df['real_delta'].max(), df['pred_delta'].max()) + 0.08
    ax.plot([lo, hi], [lo, hi], ls='--', color='grey', linewidth=0.5)
    ax.axhline(0, color='grey', linewidth=0.3, ls=':')
    ax.axvline(0, color='grey', linewidth=0.3, ls=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Observed Δ (ibrutinib − control, 180 min)')
    ax.set_ylabel('Predicted Δ')
    ax.set_aspect('equal')
    ax.text(0.05, 0.95, 'ρ = 0.907\nsign acc = 0.893\nn = 28',
            transform=ax.transAxes, va='top', ha='left', fontsize=6)
    ax.legend(loc='lower right', frameon=False, fontsize=5.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', length=2)


# ---------- Panel A (schematic) ----------
def panel_a_plot(ax, compact=False):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect('auto')
    ax.axis('off')

    fs_box = 5.5 if compact else 6.5
    fs_sub = 5.0 if compact else 5.5
    fs_bottom = 5.5 if compact else 6.0

    boxes = [
        {'x': 1,  'y': 22, 'w': 17, 'h': 16, 'label': 'scRNA',          'color': '#E8E8E8'},
        {'x': 20, 'y': 22, 'w': 17, 'h': 16, 'label': 'scFound\nencoder', 'color': '#B3DDE0'},
        {'x': 39, 'y': 22, 'w': 17, 'h': 16, 'label': 'Pathway\nattn',  'color': COLORS['sc_purple']},
        {'x': 58, 'y': 22, 'w': 17, 'h': 16, 'label': 'Expanded\nGNN',  'color': COLORS['graph_residual_gold']},
        {'x': 77, 'y': 22, 'w': 17, 'h': 16, 'label': 'Fusion\noutput', 'color': COLORS['base_pred_green']},
    ]
    for b in boxes:
        rect = FancyBboxPatch(
            (b['x'], b['y']), b['w'], b['h'],
            boxstyle='round,pad=0.3,rounding_size=0.8',
            facecolor=b['color'], edgecolor='black', linewidth=0.6
        )
        ax.add_patch(rect)
        ax.text(b['x'] + b['w'] / 2, b['y'] + b['h'] / 2, b['label'],
                ha='center', va='center', fontsize=fs_box)

    # Inter-box arrows
    for x1, x2 in [(18, 20), (37, 39), (56, 58), (75, 77)]:
        ax.annotate('', xy=(x2, 30), xytext=(x1, 30),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.6))

    # GNN sub-label (under the GNN box)
    ax.text(66.5, 19.5, '7,369 nodes · 882,959 edges',
            ha='center', va='top', fontsize=fs_sub)

    # Bulk box (above the GNN box) + transfer arrow (no annotation text)
    bbox = FancyBboxPatch(
        (54, 47), 25, 9,
        boxstyle='round,pad=0.3,rounding_size=0.8',
        facecolor='white', edgecolor=COLORS['external_blue'],
        linewidth=0.6, linestyle='--'
    )
    ax.add_patch(bbox)
    ax.text(66.5, 51.5, 'Bulk SCP682', ha='center', va='center',
            fontsize=fs_box, color=COLORS['external_blue'])
    arrow = FancyArrowPatch(
        (66.5, 47), (66.5, 38),
        arrowstyle='->', color=COLORS['external_blue'],
        linewidth=0.8, linestyle='--', connectionstyle='arc3,rad=0.0'
    )
    ax.add_patch(arrow)

    # Bottom label
    ax.text(50, 9,
            '121,847 cells (icCITE + QuRIE) · 56 supervised readouts',
            ha='center', va='center', fontsize=fs_bottom)


# ---------- Combined layout helpers ----------
def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


# ---------- Main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_targets, ablation_per_target, qurie_per_target, stat3, ctnnd1 = load_data()

    df_b = panel_b_data(per_targets)
    df_c_left = panel_c_left_data(ctnnd1)
    df_c_right = panel_c_right_data(stat3)
    df_d, w_stat, w_p, n_pairs = panel_d_data(None, ablation_per_target)
    df_e = panel_e_data(qurie_per_target)

    print(f'[Panel B] cohorts: {list(df_b["display_cohort"])}')
    print(f'[Panel B] site_counts: {list(df_b["site_count"])}')
    print(f'[Panel B] medians: {list(df_b["median_spearman"].round(4))}')
    print(f'[Panel C left] CTNND1_T310 HeLa n_cells: {len(df_c_left)}')
    print(f'[Panel C right] STAT3_Y705 platforms: {list(df_c_right["display_platform"])}')
    print(f'[Panel D] groups: {list(df_d["group"])}')
    print(f'[Panel D] site_counts: {list(df_d["site_count"])}')
    print(f'[Panel D] deltas: {list(df_d["delta_no_graph_minus_graph"].round(4))}')
    print(f'[Panel D] Wilcoxon stat={w_stat:.4f}, p={w_p:.4e}, n_pairs={n_pairs}')
    print(f'[Panel E] time180 n_readouts: {len(df_e)}')

    # Individual SVGs
    fig, ax = plt.subplots(figsize=(2.4, 1.5))
    panel_a_plot(ax)
    fig.savefig(os.path.join(OUT_DIR, 'fig3_panel_a.svg'), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    panel_b_plot(df_b, ax)
    fig.savefig(os.path.join(OUT_DIR, 'fig3_panel_b.svg'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(2.6, 1.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.55)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    panel_c_plot_left(df_c_left, ax_l)
    panel_c_plot_right(df_c_right, ax_r)
    fig.savefig(os.path.join(OUT_DIR, 'fig3_panel_c.svg'), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    panel_d_plot(df_d, ax)
    fig.savefig(os.path.join(OUT_DIR, 'fig3_panel_d.svg'), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.8, 2.2))
    panel_e_plot(df_e, ax)
    fig.savefig(os.path.join(OUT_DIR, 'fig3_panel_e.svg'), bbox_inches='tight')
    plt.close(fig)

    # Combined: 180mm x 100mm = 7.087" x 3.937"
    fig = plt.figure(figsize=(7.087, 3.937))
    # 12-col grid: top a(0:5) wider for schematic, b(5:8), c(8:12)
    #              bottom d(0:6), e(6:12) equal
    gs = GridSpec(2, 12, figure=fig, wspace=1.4, hspace=0.95,
                  left=0.05, right=0.98, top=0.92, bottom=0.14)
    ax_a = fig.add_subplot(gs[0, 0:5])
    panel_a_plot(ax_a, compact=True)
    add_panel_label(ax_a, 'a', x=-0.02, y=1.08)

    ax_b = fig.add_subplot(gs[0, 5:8])
    panel_b_plot(df_b, ax_b)
    add_panel_label(ax_b, 'b', x=-0.30, y=1.08)

    gs_c = gs[0, 8:12].subgridspec(1, 2, wspace=0.6)
    ax_cl = fig.add_subplot(gs_c[0])
    ax_cr = fig.add_subplot(gs_c[1])
    panel_c_plot_left(df_c_left, ax_cl)
    panel_c_plot_right(df_c_right, ax_cr)
    add_panel_label(ax_cl, 'c', x=-0.30, y=1.08)

    ax_d = fig.add_subplot(gs[1, 0:6])
    panel_d_plot(df_d, ax_d)
    add_panel_label(ax_d, 'd', x=-0.07, y=1.08)

    ax_e = fig.add_subplot(gs[1, 6:12])
    panel_e_plot(df_e, ax_e)
    add_panel_label(ax_e, 'e', x=-0.08, y=1.08)

    fig.savefig(os.path.join(OUT_DIR, 'fig3_combined.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'fig3_combined_preview.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    print('All SVGs saved.')


if __name__ == '__main__':
    main()
