"""
LOO ComBat — Harmonization Performance Visualization
All figures use the Leave-One-Cohort-Out ComBat strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpl_path_effects
import seaborn as sns
from pathlib import Path
from scipy.stats import rankdata

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

Path("results/figures").mkdir(parents=True, exist_ok=True)


def add_caption(fig, text, fontsize=9, bottom=None):
    """Render a multi-line italic caption beneath a figure. Reserves space
    via subplots_adjust so it doesn't collide with axis tick labels."""
    n_lines = text.count('\n') + 1
    if bottom is None:
        # Roughly: room for x tick labels (~0.10) + caption lines.
        bottom = 0.18 + 0.035 * n_lines
    fig.subplots_adjust(bottom=bottom)
    caption_y = 0.015
    fig.text(0.5, caption_y, text, ha='center', va='bottom',
             fontsize=fontsize, style='italic', color='#333333',
             wrap=True, linespacing=1.35)

# ── Load tables ──────────────────────────────────────────────
print("=" * 60)
print("LOADING PRE-COMPUTED TABLES")
print("=" * 60)

scenario_egfr_pivot = pd.read_csv('results/tables/egfr_scenario_summary.csv', index_col=0)
scenario_dgf_pivot  = pd.read_csv('results/tables/dgf_scenario_summary.csv',  index_col=0)
print("✓ Loaded scenario summary tables")

try:
    ckd_distribution = pd.read_csv('results/tables/ckd_stage_distribution.csv')
    has_ckd = True
    print("✓ Loaded CKD stage distribution")
except FileNotFoundError:
    has_ckd = False
    print("⚠ CKD stage distribution not found")

try:
    egfr_importance = pd.read_csv('results/tables/egfr_permutation_importance.csv')
    dgf_importance  = pd.read_csv('results/tables/dgf_permutation_importance.csv')
    has_importance  = True
    print("✓ Loaded permutation importance tables")
except FileNotFoundError:
    has_importance = False
    print("⚠ Permutation importance tables not found")

# ── LOO key groups ────────────────────────────────────────────
FOLDS = ['UC → M', 'UM → C', 'CM → U']
FOLD_LABELS = {
    'UC → M': 'UC Davis + Coimbra\n→ Mayo',
    'UM → C': 'UC Davis + Mayo\n→ Coimbra',
    'CM → U': 'Coimbra + Mayo\n→ UC Davis',
}
STRATEGIES = {
    'Raw':       'LOO Raw: {}',
    'ComBat':    'LOO ComBat: {}',
    'Harm→Raw':  'LOO Harm→Raw: {}',
}
STRATEGY_COLORS = {'Raw': '#e74c3c', 'ComBat': '#3498db', 'Harm→Raw': '#f39c12'}

# Build per-fold, per-strategy sub-tables
def get_loo_table(strategy_fmt, outcome_pivot):
    keys = [strategy_fmt.format(f) for f in
            [f'UC → M', f'UM → C', f'CM → U']]
    available = [k for k in keys if k in outcome_pivot.columns]
    if not available:
        return None
    df = outcome_pivot[available].copy()
    df.columns = [FOLD_LABELS[k.split(': ', 1)[1]] for k in available]
    return df

combat_egfr = get_loo_table('LOO ComBat: {}', scenario_egfr_pivot)
combat_dgf  = get_loo_table('LOO ComBat: {}', scenario_dgf_pivot)

if combat_egfr is None:
    print("⚠ No LOO ComBat results found — run 03_train_models.py first")
    raise SystemExit

fold_labels = list(combat_egfr.columns)
print(f"✓ LOO data: {len(combat_egfr)} models × {len(fold_labels)} folds")

# ── 1. LOO Heatmaps ──────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING LOO HEATMAPS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# eGFR heatmap — absolute MSE colored, green = low. Clipped so Ridge tail
# doesn't compress contrast between the interesting cells.
ax = axes[0]
egfr_best = float(combat_egfr.values.min())
egfr_vmax = max(750.0, egfr_best * 2.0)
sns.heatmap(combat_egfr.astype(float), annot=True, fmt='.0f', cmap='RdYlGn_r',
            vmin=egfr_best, vmax=egfr_vmax,
            cbar_kws={'label': 'Test MSE (lower = better)'},
            ax=ax, linewidths=0.5)
ax.set_title('eGFR Test MSE', fontsize=13, fontweight='bold')
ax.set_xlabel('Held-Out Test Cohort', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=20)

# DGF heatmap — AUC colored, green = high.
ax = axes[1]
sns.heatmap(combat_dgf.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.45, vmax=float(combat_dgf.values.max()),
            cbar_kws={'label': 'Test AUC (higher = better)'},
            ax=ax, linewidths=0.5)
ax.set_title('DGF Test AUC', fontsize=13, fontweight='bold')
ax.set_xlabel('Held-Out Test Cohort', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=20)

fig.suptitle('LOO ComBat Performance Across Held-Out Cohorts',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
add_caption(fig,
    "Each cell is one (model, held-out cohort) result under leave-one-cohort-out ComBat harmonization.\n"
    f"Left panel: eGFR Test MSE (lower = better), color-scale clipped at MSE = {egfr_vmax:.0f} so the linear-model tail does not\n"
    "compress contrast in the low-MSE region. Right panel: DGF Test AUC (higher = better), color-scale starts at 0.45 ≈ chance.\n"
    "Cohorts: UC Davis (U), Coimbra (C), Mayo (M).")
plt.savefig('results/figures/loo_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_heatmap.png")

# ── 2. LOO Average Performance — all 3 strategies, grouped by model ──
print("\n" + "=" * 60)
print("GENERATING LOO AVERAGE PERFORMANCE")
print("=" * 60)

raw_egfr       = get_loo_table('LOO Raw: {}',      scenario_egfr_pivot)
harmtrain_egfr = get_loo_table('LOO Harm→Raw: {}', scenario_egfr_pivot)
raw_dgf        = get_loo_table('LOO Raw: {}',      scenario_dgf_pivot)
harmtrain_dgf  = get_loo_table('LOO Harm→Raw: {}', scenario_dgf_pivot)

strat_avgs_egfr = {
    'Unharmonized': raw_egfr.mean(axis=1)       if raw_egfr       is not None else None,
    'LOO ComBat':   combat_egfr.mean(axis=1),
}
strat_avgs_dgf = {
    'Unharmonized': raw_dgf.mean(axis=1)        if raw_dgf        is not None else None,
    'LOO ComBat':   combat_dgf.mean(axis=1),
}
strat_colors = {'Unharmonized': '#e74c3c', 'LOO ComBat': '#3498db'}

models = combat_egfr.index.tolist()
x = np.arange(len(models))
n_strats = sum(v is not None for v in strat_avgs_egfr.values())
bar_w = 0.3

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, avgs, metric, better, fmt, ylim, use_log in [
    (axes[0], strat_avgs_egfr, 'eGFR Test MSE', 'Lower', '.0f', None,        False),
    (axes[1], strat_avgs_dgf,  'DGF Test AUC',  'Higher', '.3f', (0, 1.05), False),
]:
    i = 0
    for name, series in avgs.items():
        if series is None:
            continue
        plot_vals = np.log10(series.values) if use_log else series.values
        bars = ax.bar(x + i * bar_w, plot_vals, bar_w,
                      label=name, color=strat_colors[name],
                      edgecolor='black', linewidth=0.5)
        for bar, raw, pv in zip(bars, series.values, plot_vals):
            ax.annotate(f'{raw:{fmt}}',
                        xy=(bar.get_x() + bar.get_width() / 2, pv),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold')
        i += 1
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel(f'Avg {metric}', fontsize=11, fontweight='bold')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.legend(title='Strategy', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

fig.suptitle('Average Performance: Unharmonized vs LOO ComBat',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
add_caption(fig,
    "Bars show each model's average Test MSE (eGFR, left) or Test AUC (DGF, right) across the 3 LOO scenarios\n"
    "(UC→M, UM→C, CM→U). Red bars = Unharmonized (LOO Raw) baseline. Blue bars = LOO ComBat harmonization.\n"
    "Lower bars are better for eGFR (MSE); higher bars are better for DGF (AUC). Values printed above each bar.")
plt.savefig('results/figures/loo_avg_performance.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_avg_performance.png")

# ── 3. LOO Fold Comparison (grouped bar per model, one bar per fold) ──
print("\n" + "=" * 60)
print("GENERATING LOO FOLD COMPARISON")
print("=" * 60)

fold_colors = ['#2ecc71', '#3498db', '#9b59b6']
models = combat_egfr.index.tolist()
x = np.arange(len(models))
width = 0.25

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

for ax, pivot, metric, better, fmt, ylim, use_log in [
    (axes[0], combat_egfr, 'eGFR Test MSE', 'Lower', '.0f', None,        True),
    (axes[1], combat_dgf,  'DGF Test AUC',  'Higher', '.3f', (0, 1.05), False),
]:
    for i, fold in enumerate(pivot.columns):
        vals = np.log10(pivot[fold].values) if use_log else pivot[fold].values
        ax.bar(x + i * width, vals, width,
               label=fold, color=fold_colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
    if use_log:
        ax.set_ylabel(f'{metric} (log10)', fontsize=11, fontweight='bold')
    else:
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.legend(title='Held-Out Cohort', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

fig.suptitle('LOO ComBat — Per-Fold Performance',
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
add_caption(fig,
    "Bars within each model group show that model's Test MSE / AUC for the three LOO held-out cohorts: UC→M, UM→C, CM→U.\n"
    "Top: eGFR Test MSE (log10 axis so Ridge's tail does not compress the rest). Bottom: DGF Test AUC on a linear axis.\n"
    "All numbers are LOO ComBat (harmonized train, harmonized test). Lower = better for eGFR, higher = better for DGF.",
    fontsize=8.5)
plt.savefig('results/figures/loo_fold_comparison.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_fold_comparison.png")

# ── 4. Per-Model Improvement Heatmap (LOO methods average) ──
print("\n" + "=" * 60)
print("GENERATING PER-MODEL IMPROVEMENT HEATMAP")
print("=" * 60)

# LOO-averaged method summary (mean across UC→M, UM→C, CM→U).
methods_egfr = pd.read_csv('results/tables/egfr_loo_method_avg.csv', index_col=0)
methods_dgf  = pd.read_csv('results/tables/dgf_loo_method_avg.csv',  index_col=0)

# RAVEL is excluded — no LOO implementation in repo.
harm_methods = [m for m in ['Z-Score', 'CORAL', 'CovBat', 'ComBat'] if m in methods_egfr.columns]

egfr_diff = pd.DataFrame(index=methods_egfr.index, columns=harm_methods, dtype=float)
dgf_diff  = pd.DataFrame(index=methods_dgf.index,  columns=harm_methods, dtype=float)
for method in harm_methods:
    egfr_diff[method] = methods_egfr['Unharmonized'] - methods_egfr[method]
    dgf_diff[method]  = methods_dgf[method] - methods_dgf['Unharmonized']
egfr_diff = egfr_diff.astype(float)
dgf_diff  = dgf_diff.astype(float)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, diff_df, title, label, fmt_str in [
    (axes[0], egfr_diff, 'eGFR MSE Reduction', 'MSE Reduction', '+.0f'),
    (axes[1], dgf_diff,  'DGF AUC Gain',       'AUC Gain',      '+.3f'),
]:
    lim = np.nanmax(np.abs(diff_df.values))
    sns.heatmap(
        diff_df,
        annot=diff_df.map(lambda v: f'{v:{fmt_str}}'),
        fmt='',
        cmap='RdYlGn',
        center=0,
        vmin=-lim, vmax=lim,
        cbar_kws={'label': label},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Harmonization Method', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=20)

fig.suptitle('Harmonization Effect Across Methods',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
add_caption(fig,
    "Each cell averages results across the 3 LOO scenarios (UC→M, UM→C, CM→U).\n"
    "Left: eGFR MSE reduction (Unharmonized − Method); positive (green) means harmonization lowered MSE.\n"
    "Right: DGF AUC gain (Method − Unharmonized); positive (green) means harmonization raised AUC.\n"
    "RAVEL is excluded — no LOO implementation in the repo. Symmetric ±max color scaling per panel.")
plt.savefig('results/figures/loo_improvement_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_improvement_heatmap.png")

# ── 4a. eGFR Methods Heatmap — Two Baseline Variants ────────
# Apply Variant A (annotated per-model baseline) and Variant B (skill vs best
# Unharmonized) to the same models × methods data used above so Ridge no longer
# looks like the biggest winner just because its Raw baseline is inflated.
unharm  = methods_egfr['Unharmonized']
best_un = unharm.min()
best_un_model = unharm.idxmin()
method_only_cols = [c for c in methods_egfr.columns if c != 'Unharmonized']
method_only = methods_egfr[method_only_cols]

reduction_methods_abs = methods_egfr[method_only_cols].rsub(unharm, axis=0)
reduction_methods     = reduction_methods_abs.div(unharm, axis=0) * 100  # % reduction
skill_methods         = (best_un - method_only) / best_un * 100

def _sym_lim(df):
    v = np.nanmax(np.abs(df.values))
    return -v, v

# Variant A — color by absolute final MSE (green=low, red=high) so the actually
# best (model, method) cells dominate visually. Cell text shows both the final
# MSE (color-determining) and the per-model % reduction for context.
fig, ax = plt.subplots(figsize=(9, 5))
annot_a = pd.DataFrame(index=method_only.index, columns=method_only.columns, dtype=object)
for m in method_only.index:
    for c in method_only.columns:
        annot_a.loc[m, c] = f'MSE={method_only.loc[m, c]:.0f}\n({reduction_methods.loc[m, c]:+.1f}%)'
row_labels = [f"{m}\n(Unharm avg = {unharm[m]:.0f})" for m in method_only.index]
# Clip the upper end so Ridge's 900 doesn't drown out the tree-model contrast.
best_mse = float(method_only.values.min())
vmax_cap = max(750.0, best_mse * 2.0)
sns.heatmap(method_only.astype(float), annot=annot_a, fmt='', cmap='RdYlGn_r',
            vmin=best_mse, vmax=vmax_cap,
            cbar_kws={'label': 'Avg Test MSE (lower = better)'},
            linewidths=0.5, ax=ax, yticklabels=row_labels)
ax.set_title('eGFR — Final MSE with % Improvement',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Harmonization Method', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
add_caption(fig,
    "Cells colored by absolute final Test MSE (green = low, red = high), with color scale clipped at\n"
    f"MSE = {vmax_cap:.0f} so Ridge/Lasso outliers do not compress contrast among the better cells.\n"
    "Each cell labels the final method MSE and the % reduction vs. that model's own Unharmonized baseline.\n"
    "Row labels show each model's avg Unharmonized MSE for reference. Averaged across 3 LOO scenarios.")
plt.savefig('results/figures/loo_improvement_heatmap_methods_avg_current.png',
            bbox_inches='tight')
plt.close()
print("✓ Saved: loo_improvement_heatmap_methods_avg_current.png")

# Variant B — skill score vs best Unharmonized model (single shared baseline).
# Clip the negative end so the positive cells (the actual harmonization wins)
# don't get washed out by Ridge's deep negative tail.
fig, ax = plt.subplots(figsize=(9, 5))
annot_b = pd.DataFrame(index=skill_methods.index, columns=skill_methods.columns, dtype=object)
for m in skill_methods.index:
    for c in skill_methods.columns:
        annot_b.loc[m, c] = f'{skill_methods.loc[m, c]:+.1f}%\n(MSE={method_only.loc[m, c]:.0f})'
vmin_b, vmax_b = -60.0, 60.0
sns.heatmap(skill_methods, annot=annot_b, fmt='', cmap='RdYlGn',
            center=0, vmin=vmin_b, vmax=vmax_b,
            cbar_kws={'label': 'Skill Score (%) — clipped at ±60%'},
            linewidths=0.5, ax=ax)
ax.set_title('eGFR — Skill Score vs Best Unharmonized',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Harmonization Method', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
add_caption(fig,
    f"Single shared baseline = best Unharmonized model = {best_un_model} ({best_un:.0f} MSE).\n"
    "Skill score = (best Unharmonized MSE − Method MSE) / best Unharmonized MSE × 100.\n"
    "Positive (green) means this (model, method) cell beats the strongest unharmonized baseline. Cell text shows\n"
    "skill score and final method MSE. Color scale clipped at ±60 %. Averaged across 3 LOO scenarios.")
plt.savefig('results/figures/loo_improvement_heatmap_methods_avg_skill_vs_best_raw.png',
            bbox_inches='tight')
plt.close()
print("✓ Saved: loo_improvement_heatmap_methods_avg_skill_vs_best_raw.png")

# Variant C — absolute final MSE (green = lower). Mirrors the DGF panel's
# "winners pop" effect because tree-based models + ComBat genuinely have the
# lowest MSE across the grid; Ridge/Lasso rows are uniformly high so they
# correctly read as worse without the baseline-inflation artifact.
abs_mse = methods_egfr[['Unharmonized'] + method_only_cols]
# Cap upper end of color scale so the Ridge tail doesn't compress the
# interesting low-MSE contrast. Cells above the cap still read as deep red.
best_mse  = float(abs_mse.values.min())
vmax_cap  = max(750.0, best_mse * 2.0)
fig, ax = plt.subplots(figsize=(10, 5))
annot_c = abs_mse.map(lambda v: f'{v:.0f}')
sns.heatmap(abs_mse.astype(float), annot=annot_c, fmt='', cmap='RdYlGn_r',
            vmin=best_mse, vmax=vmax_cap,
            cbar_kws={'label': 'Avg Test MSE (3 LOO scenarios)'},
            linewidths=0.5, ax=ax)
ax.set_title('eGFR — Final Test MSE by Model × Method',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Harmonization Method', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
add_caption(fig,
    "Absolute final Test MSE for each (model, method) combination, including the Unharmonized baseline column.\n"
    f"Color: green = low MSE (better), red = high. Scale clipped at MSE = {vmax_cap:.0f} so Ridge/Lasso outliers do not\n"
    "compress contrast among the better cells. Mirrors the DGF AUC heatmap style: the actually-best (model, method)\n"
    "combinations dominate visually. Averaged across 3 LOO scenarios.")
plt.savefig('results/figures/loo_improvement_heatmap_methods_avg_absolute_mse.png',
            bbox_inches='tight')
plt.close()
print("✓ Saved: loo_improvement_heatmap_methods_avg_absolute_mse.png")

# Variant D — slope plot (parallel coordinates). One line per model across
# methods in order [Unharm → Z-Score → CORAL → CovBat → ComBat]. Y-axis = final
# Test MSE. Vertical position shows totals; line slope shows improvement.
method_order = ['Unharmonized', 'Z-Score', 'CORAL', 'CovBat', 'ComBat']
slope_data = methods_egfr[method_order]
fig, ax = plt.subplots(figsize=(10, 6))
model_styles = {
    # Linear family — warm reds/oranges
    'Lasso':         {'color': '#d62728', 'marker': 's', 'family': 'Linear'},
    'Ridge':         {'color': '#b34700', 'marker': 'D', 'family': 'Linear'},
    'Elastic Net':   {'color': '#ff7f0e', 'marker': '^', 'family': 'Linear'},
    # Tree family — cool blues/teals
    'Random Forest': {'color': '#17becf', 'marker': 'o', 'family': 'Tree'},
    'XGBoost':       {'color': '#1f77b4', 'marker': 'o', 'family': 'Tree'},
}
x = np.arange(len(method_order))
for model in slope_data.index:
    style = model_styles.get(model, {'color': 'gray', 'marker': 'o'})
    y = slope_data.loc[model].values
    ax.plot(x, y, marker=style['marker'], color=style['color'],
            linewidth=2.2, markersize=9, label=model, alpha=0.9)
    # Label endpoints
    ax.annotate(f'{y[0]:.0f}', (x[0], y[0]), textcoords='offset points',
                xytext=(-12, 0), ha='right', fontsize=8, color=style['color'])
    ax.annotate(f'{y[-1]:.0f}', (x[-1], y[-1]), textcoords='offset points',
                xytext=(12, 0), ha='left', fontsize=8, color=style['color'], fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(method_order, fontsize=11)
ax.set_xlabel('Harmonization Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg Test MSE (log scale)', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_title('eGFR — Model Trajectories Across Methods',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', which='both', alpha=0.3)
ax.legend(title='Model', loc='upper right', fontsize=10, framealpha=0.95)
ax.margins(x=0.08)
plt.tight_layout()
add_caption(fig,
    "Each line traces one model's average Test MSE across the 5 method conditions, left to right.\n"
    "Warm colors = linear models (Lasso/Ridge/Elastic Net); cool colors = tree models (Random Forest/XGBoost).\n"
    "Y-axis is log-scaled so the Ridge tail does not crush the lower band. Lower y = better; slope = harmonization effect.\n"
    "Endpoint values labelled. Tree models reach their lowest MSE at ComBat. Averaged across 3 LOO scenarios.")
plt.savefig('results/figures/loo_improvement_slope_methods_avg.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_improvement_slope_methods_avg.png")

# Variant E — 3D bar chart. x=method, y=model, z=avg Test MSE.
# Ordering: tallest bars at back, shortest in front to minimize occlusion.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

# Methods in conventional left-to-right order; model order keeps XGBoost in
# front (so the smallest tree-model bars are closest to the viewer).
bar_method_order = ['Unharmonized', 'Z-Score', 'CORAL', 'CovBat', 'ComBat']
bar_model_order  = ['XGBoost', 'Random Forest', 'Elastic Net', 'Lasso', 'Ridge']
Z = methods_egfr.loc[bar_model_order, bar_method_order].values.astype(float)

fig = plt.figure(figsize=(13, 8))
ax3d = fig.add_subplot(111, projection='3d')

xs, ys = np.meshgrid(np.arange(len(bar_method_order)),
                     np.arange(len(bar_model_order)))
xpos = xs.ravel(); ypos = ys.ravel(); zpos = np.zeros_like(xpos, dtype=float)
dx = dy = 0.6
dz = Z.ravel()

colors = []
for m in bar_model_order:
    for _ in bar_method_order:
        colors.append(model_styles[m]['color'])

ax3d.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors,
           edgecolor='black', linewidth=0.4, alpha=0.92, shade=True)

# Value labels on top of each bar — bold black with white halo for legibility.
for x_, y_, z_ in zip(xpos, ypos, dz):
    txt = ax3d.text(x_ + dx/2, y_ + dy/2, z_ + 35, f'{z_:.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='black', zorder=20)
    txt.set_path_effects([
        mpl_path_effects.Stroke(linewidth=2.5, foreground='white'),
        mpl_path_effects.Normal()
    ])

ax3d.set_xticks(np.arange(len(bar_method_order)) + dx/2)
ax3d.set_xticklabels(bar_method_order, fontsize=10, rotation=15, ha='right')
ax3d.set_yticks(np.arange(len(bar_model_order)) + dy/2)
ax3d.set_yticklabels(bar_model_order, fontsize=10)
ax3d.set_zlabel('Avg Test MSE (lower = better)', fontsize=11, fontweight='bold')
ax3d.set_title('eGFR — Model × Method Performance (3D)',
               fontsize=14, fontweight='bold')
ax3d.view_init(elev=24, azim=-60)
plt.tight_layout()
add_caption(fig,
    "Bar height = avg Test MSE across 3 LOO scenarios (lower / shorter = better). Bars colored by model: warm = linear,\n"
    "cool = tree. Model axis ordered so XGBoost sits in front (smallest values closest to the viewer).\n"
    "Linear models (Ridge towering at the back) sit above tree models regardless of harmonization method.\n"
    "ComBat brings the tree models to their lowest point. Value labels rendered with white halo for legibility.")
plt.savefig('results/figures/loo_improvement_3d_methods_avg.png', bbox_inches='tight', dpi=200)
plt.close()
print("✓ Saved: loo_improvement_3d_methods_avg.png")

# ── 5. CKD Stage Distribution ────────────────────────────────
if has_ckd:
    print("\n" + "=" * 60)
    print("GENERATING CKD STAGE DISTRIBUTION")
    print("=" * 60)

    ckd_order = ckd_distribution['CKD_Stage'].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, actual_col, pred_col, panel in [
        (axes[0], 'Train_Actual', 'Train_Predicted', 'Training Set'),
        (axes[1], 'Test_Actual',  'Test_Predicted',  'Test Set'),
    ]:
        x = np.arange(len(ckd_order))
        w = 0.35
        ax.bar(x - w/2, ckd_distribution[actual_col].values, w,
               label='Actual', color='#2ecc71', edgecolor='black')
        ax.bar(x + w/2, ckd_distribution[pred_col].values, w,
               label='Predicted', color='#9b59b6', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(ckd_order, rotation=45, ha='right')
        ax.set_xlabel('CKD Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(panel, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('CKD Stage Distribution — Actual vs Predicted',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    add_caption(fig,
        "Predictions come from the XGBoost regressor under LOO ComBat with the held-out cohort = UC Davis (Coimbra + Mayo → UC Davis).\n"
        "eGFR predictions are bucketed into the standard CKD stages 1–5. Green bars = actual stage counts; purple = predicted.\n"
        "Left: training set (Coimbra + Mayo). Right: held-out test set (UC Davis).")
    plt.savefig('results/figures/ckd_stage_distribution.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: ckd_stage_distribution.png")

# ── 6. Permutation Importance ────────────────────────────────
if has_importance:
    print("\n" + "=" * 60)
    print("GENERATING PERMUTATION IMPORTANCE PLOTS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, df, color, outcome in [
        (axes[0], egfr_importance, '#3498db', 'eGFR'),
        (axes[1], dgf_importance,  '#e74c3c', 'DGF'),
    ]:
        top = df.head(20)
        ax.barh(range(len(top)), top['Importance'].values,
                color=color, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['Feature'].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance (L1 Normalized)', fontsize=12, fontweight='bold')
        ax.set_title(outcome, fontsize=13, fontweight='bold')

    fig.suptitle('Top 20 Permutation-Importance Features',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    add_caption(fig,
        "Permutation importance computed for the XGBoost model under LOO ComBat with held-out cohort = UC Davis (CM→U).\n"
        "Importances are L1-normalized so each panel sums to 1, then truncated to the top 20 features.\n"
        "Left: eGFR regressor (predicting 12-month eGFR). Right: DGF classifier (predicting delayed graft function).")
    plt.savefig('results/figures/permutation_importance.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: permutation_importance.png")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Figures saved to results/figures/")
print("=" * 60)
print("  - loo_heatmap.png")
print("  - loo_avg_performance.png")
print("  - loo_fold_comparison.png")
print("  - loo_improvement_heatmap.png")
print("  - loo_improvement_heatmap_methods_avg_current.png")
print("  - loo_improvement_heatmap_methods_avg_skill_vs_best_raw.png")
print("  - loo_improvement_heatmap_methods_avg_absolute_mse.png")
print("  - loo_improvement_slope_methods_avg.png")
print("  - loo_improvement_3d_methods_avg.png")
if has_ckd:        print("  - ckd_stage_distribution.png")
if has_importance: print("  - permutation_importance.png")
print("=" * 60)
