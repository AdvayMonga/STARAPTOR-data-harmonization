"""
LOO ComBat — Harmonization Performance Visualization
All figures use the Leave-One-Cohort-Out ComBat strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import rankdata

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

Path("results/figures").mkdir(parents=True, exist_ok=True)

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

# eGFR heatmap (rank-colored, values annotated)
ax = axes[0]
egfr_ranks = pd.DataFrame(
    rankdata(combat_egfr.values.flatten()).reshape(combat_egfr.shape),
    index=combat_egfr.index, columns=combat_egfr.columns
)
sns.heatmap(egfr_ranks, annot=combat_egfr, fmt='.0f', cmap='viridis_r',
            cbar=False, ax=ax, linewidths=0.5)
ax.set_title('LOO ComBat — eGFR Test MSE\n(Lower / Lighter = Better, colors by rank)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Held-Out Test Cohort', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=20)

# DGF heatmap
ax = axes[1]
sns.heatmap(combat_dgf, annot=True, fmt='.3f', cmap='viridis',
            cbar=False, ax=ax, linewidths=0.5)
ax.set_title('LOO ComBat — DGF Test AUC\n(Higher / Lighter = Better)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Held-Out Test Cohort', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
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
    ax.set_ylabel(f'Avg {metric} (across folds)', fontsize=11, fontweight='bold')
    ax.set_title(f'{metric}: Strategy Comparison per Model\n({better} = Better  |  avg across 3 folds)',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Strategy', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

plt.tight_layout()
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
    ax.set_title(f'LOO ComBat — {metric} per Fold\n({better} = Better)',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Held-Out Cohort', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

plt.tight_layout()
plt.savefig('results/figures/loo_fold_comparison.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_fold_comparison.png")

# ── 4. Strategy Comparison: Raw vs ComBat vs Harm→Raw ────────
print("\n" + "=" * 60)
print("GENERATING LOO STRATEGY COMPARISON")
print("=" * 60)

raw_egfr      = get_loo_table('LOO Raw: {}',      scenario_egfr_pivot)
harmtrain_egfr = get_loo_table('LOO Harm→Raw: {}', scenario_egfr_pivot)
raw_dgf        = get_loo_table('LOO Raw: {}',      scenario_dgf_pivot)
harmtrain_dgf  = get_loo_table('LOO Harm→Raw: {}', scenario_dgf_pivot)

if raw_egfr is not None and harmtrain_egfr is not None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 11))

    strategy_data_egfr = [
        ('Unharmonized', raw_egfr,      '#e74c3c'),
        ('LOO ComBat',   combat_egfr,   '#3498db'),
        ('Harm→Raw',     harmtrain_egfr,'#f39c12'),
    ]
    strategy_data_dgf = [
        ('Unharmonized', raw_dgf,       '#e74c3c'),
        ('LOO ComBat',   combat_dgf,    '#3498db'),
        ('Harm→Raw',     harmtrain_dgf, '#f39c12'),
    ]

    for ax, strat_data, metric, better, fmt, ylim, use_log in [
        (axes[0], strategy_data_egfr, 'eGFR Test MSE', 'Lower', '.0f', None,        True),
        (axes[1], strategy_data_dgf,  'DGF Test AUC',  'Higher', '.3f', (0, 1.05), False),
    ]:
        n_folds  = len(fold_labels)
        fold_x = np.arange(n_folds)
        bar_w  = 0.25

        for i, (name, df, color) in enumerate(strat_data):
            if df is None:
                continue
            avgs = df.mean(axis=0).values  # avg across models per fold
            plot_vals = np.log10(avgs) if use_log else avgs
            bars = ax.bar(fold_x + i * bar_w, plot_vals, bar_w,
                          label=name, color=color, edgecolor='black', linewidth=0.5)
            for bar, raw, pv in zip(bars, avgs, plot_vals):
                ax.annotate(f'{raw:{fmt}}',
                            xy=(bar.get_x() + bar.get_width() / 2, pv),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', fontsize=8, fontweight='bold')

        ax.set_xticks(fold_x + bar_w)
        ax.set_xticklabels(fold_labels, fontsize=10)
        if use_log:
            ax.set_ylabel(f'Avg {metric} (log10)', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel(f'Avg {metric} (across models)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}: Unharmonized vs LOO ComBat vs Harm→Raw\n({better} = Better)',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Strategy', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig('results/figures/loo_strategy_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: loo_strategy_comparison.png")
else:
    print("⚠ Raw/Harm→Raw results not found — run 03_train_models.py first")

# ── 5. % Improvement Over Baselines ──────────────────────────
print("\n" + "=" * 60)
print("GENERATING IMPROVEMENT VS BASELINES")
print("=" * 60)

if raw_egfr is not None and harmtrain_egfr is not None:
    # eGFR: (baseline - combat) / baseline * 100  (lower MSE = better → positive = ComBat wins)
    # DGF:  (combat - baseline) / |baseline| * 100 (higher AUC = better → positive = ComBat wins)
    def pct_improve_egfr(baseline_df):
        return ((baseline_df - combat_egfr) / baseline_df * 100).mean(axis=0)

    def pct_improve_dgf(baseline_df):
        return ((combat_dgf - baseline_df) / baseline_df.abs() * 100).mean(axis=0)

    impr_data = [
        ('vs Unharmonized', pct_improve_egfr(raw_egfr),       pct_improve_dgf(raw_dgf),       '#e74c3c'),
        ('vs Harm→Raw',     pct_improve_egfr(harmtrain_egfr), pct_improve_dgf(harmtrain_dgf), '#f39c12'),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    for ax, metric, getter_idx, better in [
        (axes[0], 'eGFR % MSE Reduction vs Baseline', 0, 'Positive = ComBat has lower MSE'),
        (axes[1], 'DGF % AUC Gain vs Baseline',       1, 'Positive = ComBat has higher AUC'),
    ]:
        n = len(impr_data)
        fold_x = np.arange(len(fold_labels))
        bar_w = 0.35

        for i, (label, egfr_vals, dgf_vals, color) in enumerate(impr_data):
            vals = egfr_vals.values if getter_idx == 0 else dgf_vals.values
            bars = ax.bar(fold_x + i * bar_w, vals, bar_w,
                          label=label, color=color, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.annotate(f'{val:+.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() if val >= 0 else 0),
                            xytext=(0, 4 if val >= 0 else -14),
                            textcoords='offset points',
                            ha='center', fontsize=9, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1.0)
        ax.set_xticks(fold_x + bar_w / 2)
        ax.set_xticklabels(fold_labels, fontsize=10)
        ax.set_ylabel('% Improvement', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}\n({better}  |  averaged across models per fold)',
                     fontsize=12, fontweight='bold')
        ax.legend(title='Baseline', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/loo_improvement_vs_baselines.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: loo_improvement_vs_baselines.png")
else:
    print("⚠ Baseline results not found — skipping improvement chart")

# ── 6. Per-Model Improvement Heatmap ─────────────────────────
print("\n" + "=" * 60)
print("GENERATING PER-MODEL IMPROVEMENT HEATMAP")
print("=" * 60)

methods_egfr = pd.read_csv('results/tables/archived/egfr_all_methods_summary.csv', index_col='Model')
methods_dgf  = pd.read_csv('results/tables/archived/dgf_all_methods_summary.csv', index_col='Model')

harm_methods = ['Z-Score', 'RAVEL', 'CORAL', 'CovBat', 'ComBat']

egfr_diff = pd.DataFrame(index=methods_egfr.index, columns=harm_methods)
dgf_diff  = pd.DataFrame(index=methods_dgf.index,  columns=harm_methods)
for method in harm_methods:
    egfr_diff[method] = methods_egfr['Unharmonized'] - methods_egfr[method]
    dgf_diff[method]  = methods_dgf[method] - methods_dgf['Unharmonized']
egfr_diff = egfr_diff.astype(float)
dgf_diff  = dgf_diff.astype(float)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, diff_df, title, label, fmt_str in [
    (axes[0], egfr_diff, 'eGFR: MSE Reduction vs Unharmonized', 'MSE Reduction', '+.0f'),
    (axes[1], dgf_diff,  'DGF: AUC Gain vs Unharmonized',       'AUC Gain',      '+.3f'),
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
    ax.set_title(f'{title}\nGreen = better  |  Red = worse',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Harmonization Method', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('results/figures/loo_improvement_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_improvement_heatmap.png")

# ── 7. CKD Stage Distribution ────────────────────────────────
if has_ckd:
    print("\n" + "=" * 60)
    print("GENERATING CKD STAGE DISTRIBUTION")
    print("=" * 60)

    ckd_order = ckd_distribution['CKD_Stage'].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, actual_col, pred_col, title in [
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
        ax.set_title(f'{title}: CKD Stage Distribution\nXGBoost — LOO ComBat (CM → U)',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/ckd_stage_distribution.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: ckd_stage_distribution.png")

# ── 8. Permutation Importance ────────────────────────────────
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
        ax.set_title(f'{outcome}: Top 20 Features\nXGBoost — LOO ComBat (CM → U)',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()
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
print("  - loo_strategy_comparison.png")
print("  - loo_improvement_vs_baselines.png")
print("  - loo_improvement_heatmap.png")
if has_ckd:        print("  - ckd_stage_distribution.png")
if has_importance: print("  - permutation_importance.png")
print("=" * 60)
