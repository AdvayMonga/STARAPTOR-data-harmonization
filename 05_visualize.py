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

# ── Filter to LOO columns ─────────────────────────────────────
LOO_KEYS   = ['LOO: UC → M', 'LOO: UM → C', 'LOO: CM → U']
LOO_LABELS = {
    'LOO: UC → M': 'UC Davis + Coimbra\n→ Mayo',
    'LOO: UM → C': 'UC Davis + Mayo\n→ Coimbra',
    'LOO: CM → U': 'Coimbra + Mayo\n→ UC Davis',
}

available = [k for k in LOO_KEYS if k in scenario_egfr_pivot.columns]
if not available:
    print("⚠ No LOO results found — run 03_train_models.py first")
    raise SystemExit

loo_egfr = scenario_egfr_pivot[available].copy()
loo_dgf  = scenario_dgf_pivot[available].copy()
loo_egfr.columns = [LOO_LABELS[k] for k in available]
loo_dgf.columns  = [LOO_LABELS[k] for k in available]
fold_labels = list(loo_egfr.columns)

print(f"✓ LOO data: {len(loo_egfr)} models × {len(fold_labels)} folds")

# ── 1. LOO Heatmaps ──────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING LOO HEATMAPS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# eGFR heatmap (rank-colored, values annotated)
ax = axes[0]
egfr_ranks = pd.DataFrame(
    rankdata(loo_egfr.values.flatten()).reshape(loo_egfr.shape),
    index=loo_egfr.index, columns=loo_egfr.columns
)
sns.heatmap(egfr_ranks, annot=loo_egfr, fmt='.0f', cmap='viridis_r',
            cbar=False, ax=ax, linewidths=0.5)
ax.set_title('LOO ComBat — eGFR Test MSE\n(Lower / Lighter = Better, colors by rank)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Held-Out Test Cohort', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=20)

# DGF heatmap
ax = axes[1]
sns.heatmap(loo_dgf, annot=True, fmt='.3f', cmap='viridis',
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

# ── 2. LOO Average Performance (per model, averaged across folds) ──
print("\n" + "=" * 60)
print("GENERATING LOO AVERAGE PERFORMANCE")
print("=" * 60)

egfr_avg = loo_egfr.mean(axis=1)   # avg MSE per model
dgf_avg  = loo_dgf.mean(axis=1)    # avg AUC per model

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

colors_egfr = ['#2ecc71' if v == egfr_avg.min() else '#3498db' for v in egfr_avg]
colors_dgf  = ['#2ecc71' if v == dgf_avg.max()  else '#3498db' for v in dgf_avg]

ax = axes[0]
bars = ax.bar(egfr_avg.index, egfr_avg.values, color=colors_egfr,
              edgecolor='black', linewidth=0.5)
ax.set_yscale('function', functions=(lambda x: np.sqrt(x), lambda x: x**2))
for bar, val in zip(bars, egfr_avg.values):
    ax.annotate(f'{val:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Test MSE (sqrt scale)', fontsize=12, fontweight='bold')
ax.set_title('eGFR: Average Test MSE Across LOO Folds\n(Lower = Better, sqrt scale  |  Green = best)',
             fontsize=13, fontweight='bold')

ax = axes[1]
bars = ax.bar(dgf_avg.index, dgf_avg.values, color=colors_dgf,
              edgecolor='black', linewidth=0.5)
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, dgf_avg.values):
    ax.annotate(f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Test AUC', fontsize=12, fontweight='bold')
ax.set_title('DGF: Average Test AUC Across LOO Folds\n(Higher = Better  |  Green = best)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/loo_avg_performance.png', bbox_inches='tight')
plt.close()
print("✓ Saved: loo_avg_performance.png")

# ── 3. LOO Fold Comparison (grouped bar per model, one bar per fold) ──
print("\n" + "=" * 60)
print("GENERATING LOO FOLD COMPARISON")
print("=" * 60)

fold_colors = ['#2ecc71', '#3498db', '#9b59b6']
models = loo_egfr.index.tolist()
x = np.arange(len(models))
width = 0.25

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

for ax, pivot, metric, better, fmt, ylim in [
    (axes[0], loo_egfr, 'eGFR Test MSE', 'Lower', '.0f', None),
    (axes[1], loo_dgf,  'DGF Test AUC',  'Higher', '.3f', (0, 1.05)),
]:
    for i, fold in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[fold].values, width,
               label=fold, color=fold_colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
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

# ── 4. CKD Stage Distribution ────────────────────────────────
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
               label='Actual', color='#3498db', edgecolor='black')
        ax.bar(x + w/2, ckd_distribution[pred_col].values, w,
               label='Predicted', color='#e74c3c', edgecolor='black')
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

# ── 5. Permutation Importance ────────────────────────────────
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
if has_ckd:        print("  - ckd_stage_distribution.png")
if has_importance: print("  - permutation_importance.png")
print("=" * 60)
