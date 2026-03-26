"""
Harmonization Performance Visualization
Renders all visualizations using pre-computed tables from 04_process_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
Path("results/figures").mkdir(parents=True, exist_ok=True)

# LOAD PRE-COMPUTED TABLES
print("="*60)
print("LOADING PRE-COMPUTED TABLES")
print("="*60)

# Summary pivot tables
scenario_egfr_pivot = pd.read_csv('results/tables/egfr_scenario_summary.csv', index_col=0)
scenario_dgf_pivot = pd.read_csv('results/tables/dgf_scenario_summary.csv', index_col=0)
method_egfr_pivot = pd.read_csv('results/tables/egfr_method_summary.csv', index_col=0)
method_dgf_pivot = pd.read_csv('results/tables/dgf_method_summary.csv', index_col=0)
print("✓ Loaded summary pivot tables")

# Average performance tables
scenario_egfr_avg = pd.read_csv('results/tables/egfr_scenario_avg.csv', index_col=0).squeeze()
scenario_dgf_avg = pd.read_csv('results/tables/dgf_scenario_avg.csv', index_col=0).squeeze()
method_egfr_avg = pd.read_csv('results/tables/egfr_method_avg.csv', index_col=0).squeeze()
method_dgf_avg = pd.read_csv('results/tables/dgf_method_avg.csv', index_col=0).squeeze()
print("✓ Loaded average performance tables")

# Improvement tables
try:
    egfr_improve = pd.read_csv('results/tables/egfr_improvement_avg.csv', index_col=0).squeeze()
    dgf_improve = pd.read_csv('results/tables/dgf_improvement_avg.csv', index_col=0).squeeze()
    egfr_model_improve = pd.read_csv('results/tables/egfr_improvement_by_model.csv', index_col=0)
    dgf_model_improve = pd.read_csv('results/tables/dgf_improvement_by_model.csv', index_col=0)
    print("✓ Loaded improvement tables")
    has_improvement_data = True
except FileNotFoundError:
    print("⚠ Improvement tables not found")
    has_improvement_data = False

# CKD stage distribution
try:
    ckd_distribution = pd.read_csv('results/tables/ckd_stage_distribution.csv')
    print("✓ Loaded CKD stage distribution")
    has_ckd_data = True
except FileNotFoundError:
    print("⚠ CKD stage distribution not found")
    has_ckd_data = False

# Permutation importance
try:
    egfr_importance = pd.read_csv('results/tables/egfr_permutation_importance.csv')
    dgf_importance = pd.read_csv('results/tables/dgf_permutation_importance.csv')
    print("✓ Loaded permutation importance tables")
    has_importance_data = True
except FileNotFoundError:
    print("⚠ Permutation importance tables not found")
    has_importance_data = False

# SCENARIO HEATMAPS
print("\n" + "="*60)
print("GENERATING SCENARIO HEATMAPS")
print("="*60)

from scipy.stats import rankdata

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# eGFR Heatmap 
ax = axes[0]
# Create rank-based data for coloring (lower MSE = lower rank = better)
egfr_ranks = pd.DataFrame(
    rankdata(scenario_egfr_pivot.values.flatten()).reshape(scenario_egfr_pivot.shape),
    index=scenario_egfr_pivot.index,
    columns=scenario_egfr_pivot.columns
)
sns.heatmap(egfr_ranks, annot=scenario_egfr_pivot, fmt='.0f', cmap='viridis_r',
            cbar=False, ax=ax, linewidths=0.5)
ax.set_title('eGFR Test MSE by Model and Scenario\n(Lower/Lighter = Better, colors by rank)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

# DGF Heatmap (viridis palette)
ax = axes[1]
sns.heatmap(scenario_dgf_pivot, annot=True, fmt='.3f', cmap='viridis',
            cbar=False, ax=ax, linewidths=0.5)
ax.set_title('DGF Test AUC by Model and Scenario\n(Higher/Lighter = Better)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/scenario_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: scenario_heatmap.png")

# METHOD HEATMAPS
print("\n" + "="*60)
print("GENERATING METHOD HEATMAPS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# eGFR Heatmap 
ax = axes[0]
# Create rank-based data for coloring (lower MSE = lower rank = better)
egfr_ranks = pd.DataFrame(
    rankdata(method_egfr_pivot.values.flatten()).reshape(method_egfr_pivot.shape),
    index=method_egfr_pivot.index,
    columns=method_egfr_pivot.columns
)
sns.heatmap(egfr_ranks, annot=method_egfr_pivot, fmt='.0f', cmap='viridis_r',
            cbar=False, ax=ax, linewidths=0.5)
ax.set_title('eGFR Test MSE by Model and Harmonization Method\n(Lower/Lighter = Better, colors by rank)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Harmonization Method', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

# DGF Heatmap (viridis palette)
ax = axes[1]
sns.heatmap(method_dgf_pivot, annot=True, fmt='.3f', cmap='viridis',
            cbar=False, ax=ax, linewidths=0.5)
ax.set_title('DGF Test AUC by Model and Harmonization Method\n(Higher/Lighter = Better)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Harmonization Method', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/method_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: method_heatmap.png")

# AVERAGE PERFORMANCE BY SCENARIO
print("\n" + "="*60)
print("GENERATING AVERAGE PERFORMANCE BY SCENARIO")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors: highlight best
colors_egfr = ['#2ecc71' if v == scenario_egfr_avg.min() else '#3498db' for v in scenario_egfr_avg]
colors_dgf = ['#2ecc71' if v == scenario_dgf_avg.max() else '#3498db' for v in scenario_dgf_avg]

# eGFR Average MSE (sqrt scale)
ax = axes[0]
bars = ax.bar(scenario_egfr_avg.index, scenario_egfr_avg.values, color=colors_egfr, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Test MSE (sqrt scale)', fontsize=12, fontweight='bold')
ax.set_title('eGFR: Average Test MSE by Scenario\n(Lower = Better, sqrt scale)', fontsize=14, fontweight='bold')
ax.set_yscale('function', functions=(lambda x: np.sqrt(x), lambda x: x**2))
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, scenario_egfr_avg.values):
    ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# DGF Average AUC
ax = axes[1]
bars = ax.bar(scenario_dgf_avg.index, scenario_dgf_avg.values, color=colors_dgf, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Test AUC', fontsize=12, fontweight='bold')
ax.set_title('DGF: Average Test AUC by Scenario\n(Higher = Better)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, scenario_dgf_avg.values):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/average_performance_by_scenario.png', bbox_inches='tight')
plt.close()
print("✓ Saved: average_performance_by_scenario.png")

# AVERAGE PERFORMANCE BY METHOD
print("\n" + "="*60)
print("GENERATING AVERAGE PERFORMANCE BY METHOD")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors: highlight best
colors_egfr = ['#2ecc71' if v == method_egfr_avg.min() else '#3498db' for v in method_egfr_avg]
colors_dgf = ['#2ecc71' if v == method_dgf_avg.max() else '#3498db' for v in method_dgf_avg]

# eGFR Average MSE
ax = axes[0]
bars = ax.bar(method_egfr_avg.index, method_egfr_avg.values, color=colors_egfr, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Harmonization Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Test MSE', fontsize=12, fontweight='bold')
ax.set_title('eGFR: Average Test MSE by Method\n(Lower = Better)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
if 'Unharmonized' in method_egfr_avg.index:
    ax.axhline(y=method_egfr_avg['Unharmonized'], color='red', linestyle='--', linewidth=2,
               label=f'Unharmonized Baseline ({method_egfr_avg["Unharmonized"]:.0f})')
    ax.legend(fontsize=10)
for bar, val in zip(bars, method_egfr_avg.values):
    ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# DGF Average AUC
ax = axes[1]
bars = ax.bar(method_dgf_avg.index, method_dgf_avg.values, color=colors_dgf, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Harmonization Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Test AUC', fontsize=12, fontweight='bold')
ax.set_title('DGF: Average Test AUC by Method\n(Higher = Better)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 1.0)
if 'Unharmonized' in method_dgf_avg.index:
    ax.axhline(y=method_dgf_avg['Unharmonized'], color='red', linestyle='--', linewidth=2,
               label=f'Unharmonized Baseline ({method_dgf_avg["Unharmonized"]:.3f})')
    ax.legend(fontsize=10)
for bar, val in zip(bars, method_dgf_avg.values):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/average_performance_by_method.png', bbox_inches='tight')
plt.close()
print("✓ Saved: average_performance_by_method.png")

# IMPROVEMENT OVER UNHARMONIZED (METHODS)
if has_improvement_data:
    print("\n" + "="*60)
    print("GENERATING IMPROVEMENT OVER UNHARMONIZED")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # eGFR Improvement
    ax = axes[0]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in egfr_improve]
    bars = ax.bar(egfr_improve.index, egfr_improve.values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Harmonization Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Improvement in Test MSE', fontsize=12, fontweight='bold')
    ax.set_title('eGFR: % Improvement Over Unharmonized\n(Positive = Better)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, egfr_improve.values):
        ax.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if val >= 0 else -12), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')

    # DGF Improvement
    ax = axes[1]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in dgf_improve]
    bars = ax.bar(dgf_improve.index, dgf_improve.values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Harmonization Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Improvement in Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('DGF: % Improvement Over Unharmonized\n(Positive = Better)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, dgf_improve.values):
        ax.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if val >= 0 else -12), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/improvement_over_unharmonized.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: improvement_over_unharmonized.png")

    # IMPROVEMENT BY MODEL (Grouped Bar Chart)
    print("\n" + "="*60)
    print("GENERATING IMPROVEMENT BY MODEL")
    print("="*60)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # eGFR Improvement by Model
    ax = axes[0]
    x = np.arange(len(egfr_model_improve.index))
    width = 0.15
    methods = egfr_model_improve.columns.tolist()
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    for i, method in enumerate(methods):
        vals = egfr_model_improve[method].values
        bars = ax.bar(x + i*width, vals, width, label=method, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Improvement in Test MSE', fontsize=12, fontweight='bold')
    ax.set_title('eGFR: % Improvement Over Unharmonized by Model\n(Positive = Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(methods)-1) / 2)
    ax.set_xticklabels(egfr_model_improve.index, fontsize=10)
    ax.legend(title='Harmonization Method', fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # DGF Improvement by Model
    ax = axes[1]
    for i, method in enumerate(methods):
        vals = dgf_model_improve[method].values
        bars = ax.bar(x + i*width, vals, width, label=method, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Improvement in Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('DGF: % Improvement Over Unharmonized by Model\n(Positive = Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(methods)-1) / 2)
    ax.set_xticklabels(dgf_model_improve.index, fontsize=10)
    ax.legend(title='Harmonization Method', fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/improvement_by_model.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: improvement_by_model.png")
else:
    print("⚠ Skipping improvement charts - data not available")

# CKD STAGE DISTRIBUTION
if has_ckd_data:
    print("\n" + "="*60)
    print("GENERATING CKD STAGE DISTRIBUTION")
    print("="*60)

    ckd_order = ckd_distribution['CKD_Stage'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training set
    ax = axes[0]
    x = np.arange(len(ckd_order))
    width = 0.35
    ax.bar(x - width/2, ckd_distribution['Train_Actual'].values, width, label='Actual', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, ckd_distribution['Train_Predicted'].values, width, label='Predicted', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('CKD Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Training Set: CKD Stage Distribution\n(XGBoost on ComBat Harmonized Data)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ckd_order, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Test set
    ax = axes[1]
    ax.bar(x - width/2, ckd_distribution['Test_Actual'].values, width, label='Actual', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, ckd_distribution['Test_Predicted'].values, width, label='Predicted', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('CKD Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Test Set: CKD Stage Distribution\n(XGBoost on ComBat Harmonized Data)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ckd_order, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/ckd_stage_distribution.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: ckd_stage_distribution.png")
else:
    print("⚠ Skipping CKD stage distribution - data not available")

# PERMUTATION IMPORTANCE
if has_importance_data:
    print("\n" + "="*60)
    print("GENERATING PERMUTATION IMPORTANCE PLOTS")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # eGFR Permutation Importance
    ax = axes[0]
    top_egfr = egfr_importance.head(20)
    bars = ax.barh(range(len(top_egfr)), top_egfr['Importance'].values, color='#3498db', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_egfr)))
    ax.set_yticklabels(top_egfr['Feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance (L1 Normalized)', fontsize=12, fontweight='bold')
    ax.set_title('eGFR: Top 20 Features\n(Permutation Importance, XGBoost)', fontsize=14, fontweight='bold')

    # DGF Permutation Importance
    ax = axes[1]
    top_dgf = dgf_importance.head(20)
    bars = ax.barh(range(len(top_dgf)), top_dgf['Importance'].values, color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_dgf)))
    ax.set_yticklabels(top_dgf['Feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance (L1 Normalized)', fontsize=12, fontweight='bold')
    ax.set_title('DGF: Top 20 Features\n(Permutation Importance, XGBoost)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/permutation_importance.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: permutation_importance.png")
else:
    print("⚠ Skipping permutation importance - data not available")

# LOO COMBAT VISUALIZATIONS
print("\n" + "="*60)
print("GENERATING LOO COMBAT VISUALIZATIONS")
print("="*60)

LOO_KEYS  = ['LOO: UC → M', 'LOO: UM → C', 'LOO: CM → U']
LOO_LABELS = {
    'LOO: UC → M': 'UC Davis + Coimbra\n→ Mayo',
    'LOO: UM → C': 'UC Davis + Mayo\n→ Coimbra',
    'LOO: CM → U': 'Coimbra + Mayo\n→ UC Davis',
}

loo_egfr_available = [k for k in LOO_KEYS if k in scenario_egfr_pivot.columns]
loo_dgf_available  = [k for k in LOO_KEYS if k in scenario_dgf_pivot.columns]

if loo_egfr_available and loo_dgf_available:

    # ── LOO Heatmap: performance per model per fold ───────────────
    loo_egfr_pivot = scenario_egfr_pivot[loo_egfr_available].copy()
    loo_dgf_pivot  = scenario_dgf_pivot[loo_dgf_available].copy()
    loo_egfr_pivot.columns = [LOO_LABELS[k] for k in loo_egfr_available]
    loo_dgf_pivot.columns  = [LOO_LABELS[k] for k in loo_dgf_available]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    loo_egfr_ranks = pd.DataFrame(
        rankdata(loo_egfr_pivot.values.flatten()).reshape(loo_egfr_pivot.shape),
        index=loo_egfr_pivot.index, columns=loo_egfr_pivot.columns
    )
    sns.heatmap(loo_egfr_ranks, annot=loo_egfr_pivot, fmt='.0f', cmap='viridis_r',
                cbar=False, ax=ax, linewidths=0.5)
    ax.set_title('LOO ComBat — eGFR Test MSE\n(Lower/Lighter = Better, colors by rank)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Cohort (held out)', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=20)

    ax = axes[1]
    sns.heatmap(loo_dgf_pivot, annot=True, fmt='.3f', cmap='viridis',
                cbar=False, ax=ax, linewidths=0.5)
    ax.set_title('LOO ComBat — DGF Test AUC\n(Higher/Lighter = Better)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Cohort (held out)', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig('results/figures/loo_heatmap.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: loo_heatmap.png")

    # ── LOO vs existing scenarios: average performance comparison ─
    loo_egfr_avg = scenario_egfr_avg[loo_egfr_available]
    loo_dgf_avg  = scenario_dgf_avg[loo_dgf_available]

    # All scenarios split into LOO and non-LOO for grouped comparison
    non_loo_keys  = [k for k in scenario_egfr_avg.index if k not in LOO_KEYS]
    non_loo_egfr  = scenario_egfr_avg[non_loo_keys]
    non_loo_dgf   = scenario_dgf_avg[non_loo_keys]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, avg_non_loo, avg_loo, metric, better, fmt in [
        (axes[0], non_loo_egfr, loo_egfr_avg, 'eGFR Test MSE', 'Lower', '.0f'),
        (axes[1], non_loo_dgf,  loo_dgf_avg,  'DGF Test AUC',  'Higher', '.3f'),
    ]:
        x_non = np.arange(len(avg_non_loo))
        x_loo = np.arange(len(avg_non_loo), len(avg_non_loo) + len(avg_loo))

        bars1 = ax.bar(x_non, avg_non_loo.values, color='#3498db',
                       edgecolor='black', linewidth=0.5, label='Existing scenarios')
        bars2 = ax.bar(x_loo, avg_loo.values, color='#e67e22',
                       edgecolor='black', linewidth=0.5, label='LOO ComBat (new)')

        all_vals  = np.concatenate([avg_non_loo.values, avg_loo.values])
        all_bars  = list(bars1) + list(bars2)
        all_labels = list(avg_non_loo.index) + [LOO_LABELS[k] for k in loo_egfr_available]

        for bar, val in zip(all_bars, all_vals):
            ax.annotate(f'{val:{fmt}}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks(np.concatenate([x_non, x_loo]))
        ax.set_xticklabels(all_labels, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel(f'Average {metric}', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}: All Scenarios\n({better} = Better)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # vertical separator between existing and LOO
        ax.axvline(x=len(avg_non_loo) - 0.5, color='gray', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.savefig('results/figures/loo_vs_existing_scenarios.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: loo_vs_existing_scenarios.png")

    # ── LOO fold consistency: how stable is performance across folds ─
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models = loo_egfr_pivot.index.tolist()
    x = np.arange(len(models))
    width = 0.25
    fold_colors = ['#2ecc71', '#3498db', '#9b59b6']

    for ax, pivot, metric, better, fmt in [
        (axes[0], loo_egfr_pivot, 'eGFR Test MSE', 'Lower', '.0f'),
        (axes[1], loo_dgf_pivot,  'DGF Test AUC',  'Higher', '.3f'),
    ]:
        for i, fold in enumerate(pivot.columns):
            ax.bar(x + i * width, pivot[fold].values, width,
                   label=fold, color=fold_colors[i], edgecolor='black', linewidth=0.5)

        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'LOO ComBat — {metric} per Fold\n({better} = Better)',
                     fontsize=12, fontweight='bold')
        ax.legend(title='Held-out Cohort', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/loo_fold_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ Saved: loo_fold_comparison.png")

else:
    print("⚠ LOO results not found in scenario tables — run 03_train_models.py first")

# SUMMARY
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\n--- Figures saved to results/figures/ ---")
print("  - scenario_heatmap.png")
print("  - method_heatmap.png")
print("  - average_performance_by_scenario.png")
print("  - average_performance_by_method.png")
if has_improvement_data:
    print("  - improvement_over_unharmonized.png")
    print("  - improvement_by_model.png")
if has_ckd_data:
    print("  - ckd_stage_distribution.png")
if has_importance_data:
    print("  - permutation_importance.png")
print("\n" + "="*60)
