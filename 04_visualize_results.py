"""
Harmonization Performance Visualization
Compares all harmonization methods across all models and outcomes
Generates two sets of visualizations:
  1. SCENARIOS: Cross-cohort comparisons (U→C, C→U, Pooled variants)
  2. METHODS: Harmonization method comparisons (all 6 methods on pooled data)
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

# ========== DEFINE COMPARISON GROUPS ==========

# SCENARIOS: Cross-cohort training strategies
SCENARIOS = {
    'U → C': 'U_to_C',
    'C → U': 'C_to_U',
    'Pooled Unharmonized': 'unharmonized',
    'Pooled ComBat': 'harmonized',
    'Pooled CovBat': 'covbat',
}

# METHODS: All harmonization methods for pooled cohort comparison
METHODS = {
    'Unharmonized': 'unharmonized',
    'ComBat': 'harmonized',
    'CovBat': 'covbat',
    'Z-Score': 'zscore',
    'RAVEL': 'ravel',
    'CORAL': 'coral',
}

# ========== LOAD ALL RESULTS ==========
print("="*60)
print("LOADING RESULTS")
print("="*60)

def load_results(comparison_dict, outcome):
    """Load results for a given comparison (scenarios or methods)"""
    results = {}
    for name, key in comparison_dict.items():
        try:
            df = pd.read_csv(f'results/{outcome}_results_{key}.csv', index_col=0)
            df['Group'] = name
            results[name] = df
            print(f"  ✓ Loaded {outcome.upper()} {name}")
        except FileNotFoundError:
            print(f"  ✗ Missing {outcome.upper()} {name}")
    return results

# Load SCENARIOS results
print("\n--- Loading Scenario Results ---")
scenario_egfr = load_results(SCENARIOS, 'egfr')
scenario_dgf = load_results(SCENARIOS, 'dgf')

# Load METHODS results  
print("\n--- Loading Method Results ---")
method_egfr = load_results(METHODS, 'egfr')
method_dgf = load_results(METHODS, 'dgf')

# Combine into DataFrames
df_scenario_egfr = pd.concat(scenario_egfr.values()).reset_index().rename(columns={'index': 'Model'})
df_scenario_dgf = pd.concat(scenario_dgf.values()).reset_index().rename(columns={'index': 'Model'})
df_method_egfr = pd.concat(method_egfr.values()).reset_index().rename(columns={'index': 'Model'})
df_method_dgf = pd.concat(method_dgf.values()).reset_index().rename(columns={'index': 'Model'})

# ========== 1. SUMMARY TABLES ==========
print("\n" + "="*60)
print("SUMMARY TABLES")
print("="*60)

# SCENARIOS Tables
print("\n--- SCENARIO COMPARISON ---")
scenario_egfr_pivot = df_scenario_egfr.pivot(index='Model', columns='Group', values='Test MSE')
scenario_egfr_pivot = scenario_egfr_pivot[[k for k in SCENARIOS.keys() if k in scenario_egfr_pivot.columns]]
print("\neGFR Test MSE by Scenario (Lower = Better):")
print(scenario_egfr_pivot.round(1).to_string())

scenario_dgf_pivot = df_scenario_dgf.pivot(index='Model', columns='Group', values='Test AUC')
scenario_dgf_pivot = scenario_dgf_pivot[[k for k in SCENARIOS.keys() if k in scenario_dgf_pivot.columns]]
print("\nDGF Test AUC by Scenario (Higher = Better):")
print(scenario_dgf_pivot.round(3).to_string())

# METHODS Tables
print("\n--- METHOD COMPARISON ---")
method_egfr_pivot = df_method_egfr.pivot(index='Model', columns='Group', values='Test MSE')
method_egfr_pivot = method_egfr_pivot[[k for k in METHODS.keys() if k in method_egfr_pivot.columns]]
print("\neGFR Test MSE by Method (Lower = Better):")
print(method_egfr_pivot.round(1).to_string())

method_dgf_pivot = df_method_dgf.pivot(index='Model', columns='Group', values='Test AUC')
method_dgf_pivot = method_dgf_pivot[[k for k in METHODS.keys() if k in method_dgf_pivot.columns]]
print("\nDGF Test AUC by Method (Higher = Better):")
print(method_dgf_pivot.round(3).to_string())

# Save tables
scenario_egfr_pivot.to_csv('results/egfr_scenario_summary.csv')
scenario_dgf_pivot.to_csv('results/dgf_scenario_summary.csv')
method_egfr_pivot.to_csv('results/egfr_method_summary.csv')
method_dgf_pivot.to_csv('results/dgf_method_summary.csv')
print("\n✓ Saved summary tables to results/")

# ========== 2. SCENARIO HEATMAPS ==========
print("\n" + "="*60)
print("GENERATING SCENARIO HEATMAPS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# eGFR Heatmap
ax = axes[0]
sns.heatmap(scenario_egfr_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Test MSE'}, ax=ax, linewidths=0.5)
ax.set_title('eGFR Test MSE by Model and Scenario\n(Lower = Better)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

# DGF Heatmap
ax = axes[1]
sns.heatmap(scenario_dgf_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'Test AUC'}, ax=ax, linewidths=0.5)
ax.set_title('DGF Test AUC by Model and Scenario\n(Higher = Better)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/scenario_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: scenario_heatmap.png")

# ========== 3. METHOD HEATMAPS ==========
print("\n" + "="*60)
print("GENERATING METHOD HEATMAPS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# eGFR Heatmap
ax = axes[0]
sns.heatmap(method_egfr_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Test MSE'}, ax=ax, linewidths=0.5)
ax.set_title('eGFR Test MSE by Model and Harmonization Method\n(Lower = Better)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Harmonization Method', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

# DGF Heatmap
ax = axes[1]
sns.heatmap(method_dgf_pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'Test AUC'}, ax=ax, linewidths=0.5)
ax.set_title('DGF Test AUC by Model and Harmonization Method\n(Higher = Better)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Harmonization Method', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/method_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: method_heatmap.png")

# ========== 4. AVERAGE PERFORMANCE BY SCENARIO ==========
print("\n" + "="*60)
print("GENERATING AVERAGE PERFORMANCE BY SCENARIO")
print("="*60)

# Calculate averages across all models for SCENARIOS
scenario_egfr_avg = df_scenario_egfr.groupby('Group')['Test MSE'].mean()
scenario_egfr_avg = scenario_egfr_avg.reindex([k for k in SCENARIOS.keys() if k in scenario_egfr_avg.index])

scenario_dgf_avg = df_scenario_dgf.groupby('Group')['Test AUC'].mean()
scenario_dgf_avg = scenario_dgf_avg.reindex([k for k in SCENARIOS.keys() if k in scenario_dgf_avg.index])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors: highlight best
colors_egfr = ['#2ecc71' if v == scenario_egfr_avg.min() else '#3498db' for v in scenario_egfr_avg]
colors_dgf = ['#2ecc71' if v == scenario_dgf_avg.max() else '#3498db' for v in scenario_dgf_avg]

# eGFR Average MSE
ax = axes[0]
bars = ax.bar(scenario_egfr_avg.index, scenario_egfr_avg.values, color=colors_egfr, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Test MSE', fontsize=12, fontweight='bold')
ax.set_title('eGFR: Average Test MSE by Scenario\n(Lower = Better)', fontsize=14, fontweight='bold')
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

# ========== 5. AVERAGE PERFORMANCE BY METHOD ==========
print("\n" + "="*60)
print("GENERATING AVERAGE PERFORMANCE BY METHOD")
print("="*60)

# Calculate averages across all models for METHODS
method_egfr_avg = df_method_egfr.groupby('Group')['Test MSE'].mean()
method_egfr_avg = method_egfr_avg.reindex([k for k in METHODS.keys() if k in method_egfr_avg.index])

method_dgf_avg = df_method_dgf.groupby('Group')['Test AUC'].mean()
method_dgf_avg = method_dgf_avg.reindex([k for k in METHODS.keys() if k in method_dgf_avg.index])

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

# ========== 6. IMPROVEMENT OVER UNHARMONIZED (METHODS) ==========
print("\n" + "="*60)
print("GENERATING IMPROVEMENT OVER UNHARMONIZED")
print("="*60)

if 'Unharmonized' in method_egfr_avg.index and 'Unharmonized' in method_dgf_avg.index:
    # Calculate % improvement over unharmonized
    egfr_improve = ((method_egfr_avg['Unharmonized'] - method_egfr_avg) / method_egfr_avg['Unharmonized']) * 100
    dgf_improve = ((method_dgf_avg - method_dgf_avg['Unharmonized']) / method_dgf_avg['Unharmonized']) * 100

    # Remove unharmonized from comparison
    egfr_improve = egfr_improve.drop('Unharmonized')
    dgf_improve = dgf_improve.drop('Unharmonized')

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
else:
    print("⚠ Skipping improvement chart - Unharmonized baseline not available")

# ========== 7. COMBINED RESULTS TABLES ==========
print("\n" + "="*60)
print("GENERATING COMBINED RESULTS TABLES")
print("="*60)

# Create comprehensive scenario results table
scenario_results = []
for group in df_scenario_egfr['Group'].unique():
    for model in df_scenario_egfr['Model'].unique():
        egfr_row = df_scenario_egfr[(df_scenario_egfr['Group'] == group) & (df_scenario_egfr['Model'] == model)]
        dgf_row = df_scenario_dgf[(df_scenario_dgf['Group'] == group) & (df_scenario_dgf['Model'] == model)]
        
        if len(egfr_row) > 0 and len(dgf_row) > 0:
            scenario_results.append({
                'Scenario': group,
                'Model': model,
                'eGFR Train MSE': egfr_row['Train MSE'].values[0],
                'eGFR Test MSE': egfr_row['Test MSE'].values[0],
                'DGF Train AUC': dgf_row['Train AUC'].values[0],
                'DGF Test AUC': dgf_row['Test AUC'].values[0],
            })

df_scenario_combined = pd.DataFrame(scenario_results)
df_scenario_combined.to_csv('results/all_scenario_results.csv', index=False)
print("✓ Saved: all_scenario_results.csv")

# Create comprehensive method results table
method_results = []
for group in df_method_egfr['Group'].unique():
    for model in df_method_egfr['Model'].unique():
        egfr_row = df_method_egfr[(df_method_egfr['Group'] == group) & (df_method_egfr['Model'] == model)]
        dgf_row = df_method_dgf[(df_method_dgf['Group'] == group) & (df_method_dgf['Model'] == model)]
        
        if len(egfr_row) > 0 and len(dgf_row) > 0:
            method_results.append({
                'Method': group,
                'Model': model,
                'eGFR Train MSE': egfr_row['Train MSE'].values[0],
                'eGFR Test MSE': egfr_row['Test MSE'].values[0],
                'DGF Train AUC': dgf_row['Train AUC'].values[0],
                'DGF Test AUC': dgf_row['Test AUC'].values[0],
            })

df_method_combined = pd.DataFrame(method_results)
df_method_combined.to_csv('results/all_method_results.csv', index=False)
print("✓ Saved: all_method_results.csv")

# ========== SUMMARY ==========
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\n--- SCENARIO COMPARISON ---")
if len(scenario_egfr_avg) > 0:
    best_scenario_egfr = scenario_egfr_avg.idxmin()
    print(f"Best Scenario for eGFR (lowest avg Test MSE): {best_scenario_egfr} ({scenario_egfr_avg[best_scenario_egfr]:.1f})")
if len(scenario_dgf_avg) > 0:
    best_scenario_dgf = scenario_dgf_avg.idxmax()
    print(f"Best Scenario for DGF (highest avg Test AUC): {best_scenario_dgf} ({scenario_dgf_avg[best_scenario_dgf]:.3f})")

print("\n--- METHOD COMPARISON ---")
if len(method_egfr_avg) > 0:
    best_method_egfr = method_egfr_avg.idxmin()
    print(f"Best Method for eGFR (lowest avg Test MSE): {best_method_egfr} ({method_egfr_avg[best_method_egfr]:.1f})")
if len(method_dgf_avg) > 0:
    best_method_dgf = method_dgf_avg.idxmax()
    print(f"Best Method for DGF (highest avg Test AUC): {best_method_dgf} ({method_dgf_avg[best_method_dgf]:.3f})")

print("\n--- Figures saved to results/figures/ ---")
print("  - scenario_heatmap.png")
print("  - method_heatmap.png")
print("  - average_performance_by_scenario.png")
print("  - average_performance_by_method.png")
print("  - improvement_over_unharmonized.png")

print("\n--- Tables saved to results/ ---")
print("  - egfr_scenario_summary.csv")
print("  - dgf_scenario_summary.csv")
print("  - egfr_method_summary.csv")
print("  - dgf_method_summary.csv")
print("  - all_scenario_results.csv")
print("  - all_method_results.csv")

print("\n" + "="*60)