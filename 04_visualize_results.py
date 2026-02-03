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

# LOAD ALL RESULTS
print("="*60)
print("LOADING RESULTS")
print("="*60)

def load_results(comparison_dict, outcome):
    """Load results for a given comparison (scenarios or methods)"""
    results = {}
    for name, key in comparison_dict.items():
        try:
            df = pd.read_csv(f'results/tables/{outcome}_results_{key}.csv', index_col=0)
            df['Group'] = name
            results[name] = df
            print(f"  ✓ Loaded {outcome.upper()} {name}")
        except FileNotFoundError:
            print(f"  ✗ Missing {outcome.upper()} {name}")
    return results

# Load SCENARIOS results
scenario_egfr = load_results(SCENARIOS, 'egfr')
scenario_dgf = load_results(SCENARIOS, 'dgf')

# Load METHODS results  
method_egfr = load_results(METHODS, 'egfr')
method_dgf = load_results(METHODS, 'dgf')

# Combine into DataFrames
df_scenario_egfr = pd.concat(scenario_egfr.values()).reset_index().rename(columns={'index': 'Model'})
df_scenario_dgf = pd.concat(scenario_dgf.values()).reset_index().rename(columns={'index': 'Model'})
df_method_egfr = pd.concat(method_egfr.values()).reset_index().rename(columns={'index': 'Model'})
df_method_dgf = pd.concat(method_dgf.values()).reset_index().rename(columns={'index': 'Model'})

# SUMMARY TABLES
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
scenario_egfr_pivot.to_csv('results/tables/egfr_scenario_summary.csv')
scenario_dgf_pivot.to_csv('results/tables/dgf_scenario_summary.csv')
method_egfr_pivot.to_csv('results/tables/egfr_method_summary.csv')
method_dgf_pivot.to_csv('results/tables/dgf_method_summary.csv')
print("\n✓ Saved summary tables to results/tables/")

# SCENARIO HEATMAPS
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

# METHOD HEATMAPS
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

# AVERAGE PERFORMANCE BY SCENARIO
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

# AVERAGE PERFORMANCE BY METHOD
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

# IMPROVEMENT OVER UNHARMONIZED (METHODS)
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
    
    # IMPROVEMENT BY MODEL (Grouped Bar Chart)
    print("\n" + "="*60)
    print("GENERATING IMPROVEMENT BY MODEL")
    print("="*60)
    
    # Calculate per-model improvement for eGFR
    egfr_model_improve = pd.DataFrame()
    for method in [m for m in METHODS.keys() if m != 'Unharmonized']:
        method_mse = method_egfr_pivot[method]
        unharm_mse = method_egfr_pivot['Unharmonized']
        improve = ((unharm_mse - method_mse) / unharm_mse) * 100
        egfr_model_improve[method] = improve
    
    # Calculate per-model improvement for DGF
    dgf_model_improve = pd.DataFrame()
    for method in [m for m in METHODS.keys() if m != 'Unharmonized']:
        method_auc = method_dgf_pivot[method]
        unharm_auc = method_dgf_pivot['Unharmonized']
        improve = ((method_auc - unharm_auc) / unharm_auc) * 100
        dgf_model_improve[method] = improve
    
    # Plot grouped bar charts
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
    
    # Print per-model improvements
    print("\neGFR % Improvement by Model:")
    print(egfr_model_improve.round(1).to_string())
    print("\nDGF % Improvement by Model:")
    print(dgf_model_improve.round(1).to_string())
else:
    print("⚠ Skipping improvement chart - Unharmonized baseline not available")

# COMBINED RESULTS TABLES
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
df_scenario_combined.to_csv('results/tables/all_scenario_results.csv', index=False)
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
df_method_combined.to_csv('results/tables/all_method_results.csv', index=False)
print("✓ Saved: all_method_results.csv")

# SUMMARY
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

print("\n--- Tables saved to results/tables/ ---")
print("  - egfr_scenario_summary.csv")
print("  - dgf_scenario_summary.csv")
print("  - egfr_method_summary.csv")
print("  - dgf_method_summary.csv")
print("  - all_scenario_results.csv")
print("  - all_method_results.csv")

# FEATURE IMPORTANCE (Best Models)
print("\n" + "="*60)
print("GENERATING FEATURE IMPORTANCE PLOTS")
print("="*60)

try:
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
    
    # Load ComBat harmonized training and test data (best performing)
    train_data = pd.read_csv('data/harmonized_combined/harmonized_train.csv')
    test_data = pd.read_csv('data/harmonized_combined/harmonized_test.csv')
    print(f"Loaded training data: {train_data.shape}")
    print(f"Loaded test data: {test_data.shape}")
    
    # Prepare features (exclude ID and outcomes)
    exclude_cols = ['Subject_ID', 'eGFR_12M', 'DGF', 'Site']
    feature_cols = [c for c in train_data.columns if c not in exclude_cols]
    
    X_train = train_data[feature_cols].copy()
    X_test = test_data[feature_cols].copy()
    y_train_egfr = train_data['eGFR_12M']
    y_test_egfr = test_data['eGFR_12M']
    y_train_dgf = train_data['DGF']
    y_test_dgf = test_data['DGF']
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost for eGFR (regression)
    print("Training XGBoost for eGFR...")
    xgb_egfr = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_egfr.fit(X_train_scaled, y_train_egfr)
    
    # Train XGBoost for DGF (classification)
    print("Training XGBoost for DGF...")
    xgb_dgf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_dgf.fit(X_train_scaled, y_train_dgf)
    
    # CKD STAGE DISTRIBUTION (eGFR)
    print("\n" + "="*60)
    print("GENERATING CKD STAGE DISTRIBUTION")
    print("="*60)
    
    # Get predictions
    y_train_pred = xgb_egfr.predict(X_train_scaled)
    y_test_pred = xgb_egfr.predict(X_test_scaled)
    
    # Define CKD stages based on eGFR
    def egfr_to_ckd_stage(egfr):
        """Map eGFR to CKD stage"""
        if egfr >= 90:
            return 'G1 (≥90)'
        elif egfr >= 60:
            return 'G2 (60-89)'
        elif egfr >= 45:
            return 'G3a (45-59)'
        elif egfr >= 30:
            return 'G3b (30-44)'
        elif egfr >= 15:
            return 'G4 (15-29)'
        else:
            return 'G5 (<15)'
    
    ckd_order = ['G1 (≥90)', 'G2 (60-89)', 'G3a (45-59)', 'G3b (30-44)', 'G4 (15-29)', 'G5 (<15)']
    
    # Map to CKD stages
    train_true_ckd = pd.Series([egfr_to_ckd_stage(e) for e in y_train_egfr])
    train_pred_ckd = pd.Series([egfr_to_ckd_stage(e) for e in y_train_pred])
    test_true_ckd = pd.Series([egfr_to_ckd_stage(e) for e in y_test_egfr])
    test_pred_ckd = pd.Series([egfr_to_ckd_stage(e) for e in y_test_pred])
    
    # Create CKD distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set
    ax = axes[0]
    train_true_counts = train_true_ckd.value_counts().reindex(ckd_order, fill_value=0)
    train_pred_counts = train_pred_ckd.value_counts().reindex(ckd_order, fill_value=0)
    
    x = np.arange(len(ckd_order))
    width = 0.35
    ax.bar(x - width/2, train_true_counts.values, width, label='Actual', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, train_pred_counts.values, width, label='Predicted', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('CKD Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Training Set: CKD Stage Distribution\n(XGBoost on ComBat Harmonized Data)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ckd_order, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Test set
    ax = axes[1]
    test_true_counts = test_true_ckd.value_counts().reindex(ckd_order, fill_value=0)
    test_pred_counts = test_pred_ckd.value_counts().reindex(ckd_order, fill_value=0)
    
    ax.bar(x - width/2, test_true_counts.values, width, label='Actual', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, test_pred_counts.values, width, label='Predicted', color='#e74c3c', edgecolor='black')
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
    
    # Print CKD stage comparison
    print("\nCKD Stage Distribution (Training):")
    for stage in ckd_order:
        true_n = train_true_counts.get(stage, 0)
        pred_n = train_pred_counts.get(stage, 0)
        print(f"  {stage}: Actual={true_n}, Predicted={pred_n}")
    
    print("\nCKD Stage Distribution (Test):")
    for stage in ckd_order:
        true_n = test_true_counts.get(stage, 0)
        pred_n = test_pred_counts.get(stage, 0)
        print(f"  {stage}: Actual={true_n}, Predicted={pred_n}")
    
    # CONDITIONAL PERMUTATION IMPORTANCE
    print("\n" + "="*60)
    print("GENERATING PERMUTATION IMPORTANCE PLOTS")
    print("="*60)
    
    # Compute permutation importance for eGFR (on test set)
    print("Computing permutation importance for eGFR...")
    perm_egfr = permutation_importance(
        xgb_egfr, X_test_scaled, y_test_egfr, 
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    # Compute permutation importance for DGF (on test set)
    print("Computing permutation importance for DGF...")
    perm_dgf = permutation_importance(
        xgb_dgf, X_test_scaled, y_test_dgf,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring='roc_auc'
    )
    
    # Process importance: zero out negatives, L1 normalize
    def process_importance(importance_means, feature_names):
        """Zero out negatives and L1 normalize"""
        # Zero out negative values
        importance = np.maximum(importance_means, 0)
        # L1 normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        return df
    
    egfr_perm_importance = process_importance(perm_egfr.importances_mean, feature_cols)
    dgf_perm_importance = process_importance(perm_dgf.importances_mean, feature_cols)
    
    # Plot top 20 features for each
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # eGFR Permutation Importance
    ax = axes[0]
    top_egfr = egfr_perm_importance.head(20)
    bars = ax.barh(range(len(top_egfr)), top_egfr['Importance'].values, color='#3498db', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_egfr)))
    ax.set_yticklabels(top_egfr['Feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance (L1 Normalized)', fontsize=12, fontweight='bold')
    ax.set_title('eGFR: Top 20 Features\n(Permutation Importance, XGBoost)', fontsize=14, fontweight='bold')
    
    # DGF Permutation Importance
    ax = axes[1]
    top_dgf = dgf_perm_importance.head(20)
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
    
    # Save permutation importance tables
    egfr_perm_importance.to_csv('results/tables/egfr_permutation_importance.csv', index=False)
    dgf_perm_importance.to_csv('results/tables/dgf_permutation_importance.csv', index=False)
    print("✓ Saved: egfr_permutation_importance.csv")
    print("✓ Saved: dgf_permutation_importance.csv")
    
    # Print top 10 features
    print("\nTop 10 Features for eGFR (Permutation Importance):")
    for _, row in egfr_perm_importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    print("\nTop 10 Features for DGF (Permutation Importance):")
    for _, row in dgf_perm_importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

except Exception as e:
    import traceback
    print(f"⚠ Could not generate feature importance plots: {e}")
    traceback.print_exc()

print("\n" + "="*60)