"""
Process Results for Visualization
Creates all data tables needed for visualization scripts (05_visualize.py and visualize_test.Rmd)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create output directory
Path("results/tables").mkdir(parents=True, exist_ok=True)

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
print("CREATING SUMMARY TABLES")
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

# Save pivot tables
scenario_egfr_pivot.to_csv('results/tables/egfr_scenario_summary.csv')
scenario_dgf_pivot.to_csv('results/tables/dgf_scenario_summary.csv')
method_egfr_pivot.to_csv('results/tables/egfr_method_summary.csv')
method_dgf_pivot.to_csv('results/tables/dgf_method_summary.csv')
print("\n✓ Saved summary pivot tables")

# AVERAGE PERFORMANCE TABLES
print("\n" + "="*60)
print("CREATING AVERAGE PERFORMANCE TABLES")
print("="*60)

# Calculate averages across all models for SCENARIOS
scenario_egfr_avg = df_scenario_egfr.groupby('Group')['Test MSE'].mean()
scenario_egfr_avg = scenario_egfr_avg.reindex([k for k in SCENARIOS.keys() if k in scenario_egfr_avg.index])

scenario_dgf_avg = df_scenario_dgf.groupby('Group')['Test AUC'].mean()
scenario_dgf_avg = scenario_dgf_avg.reindex([k for k in SCENARIOS.keys() if k in scenario_dgf_avg.index])

# Calculate averages across all models for METHODS
method_egfr_avg = df_method_egfr.groupby('Group')['Test MSE'].mean()
method_egfr_avg = method_egfr_avg.reindex([k for k in METHODS.keys() if k in method_egfr_avg.index])

method_dgf_avg = df_method_dgf.groupby('Group')['Test AUC'].mean()
method_dgf_avg = method_dgf_avg.reindex([k for k in METHODS.keys() if k in method_dgf_avg.index])

# Save averages
scenario_egfr_avg.to_csv('results/tables/egfr_scenario_avg.csv', header=['Avg Test MSE'])
scenario_dgf_avg.to_csv('results/tables/dgf_scenario_avg.csv', header=['Avg Test AUC'])
method_egfr_avg.to_csv('results/tables/egfr_method_avg.csv', header=['Avg Test MSE'])
method_dgf_avg.to_csv('results/tables/dgf_method_avg.csv', header=['Avg Test AUC'])
print("✓ Saved average performance tables")

# IMPROVEMENT TABLES
print("\n" + "="*60)
print("CREATING IMPROVEMENT TABLES")
print("="*60)

if 'Unharmonized' in method_egfr_avg.index and 'Unharmonized' in method_dgf_avg.index:
    # Calculate % improvement over unharmonized (averages)
    egfr_improve = ((method_egfr_avg['Unharmonized'] - method_egfr_avg) / method_egfr_avg['Unharmonized']) * 100
    dgf_improve = ((method_dgf_avg - method_dgf_avg['Unharmonized']) / method_dgf_avg['Unharmonized']) * 100

    # Remove unharmonized from comparison
    egfr_improve = egfr_improve.drop('Unharmonized')
    dgf_improve = dgf_improve.drop('Unharmonized')

    egfr_improve.to_csv('results/tables/egfr_improvement_avg.csv', header=['% Improvement'])
    dgf_improve.to_csv('results/tables/dgf_improvement_avg.csv', header=['% Improvement'])

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

    egfr_model_improve.to_csv('results/tables/egfr_improvement_by_model.csv')
    dgf_model_improve.to_csv('results/tables/dgf_improvement_by_model.csv')
    print("✓ Saved improvement tables")

    print("\neGFR % Improvement by Model:")
    print(egfr_model_improve.round(1).to_string())
    print("\nDGF % Improvement by Model:")
    print(dgf_model_improve.round(1).to_string())
else:
    print("⚠ Skipping improvement tables - Unharmonized baseline not available")

# COMBINED RESULTS TABLES
print("\n" + "="*60)
print("CREATING COMBINED RESULTS TABLES")
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

# CKD STAGE DISTRIBUTION AND PERMUTATION IMPORTANCE
print("\n" + "="*60)
print("CREATING CKD STAGE AND PERMUTATION IMPORTANCE DATA")
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

    # CKD STAGE DISTRIBUTION
    print("\nCreating CKD stage distribution data...")

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

    # Create CKD distribution dataframe
    train_true_counts = train_true_ckd.value_counts().reindex(ckd_order, fill_value=0)
    train_pred_counts = train_pred_ckd.value_counts().reindex(ckd_order, fill_value=0)
    test_true_counts = test_true_ckd.value_counts().reindex(ckd_order, fill_value=0)
    test_pred_counts = test_pred_ckd.value_counts().reindex(ckd_order, fill_value=0)

    ckd_distribution = pd.DataFrame({
        'CKD_Stage': ckd_order,
        'Train_Actual': train_true_counts.values,
        'Train_Predicted': train_pred_counts.values,
        'Test_Actual': test_true_counts.values,
        'Test_Predicted': test_pred_counts.values
    })
    ckd_distribution.to_csv('results/tables/ckd_stage_distribution.csv', index=False)
    print("✓ Saved: ckd_stage_distribution.csv")

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

    # PERMUTATION IMPORTANCE
    print("\nComputing permutation importance for eGFR...")
    perm_egfr = permutation_importance(
        xgb_egfr, X_test_scaled, y_test_egfr,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    print("Computing permutation importance for DGF...")
    perm_dgf = permutation_importance(
        xgb_dgf, X_test_scaled, y_test_dgf,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring='roc_auc'
    )

    # Process importance: zero out negatives, L1 normalize
    def process_importance(importance_means, feature_names):
        """Zero out negatives and L1 normalize"""
        importance = np.maximum(importance_means, 0)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        return df

    egfr_perm_importance = process_importance(perm_egfr.importances_mean, feature_cols)
    dgf_perm_importance = process_importance(perm_dgf.importances_mean, feature_cols)

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
    print(f"⚠ Could not generate CKD/permutation importance data: {e}")
    traceback.print_exc()

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

print("\n--- Tables saved to results/tables/ ---")
print("  Summary pivots:")
print("    - egfr_scenario_summary.csv")
print("    - dgf_scenario_summary.csv")
print("    - egfr_method_summary.csv")
print("    - dgf_method_summary.csv")
print("  Averages:")
print("    - egfr_scenario_avg.csv")
print("    - dgf_scenario_avg.csv")
print("    - egfr_method_avg.csv")
print("    - dgf_method_avg.csv")
print("  Improvements:")
print("    - egfr_improvement_avg.csv")
print("    - dgf_improvement_avg.csv")
print("    - egfr_improvement_by_model.csv")
print("    - dgf_improvement_by_model.csv")
print("  Combined results:")
print("    - all_scenario_results.csv")
print("    - all_method_results.csv")
print("  CKD and importance:")
print("    - ckd_stage_distribution.csv")
print("    - egfr_permutation_importance.csv")
print("    - dgf_permutation_importance.csv")

print("\n" + "="*60)
