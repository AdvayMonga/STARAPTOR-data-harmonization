import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from mrmr import mrmr_classif, mrmr_regression
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("mRMR FEATURE SELECTION OPTIMIZATION")
print("="*60)

# Load harmonized combined data
df_harmonized = pd.read_csv('data/harmonized_combined/harmonized_all.csv')
df_harmonized = df_harmonized.drop(columns=['batch'])

# Split into train/test (same random state as before for consistency)
df_train, df_test = train_test_split(
    df_harmonized, test_size=0.2, random_state=42
)

# Prepare feature columns
feature_cols = [col for col in df_train.columns 
                if col not in ['Subject_ID', 'eGFR_12M', 'DGF']]

X_train = df_train[feature_cols]
y_train_egfr = df_train['eGFR_12M'].values
y_train_dgf = df_train['DGF'].values

X_test = df_test[feature_cols]
y_test_egfr = df_test['eGFR_12M'].values
y_test_dgf = df_test['DGF'].values

print(f"\nOriginal feature count: {len(feature_cols)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Run mRMR for both outcomes
print("\nRunning mRMR feature selection...")
print("  - For eGFR (regression)...")
selected_features_egfr = mrmr_regression(X_train, y_train_egfr, K=50)

print("  - For DGF (classification)...")
selected_features_dgf = mrmr_classif(X_train, y_train_dgf, K=50)

# Test different feature counts
feature_counts = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]

results_egfr = []
results_dgf = []

print("\n" + "="*60)
print("TESTING DIFFERENT FEATURE COUNTS")
print("="*60)

for k in feature_counts:
    print(f"\n>>> Testing with {k} features...")
    
    # eGFR MODELS
    top_k_egfr = selected_features_egfr[:k]
    X_train_egfr = X_train[top_k_egfr]
    X_test_egfr = X_test[top_k_egfr]
    
    scaler_egfr = StandardScaler()
    X_train_egfr_scaled = scaler_egfr.fit_transform(X_train_egfr)
    X_test_egfr_scaled = scaler_egfr.transform(X_test_egfr)
    
    # Elastic Net
    model_en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, 
                          random_state=42)
    model_en.fit(X_train_egfr_scaled, y_train_egfr)
    train_mse_en = mean_squared_error(y_train_egfr, 
                                      model_en.predict(X_train_egfr_scaled))
    test_mse_en = mean_squared_error(y_test_egfr, 
                                     model_en.predict(X_test_egfr_scaled))
    
    # Random Forest (with stronger regularization)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=3, 
                                     min_samples_leaf=10, max_features='sqrt',
                                     random_state=42)
    model_rf.fit(X_train_egfr, y_train_egfr)
    train_mse_rf = mean_squared_error(y_train_egfr, 
                                      model_rf.predict(X_train_egfr))
    test_mse_rf = mean_squared_error(y_test_egfr, 
                                     model_rf.predict(X_test_egfr))
    
    # XGBoost (with stronger regularization)
    model_xgb = XGBRegressor(n_estimators=200, max_depth=3, 
                            learning_rate=0.05, min_child_weight=5,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42)
    model_xgb.fit(X_train_egfr, y_train_egfr)
    train_mse_xgb = mean_squared_error(y_train_egfr, 
                                       model_xgb.predict(X_train_egfr))
    test_mse_xgb = mean_squared_error(y_test_egfr, 
                                      model_xgb.predict(X_test_egfr))
    
    # Average performance
    avg_train_mse = np.mean([train_mse_en, train_mse_rf, train_mse_xgb])
    avg_test_mse = np.mean([test_mse_en, test_mse_rf, test_mse_xgb])
    
    results_egfr.append({
        'K': k,
        'Elastic Net Train': train_mse_en,
        'Elastic Net Test': test_mse_en,
        'Random Forest Train': train_mse_rf,
        'Random Forest Test': test_mse_rf,
        'XGBoost Train': train_mse_xgb,
        'XGBoost Test': test_mse_xgb,
        'Avg Train MSE': avg_train_mse,
        'Avg Test MSE': avg_test_mse,
        'Overfitting Gap': test_mse_xgb - train_mse_xgb
    })
    
    print(f"  eGFR - Avg Test MSE: {avg_test_mse:.1f}")
    
    # DGF MODELS
    top_k_dgf = selected_features_dgf[:k]
    X_train_dgf = X_train[top_k_dgf]
    X_test_dgf = X_test[top_k_dgf]
    
    scaler_dgf = StandardScaler()
    X_train_dgf_scaled = scaler_dgf.fit_transform(X_train_dgf)
    X_test_dgf_scaled = scaler_dgf.transform(X_test_dgf)
    
    # Elastic Net Logistic Regression
    model_lr = LogisticRegression(penalty='elasticnet', solver='saga', 
                                  l1_ratio=0.5, C=1.0, max_iter=10000, 
                                  random_state=42)
    model_lr.fit(X_train_dgf_scaled, y_train_dgf)
    train_auc_lr = roc_auc_score(y_train_dgf, 
                                 model_lr.predict_proba(X_train_dgf_scaled)[:, 1])
    test_auc_lr = roc_auc_score(y_test_dgf, 
                                model_lr.predict_proba(X_test_dgf_scaled)[:, 1])
    
    # Random Forest Classifier (with stronger regularization)
    model_rf_clf = RandomForestClassifier(n_estimators=100, max_depth=3, 
                                         min_samples_leaf=10, max_features='sqrt',
                                         random_state=42)
    model_rf_clf.fit(X_train_dgf, y_train_dgf)
    train_auc_rf = roc_auc_score(y_train_dgf, 
                                 model_rf_clf.predict_proba(X_train_dgf)[:, 1])
    test_auc_rf = roc_auc_score(y_test_dgf, 
                                model_rf_clf.predict_proba(X_test_dgf)[:, 1])
    
    # XGBoost Classifier (with stronger regularization)
    model_xgb_clf = XGBClassifier(n_estimators=200, max_depth=3, 
                                 learning_rate=0.05, min_child_weight=5,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=0.1, reg_lambda=1.0,
                                 random_state=42)
    model_xgb_clf.fit(X_train_dgf, y_train_dgf)
    train_auc_xgb = roc_auc_score(y_train_dgf, 
                                  model_xgb_clf.predict_proba(X_train_dgf)[:, 1])
    test_auc_xgb = roc_auc_score(y_test_dgf, 
                                 model_xgb_clf.predict_proba(X_test_dgf)[:, 1])
    
    # Average performance
    avg_train_auc = np.mean([train_auc_lr, train_auc_rf, train_auc_xgb])
    avg_test_auc = np.mean([test_auc_lr, test_auc_rf, test_auc_xgb])
    
    results_dgf.append({
        'K': k,
        'Elastic Net Train': train_auc_lr,
        'Elastic Net Test': test_auc_lr,
        'Random Forest Train': train_auc_rf,
        'Random Forest Test': test_auc_rf,
        'XGBoost Train': train_auc_xgb,
        'XGBoost Test': test_auc_xgb,
        'Avg Train AUC': avg_train_auc,
        'Avg Test AUC': avg_test_auc,
        'Overfitting Gap': train_auc_xgb - test_auc_xgb
    })
    
    print(f"  DGF - Avg Test AUC: {avg_test_auc:.3f}")

# Convert to DataFrames
df_egfr_results = pd.DataFrame(results_egfr)
df_dgf_results = pd.DataFrame(results_dgf)

# Save results
df_egfr_results.to_csv('results/mrmr_egfr_optimization.csv', index=False)
df_dgf_results.to_csv('results/mrmr_dgf_optimization.csv', index=False)

print("\n" + "="*60)
print("OPTIMAL FEATURE COUNT ANALYSIS")
print("="*60)

# Find optimal K
optimal_egfr = df_egfr_results.loc[df_egfr_results['Avg Test MSE'].idxmin()]
optimal_dgf = df_dgf_results.loc[df_dgf_results['Avg Test AUC'].idxmax()]

print(f"\neGFR Optimal Features: {int(optimal_egfr['K'])}")
print(f"  Best Avg Test MSE: {optimal_egfr['Avg Test MSE']:.1f}")
print(f"  Overfitting Gap: {optimal_egfr['Overfitting Gap']:.1f}")

print(f"\nDGF Optimal Features: {int(optimal_dgf['K'])}")
print(f"  Best Avg Test AUC: {optimal_dgf['Avg Test AUC']:.3f}")
print(f"  Overfitting Gap: {optimal_dgf['Overfitting Gap']:.3f}")

# VISUALIZATIONS
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# eGFR: Test MSE vs K
ax = axes[0, 0]
ax.plot(df_egfr_results['K'], df_egfr_results['Avg Test MSE'], 
        'o-', linewidth=2, markersize=8, label='Test MSE')
ax.plot(df_egfr_results['K'], df_egfr_results['Avg Train MSE'], 
        's--', linewidth=2, markersize=6, alpha=0.6, label='Train MSE')
ax.axvline(optimal_egfr['K'], color='red', linestyle='--', alpha=0.5, 
           label=f'Optimal K={int(optimal_egfr["K"])}')
ax.set_xlabel('Number of Features (K)', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('eGFR: Performance vs Feature Count\n(Lower = Better)', 
             fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# DGF: Test AUC vs K
ax = axes[0, 1]
ax.plot(df_dgf_results['K'], df_dgf_results['Avg Test AUC'], 
        'o-', linewidth=2, markersize=8, label='Test AUC')
ax.plot(df_dgf_results['K'], df_dgf_results['Avg Train AUC'], 
        's--', linewidth=2, markersize=6, alpha=0.6, label='Train AUC')
ax.axvline(optimal_dgf['K'], color='red', linestyle='--', alpha=0.5, 
           label=f'Optimal K={int(optimal_dgf["K"])}')
ax.set_xlabel('Number of Features (K)', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('DGF: Performance vs Feature Count\n(Higher = Better)', 
             fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# eGFR: Overfitting Gap
ax = axes[1, 0]
ax.plot(df_egfr_results['K'], df_egfr_results['Overfitting Gap'], 
        'o-', linewidth=2, markersize=8, color='orange')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(optimal_egfr['K'], color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Features (K)', fontsize=12)
ax.set_ylabel('Overfitting Gap (Test - Train)', fontsize=12)
ax.set_title('eGFR: Overfitting vs Feature Count\n(Closer to 0 = Better)', 
             fontsize=14)
ax.grid(alpha=0.3)

# DGF: Overfitting Gap
ax = axes[1, 1]
ax.plot(df_dgf_results['K'], df_dgf_results['Overfitting Gap'], 
        'o-', linewidth=2, markersize=8, color='purple')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(optimal_dgf['K'], color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Features (K)', fontsize=12)
ax.set_ylabel('Overfitting Gap (Train - Test)', fontsize=12)
ax.set_title('DGF: Overfitting vs Feature Count\n(Closer to 0 = Better)', 
             fontsize=14)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/mrmr_optimization.png', dpi=300, 
            bbox_inches='tight')
plt.close()
print("Saved: mrmr_optimization.png")

# Model-specific comparison at optimal K
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# eGFR model comparison
ax = axes[0]
optimal_k_egfr = int(optimal_egfr['K'])
row_egfr = df_egfr_results[df_egfr_results['K'] == optimal_k_egfr].iloc[0]
models = ['Elastic Net', 'Random Forest', 'XGBoost']
test_mses = [row_egfr['Elastic Net Test'], row_egfr['Random Forest Test'], 
             row_egfr['XGBoost Test']]
train_mses = [row_egfr['Elastic Net Train'], row_egfr['Random Forest Train'], 
              row_egfr['XGBoost Train']]

x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, train_mses, width, label='Train', alpha=0.8)
bars2 = ax.bar(x + width/2, test_mses, width, label='Test', alpha=0.8)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title(f'eGFR: Model Comparison at Optimal K={optimal_k_egfr}', 
             fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# DGF model comparison
ax = axes[1]
optimal_k_dgf = int(optimal_dgf['K'])
row_dgf = df_dgf_results[df_dgf_results['K'] == optimal_k_dgf].iloc[0]
test_aucs = [row_dgf['Elastic Net Test'], row_dgf['Random Forest Test'], 
             row_dgf['XGBoost Test']]
train_aucs = [row_dgf['Elastic Net Train'], row_dgf['Random Forest Train'], 
              row_dgf['XGBoost Train']]

bars1 = ax.bar(x - width/2, train_aucs, width, label='Train', alpha=0.8)
bars2 = ax.bar(x + width/2, test_aucs, width, label='Test', alpha=0.8)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title(f'DGF: Model Comparison at Optimal K={optimal_k_dgf}', 
             fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/mrmr_optimal_model_comparison.png', dpi=300, 
            bbox_inches='tight')
plt.close()
print("Saved: mrmr_optimal_model_comparison.png")

# COMPARE WITH BASELINE (ALL FEATURES)
print("\nGenerating comparison with baseline (all features)...")

# Load baseline harmonized results
baseline_egfr = pd.read_csv('results/egfr_results_harmonized.csv', index_col=0)
baseline_dgf = pd.read_csv('results/dgf_results_harmonized.csv', index_col=0)

# Create comparison figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# eGFR Chart
x_pos = np.arange(len(models))
bar_width = 0.2

# Get baseline data (all 165 features)
egfr_train_all = [baseline_egfr.loc[m, 'Train MSE'] for m in models]
egfr_test_all = [baseline_egfr.loc[m, 'Test MSE'] for m in models]

# Get mRMR data (optimal K=25)
optimal_k_egfr = int(optimal_egfr['K'])
row_egfr = df_egfr_results[df_egfr_results['K'] == optimal_k_egfr].iloc[0]
egfr_train_mrmr = [row_egfr['Elastic Net Train'], row_egfr['Random Forest Train'], 
                   row_egfr['XGBoost Train']]
egfr_test_mrmr = [row_egfr['Elastic Net Test'], row_egfr['Random Forest Test'], 
                  row_egfr['XGBoost Test']]

# Plot bars
ax1.bar(x_pos - 1.5*bar_width, egfr_train_all, bar_width, 
        label='Train - All Features (165)', color='#3498db', alpha=0.7)
ax1.bar(x_pos - 0.5*bar_width, egfr_test_all, bar_width, 
        label='Test - All Features (165)', color='#3498db')
ax1.bar(x_pos + 0.5*bar_width, egfr_train_mrmr, bar_width, 
        label=f'Train - mRMR (K={optimal_k_egfr})', color='#e74c3c', alpha=0.7)
ax1.bar(x_pos + 1.5*bar_width, egfr_test_mrmr, bar_width, 
        label=f'Test - mRMR (K={optimal_k_egfr})', color='#e74c3c')

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
ax1.set_title('eGFR: Harmonized All Features vs mRMR Selected Features', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# DGF Chart
# Get baseline data (all 165 features)
dgf_train_all = [baseline_dgf.loc[m, 'Train AUC'] for m in models]
dgf_test_all = [baseline_dgf.loc[m, 'Test AUC'] for m in models]

# Get mRMR data (optimal K=50)
optimal_k_dgf = int(optimal_dgf['K'])
row_dgf = df_dgf_results[df_dgf_results['K'] == optimal_k_dgf].iloc[0]
dgf_train_mrmr = [row_dgf['Elastic Net Train'], row_dgf['Random Forest Train'], 
                  row_dgf['XGBoost Train']]
dgf_test_mrmr = [row_dgf['Elastic Net Test'], row_dgf['Random Forest Test'], 
                 row_dgf['XGBoost Test']]

# Plot bars
ax2.bar(x_pos - 1.5*bar_width, dgf_train_all, bar_width, 
        label='Train - All Features (165)', color='#3498db', alpha=0.7)
ax2.bar(x_pos - 0.5*bar_width, dgf_test_all, bar_width, 
        label='Test - All Features (165)', color='#3498db')
ax2.bar(x_pos + 0.5*bar_width, dgf_train_mrmr, bar_width, 
        label=f'Train - mRMR (K={optimal_k_dgf})', color='#e74c3c', alpha=0.7)
ax2.bar(x_pos + 1.5*bar_width, dgf_test_mrmr, bar_width, 
        label=f'Test - mRMR (K={optimal_k_dgf})', color='#e74c3c')

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('AUC', fontsize=12, fontweight='bold')
ax2.set_title('DGF: Harmonized All Features vs mRMR Selected Features', 
              fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('results/figures/mrmr_vs_all_features_comparison.png', dpi=300, 
            bbox_inches='tight')
plt.close()
print("Saved: mrmr_vs_all_features_comparison.png")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nResults saved to:")
print(f"  - results/mrmr_egfr_optimization.csv")
print(f"  - results/mrmr_dgf_optimization.csv")
print(f"  - results/figures/mrmr_optimization.png")
print(f"  - results/figures/mrmr_optimal_model_comparison.png")
print(f"  - results/figures/mrmr_vs_all_features_comparison.png")
print("\n" + "="*60)
