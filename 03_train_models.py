import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def run_train_test(train_csv, test_csv, egfr_outfile, dgf_outfile, print_summary=True):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    feature_cols = [col for col in df_train.columns if col not in ['Subject_ID', 'eGFR_12M', 'DGF']]

    X_train = df_train[feature_cols].values
    y_train_egfr = df_train['eGFR_12M'].values
    y_train_dgf = df_train['DGF'].values

    X_test = df_test[feature_cols].values
    y_test_egfr = df_test['eGFR_12M'].values
    y_test_dgf = df_test['DGF'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results_egfr = {}
    results_dgf = {}

    # Lasso (eGFR) - L1 regularization
    model_lasso = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    model_lasso.fit(X_train_scaled, y_train_egfr)
    y_train_pred_lasso = model_lasso.predict(X_train_scaled)
    y_test_pred_lasso = model_lasso.predict(X_test_scaled)
    mse_train_lasso = mean_squared_error(y_train_egfr, y_train_pred_lasso)
    mse_test_lasso = mean_squared_error(y_test_egfr, y_test_pred_lasso)
    results_egfr['Lasso'] = {'train_mse': mse_train_lasso, 'test_mse': mse_test_lasso}

    # Ridge (eGFR) - L2 regularization
    model_ridge = Ridge(alpha=1.0, max_iter=10000, random_state=42)
    model_ridge.fit(X_train_scaled, y_train_egfr)
    y_train_pred_ridge = model_ridge.predict(X_train_scaled)
    y_test_pred_ridge = model_ridge.predict(X_test_scaled)
    mse_train_ridge = mean_squared_error(y_train_egfr, y_train_pred_ridge)
    mse_test_ridge = mean_squared_error(y_test_egfr, y_test_pred_ridge)
    results_egfr['Ridge'] = {'train_mse': mse_train_ridge, 'test_mse': mse_test_ridge}

    # Elastic Net (eGFR) - L1 + L2 regularization
    model_en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
    model_en.fit(X_train_scaled, y_train_egfr)
    y_train_pred_en = model_en.predict(X_train_scaled)
    y_test_pred_en = model_en.predict(X_test_scaled)
    mse_train_en = mean_squared_error(y_train_egfr, y_train_pred_en)
    mse_test_en = mean_squared_error(y_test_egfr, y_test_pred_en)
    results_egfr['Elastic Net'] = {'train_mse': mse_train_en, 'test_mse': mse_test_en}

    # Random Forest (eGFR)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train_egfr)
    y_train_pred_rf = model_rf.predict(X_train)
    y_test_pred_rf = model_rf.predict(X_test)
    mse_train_rf = mean_squared_error(y_train_egfr, y_train_pred_rf)
    mse_test_rf = mean_squared_error(y_test_egfr, y_test_pred_rf)
    results_egfr['Random Forest'] = {'train_mse': mse_train_rf, 'test_mse': mse_test_rf}

    # XGBoost (eGFR)
    model_xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model_xgb.fit(X_train, y_train_egfr)
    y_train_pred_xgb = model_xgb.predict(X_train)
    y_test_pred_xgb = model_xgb.predict(X_test)
    mse_train_xgb = mean_squared_error(y_train_egfr, y_train_pred_xgb)
    mse_test_xgb = mean_squared_error(y_test_egfr, y_test_pred_xgb)
    results_egfr['XGBoost'] = {'train_mse': mse_train_xgb, 'test_mse': mse_test_xgb}

    # Lasso Logistic Regression (DGF) - L1 regularization
    model_lasso_clf = LogisticRegression(
        penalty='l1', solver='saga', C=1.0, max_iter=10000, random_state=42
    )
    model_lasso_clf.fit(X_train_scaled, y_train_dgf)
    y_train_prob_lasso = model_lasso_clf.predict_proba(X_train_scaled)[:, 1]
    y_test_prob_lasso = model_lasso_clf.predict_proba(X_test_scaled)[:, 1]
    auc_train_lasso = roc_auc_score(y_train_dgf, y_train_prob_lasso)
    auc_test_lasso = roc_auc_score(y_test_dgf, y_test_prob_lasso)
    results_dgf['Lasso'] = {'train_auc': auc_train_lasso, 'test_auc': auc_test_lasso}

    # Ridge Logistic Regression (DGF) - L2 regularization
    model_ridge_clf = LogisticRegression(
        penalty='l2', solver='saga', C=1.0, max_iter=10000, random_state=42
    )
    model_ridge_clf.fit(X_train_scaled, y_train_dgf)
    y_train_prob_ridge = model_ridge_clf.predict_proba(X_train_scaled)[:, 1]
    y_test_prob_ridge = model_ridge_clf.predict_proba(X_test_scaled)[:, 1]
    auc_train_ridge = roc_auc_score(y_train_dgf, y_train_prob_ridge)
    auc_test_ridge = roc_auc_score(y_test_dgf, y_test_prob_ridge)
    results_dgf['Ridge'] = {'train_auc': auc_train_ridge, 'test_auc': auc_test_ridge}

    # Elastic Net Logistic Regression (DGF) - L1 + L2 regularization
    model_lr = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=10000, random_state=42
    )
    model_lr.fit(X_train_scaled, y_train_dgf)
    y_train_prob_lr = model_lr.predict_proba(X_train_scaled)[:, 1]
    y_test_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
    auc_train_lr = roc_auc_score(y_train_dgf, y_train_prob_lr)
    auc_test_lr = roc_auc_score(y_test_dgf, y_test_prob_lr)
    results_dgf['Elastic Net'] = {'train_auc': auc_train_lr, 'test_auc': auc_test_lr}

    # Random Forest Classifier (DGF)
    model_rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_rf_clf.fit(X_train, y_train_dgf)
    y_train_prob_rf = model_rf_clf.predict_proba(X_train)[:, 1]
    y_test_prob_rf = model_rf_clf.predict_proba(X_test)[:, 1]
    auc_train_rf = roc_auc_score(y_train_dgf, y_train_prob_rf)
    auc_test_rf = roc_auc_score(y_test_dgf, y_test_prob_rf)
    results_dgf['Random Forest'] = {'train_auc': auc_train_rf, 'test_auc': auc_test_rf}

    # XGBoost (DGF)
    model_xgb_clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model_xgb_clf.fit(X_train, y_train_dgf)
    y_train_prob_xgb = model_xgb_clf.predict_proba(X_train)[:, 1]
    y_test_prob_xgb = model_xgb_clf.predict_proba(X_test)[:, 1]
    auc_train_xgb = roc_auc_score(y_train_dgf, y_train_prob_xgb)
    auc_test_xgb = roc_auc_score(y_test_dgf, y_test_prob_xgb)
    results_dgf['XGBoost'] = {'train_auc': auc_train_xgb, 'test_auc': auc_test_xgb}

    # Save results
    df_egfr = pd.DataFrame(results_egfr).T
    df_egfr.columns = ['Train MSE', 'Test MSE']
    df_egfr.to_csv(egfr_outfile)
    df_dgf = pd.DataFrame(results_dgf).T
    df_dgf.columns = ['Train AUC', 'Test AUC']
    df_dgf.to_csv(dgf_outfile)

    if print_summary:
        print("\neGFR Prediction:")
        print(df_egfr.to_string())
        print("\nDGF Prediction:")
        print(df_dgf.to_string())
        print(f"\nSaved: {egfr_outfile}")
        print(f"Saved: {dgf_outfile}")
    return df_egfr, df_dgf


print("="*60)
print("TRAIN MODELS: Train U -> Test C")
print("="*60)

# Train on U, test on C
run_train_test(
    'data/u_image_features.csv',
    'data/c_image_features.csv',
    'results/egfr_results_U_to_C.csv',
    'results/dgf_results_U_to_C.csv',
    print_summary=True
)

print("="*60)
print("TRAIN MODELS: Train C -> Test U")
print("="*60)

# Train on C, test on U
run_train_test(
    'data/c_image_features.csv',
    'data/u_image_features.csv',
    'results/egfr_results_C_to_U.csv',
    'results/dgf_results_C_to_U.csv',
    print_summary=True
)

print("="*60)
print("TRAIN MODELS: Combined Harmonized Data (U + C)")
print("="*60)

# Combined harmonized data
df_harmonized = pd.read_csv('data/harmonized_combined/harmonized_all.csv')
df_harmonized = df_harmonized.drop(columns=['batch'])

df_train, df_test = train_test_split(
    df_harmonized, test_size=0.2, random_state=42
)

df_train.to_csv('data/harmonized_combined/harmonized_train.csv', index=False)
df_test.to_csv('data/harmonized_combined/harmonized_test.csv', index=False)

run_train_test(
    'data/harmonized_combined/harmonized_train.csv',
    'data/harmonized_combined/harmonized_test.csv',
    'results/egfr_results_harmonized.csv',
    'results/dgf_results_harmonized.csv',
    print_summary=True
)

print("="*60)
print("TRAIN MODELS: Combined UNHARMONIZED Data (U + C)")
print("="*60)

# Combined unharmonized data
df_u = pd.read_csv('data/u_image_features.csv')
df_c = pd.read_csv('data/c_image_features.csv')

# Combine without harmonization
df_unharmonized = pd.concat([df_u, df_c], ignore_index=True)

# train/test with same random state
df_train_unharm, df_test_unharm = train_test_split(
    df_unharmonized, test_size=0.2, random_state=42
)

# Save unharmonized combined data
import os
os.makedirs('data/unharmonized_combined', exist_ok=True)
df_unharmonized.to_csv('data/unharmonized_combined/unharmonized_all.csv', index=False)
df_train_unharm.to_csv('data/unharmonized_combined/unharmonized_train.csv', index=False)
df_test_unharm.to_csv('data/unharmonized_combined/unharmonized_test.csv', index=False)

run_train_test(
    'data/unharmonized_combined/unharmonized_train.csv',
    'data/unharmonized_combined/unharmonized_test.csv',
    'results/egfr_results_unharmonized.csv',
    'results/dgf_results_unharmonized.csv',
    print_summary=True
)

print("="*60)
print("TRAIN MODELS: Combined CovBat Harmonized Data")
print("="*60)

# Combined CovBat harmonized data
df_covbat = pd.read_csv('data/covbat_harmonized_combined/covbat_harmonized_all.csv')
df_covbat = df_covbat.drop(columns=['batch'])

df_train_covbat, df_test_covbat = train_test_split(
    df_covbat, test_size=0.2, random_state=42
)

df_train_covbat.to_csv('data/covbat_harmonized_combined/covbat_train.csv', index=False)
df_test_covbat.to_csv('data/covbat_harmonized_combined/covbat_test.csv', index=False)

run_train_test(
    'data/covbat_harmonized_combined/covbat_train.csv',
    'data/covbat_harmonized_combined/covbat_test.csv',
    'results/egfr_results_covbat.csv',
    'results/dgf_results_covbat.csv',
    print_summary=True
)

print("="*60)
print("TRAIN MODELS: Z-Score Harmonized Data")
print("="*60)

# Z-Score harmonized data
df_zscore = pd.read_csv('data/zscore_harmonized_combined/zscore_harmonized_all.csv')
df_zscore = df_zscore.drop(columns=['batch'])

df_train_zscore, df_test_zscore = train_test_split(
    df_zscore, test_size=0.2, random_state=42
)

df_train_zscore.to_csv('data/zscore_harmonized_combined/zscore_train.csv', index=False)
df_test_zscore.to_csv('data/zscore_harmonized_combined/zscore_test.csv', index=False)

run_train_test(
    'data/zscore_harmonized_combined/zscore_train.csv',
    'data/zscore_harmonized_combined/zscore_test.csv',
    'results/egfr_results_zscore.csv',
    'results/dgf_results_zscore.csv',
    print_summary=True
)

print("="*60)
print("TRAIN MODELS: RAVEL Harmonized Data")
print("="*60)

# RAVEL harmonized data
df_ravel = pd.read_csv('data/ravel_harmonized_combined/ravel_harmonized_all.csv')
df_ravel = df_ravel.drop(columns=['batch'])

df_train_ravel, df_test_ravel = train_test_split(
    df_ravel, test_size=0.2, random_state=42
)

df_train_ravel.to_csv('data/ravel_harmonized_combined/ravel_train.csv', index=False)
df_test_ravel.to_csv('data/ravel_harmonized_combined/ravel_test.csv', index=False)

run_train_test(
    'data/ravel_harmonized_combined/ravel_train.csv',
    'data/ravel_harmonized_combined/ravel_test.csv',
    'results/egfr_results_ravel.csv',
    'results/dgf_results_ravel.csv',
    print_summary=True
)

print("="*60)
print("TRAIN MODELS: CORAL Harmonized Data")
print("="*60)

# CORAL harmonized data
df_coral = pd.read_csv('data/coral_harmonized_combined/coral_harmonized_all.csv')
df_coral = df_coral.drop(columns=['batch'])

df_train_coral, df_test_coral = train_test_split(
    df_coral, test_size=0.2, random_state=42
)

df_train_coral.to_csv('data/coral_harmonized_combined/coral_train.csv', index=False)
df_test_coral.to_csv('data/coral_harmonized_combined/coral_test.csv', index=False)

run_train_test(
    'data/coral_harmonized_combined/coral_train.csv',
    'data/coral_harmonized_combined/coral_test.csv',
    'results/egfr_results_coral.csv',
    'results/dgf_results_coral.csv',
    print_summary=True
)
