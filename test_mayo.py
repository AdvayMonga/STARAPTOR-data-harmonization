"""
Quick test: Train on harmonized U+C, test on Mayo (M).
Mayo is NOT harmonized — this tests cross-domain generalization.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("QUICK TEST: Harmonized U+C → Mayo (M)")
print("="*60)

# Load training data (harmonized U+C combined)
df_train = pd.read_csv('data/harmonized_combined/harmonized_all.csv')
df_train = df_train.drop(columns=['batch'], errors='ignore')

# Load test data (Mayo)
df_test = pd.read_csv('data/m_image_features.csv')

print(f"\nTrain (harmonized U+C): {df_train.shape}")
print(f"Test (Mayo): {df_test.shape}")

# R's make.names() converts spaces to dots and special chars to dots.
# Apply same transformation to Mayo columns so they match harmonized column names.
import re
def r_make_names(col):
    # Replace non-alphanumeric/underscore chars with dots (like R's make.names)
    return re.sub(r'[^A-Za-z0-9_]', '.', col)

df_test.columns = [r_make_names(c) for c in df_test.columns]

# Use only common feature columns
meta_cols = ['Subject_ID', 'eGFR_12M', 'DGF']
train_features = [c for c in df_train.columns if c not in meta_cols]
test_features = [c for c in df_test.columns if c not in meta_cols]
common_features = sorted(set(train_features) & set(test_features))

print(f"Common features: {len(common_features)}")

X_train = df_train[common_features].values
y_train_egfr = df_train['eGFR_12M'].values
y_train_dgf = df_train['DGF'].values

X_test = df_test[common_features].values
y_test_egfr = df_test['eGFR_12M'].values
y_test_dgf = df_test['DGF'].values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# eGFR REGRESSION
# ============================================================
print("\n" + "="*60)
print("eGFR PREDICTION (Regression)")
print("="*60)

results_egfr = {}

model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train_egfr)
results_egfr['Lasso'] = {
    'Train MSE': mean_squared_error(y_train_egfr, model.predict(X_train_scaled)),
    'Test MSE': mean_squared_error(y_test_egfr, model.predict(X_test_scaled)),
}

model = Ridge(alpha=1.0, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train_egfr)
results_egfr['Ridge'] = {
    'Train MSE': mean_squared_error(y_train_egfr, model.predict(X_train_scaled)),
    'Test MSE': mean_squared_error(y_test_egfr, model.predict(X_test_scaled)),
}

model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train_egfr)
results_egfr['Elastic Net'] = {
    'Train MSE': mean_squared_error(y_train_egfr, model.predict(X_train_scaled)),
    'Test MSE': mean_squared_error(y_test_egfr, model.predict(X_test_scaled)),
}

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train_egfr)
results_egfr['Random Forest'] = {
    'Train MSE': mean_squared_error(y_train_egfr, model.predict(X_train)),
    'Test MSE': mean_squared_error(y_test_egfr, model.predict(X_test)),
}

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train_egfr)
results_egfr['XGBoost'] = {
    'Train MSE': mean_squared_error(y_train_egfr, model.predict(X_train)),
    'Test MSE': mean_squared_error(y_test_egfr, model.predict(X_test)),
}

df_egfr = pd.DataFrame(results_egfr).T
print(df_egfr.to_string())

# ============================================================
# DGF CLASSIFICATION
# ============================================================
print("\n" + "="*60)
print("DGF PREDICTION (Classification)")
print("="*60)

results_dgf = {}

model = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train_dgf)
results_dgf['Lasso'] = {
    'Train AUC': roc_auc_score(y_train_dgf, model.predict_proba(X_train_scaled)[:, 1]),
    'Test AUC': roc_auc_score(y_test_dgf, model.predict_proba(X_test_scaled)[:, 1]),
}

model = LogisticRegression(penalty='l2', solver='saga', C=1.0, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train_dgf)
results_dgf['Ridge'] = {
    'Train AUC': roc_auc_score(y_train_dgf, model.predict_proba(X_train_scaled)[:, 1]),
    'Test AUC': roc_auc_score(y_test_dgf, model.predict_proba(X_test_scaled)[:, 1]),
}

model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train_dgf)
results_dgf['Elastic Net'] = {
    'Train AUC': roc_auc_score(y_train_dgf, model.predict_proba(X_train_scaled)[:, 1]),
    'Test AUC': roc_auc_score(y_test_dgf, model.predict_proba(X_test_scaled)[:, 1]),
}

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train_dgf)
results_dgf['Random Forest'] = {
    'Train AUC': roc_auc_score(y_train_dgf, model.predict_proba(X_train)[:, 1]),
    'Test AUC': roc_auc_score(y_test_dgf, model.predict_proba(X_test)[:, 1]),
}

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train_dgf)
results_dgf['XGBoost'] = {
    'Train AUC': roc_auc_score(y_train_dgf, model.predict_proba(X_train)[:, 1]),
    'Test AUC': roc_auc_score(y_test_dgf, model.predict_proba(X_test)[:, 1]),
}

df_dgf = pd.DataFrame(results_dgf).T
print(df_dgf.to_string())

# Save results
df_egfr.to_csv('results/tables/egfr_results_harmonized_UC_to_M.csv')
df_dgf.to_csv('results/tables/dgf_results_harmonized_UC_to_M.csv')

print(f"\nSaved: results/tables/egfr_results_harmonized_UC_to_M.csv")
print(f"Saved: results/tables/dgf_results_harmonized_UC_to_M.csv")
