"""
Leave-One-Cohort-Out (LOO) ComBat Pipeline

  1. Pick 2 cohorts for training, 1 for testing
  2. Fit ComBat on the 2 training cohorts → learn harmonization parameters
  3. Train ML models on the harmonized training data
  4. Apply the learned ComBat parameters to the test cohort (no refitting)
  5. Test ML models on the harmonized test data
  6. Record results for all 3 combinations

Combinations:
  UC → M : Train on U+C, Test on M
  UM → C : Train on U+M, Test on C
  CM → U : Train on C+M, Test on U
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

META_COLS = ['Subject_ID', 'eGFR_12M', 'DGF']


# ─────────────────────────────────────────────────────────────
# ComBat: fit on training batches
# ─────────────────────────────────────────────────────────────

def combat_fit(dfs, batch_labels, feature_cols):
    """
    Fit ComBat on two training batches.

    Per feature, estimates:
      grand_mean  : mean across all training samples
      pooled_std  : sqrt of the average within-batch variance
      gamma       : additive batch offset  (batch_mean - grand_mean)
      delta       : multiplicative scale   (batch_std  / pooled_std)

    Returns
    -------
    params : dict of fitted parameters
    harmonized_dfs : list of batch-corrected DataFrames (same order as dfs)
    """
    X_all = np.vstack([df[feature_cols].values.astype(float) for df in dfs])
    grand_mean = X_all.mean(axis=0)

    batch_means, batch_stds = {}, {}
    for df, label in zip(dfs, batch_labels):
        X_b = df[feature_cols].values.astype(float)
        batch_means[label] = X_b.mean(axis=0)
        batch_stds[label]  = X_b.std(axis=0)

    # pooled_std = sqrt of average within-batch variance
    pooled_std = np.sqrt(
        np.mean([s ** 2 for s in batch_stds.values()], axis=0) + 1e-8
    )

    batch_params = {}
    for label in batch_labels:
        batch_params[label] = {
            'gamma': batch_means[label] - grand_mean,
            'delta': (batch_stds[label] + 1e-8) / pooled_std,
        }

    harmonized_dfs = []
    for df, label in zip(dfs, batch_labels):
        df_h = df.copy()
        X_b  = df[feature_cols].values.astype(float)
        g    = batch_params[label]['gamma']
        d    = batch_params[label]['delta']
        df_h[feature_cols] = (X_b - g) / d
        harmonized_dfs.append(df_h)

    params = {
        'grand_mean':   grand_mean,
        'pooled_std':   pooled_std,
        'batch_params': batch_params,
        'feature_cols': feature_cols,
    }
    return params, harmonized_dfs


# ─────────────────────────────────────────────────────────────
# ComBat: apply learned parameters to held-out test cohort
# ─────────────────────────────────────────────────────────────

def combat_apply(df_test, params):
    """
    Project a held-out cohort into the training reference space.

    Uses the grand_mean and pooled_std from the training fit as the
    reference distribution. Estimates the test cohort's additive (gamma)
    and multiplicative (delta) batch effects empirically, then removes them.

    No ComBat model is re-fitted — only the training reference is used.
    """
    feature_cols = params['feature_cols']
    grand_mean   = params['grand_mean']
    pooled_std   = params['pooled_std']

    X_test     = df_test[feature_cols].values.astype(float)
    test_mean  = X_test.mean(axis=0)
    test_std   = X_test.std(axis=0)

    gamma_test = test_mean - grand_mean
    delta_test = (test_std + 1e-8) / pooled_std

    df_corrected = df_test.copy()
    df_corrected[feature_cols] = (X_test - gamma_test) / delta_test
    return df_corrected


# ─────────────────────────────────────────────────────────────
# ML training and evaluation
# ─────────────────────────────────────────────────────────────

def run_models(df_train, df_test, feature_cols, egfr_out, dgf_out, label=''):
    X_train = df_train[feature_cols].values
    X_test  = df_test[feature_cols].values
    y_train_egfr = df_train['eGFR_12M'].values
    y_test_egfr  = df_test['eGFR_12M'].values
    y_train_dgf  = df_train['DGF'].values
    y_test_dgf   = df_test['DGF'].values

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    res_egfr, res_dgf = {}, {}

    # ── eGFR (regression) ────────────────────────────────────
    m = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    m.fit(X_train_sc, y_train_egfr)
    res_egfr['Lasso'] = {
        'Train MSE': mean_squared_error(y_train_egfr, m.predict(X_train_sc)),
        'Test MSE':  mean_squared_error(y_test_egfr,  m.predict(X_test_sc)),
    }

    m = Ridge(alpha=1.0, max_iter=10000, random_state=42)
    m.fit(X_train_sc, y_train_egfr)
    res_egfr['Ridge'] = {
        'Train MSE': mean_squared_error(y_train_egfr, m.predict(X_train_sc)),
        'Test MSE':  mean_squared_error(y_test_egfr,  m.predict(X_test_sc)),
    }

    m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
    m.fit(X_train_sc, y_train_egfr)
    res_egfr['Elastic Net'] = {
        'Train MSE': mean_squared_error(y_train_egfr, m.predict(X_train_sc)),
        'Test MSE':  mean_squared_error(y_test_egfr,  m.predict(X_test_sc)),
    }

    m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    m.fit(X_train, y_train_egfr)
    res_egfr['Random Forest'] = {
        'Train MSE': mean_squared_error(y_train_egfr, m.predict(X_train)),
        'Test MSE':  mean_squared_error(y_test_egfr,  m.predict(X_test)),
    }

    m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    m.fit(X_train, y_train_egfr)
    res_egfr['XGBoost'] = {
        'Train MSE': mean_squared_error(y_train_egfr, m.predict(X_train)),
        'Test MSE':  mean_squared_error(y_test_egfr,  m.predict(X_test)),
    }

    # ── DGF (classification) ─────────────────────────────────
    m = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=10000, random_state=42)
    m.fit(X_train_sc, y_train_dgf)
    res_dgf['Lasso'] = {
        'Train AUC': roc_auc_score(y_train_dgf, m.predict_proba(X_train_sc)[:, 1]),
        'Test AUC':  roc_auc_score(y_test_dgf,  m.predict_proba(X_test_sc)[:, 1]),
    }

    m = LogisticRegression(penalty='l2', solver='saga', C=1.0, max_iter=10000, random_state=42)
    m.fit(X_train_sc, y_train_dgf)
    res_dgf['Ridge'] = {
        'Train AUC': roc_auc_score(y_train_dgf, m.predict_proba(X_train_sc)[:, 1]),
        'Test AUC':  roc_auc_score(y_test_dgf,  m.predict_proba(X_test_sc)[:, 1]),
    }

    m = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                           C=1.0, max_iter=10000, random_state=42)
    m.fit(X_train_sc, y_train_dgf)
    res_dgf['Elastic Net'] = {
        'Train AUC': roc_auc_score(y_train_dgf, m.predict_proba(X_train_sc)[:, 1]),
        'Test AUC':  roc_auc_score(y_test_dgf,  m.predict_proba(X_test_sc)[:, 1]),
    }

    m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    m.fit(X_train, y_train_dgf)
    res_dgf['Random Forest'] = {
        'Train AUC': roc_auc_score(y_train_dgf, m.predict_proba(X_train)[:, 1]),
        'Test AUC':  roc_auc_score(y_test_dgf,  m.predict_proba(X_test)[:, 1]),
    }

    m = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    m.fit(X_train, y_train_dgf)
    res_dgf['XGBoost'] = {
        'Train AUC': roc_auc_score(y_train_dgf, m.predict_proba(X_train)[:, 1]),
        'Test AUC':  roc_auc_score(y_test_dgf,  m.predict_proba(X_test)[:, 1]),
    }

    df_egfr = pd.DataFrame(res_egfr).T
    df_dgf  = pd.DataFrame(res_dgf).T
    df_egfr.to_csv(egfr_out)
    df_dgf.to_csv(dgf_out)

    print(f"\n  {label} — eGFR Test MSE:")
    print(df_egfr[['Test MSE']].round(1).to_string())
    print(f"\n  {label} — DGF Test AUC:")
    print(df_dgf[['Test AUC']].round(3).to_string())

    return df_egfr, df_dgf


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("LEAVE-ONE-COHORT-OUT COMBAT PIPELINE")
print("=" * 60)

print("\nLoading data...")
df_u = pd.read_csv('data/u_image_features.csv')
df_c = pd.read_csv('data/c_image_features.csv')
df_m = pd.read_csv('data/m_image_features.csv')

print(f"  U (UC Davis):    {df_u.shape}")
print(f"  C (Coimbra):     {df_c.shape}")
print(f"  M (Mayo Clinic): {df_m.shape}")

feature_cols = [col for col in df_u.columns if col not in META_COLS]
print(f"  Features: {len(feature_cols)}")

Path('results/tables').mkdir(parents=True, exist_ok=True)
Path('data/loo_combat').mkdir(parents=True, exist_ok=True)

COHORTS = {'U': df_u, 'C': df_c, 'M': df_m}
NAMES   = {'U': 'UC Davis', 'C': 'Coimbra', 'M': 'Mayo Clinic'}

LOO_SPLITS = [
    # (train_cohort_labels, test_cohort_label, result_key)
    (['U', 'C'], 'M', 'UC_to_M'),
    (['U', 'M'], 'C', 'UM_to_C'),
    (['C', 'M'], 'U', 'CM_to_U'),
]

all_egfr, all_dgf = {}, {}

for train_labels, test_label, key in LOO_SPLITS:
    train_str = ' + '.join(NAMES[l] for l in train_labels)
    test_str  = NAMES[test_label]

    print("\n" + "=" * 60)
    print(f"Train: {train_str}  →  Test: {test_str}")
    print("=" * 60)

    train_dfs = [COHORTS[l] for l in train_labels]
    df_test   = COHORTS[test_label]

    # Step 2: fit ComBat on training cohorts
    print("  Step 2: Fitting ComBat on training cohorts...")
    params, harmonized_train_dfs = combat_fit(train_dfs, train_labels, feature_cols)
    df_train_harmonized = pd.concat(harmonized_train_dfs, ignore_index=True)
    print(f"    Training set: {df_train_harmonized.shape[0]} subjects")

    # Step 4: apply learned parameters to test cohort (no refitting)
    print("  Step 4: Applying ComBat parameters to test cohort...")
    df_test_harmonized = combat_apply(df_test, params)
    print(f"    Test set: {df_test_harmonized.shape[0]} subjects")

    # Save harmonized splits for inspection
    out_dir = Path(f'data/loo_combat/{key}')
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train_harmonized.to_csv(out_dir / 'train_harmonized.csv', index=False)
    df_test_harmonized.to_csv(out_dir / 'test_harmonized.csv',  index=False)

    # Steps 3 & 5: train models on harmonized training data, test on harmonized test data
    print("  Steps 3 & 5: Training and evaluating models...")
    df_egfr, df_dgf = run_models(
        df_train        = df_train_harmonized,
        df_test         = df_test_harmonized,
        feature_cols    = feature_cols,
        egfr_out        = f'results/tables/egfr_results_{key}.csv',
        dgf_out         = f'results/tables/dgf_results_{key}.csv',
        label           = key,
    )

    all_egfr[key] = df_egfr
    all_dgf[key]  = df_dgf

# ─────────────────────────────────────────────────────────────
# Summary across all 3 combinations
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: SUMMARY ACROSS ALL 3 LOO COMBINATIONS")
print("=" * 60)

egfr_summary = pd.DataFrame({k: df['Test MSE']  for k, df in all_egfr.items()})
dgf_summary  = pd.DataFrame({k: df['Test AUC']  for k, df in all_dgf.items()})

print("\neGFR Test MSE (lower = better):")
print(egfr_summary.round(1).to_string())

print("\nDGF Test AUC (higher = better):")
print(dgf_summary.round(3).to_string())

egfr_summary.to_csv('results/tables/loo_egfr_summary.csv')
dgf_summary.to_csv('results/tables/loo_dgf_summary.csv')

print("\n✓ Results saved to results/tables/")
print("  loo_egfr_summary.csv")
print("  loo_dgf_summary.csv")
print("  egfr_results_{UC_to_M, UM_to_C, CM_to_U}.csv")
print("  dgf_results_{UC_to_M, UM_to_C, CM_to_U}.csv")
