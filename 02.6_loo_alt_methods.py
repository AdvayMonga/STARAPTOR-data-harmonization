"""
LOO Alternative Harmonization Methods.

For each LOO split (UC→M, UM→C, CM→U) and each method (Z-Score, CORAL, CovBat),
produce harmonized train/test CSVs and train models on them, writing results to
`results/tables/egfr_results_loo_{scenario}_{method}.csv` and the DGF counterpart.

RAVEL is not run — no implementation exists in this repo and no Python/R source
is available. Treat its column in the older pooled summary as not comparable.
"""

import os
import re
import subprocess
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

RAW_COHORT_FILES = {
    'U': 'data/u_image_features.csv',
    'C': 'data/c_image_features.csv',
    'M': 'data/m_image_features.csv',
}
LOO_TRAIN_COHORTS = {'UC_to_M': ['U', 'C'], 'UM_to_C': ['U', 'M'], 'CM_to_U': ['C', 'M']}
LOO_TEST_COHORTS  = {'UC_to_M': 'M',        'UM_to_C': 'C',        'CM_to_U': 'U'}
LOO_SPLITS = list(LOO_TRAIN_COHORTS.keys())

METADATA = ['Subject_ID', 'eGFR_12M', 'DGF']


def feature_cols_of(df):
    return [c for c in df.columns if c not in METADATA + ['batch']]


# ── Harmonizers ──────────────────────────────────────────────

def zscore_loo(train_dfs, test_df):
    """Per-cohort standardization. Each cohort scaled with its own mean/std."""
    out_train = []
    feats = feature_cols_of(train_dfs[0])
    for df in train_dfs:
        d = df.copy()
        scaler = StandardScaler()
        d[feats] = scaler.fit_transform(d[feats])
        out_train.append(d)
    train_harm = pd.concat(out_train, ignore_index=True)

    test_harm = test_df.copy()
    test_harm[feats] = StandardScaler().fit_transform(test_harm[feats])
    return train_harm, test_harm


def coral_transform(X_source, X_target):
    n_features = X_source.shape[1]
    reg = 1e-6 * np.eye(n_features)
    Cs = np.cov(X_source, rowvar=False) + reg
    Ct = np.cov(X_target, rowvar=False) + reg
    Cs_sqrt_inv = scipy.linalg.inv(scipy.linalg.sqrtm(Cs))
    Ct_sqrt = scipy.linalg.sqrtm(Ct)
    Xc = X_source - X_source.mean(axis=0)
    return (Xc @ Cs_sqrt_inv.real @ Ct_sqrt.real) + X_target.mean(axis=0)


def coral_loo(train_dfs, test_df):
    """Align each train cohort to the second train cohort's distribution
    (use first train cohort as the 'source' aligned to the second).
    Test cohort is aligned to the combined (post-alignment) train distribution."""
    feats = feature_cols_of(train_dfs[0])
    # Pick second train df as target, align first to it
    src, tgt = train_dfs[0].copy(), train_dfs[1].copy()
    src_aligned = src.copy()
    src_aligned[feats] = coral_transform(src[feats].values, tgt[feats].values)
    train_harm = pd.concat([src_aligned, tgt], ignore_index=True)
    # Align test to train (combined post-alignment)
    test_harm = test_df.copy()
    test_harm[feats] = coral_transform(test_df[feats].values, train_harm[feats].values)
    return train_harm, test_harm


def covbat_loo_via_r(train_dfs, test_df, scenario):
    """Call ComBatFamQC's covbat_harm via Rscript. Harmonizes train+test
    together using batch label; outputs harmonized dataframe split back."""
    feats = feature_cols_of(train_dfs[0])
    # Tag batches: combine, add batch label
    parts = []
    for i, df in enumerate(train_dfs):
        d = df.copy()
        d['batch'] = f'TR{i}'
        parts.append(d)
    test_tagged = test_df.copy()
    test_tagged['batch'] = 'TE'
    combined = pd.concat(parts + [test_tagged], ignore_index=True)

    tmp_in  = f'data/loo_combat/{scenario}/_covbat_in.csv'
    tmp_out = f'data/loo_combat/{scenario}/_covbat_out.csv'
    os.makedirs(os.path.dirname(tmp_in), exist_ok=True)
    combined.to_csv(tmp_in, index=False)

    r_script = f"""
suppressMessages(library(ComBatFamQC))
df <- read.csv('{tmp_in}', check.names=FALSE)
feats <- setdiff(colnames(df), c('Subject_ID','eGFR_12M','DGF','batch'))
df$batch <- as.factor(df$batch)
res <- combat_harm(df = df, batch = 'batch', features = feats,
                   type = 'lm', family = 'covfam', eb = TRUE, quiet = TRUE)
out <- res$harmonized_df
write.csv(out, '{tmp_out}', row.names=FALSE)
"""
    p = subprocess.run(['Rscript', '-e', r_script], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"CovBat R call failed for {scenario}:\nSTDOUT:{p.stdout}\nSTDERR:{p.stderr}")

    harmonized = pd.read_csv(tmp_out)
    train_harm = harmonized[harmonized['batch'].isin([f'TR{i}' for i in range(len(train_dfs))])].drop(columns=['batch']).reset_index(drop=True)
    test_harm  = harmonized[harmonized['batch'] == 'TE'].drop(columns=['batch']).reset_index(drop=True)
    os.remove(tmp_in); os.remove(tmp_out)
    return train_harm, test_harm


# ── Train/test driver (mirrors 03_train_models.run_train_test) ─

def run_train_test(train_df, test_df, egfr_outfile, dgf_outfile, label=''):
    feats = [c for c in train_df.columns if c not in METADATA]
    X_train, X_test = train_df[feats].values, test_df[feats].values
    y_tr_egfr, y_te_egfr = train_df['eGFR_12M'].values, test_df['eGFR_12M'].values
    y_tr_dgf,  y_te_dgf  = train_df['DGF'].values,    test_df['DGF'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    res_egfr, res_dgf = {}, {}

    regressors = {
        'Lasso':         Lasso(alpha=0.1, max_iter=10000, random_state=42),
        'Ridge':         Ridge(alpha=1.0, max_iter=10000, random_state=42),
        'Elastic Net':   ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':       XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1),
    }
    classifiers = {
        'Lasso':         LogisticRegression(penalty='l1', solver='liblinear', C=10, max_iter=10000, random_state=42),
        'Ridge':         LogisticRegression(penalty='l2', C=1.0, max_iter=10000, random_state=42),
        'Elastic Net':   LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=10, max_iter=10000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':       XGBClassifier(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
    }

    for name, m in regressors.items():
        m.fit(X_tr_s, y_tr_egfr)
        res_egfr[name] = {
            'Train MSE': mean_squared_error(y_tr_egfr, m.predict(X_tr_s)),
            'Test MSE':  mean_squared_error(y_te_egfr, m.predict(X_te_s)),
        }
    for name, m in classifiers.items():
        m.fit(X_tr_s, y_tr_dgf)
        proba_tr = m.predict_proba(X_tr_s)[:, 1]
        proba_te = m.predict_proba(X_te_s)[:, 1]
        res_dgf[name] = {
            'Train AUC': roc_auc_score(y_tr_dgf, proba_tr),
            'Test AUC':  roc_auc_score(y_te_dgf, proba_te),
        }

    pd.DataFrame(res_egfr).T.to_csv(egfr_outfile)
    pd.DataFrame(res_dgf ).T.to_csv(dgf_outfile)
    print(f"  ✓ {label}: saved {egfr_outfile.split('/')[-1]} / {dgf_outfile.split('/')[-1]}")


# ── Main ─────────────────────────────────────────────────────

METHODS = {
    'zscore': zscore_loo,
    'coral':  coral_loo,
    'covbat': None,  # special-cased (R)
}

os.makedirs('results/tables', exist_ok=True)

for scenario in LOO_SPLITS:
    train_cohorts = LOO_TRAIN_COHORTS[scenario]
    test_cohort   = LOO_TEST_COHORTS[scenario]
    train_dfs = [pd.read_csv(RAW_COHORT_FILES[c]) for c in train_cohorts]
    test_df   = pd.read_csv(RAW_COHORT_FILES[test_cohort])

    print(f"\n=== Scenario {scenario}: {'+'.join(train_cohorts)} → {test_cohort} ===")

    for method_key, harm_fn in METHODS.items():
        try:
            if method_key == 'covbat':
                train_h, test_h = covbat_loo_via_r(train_dfs, test_df, scenario)
            else:
                train_h, test_h = harm_fn(train_dfs, test_df)
        except Exception as e:
            print(f"  ✗ {method_key} failed for {scenario}: {e}")
            continue

        # Save harmonized CSVs for traceability
        out_dir = f'data/loo_alt/{scenario}'
        os.makedirs(out_dir, exist_ok=True)
        train_h.to_csv(f'{out_dir}/train_{method_key}.csv', index=False)
        test_h.to_csv(f'{out_dir}/test_{method_key}.csv', index=False)

        # Train + evaluate
        run_train_test(
            train_h, test_h,
            f'results/tables/egfr_results_loo_{scenario}_{method_key}.csv',
            f'results/tables/dgf_results_loo_{scenario}_{method_key}.csv',
            label=f'{scenario}/{method_key}',
        )

print("\n✓ Done.")
