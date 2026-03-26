"""
Alternative Harmonization Methods
- Z-score per batch
- Quantile Normalization
- CORAL (Correlation Alignment)

Saves results in same format as ComBat/CovBat for comparison
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
import scipy.linalg
import os

print("="*60)
print("ALTERNATIVE HARMONIZATION METHODS")
print("="*60)

# LOAD DATA
print("\nLoading data...")
df_u = pd.read_csv('data/u_image_features.csv')
df_c = pd.read_csv('data/c_image_features.csv')

print(f"  U cohort: {len(df_u)} subjects")
print(f"  C cohort: {len(df_c)} subjects")

# Add batch column
df_u['batch'] = 'U'
df_c['batch'] = 'C'

# Combine data
df_combined = pd.concat([df_u, df_c], ignore_index=True)

# Identify feature columns (exclude metadata)
metadata_cols = ['Subject_ID', 'eGFR_12M', 'DGF', 'batch']
feature_cols = [col for col in df_combined.columns if col not in metadata_cols]

print(f"  Features: {len(feature_cols)}")

# Z-SCORE PER BATCH
print("\n" + "-"*60)
print("1. Z-SCORE PER BATCH")
print("-"*60)

df_zscore = df_combined.copy()

# Z-score normalize each batch separately
for batch in ['U', 'C']:
    mask = df_zscore['batch'] == batch
    scaler = StandardScaler()
    df_zscore.loc[mask, feature_cols] = scaler.fit_transform(
        df_zscore.loc[mask, feature_cols]
    )

print("  ✓ Z-score normalization applied per batch")

# Save Z-score results
os.makedirs('data/zscore_harmonized_combined', exist_ok=True)
os.makedirs('data/zscore_harmonized_u', exist_ok=True)
os.makedirs('data/zscore_harmonized_c', exist_ok=True)

# Save combined
df_zscore.to_csv('data/zscore_harmonized_combined/zscore_harmonized_all.csv', index=False)

# Split into train/test
df_zscore_train, df_zscore_test = train_test_split(
    df_zscore, test_size=0.2, random_state=42
)
df_zscore_train.to_csv('data/zscore_harmonized_combined/zscore_train.csv', index=False)
df_zscore_test.to_csv('data/zscore_harmonized_combined/zscore_test.csv', index=False)

# Save per-cohort
df_zscore[df_zscore['batch'] == 'U'].to_csv(
    'data/zscore_harmonized_u/u_zscore_harmonized.csv', index=False
)
df_zscore[df_zscore['batch'] == 'C'].to_csv(
    'data/zscore_harmonized_c/c_zscore_harmonized.csv', index=False
)

print("  ✓ Saved to data/zscore_harmonized_*/")

# QUANTILE NORMALIZATION
print("\n" + "-"*60)
print("2. QUANTILE NORMALIZATION")
print("-"*60)

df_quantile = df_combined.copy()

# Fit quantile transformer on all data to create reference distribution
qt = QuantileTransformer(output_distribution='normal', random_state=42)
df_quantile[feature_cols] = qt.fit_transform(df_quantile[feature_cols])

print("  ✓ Quantile normalization applied (Gaussian output)")

# Save Quantile results
os.makedirs('data/quantile_harmonized_combined', exist_ok=True)
os.makedirs('data/quantile_harmonized_u', exist_ok=True)
os.makedirs('data/quantile_harmonized_c', exist_ok=True)

# Save combined
df_quantile.to_csv('data/quantile_harmonized_combined/quantile_harmonized_all.csv', index=False)

# Split train/test
df_quantile_train, df_quantile_test = train_test_split(
    df_quantile, test_size=0.2, random_state=42
)
df_quantile_train.to_csv('data/quantile_harmonized_combined/quantile_train.csv', index=False)
df_quantile_test.to_csv('data/quantile_harmonized_combined/quantile_test.csv', index=False)

# Save per-cohort
df_quantile[df_quantile['batch'] == 'U'].to_csv(
    'data/quantile_harmonized_u/u_quantile_harmonized.csv', index=False
)
df_quantile[df_quantile['batch'] == 'C'].to_csv(
    'data/quantile_harmonized_c/c_quantile_harmonized.csv', index=False
)

print("  ✓ Saved to data/quantile_harmonized_*/")

# CORAL (Correlation Alignment)
print("\n" + "-"*60)
print("3. CORAL (Correlation Alignment)")
print("-"*60)

def coral_transform(X_source, X_target):
    # Compute covariance matrices with regularization
    n_features = X_source.shape[1]
    reg = 1e-6 * np.eye(n_features)
    
    Cs = np.cov(X_source, rowvar=False) + reg
    Ct = np.cov(X_target, rowvar=False) + reg
    
    # Compute transformation: Cs^(-1/2) @ Ct^(1/2)
    Cs_sqrt_inv = scipy.linalg.inv(scipy.linalg.sqrtm(Cs))
    Ct_sqrt = scipy.linalg.sqrtm(Ct)
    
    # Center source, apply transformation, then re-center to target mean
    X_source_centered = X_source - X_source.mean(axis=0)
    X_transformed = X_source_centered @ Cs_sqrt_inv.real @ Ct_sqrt.real
    X_aligned = X_transformed + X_target.mean(axis=0)
    
    return X_aligned

# Use U as source, C as target (align U to C's distribution)
X_u = df_combined[df_combined['batch'] == 'U'][feature_cols].values
X_c = df_combined[df_combined['batch'] == 'C'][feature_cols].values

# Align U to C
X_u_aligned = coral_transform(X_u, X_c)

# Create CORAL dataframe
df_coral = df_combined.copy()

# Replace U features with aligned , keep C
df_coral.loc[df_coral['batch'] == 'U', feature_cols] = X_u_aligned
# C stays the same (it's the target distribution)

print("  ✓ CORAL alignment applied (U aligned to C distribution)")

# Save CORAL results
os.makedirs('data/coral_harmonized_combined', exist_ok=True)
os.makedirs('data/coral_harmonized_u', exist_ok=True)
os.makedirs('data/coral_harmonized_c', exist_ok=True)

# Save combined
df_coral.to_csv('data/coral_harmonized_combined/coral_harmonized_all.csv', index=False)

# Split into train/test
df_coral_train, df_coral_test = train_test_split(
    df_coral, test_size=0.2, random_state=42
)
df_coral_train.to_csv('data/coral_harmonized_combined/coral_train.csv', index=False)
df_coral_test.to_csv('data/coral_harmonized_combined/coral_test.csv', index=False)

# Save per-cohort
df_coral[df_coral['batch'] == 'U'].to_csv(
    'data/coral_harmonized_u/u_coral_harmonized.csv', index=False
)
df_coral[df_coral['batch'] == 'C'].to_csv(
    'data/coral_harmonized_c/c_coral_harmonized.csv', index=False
)

print("  ✓ Saved to data/coral_harmonized_*/")

# SUMMARY
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nHarmonization methods applied:")
print("  1. Z-score per batch  → data/zscore_harmonized_*/")
print("  2. Quantile Normal.   → data/quantile_harmonized_*/")
print("  3. CORAL              → data/coral_harmonized_*/")

print("\nEach directory contains:")
print("  - *_harmonized_all.csv  (full dataset)")
print("  - *_train.csv           (80% train split)")
print("  - *_test.csv            (20% test split)")

print("\nNext steps:")
print("  - Run 03_train_models.py with new harmonization methods")
print("  - Compare results across all methods")

print("\n" + "="*60)
