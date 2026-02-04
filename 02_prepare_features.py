import pandas as pd
from utils_imputation import impute_missing_data
from utils_feature_mapping import get_feature_mapping

print("="*60)
print("PREPARE AND MATCH FEATURES")
print("="*60)

# Load subject-level data
print("\nLoading preprocessed data...")

df_u = pd.read_csv('data/u_subject_level.csv')
df_c = pd.read_csv('data/c_subject_level.csv')

print(f"U dataset: {df_u.shape}")
print(f"C dataset: {df_c.shape}")

# U dataset: Remove only non-image clinical columns
u_drop_cols = ['R_Age', 'R_Gender', 'R_SCr_1YR', 'DGF']
df_u_clean = df_u.drop(columns=u_drop_cols)
df_u_clean.rename(columns={'R_eGFR_1YR': 'eGFR_12M', 'DGF_binary': 'DGF'}, inplace=True)

# C dataset: Remove non-image columns
c_drop_prefixes = ['Allocated', 'Slide', 'Number', 'Transplant', 'Country', 
                   'Pre_transplant', 'Pretransplant', 'Imunossupression', 'Antibody',
                   'Cr1M', 'Cr3M', 'Cr6M', 'Cr12M', 'Cr3years', 'Cr5years', 'Cr10years',
                   'eGFR_CKD_EPI_1M', 'eGFR_CKD_EPI_3M', 'eGFR_CKD_EPI_6M', 'eGFR_CKD_EPI_3years',
                   'eGFR_CKD_EPI_5years', 'eGFR_Variation', 'eGFR_less', 'eGFR_12M_ABOVE',
                   'Blood_units', 'HLA_', 'Total_HLA', 'PRA_', 'Cold_Ischemia',
                   'Graft_loss', 'Return_to_dialysis', 'Death', 'Cause_of_death', 'FollowUp',
                   'Biopsy_type', 'DONOR', 'Kidney_used', 'Expanded_criteria', 'Donor_',
                   'Donation_after', 'KDRI', 'KDPI', 'REMUZZI', 'Glomerular_GT1', 
                   'Vascular_GT1', 'IFTA_GT1', 'Fibrointimal_thickening', 'Fibro',
                   'IFTA_AI', 'Glomerulosclerosis_PERCENTAGE', 'Last_follow']

c_drop_cols = [col for col in df_c.columns if any(col.startswith(prefix) for prefix in c_drop_prefixes)]
df_c_clean = df_c.drop(columns=c_drop_cols)
df_c_clean.rename(columns={'Recipient_code': 'Subject_ID', 'eGFR_CKD_EPI_12M': 'eGFR_12M', 
                           'Delayed_Graft_Function': 'DGF'}, inplace=True)

print(f"\nU dataset after cleaning: {df_u_clean.shape}")
print(f"C dataset after cleaning: {df_c_clean.shape}")

# FEATURE MATCHING
print("\n" + "="*60)
print("MANUAL FEATURE MATCHING")
print("="*60)

# Get feature mapping from utility module
feature_mapping = get_feature_mapping()

print(f"Matched features: {len(feature_mapping)}")

# Create matched datasets
matched_u_cols = list(feature_mapping.keys())
matched_c_cols = list(feature_mapping.values())

# Create matched U dataset
df_u_matched = df_u_clean[['Subject_ID'] + matched_u_cols + ['eGFR_12M', 'DGF']].copy()

# Create matched C dataset and rename columns to match U
df_c_matched = df_c_clean[['Subject_ID'] + matched_c_cols + ['eGFR_12M', 'DGF']].copy()
rename_dict = {c_col: u_col for u_col, c_col in feature_mapping.items()}
df_c_matched.rename(columns=rename_dict, inplace=True)

# Save final matched datasets
df_u_matched.to_csv('data/u_image_features.csv', index=False)
df_c_matched.to_csv('data/c_image_features.csv', index=False)

print(f"\nSaved: data/u_image_features.csv ({df_u_matched.shape})")
print(f"Saved: data/c_image_features.csv ({df_c_matched.shape})")

# IMPUTE MISSING DATA
print("\n" + "="*60)
print("IMPUTING MISSING DATA")
print("="*60)

# Set imputation method: 'drop', 'mean', or 'median'
IMPUTATION_METHOD = 'drop'

# Apply imputation (function handles all conversion and checking)
impute_missing_data('data/u_image_features.csv', method=IMPUTATION_METHOD)
impute_missing_data('data/c_image_features.csv', method=IMPUTATION_METHOD)

# Reload imputed files
df_u_final = pd.read_csv('data/u_image_features.csv')
df_c_final = pd.read_csv('data/c_image_features.csv')

# ENSURE BOTH DATASETS HAVE SAME FEATURES
print("\nENSURING FEATURE ALIGNMENT...")

# Get feature columns (exclude Subject_ID, eGFR_12M, DGF)
u_features = [col for col in df_u_final.columns if col not in ['Subject_ID', 'eGFR_12M', 'DGF']]
c_features = [col for col in df_c_final.columns if col not in ['Subject_ID', 'eGFR_12M', 'DGF']]

# Find common features
common_features = sorted(list(set(u_features) & set(c_features)))

print(f"U features: {len(u_features)}")
print(f"C features: {len(c_features)}")
print(f"Common features: {len(common_features)}")

# Find features only in one dataset
u_only = set(u_features) - set(c_features)
c_only = set(c_features) - set(u_features)

if len(u_only) > 0:
    print(f"\nFeatures only in U dataset ({len(u_only)}): Will be dropped")

if len(c_only) > 0:
    print(f"\nFeatures only in C dataset ({len(c_only)}): Will be dropped")

# Keep only common features in both datasets
df_u_aligned = df_u_final[['Subject_ID'] + common_features + ['eGFR_12M', 'DGF']].copy()
df_c_aligned = df_c_final[['Subject_ID'] + common_features + ['eGFR_12M', 'DGF']].copy()

# REMOVE ROWS WITH MISSING SUBJECT_ID OR OUTCOMES
print("\nREMOVING INCOMPLETE ROWS...")

u_before = len(df_u_aligned)
df_u_aligned = df_u_aligned.dropna(subset=['Subject_ID', 'eGFR_12M', 'DGF'])
u_removed = u_before - len(df_u_aligned)
print(f"  U: removed {u_removed} rows with missing Subject_ID/outcomes")

c_before = len(df_c_aligned)
df_c_aligned = df_c_aligned.dropna(subset=['Subject_ID', 'eGFR_12M', 'DGF'])
c_removed = c_before - len(df_c_aligned)
print(f"  C: removed {c_removed} rows with missing Subject_ID/outcomes")

# VALIDATE OUTCOME COLUMNS BEFORE SAVING
print("\nVALIDATING OUTCOME COLUMNS...")

# Check eGFR_12M
print(f"\neGFR_12M:")
print(f"  U - Type: {df_u_aligned['eGFR_12M'].dtype}, Range: [{df_u_aligned['eGFR_12M'].min():.1f}, {df_u_aligned['eGFR_12M'].max():.1f}]")
print(f"  C - Type: {df_c_aligned['eGFR_12M'].dtype}, Range: [{df_c_aligned['eGFR_12M'].min():.1f}, {df_c_aligned['eGFR_12M'].max():.1f}]")

# Check DGF
print(f"\nDGF:")
print(f"  U - Unique values: {sorted(df_u_aligned['DGF'].dropna().unique())}")
print(f"  C - Unique values: {sorted(df_c_aligned['DGF'].dropna().unique())}")
print(f"  U - Type: {df_u_aligned['DGF'].dtype}")
print(f"  C - Type: {df_c_aligned['DGF'].dtype}")

# Check for unexpected values
u_dgf_vals = set(df_u_aligned['DGF'].dropna().unique())
c_dgf_vals = set(df_c_aligned['DGF'].dropna().unique())

if not u_dgf_vals.issubset({0, 1, 0.0, 1.0}):
    print(f"\nWARNING: U dataset DGF has unexpected values: {u_dgf_vals}")
    print("   Expected only 0 and 1 for binary outcome!")
    
if not c_dgf_vals.issubset({0, 1, 0.0, 1.0}):
    print(f"\nWARNING: C dataset DGF has unexpected values: {c_dgf_vals}")
    print("   Expected only 0 and 1 for binary outcome!")

u_egfr_vals = set(df_u_aligned['eGFR_12M'].dropna().unique())
c_egfr_vals = set(df_c_aligned['eGFR_12M'].dropna().unique())

if any(u_egfr_vals < 0) | (u_egfr_vals > 200):
    print(f"\nWARNING: U dataset eGFR_12M has values outside expected range: {u_egfr_vals}")
    print("   Range is expected to be 0-200")

if not c_egfr_vals.issubset():
    print(f"\nWARNING: C dataset eGFR_12M has values outside expected range: {c_egfr_vals}")
    print("   Range is expected to be 0-200")

# Save aligned datasets
df_u_aligned.to_csv('data/u_image_features.csv', index=False)
df_c_aligned.to_csv('data/c_image_features.csv', index=False)

print(f"\n✓ Datasets aligned with {len(common_features)} common features")
print(f"  Saved: data/u_image_features.csv ({df_u_aligned.shape})")
print(f"  Saved: data/c_image_features.csv ({df_c_aligned.shape})")

# SUMMARY
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"\nImputation method: {IMPUTATION_METHOD}")

if IMPUTATION_METHOD == 'drop':
    features_dropped_total_u = (df_u_matched.shape[1]-3) - len(common_features)
    features_dropped_total_c = (df_c_matched.shape[1]-3) - len(common_features)
    
    print(f"\nU dataset:")
    print(f"  Initial features: {df_u_matched.shape[1]-3}")
    print(f"  After imputation: {len(u_features)}")
    print(f"  After alignment: {len(common_features)}")
    print(f"  Total dropped: {features_dropped_total_u} features")
    
    print(f"\nC dataset:")
    print(f"  Initial features: {df_c_matched.shape[1]-3}")
    print(f"  After imputation: {len(c_features)}")
    print(f"  After alignment: {len(common_features)}")
    print(f"  Total dropped: {features_dropped_total_c} features")

print(f"\n{'='*60}")
print("FINAL FEATURE MATRIX DIMENSIONS")
print("="*60)
print(f"\nU cohort: {df_u_aligned.shape[0]} subjects x {len(common_features)} features")
print(f"C cohort: {df_c_aligned.shape[0]} subjects x {len(common_features)} features")
print(f"\nTotal columns per dataset: {df_u_aligned.shape[1]}")
print(f"  - Subject_ID: 1 column")
print(f"  - Image features: {len(common_features)} columns")
print(f"  - eGFR_12M: 1 column")
print(f"  - DGF: 1 column")

print(f"\n✓ Both datasets have IDENTICAL feature sets")
print(f"✓ Ready for harmonization and modeling")
print("="*60)

