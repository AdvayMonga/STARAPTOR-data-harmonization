import pandas as pd
import numpy as np
import os
from config import RAW_INPUT, RAW_DATA, DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n" + "="*60)
print("LOADING RAW DATA")
print("\n" + "="*60)

df_u = pd.read_csv(RAW_INPUT["u_raw"])
df_c = pd.read_csv(RAW_INPUT["c_raw"])

print("Data Loaded Successfully")
print(f"U Dataset Shape: {df_u.shape}")
print(f"C Dataset Shape: {df_c.shape}")

# Aggregate C data to subject-level
print("\n" + "="*60)
print("AGGREGATING C DATA TO SUBJECT-LEVEL")
print("="*60)

# Identify numeric columns
numeric_cols = df_c.select_dtypes(include=[np.number]).columns.tolist()

# Outcomes to preserve, first value
outcome_cols = ['eGFR_CKD_EPI_12M', 'Delayed_Graft_Function']

# Demographic features, first value
demographic_cols = ['Donor_age', 'Donor_Sex_0_Male_1_Female', 'Donor_race', 'Donor_height', 
                    'Donor_weight_Kg', 'Donor_BMI', 'Donor_Hypertension_History', 
                    'Donor_Diabetes_History', 'Expanded_criteria_donor', 'Allocated_Discarded']


# Image features - all numeric columns except outcomes
image_cols = [col for col in numeric_cols if col not in outcome_cols]

print(f"{len(image_cols)} image feature columns")

# Aggregation dictionary
agg_dict = {col: 'mean' for col in image_cols}
agg_dict.update({col: 'first' for col in outcome_cols})

# Aggregate by patient
df_c_subject = df_c.groupby('Recipient_code').agg(agg_dict).reset_index()

# Remove rows with empty Recipient_code
df_c_subject = df_c_subject[df_c_subject['Recipient_code'].notna() & (df_c_subject['Recipient_code'] != '')]

# Remove empty columns
df_c_subject = df_c_subject.dropna(axis=1, how='all')

print(f"\nBefore aggregation: {df_c.shape}")
print(f"After aggregation: {df_c_subject.shape}")
print(f"Subjects: {df_c_subject['Recipient_code'].nunique()}")

# Save processed data
print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)

df_u.to_csv(RAW_DATA["u_subject_level"], index=False)
df_c_subject.to_csv(RAW_DATA["c_subject_level"], index=False)

print(f"Saved: {RAW_DATA['u_subject_level']}")
print(f"Saved: {RAW_DATA['c_subject_level']}")

print("\n" + "="*60)
print("PROCESSING MAYO CLINIC (M) COHORT")
print("="*60)

mayo_base = RAW_INPUT["m_raw"]

tissue_config = {
    "Arteries": "_artery",
    "Non_Globally_Sclerotic_Gloms": "_glom",
    "Tubules": "_tubule",
    "Globally_Sclerotic_Gloms": "_gs_glom",
}

tissue_dfs = {}

for tissue_folder, suffix in tissue_config.items():
    tissue_path = mayo_base / tissue_folder
    if not tissue_path.exists():
        print(f"WARNING: {tissue_folder} directory not found, skipping")
        continue

    subject_dirs = sorted([d for d in os.listdir(tissue_path) if os.path.isdir(tissue_path / d)])
    print(f"\n{tissue_folder}: {len(subject_dirs)} subjects")

    records = []
    for subject_id in subject_dirs:
        subject_path = tissue_path / subject_id
        xlsx_files = [f for f in os.listdir(subject_path) if f.endswith('.xlsx')]
        if not xlsx_files:
            print(f"  WARNING: No xlsx file for subject {subject_id}")
            continue

        df_subj = pd.read_excel(subject_path / xlsx_files[0])

        if 'compartment_id' in df_subj.columns:
            df_subj = df_subj.drop(columns=['compartment_id'])

        numeric_cols = df_subj.select_dtypes(include=[np.number]).columns.tolist()
        agg_row = df_subj[numeric_cols].mean()
        agg_row['Subject_ID'] = subject_id
        records.append(agg_row)

    df_tissue = pd.DataFrame(records)

    rename_map = {col: col + suffix for col in df_tissue.columns if col != 'Subject_ID'}
    df_tissue = df_tissue.rename(columns=rename_map)

    tissue_dfs[tissue_folder] = df_tissue
    print(f"  Aggregated shape: {df_tissue.shape}")

print("\n" + "="*60)
print("MERGING MAYO TISSUE TYPES")
print("="*60)

df_m = None
for tissue_name, df_tissue in tissue_dfs.items():
    if df_m is None:
        df_m = df_tissue
    else:
        df_m = df_m.merge(df_tissue, on='Subject_ID', how='outer')

print(f"Merged Mayo dataset: {df_m.shape}")
print(f"Subjects: {df_m['Subject_ID'].nunique()}")

df_m = df_m.dropna(axis=1, how='all')
print(f"After dropping empty columns: {df_m.shape}")

print("\n" + "="*60)
print("MERGING MAYO CLINICAL DATA")
print("="*60)

df_clinical = pd.read_excel(RAW_INPUT["m_clinical"])
print(f"Clinical data shape: {df_clinical.shape}")

df_clinical = df_clinical.dropna(subset=['Image ID'])
df_clinical['Image ID'] = df_clinical['Image ID'].astype(int).astype(str)
print(f"After dropping empty rows: {df_clinical.shape}")

df_clinical['DGF'] = df_clinical['DGF'].map({'Y': 1, 'N': 0})

def compute_egfr_ckd_epi(row):
    scr = row['1yr SCr']
    age = row['Recipient  Age']
    sex = row['Recipient Sex']

    if pd.isna(scr) or pd.isna(age) or pd.isna(sex):
        return np.nan

    if sex == 'F':
        kappa = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        sex_factor = 1.0

    scr_kappa = scr / kappa
    egfr = 142 * (min(scr_kappa, 1) ** alpha) * (max(scr_kappa, 1) ** (-1.200)) * (0.9938 ** age) * sex_factor
    return egfr

df_clinical['eGFR_12M'] = df_clinical.apply(compute_egfr_ckd_epi, axis=1)

print(f"\neGFR_12M computed (CKD-EPI 2021):")
print(f"  Available: {df_clinical['eGFR_12M'].notna().sum()} / {len(df_clinical)}")
print(f"  Range: [{df_clinical['eGFR_12M'].min():.1f}, {df_clinical['eGFR_12M'].max():.1f}]")
print(f"  Mean: {df_clinical['eGFR_12M'].mean():.1f}")

print(f"\nDGF:")
print(f"  Available: {df_clinical['DGF'].notna().sum()} / {len(df_clinical)}")
print(f"  Distribution: {df_clinical['DGF'].value_counts().to_dict()}")

df_m = df_m.merge(
    df_clinical[['Image ID', 'eGFR_12M', 'DGF']],
    left_on='Subject_ID', right_on='Image ID', how='left'
)
df_m = df_m.drop(columns=['Image ID'])

print(f"\nMerged Mayo dataset: {df_m.shape}")
print(f"Subjects with eGFR_12M: {df_m['eGFR_12M'].notna().sum()}")
print(f"Subjects with DGF: {df_m['DGF'].notna().sum()}")

df_m.to_csv(RAW_DATA["m_subject_level"], index=False)
print(f"Saved: {RAW_DATA['m_subject_level']}")
