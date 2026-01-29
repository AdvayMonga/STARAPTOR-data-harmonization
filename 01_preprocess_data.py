import pandas as pd
import numpy as np

# Load data
print("\n" + "="*60)
print("LOADING RAW DATA")
print("\n" + "="*60)

df_u = pd.read_csv('/Users/advaymonga/Downloads/ucd_preprocessed_data.csv')
df_c = pd.read_csv('/Users/advaymonga/Downloads/Reduced_Features_NO_COLOR (1).csv')

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


# Image features = all numeric columns except outcomes
image_cols = [col for col in numeric_cols if col not in outcome_cols]

print(f"{len(image_cols)} image feature columns")

# Aggregation dictionary
agg_dict = {col: 'mean' for col in image_cols}
agg_dict.update({col: 'first' for col in outcome_cols})

# Aggregate by patient
df_c_subject = df_c.groupby('Recipient_code').agg(agg_dict).reset_index()

# Remove rows with empty Recipient_code
df_c_subject = df_c_subject[df_c_subject['Recipient_code'].notna() & (df_c_subject['Recipient_code'] != '')]

# Remove empyty columns
df_c_subject = df_c_subject.dropna(axis=1, how='all')

print(f"\nBefore aggregation: {df_c.shape}")
print(f"After aggregation: {df_c_subject.shape}")
print(f"Subjects: {df_c_subject['Recipient_code'].nunique()}")

# Save processed data
print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)

df_u.to_csv('data/u_subject_level.csv', index=False)
df_c_subject.to_csv('data/c_subject_level.csv', index=False)

print("Saved: data/u_subject_level.csv")
print("Saved: data/c_subject_level.csv")
