import pandas as pd
import numpy as np

def impute_missing_data(file_path, method='drop'):
    
    # Load data
    df = pd.read_csv(file_path)
    
    print(f"\nImputing missing data for: {file_path}")
    print(f"Original shape: {df.shape}")
    
    # Replace empty strings and whitespace with NaN first
    print("Converting empty strings/whitespace to NaN...")
    for col in df.columns:
        if col != 'Subject_ID':  # Don't modify ID column
            # Replace various missing value representations with NaN
            df[col] = df[col].replace(['', ' ', '  ', 'NA', 'N/A', 'nan', 'NaN', 'null'], np.nan)
            # Convert to numeric if possible (non-numeric becomes NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for missing values 
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) == 0:
        print("No missing values found.")
        # Save even if no changes
        df.to_csv(file_path, index=False)
        return
    
    print(f"Columns with missing values: {len(cols_with_missing)}")
    print(f"Total missing values: {missing_counts.sum()}")
    
    if method == 'drop':
        # Drop columns with any missing values (but NEVER drop Subject_ID or outcomes)
        protected_cols = ['Subject_ID', 'eGFR_12M', 'DGF']
        cols_to_drop = [col for col in cols_with_missing.index.tolist() 
                       if col not in protected_cols]
        
        if len(cols_to_drop) > 0:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns with missing values")
            
            # Show which columns were dropped
            if len(cols_to_drop) <= 5:
                for col in cols_to_drop:
                    print(f"  - {col}")
            else:
                for col in cols_to_drop[:5]:
                    print(f"  - {col}")
                print(f"  ... and {len(cols_to_drop) - 5} more")
        
        print(f"New shape: {df.shape}")
    
    elif method == 'mean':
        # Fill missing values with mean (only for numeric columns)
        for col in cols_with_missing.index:
            if df[col].dtype in ['float64', 'int64']:
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"Filled {col} with mean: {mean_val:.4f}")
            else:
                # For non-numeric, keep as-is or could drop
                print(f"Skipped non-numeric column: {col}")
        print(f"Final shape: {df.shape}")
    
    elif method == 'median':
        # Fill missing values with median (only for numeric columns)
        for col in cols_with_missing.index:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled {col} with median: {median_val:.4f}")
            else:
                # For non-numeric, keep as-is or could drop
                print(f"Skipped non-numeric column: {col}")
        print(f"Final shape: {df.shape}")
    
    else:
        raise ValueError(f"Unknown imputation method: {method}. Use 'drop', 'mean', or 'median'.")
    
    # Save back to same file
    df.to_csv(file_path, index=False)
    print(f"Saved imputed data to: {file_path}")
