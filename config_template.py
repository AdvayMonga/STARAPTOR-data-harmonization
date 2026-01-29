"""
Configuration TEMPLATE for STARAPTOR project
Copy this file to config.py and update the paths for your local machine
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ============================================================
# RAW INPUT FILES - Update these paths for your local machine
# ============================================================
RAW_INPUT = {
    "u_raw": Path("/path/to/your/ucd_preprocessed_data.csv"),
    "c_raw": Path("/path/to/your/Reduced_Features_NO_COLOR.csv"),
}

# Processed data files (in data/ directory)
RAW_DATA = {
    "c_image_features": DATA_DIR / "c_image_features.csv",
    "c_subject_level": DATA_DIR / "c_subject_level.csv",
    "u_image_features": DATA_DIR / "u_image_features.csv",
    "u_subject_level": DATA_DIR / "u_subject_level.csv",
}

# Harmonized data directories
HARMONIZED_DIRS = {
    "combat": {
        "c": DATA_DIR / "harmonized_c",
        "u": DATA_DIR / "harmonized_u",
        "combined": DATA_DIR / "harmonized_combined",
    },
    "covbat": {
        "c": DATA_DIR / "covbat_harmonized_c",
        "u": DATA_DIR / "covbat_harmonized_u",
        "combined": DATA_DIR / "covbat_harmonized_combined",
    },
    "zscore": {
        "c": DATA_DIR / "zscore_harmonized_c",
        "u": DATA_DIR / "zscore_harmonized_u",
        "combined": DATA_DIR / "zscore_harmonized_combined",
    },
    "ravel": {
        "c": DATA_DIR / "ravel_harmonized_c",
        "u": DATA_DIR / "ravel_harmonized_u",
        "combined": DATA_DIR / "ravel_harmonized_combined",
    },
    "coral": {
        "c": DATA_DIR / "coral_harmonized_c",
        "u": DATA_DIR / "coral_harmonized_u",
        "combined": DATA_DIR / "coral_harmonized_combined",
    },
    "unharmonized": {
        "combined": DATA_DIR / "unharmonized_combined",
    },
}

# Model training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature selection
MRMR_K_EGFR = 25
MRMR_K_DGF = 50

# Outcomes
OUTCOMES = {
    "regression": "eGFR_12M",
    "classification": "DGF",
}
