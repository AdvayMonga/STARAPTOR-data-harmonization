# STARAPTOR

ML pipeline for predicting kidney transplant outcomes across multi-site cohorts using batch harmonization.

## Overview

Donor renal biopsy whole-slide images (WSIs) are processed into quantitative pathomics features across three clinical sites — UC Davis, Coimbra, and Mayo Clinic. Cross-site batch effects from differences in scanners, tissue processing, and staining protocols are corrected via harmonization, enabling predictive models that generalize across institutions.

**Outcomes predicted:**
- **eGFR** — estimated glomerular filtration rate at 12 months (regression)
- **DGF** — delayed graft function (binary classification)

**Models:** Elastic Net, Lasso, Ridge, Random Forest, XGBoost

## Harmonization Method Comparison

Evaluated 5 harmonization methods on pooled data against an unharmonized baseline.

### eGFR Test MSE (lower = better)

| Model | Unharmonized | Z-Score | RAVEL | CORAL | CovBat | ComBat |
|-------|-------------|---------|-------|-------|--------|--------|
| Elastic Net | 406 | 367 | 505 | 407 | 316 | **314** |
| Lasso | 542 | 456 | 790 | 440 | 349 | **360** |
| Random Forest | 343 | 348 | 356 | 390 | **242** | 252 |
| Ridge | 804 | 582 | 1,222 | 521 | **352** | 359 |
| XGBoost | 353 | 384 | 339 | 414 | **210** | 239 |

### DGF Test AUC (higher = better)

| Model | Unharmonized | Z-Score | RAVEL | CORAL | CovBat | ComBat |
|-------|-------------|---------|-------|-------|--------|--------|
| Elastic Net | 0.685 | 0.571 | 0.557 | 0.550 | 0.811 | **0.813** |
| Lasso | 0.692 | 0.599 | 0.570 | 0.571 | 0.807 | **0.813** |
| Random Forest | 0.714 | 0.663 | 0.657 | 0.704 | 0.852 | **0.866** |
| Ridge | 0.660 | 0.524 | 0.550 | 0.519 | 0.804 | **0.805** |
| XGBoost | 0.699 | 0.684 | 0.700 | 0.687 | 0.936 | **0.961** |

**Key findings:**
- ComBat and CovBat consistently outperform all other methods
- Z-Score, RAVEL, and CORAL degrade DGF classification vs. unharmonized
- Best result: XGBoost + ComBat — DGF AUC 0.961, eGFR MSE 239

## Leave-One-Out (LOO) Cross-Cohort Validation

Tests whether harmonization generalizes to unseen clinical sites:

1. Train on 2 of 3 cohorts, hold out the 3rd as test
2. Fit ComBat on training cohorts to learn harmonization parameters
3. Train ML models on harmonized training data
4. Apply learned ComBat parameters to held-out test cohort (no refitting)
5. Evaluate on harmonized test cohort
6. Repeat for all 3 LOO combinations

### eGFR Test MSE (averaged across 3 LOO folds)

| Model | Unharmonized | LOO ComBat |
|-------|-------------|------------|
| Elastic Net | 863 | **482** |
| Lasso | 934 | **693** |
| Random Forest | 458 | **420** |
| Ridge | 1,599 | **900** |
| XGBoost | 499 | **372** |

### DGF Test AUC (averaged across 3 LOO folds)

| Model | Unharmonized | LOO ComBat |
|-------|-------------|------------|
| Elastic Net | 0.532 | **0.581** |
| Lasso | 0.532 | **0.591** |
| Random Forest | 0.561 | **0.713** |
| Ridge | 0.522 | **0.571** |
| XGBoost | 0.605 | **0.829** |

**Key findings:**
- LOO ComBat improves every model for both outcomes
- XGBoost + LOO ComBat is the best combination (MSE 372, AUC 0.829)
- DGF AUC gains: +37% for XGBoost, +27% for Random Forest
- Harm→Raw (harmonized train, raw test) performs worse than unharmonized, confirming harmonization must be applied at inference time

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_preprocess_data.py` | Load raw data, aggregate to subject level, compute outcomes |
| 2 | `02_prepare_features.py` | Match 165 features across cohorts, align naming, impute |
| 2.5 | `02.5_alt_harm_methods.py` | Z-Score, Quantile, CORAL harmonization |
| 2.5 | `02.5_harmonize.Rmd` | ComBat + CovBat harmonization (R, ComBatFamQC) |
| 3 | `03_loo_combat.py` | LOO ComBat harmonization + model training |
| 3 | `03_train_models.py` | Train models across all harmonization scenarios |
| 3.5 | `03.5_mrmr_feature_selection.py` | mRMR feature selection optimization |
| 4 | `04_process_results.py` | Aggregate results into summary tables |
| 5 | `05_visualize.py` | Generate all figures |

## Setup

1. Create and activate Python and R virtual environments
2. Install `requirements.txt` in Python virtual environment
3. Copy `config_template.py` to `config.py` and fill in local data paths
4. Run pipeline scripts in order
