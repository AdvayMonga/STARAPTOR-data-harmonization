## ABSTRACT

We compared multiple harmonization methods including ComBat, CovBat, Z-score normalization, and CORAL, and found that ComBat-family methods significantly improved cross-cohort generalization, reducing prediction error by up to 40% compared to unharmonized data. ComBat uses empirical Bayes to adjust feature means and variances across batches while preserving biological covariates, making it well-suited for multi-site radiomics data where batch effects can obscure true clinical signal.

## NEW STRATEGY - LEAVE ONE OUT

    1. Pick 2/3 cohorts for training and one for testing
    2. Use ComBat on the training cohort to estimate the ComBat model parameters, harmonizing the training features 
    3. Train ML models on the training data only
    4. Using the ComBat parameters learned on the training data, harmonize the testing data - no ComBat model fitting here, but just using learned ComBat parameters from the training data to adjust the test data
    5. Test ML models on the test data
    6 Record results for all 3 combinations of 2 training cohorts with 1 test cohort 

    RESULTS

LOO ComBat consistently improves predictive accuracy over unharmonized baselines across all models and both outcomes.

### eGFR Test MSE (lower = better, averaged across 3 LOO folds)

| Model | Raw (Unharmonized) | LOO ComBat | Harm→Raw |
|-------|-------------------|------------|----------|
| Elastic Net | 864 | **482** | 3,737 |
| Lasso | 934 | **693** | 21,973 |
| Random Forest | 458 | **420** | 511 |
| Ridge | 1,599 | **900** | 57,928 |
| XGBoost | 499 | **372** | 534 |

### DGF Test AUC (higher = better, averaged across 3 LOO folds)

| Model | Raw (Unharmonized) | LOO ComBat | Harm→Raw |
|-------|-------------------|------------|----------|
| Elastic Net | 0.531 | **0.581** | 0.453 |
| Lasso | 0.532 | **0.591** | 0.452 |
| Random Forest | 0.561 | **0.713** | 0.581 |
| Ridge | 0.522 | **0.571** | 0.433 |
| XGBoost | 0.605 | **0.829** | 0.544 |

### Key Findings

- LOO ComBat improves every model for both eGFR regression and DGF classification
- XGBoost + LOO ComBat is the best combination (avg MSE 372, AUC 0.829)
- DGF AUC gains are substantial: +37% for XGBoost, +27% for Random Forest
- Harm→Raw (harmonized train, raw test) performs worse than no harmonization, confirming that harmonization must be applied to both train and test data at inference time



## SETUP

 - create and activate python and R virtual environments 
 - install requirements.txt in python virtual environment
 - fill out config template with local data paths
 - run pipeline in order