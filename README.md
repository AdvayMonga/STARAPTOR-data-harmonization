## ABSTRACT

We compared multiple harmonization methods including ComBat, CovBat, Z-score normalization, and CORAL, and found that ComBat-family methods significantly improved cross-cohort generalization, reducing prediction error by up to 40% compared to unharmonized data. ComBat uses empirical Bayes to adjust feature means and variances across batches while preserving biological covariates, making it well-suited for multi-site radiomics data where batch effects can obscure true clinical signal.

## NEW STRATEGY - LEAVE ONE OUT

    1. Pick 2/3 cohorts for training and one for testing
    2. Use ComBat on the training cohort to estimate the ComBat model parameters, harmonizing the training features 
    3. Train ML models on the training data only
    4. Using the ComBat parameters learned on the training data, harmonize the testing data - no ComBat model fitting here, but just using learned ComBat parameters from the training data to adjust the test data
    5. Test ML models on the test data
    6 Record results for all 3 combinations of 2 training cohorts with 1 test cohort 

## SETUP

 - create and activate python and R virtual environments 
 - install requirements.txt in python virtual environment
 - fill out config template with local data paths
 - run pipeline in order