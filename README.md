ABSTRACT

We compared multiple harmonization methods including ComBat, CovBat, Z-score normalization, and CORAL, and found that ComBat-family methods significantly improved cross-cohort generalization, reducing prediction error by up to 40% compared to unharmonized data. ComBat uses empirical Bayes to adjust feature means and variances across batches while preserving biological covariates, making it well-suited for multi-site radiomics data where batch effects can obscure true clinical signal.

SETUP

 - create and activate python and R virtual environments 
 - install requirements.txt in python virtual environment
 - fill out config template with local data paths
 - run pipeline in order