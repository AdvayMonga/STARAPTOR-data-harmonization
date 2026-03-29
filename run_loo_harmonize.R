setwd("/Users/advaymonga/Desktop/STARAPTOR")

library(devtools)
if (!require("ComBatFamQC", quietly = TRUE)) {
  install_github("Zheng206/ComBatFamQC", build_vignettes = TRUE)
}
library(ComBatFamQC)
library(dplyr)

# Load data
df_u <- read.csv("data/u_image_features.csv")
df_c <- read.csv("data/c_image_features.csv")
df_m <- read.csv("data/m_image_features.csv")

feature_cols <- setdiff(colnames(df_u), c("Subject_ID", "eGFR_12M", "DGF"))
covariates   <- c("eGFR_12M", "DGF")

cat("M cohort dimensions:", dim(df_m), "\n")
cat("Feature columns match U:", all(feature_cols %in% colnames(df_m)), "\n")

# LOO helper: fit ComBat ONLY on training cohorts, then apply to held-out test cohort
run_loo_combat <- function(train_list, train_labels, df_test, test_label,
                           feature_cols, covariates, out_dir) {

  # --- Build training data frame ---
  train_dfs <- mapply(function(df, lbl) {
    df_prep       <- df[, c("Subject_ID", feature_cols, covariates)]
    df_prep$batch <- lbl
    df_prep
  }, train_list, train_labels, SIMPLIFY = FALSE)

  df_train_combined <- do.call(rbind, train_dfs)
  cat("  Training subjects:", nrow(df_train_combined),
      "(", paste(train_labels, collapse = "+"), ")\n")

  # --- Step 1: Fit ComBat on training cohorts ONLY ---
  combat_fit <- combat_harm(
    df         = df_train_combined,
    features   = feature_cols,
    batch      = "batch",
    covariates = covariates,
    type       = "lm",
    eb         = TRUE,
    quiet      = TRUE
  )
  harmonized_train <- combat_fit$harmonized_df
  cat("  ComBat fit on training cohorts complete.\n")

  # --- Step 2: Apply learned parameters to held-out test cohort (no refitting) ---
  df_test_prep       <- df_test[, c("Subject_ID", feature_cols, covariates)]
  df_test_prep$batch <- test_label

  test_result    <- combat_harm(df = df_test_prep, predict = TRUE,
                                object = combat_fit$combat.object)
  harmonized_test <- test_result$harmonized_df
  cat("  Test cohort harmonized via predict():", nrow(harmonized_test), "subjects.\n")

  # --- Save ---
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  train_out <- harmonized_train[, !(names(harmonized_train) %in% "batch")]
  test_out  <- harmonized_test[,  !(names(harmonized_test)  %in% "batch")]

  write.csv(train_out, file.path(out_dir, "train_harmonized.csv"), row.names = FALSE)
  write.csv(test_out,  file.path(out_dir, "test_harmonized.csv"),  row.names = FALSE)
  cat("  Saved to", out_dir, "\n")
  invisible(list(train = harmonized_train, test = harmonized_test))
}

# Fold 1: UC Davis + Coimbra -> Mayo
cat(strrep("=", 60), "\n")
cat("LOO Fold 1: Train UC Davis + Coimbra  ->  Test Mayo\n")
cat(strrep("=", 60), "\n")
run_loo_combat(
  train_list   = list(df_u, df_c),
  train_labels = c("U", "C"),
  df_test      = df_m,
  test_label   = "M",
  feature_cols = feature_cols,
  covariates   = covariates,
  out_dir      = "data/loo_combat/UC_to_M"
)
gc(verbose = FALSE)

# Fold 2: UC Davis + Mayo -> Coimbra
cat(strrep("=", 60), "\n")
cat("LOO Fold 2: Train UC Davis + Mayo  ->  Test Coimbra\n")
cat(strrep("=", 60), "\n")
run_loo_combat(
  train_list   = list(df_u, df_m),
  train_labels = c("U", "M"),
  df_test      = df_c,
  test_label   = "C",
  feature_cols = feature_cols,
  covariates   = covariates,
  out_dir      = "data/loo_combat/UM_to_C"
)
gc(verbose = FALSE)

# Fold 3: Coimbra + Mayo -> UC Davis
cat(strrep("=", 60), "\n")
cat("LOO Fold 3: Train Coimbra + Mayo  ->  Test UC Davis\n")
cat(strrep("=", 60), "\n")
run_loo_combat(
  train_list   = list(df_c, df_m),
  train_labels = c("C", "M"),
  df_test      = df_u,
  test_label   = "U",
  feature_cols = feature_cols,
  covariates   = covariates,
  out_dir      = "data/loo_combat/CM_to_U"
)
gc(verbose = FALSE)

cat("\nLOO ComBat harmonization complete.\n")
