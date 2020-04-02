univariate_analysis = function(data, adjusted_method = 'BH', na_var = 'N/A')
{
  #=== Installing missing packages ===
  package_list = c('pROC', 'stringr')
  for (package in package_list)
  {
    if (!require(package, character.only = TRUE)){install.packages(package, repos = 'https://cloud.r-project.org/')}
    library(package, character.only = TRUE)
  }
  #=== Preprocess data ===
  # Delete patients with N/A
  NA_indices = which(apply(data, 1, function(x)str_detect(paste(x, collapse=""), paste(na_var, collapse="")))==TRUE)
  if (length(NA_indices) != 0)
  {
    data <- data[-NA_indices, ]
  }
  output <- data$Output
  data <- as.data.frame(sapply(data[, !colnames(data) == 'Output'], as.numeric))
  output <- as.factor(output)
  #=== Statistical analysis ===
  pvalue_list = list()
  diag_perf = data.frame()
  rets <- c("specificity", "sensitivity", "accuracy")
  for (n_feature in 1:dim(data)[2])
  {
    if (length(unique(output))==2)
    {
      test_results <- wilcox.test(data[[n_feature]] ~ output)
    }else{test_results <- kruskal.test(data[[n_feature]] ~ output)}
    pvalue_list[[n_feature]] = test_results$p.value
    rocc <- roc(output, data[[n_feature]], quiet = TRUE)
    ROC_results <- coords(rocc, x="best", input = "threshold", ret=rets, best.method = "closest.topleft", transpose = TRUE)
    diag_perf[n_feature, 1] = as.numeric(rocc$auc)
    if (is.null(dim(ROC_results)) != TRUE)
    {
      diag_perf[n_feature, 2] = ROC_results['sensitivity', ][ROC_results['accuracy', ] == max(ROC_results['accuracy', ])]
      diag_perf[n_feature, 3] = ROC_results['specificity', ][ROC_results['accuracy', ] == max(ROC_results['accuracy', ])]
      diag_perf[n_feature, 4] = ROC_results['accuracy', ][ROC_results['accuracy', ] == max(ROC_results['accuracy', ])]
    }
    else
    {
      diag_perf[n_feature, 2] = ROC_results[['sensitivity']]
      diag_perf[n_feature, 3] = ROC_results[['specificity']]
      diag_perf[n_feature, 4] = ROC_results[['accuracy']]
    }
  }
  adjusted_pvalue_list <- p.adjust(pvalue_list, method = adjusted_method)
  colnames(diag_perf) <- c("AUC", "Specificity", "Sensitivity", "Accuracy")
  final_df = data.frame("pvalue" = unlist(pvalue_list), "adjusted p_value"=unlist(adjusted_pvalue_list),"AUC"=diag_perf$AUC,
                        "Sensitivity"=diag_perf$Sensitivity, "Specificity"=diag_perf$Specificity, "Accuracy"=diag_perf$Accuracy,
                        row.names = colnames(data))
  return(final_df)
}
