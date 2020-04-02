hierarchical_clustering_leger = function(data, save_dir = NULL, na_var = 'N/A')
{
  #=== Installing missing packages ===
  package_list = c('pROC', 'xlsx', 'stringr')
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
  data <- as.matrix(sapply(data, as.numeric))
  # === Run hierarchical clustering ===
  hc <- hclust(as.dist(1 - abs(cor(data,method="spearman"))), method="complete")
  # === Cut tree ===
  clusters <- cutree(hc, h=0.1)
  clusters.df <- data.frame(clusterNumber = clusters)
  # === Feature coefficient for each cluster ===
  correlation_matrix <- cor(data,method="spearman")
  clusters.df$featureCoefficient <- 0
  for (n_k in 1:max(clusters.df$clusterNumber))
  {
    n <- length(which(clusters.df$clusterNumber == n_k))
    if (n != 1)
    {
      cluster_corr_matrix <- correlation_matrix[clusters.df$clusterNumber == n_k, clusters.df$clusterNumber == n_k]
      clusters.df$featureCoefficient[clusters.df$clusterNumber == n_k] <- ifelse(test = cluster_corr_matrix[, 1] < 0, -1/n, 1/n)
    }else{clusters.df$featureCoefficient[clusters.df$clusterNumber == n_k] <- 1}
  }
  if (is.null(save_dir) != TRUE)
  {
    # === Number of features in each cluster ===
    clusters_length = vector()
    for (n_k in 1:max(clusters.df$clusterNumber))
    {
      clusters_length[n_k] <- length(which(clusters.df$clusterNumber == n_k))
    }
    clusters_length <- as.data.frame(clusters_length)
    # === Cluster compactness ===
    cluster_compactness <- data.frame('cluster'=clusters.df$cluster)
    for (n_k in 1:max(clusters.df$clusterNumber))
    {
      cluster_feature_data <- data[, clusters.df$clusterNumber == n_k]
      pairwise_corr <- cor(cluster_feature_data, method = "spearman")
      diag(pairwise_corr) = NA
      cluster_compactness[n_k, 2] <- mean(pairwise_corr, na.rm = TRUE)
      cluster_compactness[n_k, 3] <- sd(pairwise_corr, na.rm = TRUE)
    }
    colnames(cluster_compactness) <- c('cluster', 'ClusterCorrelation_mean', 'ClusterCorrelation_std')
    # === Merge and save results ===
    merged_results <- merge(cluster_compactness, cluster_length)
    save_path = file.path(dirname(save_dir), 'Hierarchical_clustering_Leger', 'Cluster_analysis.xlsx')
    write.xlsx2(merged_results, save_path, row.names = FALSE, append = TRUE)
  }
  return(clusters.df)
}