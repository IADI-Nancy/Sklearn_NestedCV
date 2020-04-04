hierarchical_clustering_leger = function(data)
{
  #=== Installing missing packages ===
  package_list = c('pROC')
  for (package in package_list)
  {
    if (!require(package, character.only = TRUE)){install.packages(package, repos = 'https://cloud.r-project.org/')}
    library(package, character.only = TRUE)
  }
  #=== Preprocess data ===
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
  return(clusters.df)
}
