# function to find medoid in cluster i
# https://www.biostars.org/p/11987/#11989
clust.medoid = function(i, distmat, clusters)
{
  ind = (clusters == i)
  names(which.min(rowSums( distmat[ind, ind] )))
}

hierarchical_clustering_parmar = function(data, max_k = 20, threshold = 0.1, seed=1, save_dir = NULL, na_var = 'N/A')
{
  set.seed(1)
  #=== Installing missing packages ===
  if (!requireNamespace("BiocManager", quietly = TRUE)){install.packages("BiocManager", repos = 'https://cloud.r-project.org/')}
  if (!require('ConsensusClusterPlus')){BiocManager::install("ConsensusClusterPlus")}
  library('ConsensusClusterPlus')
  package_list = c('pROC', 'xlsx', 'stringr')
  for (package in package_list)
  {
    if (!require(package, character.only = TRUE)){install.packages(package, repos = 'https://cloud.r-project.org/')}
    library(package, character.only = TRUE)
  }
  source('./R_scripts/ConsensusClusterPlus_utils.R')
  #=== Preprocess data ===
  # Delete patients with N/A
  NA_indices = which(apply(data, 1, function(x)str_detect(paste(x, collapse=""), paste(na_var, collapse="")))==TRUE)
  if (length(NA_indices) != 0)
  {
    data <- data[-NA_indices, ]
  }
  data <- as.matrix(sapply(data, as.numeric))
  #=== Run consensus hierarchical clustering ===
  if (is.null(save_dir) != TRUE)
  {
    setwd(save_dir)
    results <- ConsensusClusterPlus(data, maxK = max_k, reps = 10000, pItem = 0.8, pFeature = 1,
                                    clusterAlg = "hc", distance = "spearman", plot = "png", 
                                    innerLinkage = 'ward.D2', finalLinkage = 'ward.D2', seed = seed,
                                    title = paste(molecular_feature, ROI, 'Spearman_wardD2', sep = '_'))
    icl <- calcICL(results, plot='png', title = paste('Spearman_wardD2', sep = '_'))
  }else
  {
    results <- ConsensusClusterPlus(data, maxK = max_k, reps = 10000, pItem = 0.8, pFeature = 1,
                                    clusterAlg = "hc", distance = "spearman", plot = FALSE, 
                                    innerLinkage = 'ward.D2', finalLinkage = 'ward.D2', seed = seed)
      icl <- calcICL(results, plot=FALSE)
  }
  clusterConsensus <- as.data.frame(icl[['clusterConsensus']])
  # === Find the best number of cluster ===
  # Minimum to explore is found thanks to delta area plot obtain with ConsensusClusterPlus function
  n=1
  ml = vector(mode = 'list', max_k-1)
  for (i in results[-1])
  {
    ml[[n]] <- i$ml
    n = n+1
  }
  deltaK <- CDF(ml)
  diff_vector <- abs(diff(deltaK))
  min_k = min(which(diff_vector <= threshold))
  
  median_consensus <- data.frame()
  n_rows = 1
  for (n_k in min_k:max_k)  
  {
    median_consensus[n_rows, 1] <- n_k
    median_consensus[n_rows, 2] <- median(clusterConsensus$clusterConsensus[clusterConsensus$k == n_k], na.rm = FALSE)
    n_rows = n_rows + 1
  }
  colnames(median_consensus) <- c('clusterNumber', 'medianConsensus')
  best_k = median_consensus$clusterNumber[median_consensus$medianConsensus == max(median_consensus$medianConsensus, na.rm = TRUE)]
  best_k <- best_k[!is.na(best_k)]
  if(length(best_k) > 1){best_k <- min(best_k)}
  consensus_matrix_best_k <- results[[best_k]][["consensusMatrix"]]
  consensus_class_best_k <- as.data.frame(results[[best_k]][['consensusClass']])
  colnames(consensus_class_best_k) <- c('clusterNumber')
  # === Feature coefficient for each cluster ===
  dissimilarity_matrix <- 1 - cor(data, method = "spearman")
  consensus_class_best_k$featureCoefficient <- 0
  for (n_k in 1:best_k)
  {
    cluster_medoids <- clust.medoid(n_k, dissimilarity_matrix, consensus_class_best_k[, 1])
    consensus_class_best_k[cluster_medoids, 2] <- 1
  }
  if (is.null(save_dir) != TRUE)
  {
    # === Number of features in each cluster ===
    clusters_length = vector()
    for (n_k in 1:best_k)
    {
      clusters_length[n_k] <- length(consensus_class_best_k$clusterNumber[consensus_class_best_k$clusterNumber == n_k])
    }
    cluster_length <- data.frame('cluster'=cluster_consensus_best_k$cluster,
                                 'clusterLength'=clusters_length)
    # === Cluster stability ===
    cluster_consensus_best_k <- clusterConsensus[clusterConsensus$k == best_k, ][-1]
    for (n_k in 1:best_k)
    {
      consensus_matrix_n_k <- consensus_matrix_best_k[consensus_class_best_k$clusterNumber == n_k, consensus_class_best_k$clusterNumber == n_k]
      diag(consensus_matrix_n_k) = NA
      cluster_consensus_best_k[n_k, 3] <- sd(consensus_matrix_n_k, na.rm = TRUE)
    }
    colnames(cluster_consensus_best_k) <- c(colnames(cluster_consensus_best_k)[-3], 'clusterConsensusStd')
    row.names(cluster_consensus_best_k) <- seq(1, best_k)
    # === Cluster compactness ===
    cluster_compactness_best_k <- data.frame('cluster'=cluster_consensus_best_k$cluster)
    for (n_k in 1:best_k)
    {
      cluster_feature_data <- data[, consensus_class_best_k$clusterNumber == n_k]
      pairwise_corr <- cor(cluster_feature_data, method = "spearman")
      diag(pairwise_corr) = NA
      cluster_compactness_best_k[n_k, 2] <- mean(pairwise_corr, na.rm = TRUE)
      cluster_compactness_best_k[n_k, 3] <- sd(pairwise_corr, na.rm = TRUE)
    }
    colnames(cluster_compactness_best_k) <- c('cluster', 'ClusterCorrelation_mean', 'ClusterCorrelation_std')
    # === Merge and save cluster analysis results ===
    merged_results <- merge(cluster_consensus_best_k, cluster_compactness_best_k)
    merged_results <- merge(merged_results, cluster_length)
    save_path = file.path(dirname(save_dir), 'Hierarchical_clustering_Parmar', 'Cluster_analysis.xlsx')
    write.xlsx2(merged_results, save_path, row.names = FALSE)
  }
  return(consensus_class_best_k)
}

