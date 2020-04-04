# function to find medoid in cluster i
# https://www.biostars.org/p/11987/#11989
clust.medoid = function(i, distmat, clusters)
{
  ind = (clusters == i)
  names(which.min(rowSums( distmat[ind, ind] )))
}

hierarchical_clustering_parmar = function(data, max_k = 20, threshold = 0.1, seed=1)
{
  set.seed(1)
  #=== Installing missing packages ===
  if (!requireNamespace("BiocManager", quietly = TRUE)){install.packages("BiocManager", repos = 'https://cloud.r-project.org/')}
  if (!require('ConsensusClusterPlus')){BiocManager::install("ConsensusClusterPlus")}
  library('ConsensusClusterPlus')
  package_list = c('pROC')
  for (package in package_list)
  {
    if (!require(package, character.only = TRUE)){install.packages(package, repos = 'https://cloud.r-project.org/')}
    library(package, character.only = TRUE)
  }
  source('./R_scripts/ConsensusClusterPlus_utils.R')
  #=== Preprocess data ===
  data <- as.matrix(sapply(data, as.numeric))
  #=== Run consensus hierarchical clustering ===
  results <- ConsensusClusterPlus(data, maxK = max_k, reps = 10000, pItem = 0.8, pFeature = 1,
                                  clusterAlg = "hc", distance = "spearman", plot = FALSE, 
                                  innerLinkage = 'ward.D2', finalLinkage = 'ward.D2', seed = seed)
  icl <- calcICL(results, plot=FALSE)
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
  return(consensus_class_best_k, merged_results)
}

