######    M-ComBat   ###########################################################################################
## 
##   Overview:  Script necessary to perform M-ComBat to transform GEP data to
##              a pre-determined, 'gold-standard' subset of samples.
##
##   Requirements:
##    Load 'sva' package
##
##   Input:
##     'dat' = p by n data.frame or matrix , genomic measure matrix 
##                 ( dimensions: probe by sample )
##     'batch' = numeric vector of batch association, length n
##     'center' = numeric value of 'gold-standard' batch
##     'mod' = model matrix of potential covariates
##     'numCovs' = column number of variables in 'mod' to be treated as continuous variables 
##                   (otherwise all covariates treated as factors)
##
#############################################################################################################
M.COMBAT <- function (dat, batch, center , mod, numCovs = NULL ) 
{
  batch <- as.factor(batch)
  batchmod <- model.matrix(~-1 + batch)
  cat("Found", nlevels(batch), "batches\n")
  n.batch <- nlevels(batch)
  batches <- list()
  for (i in 1:n.batch) {
    batches[[i]] <- which(batch == levels(batch)[i])
  }
  n.batches <- sapply(batches, length)
  n.array <- sum(n.batches)
  design <- cbind(batchmod, mod)
  check <- apply(design, 2, function(x) all(x == 1))
  design <- as.matrix(design[, !check])
  n.batches <- sapply(batches, length)
  n.array <- sum(n.batches)
  NAs = any(is.na(dat))
  if (NAs) {
    cat(c("Found", sum(is.na(dat)), "Missing Data Values\n"), 
        sep = " ")
    stop()
  }
  cat("Standardizing Data across genes\n")
  
  B.hat <- solve(t(design) %*% design) %*% t(design) %*% t(as.matrix(dat))
  
  # variance of batch of interest
  var.batch <- apply(dat[, batch==center], 1, var)
  var.pooled <- ((dat - t(design %*% B.hat))^2) %*% rep(1/n.array, n.array)
  
  grand.mean <- t(n.batches/n.array) %*% B.hat[1:n.batch, ]
  stand.mean <- t(grand.mean) %*% t(rep(1, n.array))
  
  # accounts for covariates here
  if (!is.null(design)) {
    tmp <- design
    tmp[, c(1:n.batch)] <- 0
    stand.mean <- stand.mean + t(tmp %*% B.hat)}
  
  # standardized data
  s.data <- (dat - stand.mean)/(sqrt(var.pooled) %*% t(rep(1, n.array)))
  
  cat("Fitting L/S model and finding priors\n")
  batch.design <- design[, 1:n.batch]
  
  gamma.hat <- solve(t(batch.design) %*% batch.design) %*% t(batch.design) %*% t(as.matrix(s.data))
  
  delta.hat <- NULL
  for (i in batches) {
    delta.hat <- rbind(delta.hat, apply(s.data[, i], 1, var,  na.rm = T)) }
  
  gamma.bar <- apply(gamma.hat, 1, mean)
  t2 <- apply(gamma.hat, 1, var)
  a.prior <- apply(delta.hat, 1, sva:::aprior)
  b.prior <- apply(delta.hat, 1, sva:::bprior)
  
  gamma.star <- delta.star <- NULL
  
  cat("Finding parametric adjustments\n")
  for (i in 1:n.batch) {
    
    temp <- sva:::it.sol(s.data[, batches[[i]]], gamma.hat[i,], delta.hat[i, ], gamma.bar[i], t2[i], a.prior[i], b.prior[i])
    
    gamma.star <- rbind(gamma.star, temp[1, ])
    delta.star <- rbind(delta.star, temp[2, ])
  }
  
  cat("Adjusting the Data\n")
  bayesdata <- s.data
  j <- 1
  for (i in batches) {
    bayesdata[, i] <- (bayesdata[, i] - t(batch.design[i,] %*% gamma.star))/(sqrt(delta.star[j, ]) %*% t(rep(1, n.batches[j])))
    j <- j + 1
  }
  
  bayesdata <- (bayesdata * (sqrt(var.batch) %*% t(rep(1, n.array)))) + matrix( B.hat[center,] , nrow(dat) , ncol(dat))
  
  return(bayesdata)
}

MComBat_harmonization = function(data, covariate, batch, ref_batch, numCovs, na_var = 'N/A')
{
  #=== Installing missing packages ===
  if (!requireNamespace("BiocManager", quietly = TRUE)){install.packages("BiocManager", repos = 'https://cloud.r-project.org/')}
  package_list = c("RCurl", "sva")
  for (package in package_list)
  {
    print(sprintf('%s', package))
    if (!require(package, character.only = TRUE)){BiocManager::install(package)}
    library(package, character.only = TRUE)
  }
  package_list = c('stringr')
  for (package in package_list)
  {
    print(sprintf('%s', package))
    if (!require(package, character.only = TRUE)){install.packages(package, repos = 'https://cloud.r-project.org/')}
    library(package, character.only = TRUE)
  }
  #=== Preprocess data ===
  # Delete patients with N/A
  NA_indices = which(apply(data, 1, function(x)str_detect(paste(x, collapse=""), paste(na_var, collapse="")))==TRUE)
  if (length(NA_indices) != 0)
  {
    NA_rows <- data[NA_indices, ]
    data <- data[-NA_indices, ]
    covariate <- covariate[-NA_indices, ]
  }
  data <- as.data.frame(sapply(data, as.numeric))
  data <- t(data)
  ref_batch <- seq(length(unique(batch)))[unique(batch) == ref_batch]
  batch <- factor(x = batch, levels = unique(batch), labels = seq(length(unique(batch))))
  mod <- model.matrix(~ ., data = as.data.frame(covariate))
  #=== ComBat ===
  RES <- M.COMBAT(data, batch, ref_batch, mod, numCovs)
  Combat_data <- as.data.frame(t(RES))
  if (length(NA_indices) != 0)
  {
    Combat_data <- rbind(Combat_data, NA_rows)
  }
  return(Combat_data)
}
        



