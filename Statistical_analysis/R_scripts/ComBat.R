ComBat_harmonization = function(data, covariate, batch, parametric, empirical_bayes, na_var = 'N/A')
{
  #=== Installing missing packages ===
  package_list = c('stringr', 'matrixStats')
  for (package in package_list)
  {
    if (!require(package, character.only = TRUE)){install.packages(package, repos = 'https://cloud.r-project.org/')}
    library(package, character.only = TRUE)
  }
  source("./R_scripts/scripts/utils.R")
  source("./R_scripts/scripts/combat.R")
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
  batch <- factor(x = batch, levels = unique(batch), labels = seq(length(unique(batch))))
  mod <- model.matrix(~ ., data = as.data.frame(covariate))
  #=== ComBat ===
  data.harmonized <- combat(dat=data, batch=batch, parametric=parametric, mod=mod, eb=empirical_bayes)
  Combat_data <- as.data.frame(t(data.harmonized$dat.combat))
  if (length(NA_indices) != 0)
  {
    Combat_data <- rbind(Combat_data, NA_rows)
  }
  return(Combat_data)
}



