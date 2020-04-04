# Source : https://github.com/Jfortin1/ComBatHarmonization/tree/master/R
ComBat_harmonization = function(data, covariate, batch, parametric, empirical_bayes)
{
  #=== Installing missing packages ===
  package_list = c('matrixStats')
  for (package in package_list)
  {
    if (!require(package, character.only = TRUE)){install.packages(package, repos = 'https://cloud.r-project.org/')}
    library(package, character.only = TRUE)
  }
  source("./R_scripts/scripts/utils.R")
  source("./R_scripts/scripts/combat.R")
  #=== Preprocess data ===
  data <- as.data.frame(sapply(data, as.numeric))
  data <- t(data)
  batch <- factor(x = batch, levels = unique(batch), labels = seq(length(unique(batch))))
  mod <- model.matrix(~ ., data = as.data.frame(covariate))
  #=== ComBat ===
  data.harmonized <- combat(dat=data, batch=batch, parametric=parametric, mod=mod, eb=empirical_bayes)
  Combat_data <- as.data.frame(t(data.harmonized$dat.combat))
  return(Combat_data)
}



