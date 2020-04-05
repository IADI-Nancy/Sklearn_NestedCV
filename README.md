# Sklearn_NestedCV
Implementation of a Sklearn wrapper to handle Nested CrossValidation. 
Additionnal general classes are provided to integrate dimensionality reduction and feature selection by filtering methods in the pipeline either by choosing one of the integrated method or by providing your own one. 
In a context of radiomics studies with data coming from different devices, some methods are provided to harmonize data.


## Package needed
-numpy
-sklearn
-pandas
-rpy2
-scipy
-imblearn
-R

R can be installed automatically in your conda/anaconda environment by using `conda install -c r rpy2`.
Missing R packages used in scripts will be downloaded automatically
