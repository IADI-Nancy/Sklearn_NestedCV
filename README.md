# Sklearn_NestedCV
Implementation of a Sklearn wrapper to handle Nested CrossValidation. 
Additionnal general classes are provided to integrate dimensionality reduction and feature selection by filtering methods in the pipeline either by choosing one of the integrated method or by providing your own one. 
In a context of radiomics studies with data coming from different devices, some methods are provided to harmonize data.


## Package needed
- numpy
- sklearn
- pandas
- rpy2
- scipy
- imblearn
- skfeature
- scikit-optimize

R will be installed automatically in your conda/anaconda environment with rpy2 by using `conda install -c r rpy2`.
`xml` package from R should be installed manually using `conda install -c r r-xml`
Other missing R packages used in scripts will be downloaded automatically.
