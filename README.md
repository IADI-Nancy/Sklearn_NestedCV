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
- stability-selection
- BorutaShap
- neuroCombat

R will be installed automatically in your conda/anaconda environment with rpy2 by using `conda install -c r rpy2`.
`xml` package from R should be installed manually using `conda install -c r r-xml`
Other missing R packages used in scripts will be downloaded automatically.

If you use this wrapper in a scientific publication, we would appreciate citations to the following paper :
```
@article{zaragori_18_2021,
	title = {$^{\textrm{18}}$ {F}-{FDOPA} {PET} for the non-invasive prediction of glioma molecular parameters: a radiomics study},
	issn = {0161-5505, 2159-662X},
	shorttitle = {$^{\textrm{18}}$ {F}-{FDOPA} {PET} for the non-invasive prediction of glioma molecular parameters},
	url = {http://jnm.snmjournals.org/lookup/doi/10.2967/jnumed.120.261545},
	doi = {10.2967/jnumed.120.261545},
	language = {en},
	urldate = {2021-05-21},
	journal = {Journal of Nuclear Medicine},
	author = {Zaragori, Timothée and Oster, Julien and Roch, Veronique and Hossu, Gabriela and Chawki, Mohammad Bilal and Grignon, Rachel and Pouget, Celso and Gauchotte, Guillaume and Rech, Fabien and Blonski, Marie and Taillandier, Luc and Imbert, Laëtitia and Verger, Antoine},
	month = may,
	year = {2021},
	pages = {jnumed.120.261545},
} ```
