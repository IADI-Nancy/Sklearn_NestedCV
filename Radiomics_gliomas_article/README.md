The code provided herein allow to reproduce the results found in the following article:  
Zaragori T, Oster J, Roch V, Hossu G, Chawki MB, Grignon R, et al. 18F-FDopa PET for the non-invasive prediction of glioma molecular parameters: a radiomics study (accepted in Journal of Nuclear Medicine)

# How to use
Assuming a directory structure similar to this directory:
- For the prediction of the IDH mutations run:  
`python modeling_pipeline_article.py --root ./data --covariates_file ./data/clinical_data.xlsx --output IDH --save_root ./data/results`

- For the prediction of the 1p/19q co-deletion run:  
`python modeling_pipeline_article.py --root ./data --covariates_file ./data/clinical_data.xlsx --output codeletion --save_root ./data/results`

The `--threads` option defines the number of processes that will be run in parallel and must be set according to your processor. By default it will use all the cores.
