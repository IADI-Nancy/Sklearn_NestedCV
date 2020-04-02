import os
import pandas as pd
import numpy as np
import sys
from rpy2.robjects import r, pandas2ri, numpy2ri, Vector, NULL
from collections.abc import Mapping
from scipy.stats import mannwhitneyu, kruskal
pandas2ri.activate()
numpy2ri.activate()


def ComBat(X, batch, covariate=None, parametric=False, empirical_bayes=True, save_dir=None):
    # Check X
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        if isinstance(X, (list, tuple, np.ndarray, Mapping)):
            df = pd.DataFrame(X)
        else:
            raise TypeError('X must be an array-like object, dictionary or pandas Dataframe/Series')
    else:
        df = X
    r_df = pandas2ri.py2ri(df)
    # Check covariate
    if covariate is None:
        covariate = np.ones((len(batch), 1))
    else:
        if not isinstance(covariate, (list, tuple, np.ndarray)):
            if isinstance(covariate, pd.DataFrame) or isinstance(covariate, pd.Series):
                covariate = covariate.to_numpy()
            else:
                raise TypeError('covariate array must be an array like or pandas Dataframe/Series')
        else:
            covariate = np.array(covariate)
    if len(covariate.shape) == 1:
        covariate = covariate.reshape(-1, 1)
    elif len(covariate.shape) > 2:
        raise ValueError('covariate array must be 1D or 2D')
    nr, nc = covariate.shape
    r_covariate = r.matrix(covariate, nrow=nr, ncol=nc)
    # Check batch
    if not isinstance(batch, (list, tuple, np.ndarray)):
        if isinstance(batch, pd.DataFrame) or isinstance(batch, pd.Series):
            batch = batch.to_numpy()
        else:
            raise TypeError('batch array must be an array like or pandas Dataframe/Series')
    else:
        batch = np.array(batch)
    if len(batch.shape) != 1:
        if len(batch.shape) == 2 and batch.shape[1] == 1:
            batch.reshape(-1)
        else:
            raise ValueError('batch array must be 1D or 2D with second dimension equal to 1')
    if len(np.unique(batch)) <= 1:
        raise ValueError('batch array must have at least 2 classes')
    r_batch = Vector(batch)
    cwd = os.path.dirname(sys.argv[0])
    r.setwd(cwd)
    r.source('./Statistical_analysis/R_scripts/ComBat.R')
    r_dr_results = r.ComBat_harmonization(r_df, r_covariate, r_batch, parametric, empirical_bayes)
    R_object_dict = {}
    keys = r_dr_results.names
    for i in range(len(keys)):
        R_object_dict[keys[i]] = np.array(r_dr_results[i])
    results = pd.DataFrame(R_object_dict)
    if save_dir is not None:
        results.to_excel(os.path.join(save_dir, 'Features_ComBat.xlsx'))
    return results


def MComBat(X, batch, ref_batch=None, covariate=None, num_covs=None, save_dir=None):
    # Check X
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        if isinstance(X, (list, tuple, np.ndarray, Mapping)):
            df = pd.DataFrame(X)
        else:
            raise TypeError('X must be an array-like object, dictionary or pandas Dataframe/Series')
    else:
        df = X
    r_df = pandas2ri.py2ri(df)
    # Check covariate
    if covariate is None:
        covariate = np.ones((len(batch), 1))
    else:
        if not isinstance(covariate, (list, tuple, np.ndarray)):
            if isinstance(covariate, pd.DataFrame) or isinstance(covariate, pd.Series):
                covariate = covariate.to_numpy()
            else:
                raise TypeError('covariate array must be an array like or pandas Dataframe/Series')
        else:
            covariate = np.array(covariate)
    if len(covariate.shape) == 1:
        covariate = covariate.reshape(-1, 1)
    elif len(covariate.shape) > 2:
        raise ValueError('covariate array must be 1D or 2D')
    nr, nc = covariate.shape
    r_covariate = r.matrix(covariate, nrow=nr, ncol=nc)
    # Check batch
    if not isinstance(batch, (list, tuple, np.ndarray)):
        if isinstance(batch, pd.DataFrame) or isinstance(batch, pd.Series):
            batch = batch.to_numpy()
        else:
            raise TypeError('batch array must be an array like or pandas Dataframe/Series')
    else:
        batch = np.array(batch)
    if len(batch.shape) != 1:
        if len(batch.shape) == 2 and batch.shape[1] == 1:
            batch.reshape(-1)
        else:
            raise ValueError('batch array must be 1D or 2D with second dimension equal to 1')
    if len(np.unique(batch)) <= 1:
        raise ValueError('batch array must have at least 2 classes')
    r_batch = Vector(batch)
    # Check ref batch
    if ref_batch is None:
        ref_batch = np.unique(batch)[0]
    else:
        if ref_batch not in np.unique(batch):
            raise ValueError('ref_batch must be one of np.unique(batch) values')
    # Check numCovs
    if num_covs is None:
        r_numCovs = NULL
    else:
        if isinstance(num_covs, int):
            num_covs = [num_covs]
        if not isinstance(num_covs, (list, tuple, np.ndarray)):
            raise TypeError('num_covs must be an int or array like of int equal to the index of numerical covariates')
        r_numCovs = Vector(num_covs)
    cwd = os.path.dirname(sys.argv[0])
    r.setwd(cwd)
    r.source('./Statistical_analysis/R_scripts/MComBat.R')
    r_dr_results = r.MComBat_harmonization(r_df, r_covariate, r_batch, ref_batch, r_numCovs)
    R_object_dict = {}
    keys = r_dr_results.names
    for i in range(len(keys)):
        R_object_dict[keys[i]] = np.array(r_dr_results[i])
    results = pd.DataFrame(R_object_dict)
    if save_dir is not None:
        results.to_excel(os.path.join(save_dir, 'Feature_MComBat.xlsx'))
    return results


def before_after_comparison(X_before, X_after, batch):
    """
    Test the effect of data harmonization with values extracted from a reference/healthy region.
    Before harmonization values should be significantly different between each device but harmonization should remove
    those differences and no more significantly difference should be observed
    Test performed is Mann-Whitney for 2 classes else Kruskal-Wallis
    """
    data = {'before': X_before, 'after': X_after}
    for key in data:
        # Check X
        if not isinstance(data[key], (list, tuple, np.ndarray)):
            if isinstance(data[key], pd.DataFrame) or isinstance(data[key], pd.Series):
                data[key] = data[key].to_numpy()
            else:
                raise TypeError('X_%s array must be an array like or pandas Dataframe/Series' % key)
        else:
            data[key] = np.array(data[key])
        if len(data[key].shape) != 2:
            raise ValueError('X_%s array must 2D' % key)
    # Check batch
    if not isinstance(batch, (list, tuple, np.ndarray)):
        if isinstance(batch, pd.DataFrame) or isinstance(batch, pd.Series):
            batch = batch.to_numpy()
        else:
            raise TypeError('batch array must be an array like or pandas Dataframe/Series')
    else:
        batch = np.array(batch)
    if len(batch.shape) != 1:
        if len(batch.shape) == 2 and batch.shape[1] == 1:
            batch.reshape(-1)
        else:
            raise ValueError('batch array must be 1D or 2D with second dimension equal to 1')
    if len(np.unique(batch)) <= 1:
        raise ValueError('batch array must have at least 2 classes')
    # Statistical analysis
    results = {'pvalue_before': [], 'pvalue_after': []}
    for key in data:
        n_samples, n_features = data[key].shape
        labels = np.unique(batch)
        for i in range(n_features):
            if len(labels) == 2:
                statistic, pvalue = mannwhitneyu(data[key][:, i][batch == labels[0]],
                                                 data[key][:, i][batch == labels[1]], alternative='two-sided')
            else:
                X_by_label = [data[key][:, i][batch == i] for _ in labels]
                statistic, pvalue = kruskal(*X_by_label)
            results['pvalue_' + key].append(pvalue)
    return pd.DataFrame(results)

