import os
import pandas as pd
import numpy as np
import sys
from rpy2.robjects import r, pandas2ri, numpy2ri, Vector, NULL
from collections.abc import Mapping
from scipy.stats import mannwhitneyu, kruskal
from neuroCombat import neuroCombat
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
    row_names = df.index
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
    # cwd = os.path.dirname(sys.argv[0])
    cwd = os.path.dirname(os.path.abspath(__file__))
    r.setwd(cwd)
    # r.source('./Statistical_analysis/R_scripts/ComBat.R')
    r.source('./R_scripts/ComBat.R')
    r_dr_results = r.ComBat_harmonization(r_df, r_covariate, r_batch, parametric, empirical_bayes)
    R_object_dict = {}
    keys = r_dr_results.names
    for i in range(len(keys)):
        R_object_dict[keys[i]] = np.array(r_dr_results[i])
    results = pd.DataFrame(R_object_dict)
    results.index = row_names
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
    row_names = df.index
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
    # cwd = os.path.dirname(sys.argv[0])
    cwd = os.path.dirname(os.path.abspath(__file__))
    r.setwd(cwd)
    # r.source('./Statistical_analysis/R_scripts/MComBat.R')
    r.source('./R_scripts/MComBat.R')
    r_dr_results = r.MComBat_harmonization(r_df, r_covariate, r_batch, ref_batch, r_numCovs)
    R_object_dict = {}
    keys = r_dr_results.names
    for i in range(len(keys)):
        R_object_dict[keys[i]] = np.array(r_dr_results[i])
    results = pd.DataFrame(R_object_dict)
    results.index = row_names
    if save_dir is not None:
        results.to_excel(os.path.join(save_dir, 'Feature_MComBat.xlsx'))
    return results


def neuroComBat(X, batch, ref_batch=None, covariate=None, num_covs=None,
                parametric=False, empirical_bayes=True, mean_only=False, save_dir=None):
    """
    :param X: array like or pandas DataFrame
        Data to harmonize
    :param batch: array like or pandas DataFrame
        Batch/scanner covariates to use for harmonization
    :param ref_batch: string or None, default=None
        Batch (site or scanner) to be used as reference for batch adjustment.
    :param covariate: array like or pandas DataFrame, default=None
        Additional covariates that should be preserved during harmonization
    :param num_covs: int or array like of int, default=None
        Index(es) of numerical/continuous variables to be preserved during harmonization
    :param parametric: boolean, default=False
        Should parametric adjustements be performed?
    :param empirical_bayes: boolean, default=True
        Should Empirical Bayes be performed?
    :param mean_only: boolean, default=False
         Should only be the mean adjusted (no scaling)?
    :param save_dir: string, default=None
        Directory to save the harmonized data
    :return:
    """
    # Check X
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        if isinstance(X, (list, tuple, np.ndarray, Mapping)):
            df = pd.DataFrame(X)
        else:
            raise TypeError('X must be an array-like object, dictionary or pandas Dataframe/Series')
    else:
        df = X

    # Check covariate
    if covariate is None:
        covariate = pd.DataFrame({})
    else:
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            if isinstance(X, (list, tuple, np.ndarray, Mapping)):
                covariate = pd.DataFrame(covariate)
            else:
                raise TypeError('covariate array must be an array like or pandas Dataframe/Series')

    # Check numCovs
    if num_covs is not None:
        if isinstance(num_covs, int):
            num_covs = [num_covs]
        if not isinstance(num_covs, (list, tuple, np.ndarray)):
            raise TypeError('num_covs must be an int or array like of int equal to the index of numerical covariates')
        num_covs = covariate.index[np.array(num_covs)].to_list()
        cat_covs = [_ for _ in covariate.columns if _ not in num_covs]
    else:
        cat_covs = [_ for _ in covariate.columns]

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
    batch = pd.DataFrame(batch, columns='batch')

    # Concatenate batch and covariates
    covariate.reset_index(drop=True, inplace=True)
    batch.reset_index(drop=True, inplace=True)
    covariate = pd.concat([covariate, batch], axis=1)
    # Check ref batch
    if ref_batch not in np.unique(batch) and ref_batch is not None:
        raise ValueError('ref_batch must be one of np.unique(batch) values')

    results = neuroCombat(dat=df, covars=covariate, batch_col='batch', categorical_cols=cat_covs,
                          continuous_cols=num_covs, ref_batch=ref_batch, parametric=parametric,
                          eb=empirical_bayes, mean_only=mean_only)['data']
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

