import os
import pandas as pd
import numpy as np
import sys
from collections.abc import Mapping
from rpy2.robjects import r, pandas2ri, Vector
pandas2ri.activate()


def univariate_analysis(X, y, adjusted_method='BH', save_dir=None):
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        if isinstance(X, (list, tuple, np.ndarray, Mapping)):
            if len(np.array(X).shape) != 2:
                raise ValueError('X array must 2D')
            X = pd.DataFrame(X)
        else:
            raise TypeError('X must be an array-like object, dictionary or pandas Dataframe/Series')
    # else:
    #     df_X = X
    if not isinstance(y, (list, tuple, np.ndarray)):
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        else:
            raise TypeError('y array must be an array like or pandas Dataframe/Series')
    else:
        y = np.array(y)
    if len(y.shape) != 1:
        if len(y.shape) == 2 and y.shape[1] == 1:
            y.reshape(-1)
        else:
            raise ValueError('y array must be 1D or 2D with second dimension equal to 1')
    if len(np.unique(y)) <= 1:
        raise ValueError('y array must have at least 2 classes')
    r_X = pandas2ri.py2ri(X)
    r_y = Vector(y)
    cwd = os.path.dirname(sys.argv[0])
    r.setwd(cwd)
    r.source('./Statistical_analysis/R_scripts/univariate_analysis.R')
    r_dr_results = r.univariate_analysis(r_X, r_y, adjusted_method=adjusted_method)
    R_object_dict = {}
    keys = r_dr_results.names
    for i in range(len(keys)):
        R_object_dict[keys[i]] = np.array(r_dr_results[i])
    results = pd.DataFrame(R_object_dict)
    if save_dir is not None:
        results.to_excel(os.path.join(save_dir, 'univariate_stats_analysis.xlsx'))
    return results
