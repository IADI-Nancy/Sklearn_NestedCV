import os
import warnings
import pandas as pd
import numpy as np
import sys
from rpy2.robjects import r, pandas2ri
from Statistical_analysis.univariate_statistical_analysis import univariate_analysis
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
pandas2ri.activate()


class DimensionalityReduction(BaseEstimator):
    """A general class to handle dimensionality reduction.
        Parameters
        ----------
        method: str or callable
            Method used to compute dimensionality reduction. Either str or callable
            If str inbuild function named as str is called, must be one of following:
                'hierarchical_clust_parmar': Consensus Clustering with hierarchical clustering as described in :
                    Radiomic feature clusters and Prognostic Signatures specific for Lung and Head & Neck cancer.
                    Parmar et al., Scientific Reports, 2015
                'hierarchical_clust_leger': Hierarchical clustering as described in :
                    A comparative study of machine learning methods for time-to-event survival data for
                    radiomics risk modelling. Leger et al., Scientific Reports, 2017
            If callable method must take feature_array as arguments (and label_array if needed)
            and return the coefficient matrix which is an array of shape (n_components, n_features)
            with for each features the coefficient associated with each components of reduced features dataset
        sklearn_kwargs: dict (default={})
            Keyword arguments that will be used to create object if a sklearn class is used as method
            Ex: method = sklearn.decomposition.PCA()
                sklearn_kwargs={'n_components': 0.95, 'svd_solver': 'full'}
        save_dir: str (default=None)
            Path to the directory where optional clustering analysis could be saved as an excel file
            If None, analysis won't be performed
    """
    dr_methods = ['hierarchical_clust_parmar', 'hierarchical_clust_leger']

    def __init__(self, method, sklearn_kwargs={}, save_dir=None):
        self.sklearn_kwargs = sklearn_kwargs
        if callable(method):
            if not method.__name__ in self.dr_methods:
                if not self.sklearn_kwargs:
                    warnings.warn('WARNING : No keywords arguments were found.'
                                  'The method object will be created with default parameter.')
                self.method = method(**self.sklearn_kwargs)
            else:
                self.method = method
        elif isinstance(method, str):
            method = method.lower()
            if method not in self.dr_methods:
                raise ValueError('If string method must be one of : {0}. '
                                 '%s was passed'.format(str(self.dr_methods), method))
            self.method = getattr(self, method)
        else:
            raise TypeError('method argument must be a callable or a string')
        self.save_dir = save_dir
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
        self.is_reduced = False
        self.is_fitted = True

    @staticmethod
    def _check_X_Y(X, y):
        # Check X
        if not isinstance(X, (list, tuple, np.ndarray)):
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X = X.to_numpy()
            else:
                raise TypeError('X array must be an array like or pandas Dataframe/Series')
        else:
            X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError('X array must 2D')
        # Check y
        if y is not None:
            if not isinstance(y, (list, tuple, np.ndarray)):
                if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                    y = y.to_numpy()
                else:
                    raise TypeError('X array must be an array like or pandas Dataframe/Series')
            else:
                y = np.array(y)
            if len(y.shape) != 1:
                if len(y.shape) == 2 and y.shape[1] == 1:
                    y.reshape(-1)
                else:
                    raise ValueError('y array must be 1D or 2D with second dimension equal to 1')
            if len(np.unique(y)) <= 1:
                raise ValueError('y array must have at least 2 classes')
        return X, y

    @staticmethod
    def hierarchical_clust_parmar(X, save_dir, y=None):
        """
        Consensus Clustering with hierarchical clustering as described in :
            Radiomic feature clusters and Prognostic Signatures specific for Lung and Head & Neck cancer.
            Parmar et al., Scientific Reports, 2015
        save_dir must not be None to perform cluster analysis (cluster stability, cluster compactness) and get
        ConsensusClusterPlus plots.
        Cluster similarity between two different datasets and Cluster validation by comparison with test set need to be
        calculated out of the pipeline.
        """
        df = pd.DataFrame(X)
        r_df = pandas2ri.py2ri(df)
        cwd = os.path.dirname(sys.argv[0])
        r.setwd(cwd)
        r.source('./Statistical_analysis/R_scripts/hierarchical_clustering_Parmar.R')
        if save_dir:
            r_dr_results = r.hierarchical_clustering_parmar(r_df, max_k=20, threshold=0.1, save_dir=save_dir)
        else:
            r_dr_results = r.hierarchical_clustering_parmar(r_df, max_k=20, threshold=0.1)
        R_object_dict = {}
        keys = r_dr_results.names
        for i in range(len(keys)):
            R_object_dict[keys[i]] = np.array(r_dr_results[i])
        dr_results = pd.DataFrame(R_object_dict).to_numpy()
        nb_cluster = np.amax(dr_results[:, 0]).astype(int)
        coefficient_matrix = np.zeros((dr_results.shape[0], nb_cluster))  # Shape of (n_features, nb cluster)
        for i in range(nb_cluster):
            coefficient_matrix[:, i] = np.where(dr_results[:, 0] == i + 1, dr_results[:, 1], 0)
        coefficient_matrix = coefficient_matrix.T
        return coefficient_matrix

    @staticmethod
    def hierarchical_clust_leger(X, save_dir, y=None):
        """
        Hierarchical clustering as described in :
            A comparative study of machine learning methods for time-to-event survival data for
            radiomics risk modelling. Leger et al., Scientific Reports, 2017
        save_dir must not be None to perform cluster analysis (cluster compactness)
        Cluster similarity between two different datasets and Cluster validation by comparison with test set need to be
        calculated out of the pipeline.
        """
        df = pd.DataFrame(X)
        r_df = pandas2ri.py2ri(df)
        cwd = os.path.dirname(sys.argv[0])
        r.setwd(cwd)
        r.source('./Statistical_analysis/R_scripts/hierarchical_clustering_Leger.R')
        if save_dir:
            r_dr_results = r.hierarchical_clustering_leger(r_df, save_dir=save_dir)
        else:
            r_dr_results = r.hierarchical_clustering_leger(r_df)
        R_object_dict = {}
        keys = r_dr_results.names
        for i in range(len(keys)):
            R_object_dict[keys[i]] = np.array(r_dr_results[i])
        dr_results = pd.DataFrame(R_object_dict).to_numpy()
        nb_cluster = np.amax(dr_results[:, 0]).astype(int)
        coefficient_matrix = np.zeros((dr_results.shape[0], nb_cluster))  # Shape of (n_features, nb cluster)
        for i in range(nb_cluster):
            coefficient_matrix[:, i] = np.where(dr_results[:, 0] == i + 1, dr_results[:, 1], 0)
        coefficient_matrix = coefficient_matrix.T
        return coefficient_matrix

    def univariate_analysis(self, y, adjusted_method='BH', save_dir=None):
        """
        Function to perform statistical univarite analysis on reduced feature dataset
        """
        if self.is_reduced:
            self.univariate_results = univariate_analysis(self.reduced_features, y, adjusted_method, save_dir)
        else:
            raise NotFittedError('transform() or fit_transform() must be call before calling univariate_analysis()')
        return self.univariate_results

    # === Applying dimensionnality reduction ===
    def fit(self, X, y=None):
        """Fit dimensionality reduction.
            Parameters
            ----------
            X : pandas dataframe or array-like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y : optional, pandas dataframe or array-like of shape (n_samples,) (default = None)
                Target vector relative to X.
            Returns
            -------
            It will not return directly the values, but it's accessable from the class object it self.
            You should be able to access:
            coefficient_matrix
                 array of shape (n_components, n_features) with for each features the coefficient associated with
                 each components of reduced features dataset
        """
        X, y = self._check_X_Y(X, y)
        # Test whether the method has a fit function like sklearn classes
        if hasattr(self.method, 'fit'):
            self.method.fit(X, y)
        else:
            self.coefficient_matrix = self.method(X, self.save_dir, y)
            self.is_fitted = True

    def transform(self, X):
        """Reduce X dataset to create a new dataset.
            Parameters
            ----------
            X : pandas dataframe or array-like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            Returns
            -------
            reduced_features
                 array of shape (n_samples, n_components) containing the reduced dataset
        """
        # Check X
        if not isinstance(X, (list, tuple, np.ndarray)):
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X = X.to_numpy()
            else:
                raise TypeError('X array must be an array like or pandas Dataframe/Series')
        else:
            X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError('X array must 2D')
        # Test whether the method has a transform function like sklearn classes
        if hasattr(self.method, 'transform'):
            self.reduced_features = self.method.transform(X)
        else:
            if self.is_fitted:
                self.reduced_features = np.array([self.coefficient_matrix.dot(X[_, :]) for _ in range(X.shape[0])])
                self.is_reduced = True
                return self.reduced_features
            else:
                raise NotFittedError('Fit method must be used before calling transform')

    def fit_transform(self, X, y=None):
        """Fit dimensionality reduction and reduce X dataset to create a new dataset
            Parameters
            ----------
            X : pandas dataframe or array-like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y : optional, pandas dataframe or array-like of shape (n_samples,) (default = None)
                Target vector relative to X.
            Returns
            -------
            reduced_features
                 array of shape (n_samples, n_components) containing the reduced dataset
            You should be able to access as class attribute to:
            coefficient_matrix
                 array of shape (n_components, n_features) with for each features the coefficient associated with
                 each components of reduced features dataset
            """
        # Test whether the method has a fit_transform function like sklearn classes
        if hasattr(self.method, 'fit_transform'):
            return self.method.fit_transform(X, y)
        else:
            self.coefficient_matrix = self.method(X, self.save_dir, y)
            return self.transform(X)
