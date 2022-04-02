import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform


class DimensionalityReduction(BaseEstimator):
    """A general class to handle dimensionality reduction.
        Parameters
        ----------
        method: str or transform
            Method used to compute dimensionality reduction. Either str or callable
            If str inbuild function named as str is called, must be one of following:
                'hierarchical_clust_parmar': Consensus Clustering with hierarchical clustering as described in :
                    Radiomic feature clusters and Prognostic Signatures specific for Lung and Head & Neck cancer.
                    Parmar et al., Scientific Reports, 2015
                'hierarchical_clust_leger': Hierarchical clustering as described in :
                    A comparative study of machine learning methods for time-to-event survival data for
                    radiomics risk modelling. Leger et al., Scientific Reports, 2017
            If transform method must inherit from TransformMixin (like sklearn transformers) or have a fit + a transform
            method that will be called successively with the latter returning the reduce dataset.
            Ex :
                str: method = 'hierarchical_clust_leger'
                transform: method = sklearn.decomposition.PCA(n_components=0.95, solver='svd_full')
        corr_metric: str
            Correlation metric used to compute distance between features
        threshold: float
            Correlation threshold used assign clusters to feature. Tree will be cut at a height of 1 - threshold
        cluster_reduction: str
            Method used to combine features in the same cluster. Currently implemented : mean and medoid
    """
    dr_methods = ['hierarchical_clust_parmar', 'hierarchical_clust_leger']
    cluster_reduction_methods = ['mean', 'medoid']

    def __init__(self, method='hierarchical_clust_leger', corr_metric='spearman', threshold=0.9, cluster_reduction='mean'):
        self.method = method
        self.corr_metric = corr_metric
        self.threshold = threshold
        self.cluster_reduction = cluster_reduction
        self.is_reduced = False
        self.is_fitted = False

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
        return X, y

    def _get_dr_func(self):
        if isinstance(self.method, TransformerMixin) or (hasattr(self.method, 'fit') and hasattr(self.method, 'transform')):
            return self.method
        elif isinstance(self.method, str):
            method_name = self.method.lower()
            if method_name not in self.dr_methods:
                raise ValueError('If string method must be one of : {0}. '
                                 '%s was passed'.format(str(self.dr_methods), self.method))
            return getattr(self, self.method)
        else:
            raise TypeError('method argument must be a callable or a string')

    @staticmethod
    def _get_medoid(n_k, distance_matrix, cluster_labels):
        df_distance_matrix = pd.DataFrame(distance_matrix)
        cluster_distance_matrix = df_distance_matrix.loc[cluster_labels == n_k, cluster_labels == n_k]
        return cluster_distance_matrix.sum(axis=0).idxmin()

    def hierarchical_clust_leger(self, X, y=None):
        """
        Hierarchical clustering as described in :
            A comparative study of machine learning methods for time-to-event survival data for
            radiomics risk modelling. Leger et al., Scientific Reports, 2017
        """
        # df = pd.DataFrame(X)
        # r_df = pandas2ri.py2ri(df)
        # cwd = os.path.dirname(sys.argv[0])
        # r.setwd(cwd)
        # r.source('./Statistical_analysis/R_scripts/hierarchical_clustering_Leger.R')
        # r_dr_results = r.hierarchical_clustering_leger(r_df)
        # R_object_dict = {}
        # keys = r_dr_results.names
        # for i in range(len(keys)):
        #     R_object_dict[keys[i]] = np.array(r_dr_results[i])
        # dr_results = pd.DataFrame(R_object_dict).to_numpy()
        # nb_cluster = np.amax(dr_results[:, 0]).astype(int)
        # coefficient_matrix = np.zeros((dr_results.shape[0], nb_cluster))  # Shape of (n_features, nb cluster)
        # for i in range(nb_cluster):
        #     coefficient_matrix[:, i] = np.where(dr_results[:, 0] == i + 1, dr_results[:, 1], 0)
        # coefficient_matrix = coefficient_matrix.T

        dissimilarity_matrix = 1 - np.abs(pd.DataFrame(X).corr(method=self.corr_metric).to_numpy())
        distance_matrix = squareform(dissimilarity_matrix)
        Z = linkage(distance_matrix, method='complete')
        labels = cut_tree(Z, height=1 - self.threshold).reshape(-1)
        self.cluster_labels = labels
        feature_coefficient = np.zeros(np.size(labels))
        if self.cluster_reduction == 'mean':
            corr_matrix = pd.DataFrame(X).corr(method=self.corr_metric).to_numpy()
            for n_k in range(np.amax(labels) + 1):
                n = np.sum(labels == n_k)
                if n != 1:
                    cluster_corr_matrix = corr_matrix[labels == n_k, :][:, labels == n_k]
                    feature_coefficient[labels == n_k] = np.where(cluster_corr_matrix[:, 0] < 0, -1 / n, 1 / n)
                else:
                    feature_coefficient[labels == n_k] = 1
        elif self.cluster_reduction == 'medoid':
            for n_k in range(np.amax(labels) + 1):
                medoid_idx = self._get_medoid(n_k, dissimilarity_matrix, labels)
                feature_coefficient[medoid_idx] = 1
        else:
            raise ValueError('cluster_reduction must be one of : %s. '
                             '%s was passed' % (self.cluster_reduction_methods, self.cluster_reduction))
        coefficient_matrix = np.zeros((X.shape[1], np.amax(labels) + 1))  # Shape of (n_features, nb cluster)
        for i in range(np.amax(labels) + 1):
            coefficient_matrix[:, i] = np.where(labels == i, feature_coefficient, 0)
        coefficient_matrix = coefficient_matrix.T

        return coefficient_matrix

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
        self.dr_func = self._get_dr_func()

        # Test whether the method has a fit function like sklearn classes
        if hasattr(self.dr_func, 'fit'):
            self.dr_func.fit(X, y)
        else:
            self.coefficient_matrix = self.dr_func(X, y=y)
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
        X, _ = self._check_X_Y(X, None)
        # Test whether the method has a transform function like sklearn classes
        if self.is_fitted:
            if hasattr(self.dr_func, 'transform'):
                self.reduced_features = self.dr_func.transform(X)
            else:
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
        self.fit(X, y)
        return self.transform(X)
