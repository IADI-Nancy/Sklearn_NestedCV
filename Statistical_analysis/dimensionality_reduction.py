"""Scikit-learn compatible dimensionality reduction by correlation clustering.

This module implements the hierarchical feature clustering described by
Leger et al., with an optional bootstrap consensus step for small datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data, check_is_fitted, check_consistent_length


class DimensionalityReduction(TransformerMixin, BaseEstimator, ABC):
    """
    Abstract base class for custom dimensionality-reduction transformers.

    Subclasses must implement:
    - _fit_reducer
    - _transform_reducer
    - _get_feature_names_out_reducer
    """

    def fit(self, X, y=None):
        """
        Fit the dimensionality-reduction method.

        The base class validates the input data and delegates the actual
        algorithm to `_fit_reducer`.
        """
        X_validated  = validate_data(self, X=X, reset=True, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        if y is not None:
            check_consistent_length(X, y)
        self._fit_reducer(X_validated, y)
        self._dimensionality_reduction_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform input data using the fitted dimensionality reducer.
        """
        check_is_fitted(self, "_dimensionality_reduction_fitted_")
        X_validated = validate_data(self, X=X, reset=False, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        X_reduced = self._transform_reducer(X_validated)
        X_reduced = np.asarray(X_reduced)
        
        if X_reduced.ndim != 2:
            raise RuntimeError("_transform_reducer must return a two-dimensional array.")
        if X_reduced.shape[0] != X_validated.shape[0]:
            raise RuntimeError("_transform_reducer changed the number of samples.")

        return X_reduced
    
    def _validate_input_features(self, input_features=None) -> np.ndarray:
        """
        Resolve and validate input feature names.
        """
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                return np.asarray(self.feature_names_in_, dtype=object)
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

        input_features = np.asarray(input_features, dtype=object)

        if input_features.ndim != 1:
            raise ValueError("input_features must be one-dimensional.")

        if len(input_features) != self.n_features_in_:
            raise ValueError("input_features must contain one name per input feature.")

        if hasattr(self, "feature_names_in_") and not np.array_equal(input_features, self.feature_names_in_):
                raise ValueError("input_features does not match feature_names_in_.")
        return input_features

    def get_feature_names_out(self, input_features=None):
        """
        Return names for transformed features.
        """
        check_is_fitted(self, "_dimensionality_reduction_fitted_")

        input_features = self._validate_input_features(input_features)
        output_features = self._get_feature_names_out_reducer(input_features)
        output_features = np.asarray(output_features, dtype=object)

        if output_features.ndim != 1:
            raise RuntimeError("_get_feature_names_out_reducer must return a one-dimensional array.")
        return output_features
    
    @abstractmethod
    def _fit_reducer(self, X: np.ndarray, y=None) -> None:
        """
        Fit the method-specific reduction model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _transform_reducer(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted method-specific transformation.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_feature_names_out_reducer(self, input_features: np.ndarray) -> np.ndarray:
        """
        Output feature names.
        """
        raise NotImplementedError


class HierarchicalClusteringLeger(DimensionalityReduction):
    """Reduce correlated features using hierarchical clustering of Leger et al. (Sci Rep, 2017).

    Parameters
    ----------
    corr_metric : {"pearson", "spearman", "kendall"}, default="spearman"
        Correlation used between features. Clustering uses ``1 - abs(corr)``.

    threshold : float in (0, 1], default=0.9
        Minimum absolute correlation used to define feature clusters.

    cluster_reduction : {"mean", "medoid"}, default="mean"
        How each cluster is represented. ``"mean"`` computes a sign-aligned
        average. ``"medoid"`` retains the most central original feature.

    bootstrap : bool, default=False
        Whether to estimate clusters by bootstrap consensus clustering.

    n_bootstraps : int, default=500
        Number of bootstrap samples. Used only when ``bootstrap=True``.

    consensus_threshold : float in [0, 1], default=0.5
        Minimum co-association probability used for final consensus clusters.

    consensus_method : {"connected_components", "complete_linkage"},
            default="connected_components"
        ``"connected_components"`` reproduces the graph-based implementation
        used in version B. It can merge features by transitive chaining.
        ``"complete_linkage"`` is stricter: every pair in a final cluster must
        satisfy the consensus threshold.

    random_state : int, RandomState instance or None, default=None
        Controls bootstrap sampling.

    Attributes
    ----------
    coefficient_matrix_ : ndarray of shape (n_output_features, n_features_in_)
        Linear transformation used by the built-in clustering method.

    cluster_labels_ : ndarray of shape (n_features_in_,)
        Final cluster assignment of each input feature.

    consensus_matrix_ : ndarray of shape (n_features_in_, n_features_in_)
        Co-association probabilities. Present only when ``bootstrap=True``.

    medoid_indices_ : ndarray of shape (n_output_features,)
        Index of the medoid feature in every final cluster.
    """
    def __init__(
        self,
        *,
        corr_metric: Literal["pearson", "spearman", "kendall"] = "spearman",
        correlation_threshold: float = 0.9,
        cluster_reduction: Literal["mean", "medoid"] = "medoid",
        bootstrap: bool = False,
        n_bootstraps: int = 500,
        consensus_threshold: float = 0.5,
        consensus_method: Literal["connected_components", "complete_linkage"] = "connected_components",
        random_state: int | np.random.RandomState | None = None,
    ):
        self.corr_metric = corr_metric
        self.correlation_threshold = correlation_threshold
        self.cluster_reduction = cluster_reduction
        self.bootstrap = bootstrap
        self.n_bootstraps = n_bootstraps
        self.consensus_threshold = consensus_threshold
        self.consensus_method = consensus_method
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if self.corr_metric not in {"pearson", "spearman", "kendall"}:
            raise ValueError("corr_metric must be one of {'pearson', 'spearman', 'kendall'}.")
        if not isinstance(self.correlation_threshold, Real) or isinstance(self.correlation_threshold, bool) or not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be a real number in (0, 1].")
        if self.cluster_reduction not in {"mean", "medoid"}:
            raise ValueError("cluster_reduction must be either 'mean' or 'medoid'.")
        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise TypeError("bootstrap must be a boolean.")
        if not isinstance(self.n_bootstraps, Integral) or isinstance(self.n_bootstraps, (bool, np.bool_)) or self.n_bootstraps < 1:
            raise ValueError("n_bootstraps must be a positive integer.")
        if not isinstance(self.consensus_threshold, Real) or isinstance(self.consensus_threshold, bool) or not 0 <= self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be a real number in [0, 1].")
        if self.consensus_method not in {"connected_components", "complete_linkage"}:
            raise ValueError("consensus_method must be either 'connected_components' or 'complete_linkage'.")
        check_random_state(self.random_state)
    
    def _correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        corr = pd.DataFrame(X).corr(method=self.corr_metric).to_numpy(dtype=float)
        # NaN correlations arise for zero-variance features, including bootstrap
        # samples where an otherwise varying feature can become constant.
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)
    
    @staticmethod
    def _dissimilarity_from_correlation(correlation_matrix: np.ndarray) -> np.ndarray:
        dissimilarity = 1.0 - np.abs(correlation_matrix)
        # Ensure dissimilarity matrix is symmetric
        dissimilarity = (dissimilarity + dissimilarity.T) / 2.0
        np.fill_diagonal(dissimilarity, 0.0)
        return np.clip(dissimilarity, 0.0, 1.0)

    def _labels_from_dissimilarity(self, dissimilarity: np.ndarray) -> np.ndarray:
        n_features = dissimilarity.shape[0]
        if n_features == 1:
            return np.zeros(1, dtype=int)
        condensed = squareform(dissimilarity, checks=False)
        tree = linkage(condensed, method="complete")
        return cut_tree(tree, height=1.0 - self.correlation_threshold).ravel().astype(int)

    def _single_clustering(self, X: np.ndarray) -> np.ndarray:
        corr = self._correlation_matrix(X)
        dissimilarity = self._dissimilarity_from_correlation(corr)
        return self._labels_from_dissimilarity(dissimilarity)

    def _bootstrap_consensus(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        coassociation_counts = np.zeros((n_features, n_features), dtype=np.uint32)

        for _ in range(self.n_bootstraps):
            row_indices = rng.randint(0, n_samples, size=n_samples)
            labels = self._single_clustering(X[row_indices])
            coassociation_counts += labels[:, None] == labels[None, :]

        consensus = coassociation_counts.astype(float) / float(self.n_bootstraps)
        np.fill_diagonal(consensus, 1.0)

        if self.consensus_method == "connected_components":
            # >= is the natural interpretation of a minimum threshold.
            graph = consensus >= self.consensus_threshold
            _, labels = connected_components(graph.astype(np.uint8), directed=False, return_labels=True)
            return labels.astype(int), consensus

        consensus_dissimilarity = 1.0 - consensus
        np.fill_diagonal(consensus_dissimilarity, 0.0)
        if n_features == 1:
            labels = np.zeros(1, dtype=int)
        else:
            tree = linkage(squareform(consensus_dissimilarity, checks=False), method="complete",)
            labels = cut_tree(tree, height=1.0 - self.consensus_threshold).ravel().astype(int)
        return labels, consensus

    @staticmethod
    def _medoid_index(cluster_indices: np.ndarray, dissimilarity: np.ndarray) -> int:
        within_cluster = dissimilarity[np.ix_(cluster_indices, cluster_indices)]
        return int(cluster_indices[np.argmin(within_cluster.sum(axis=1))])

    def _build_coefficient_matrix(self, corr: np.ndarray, dissimilarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coefficients = np.zeros((len(self.unique_cluster_labels_), self.cluster_labels_.size), dtype=float)
        medoids = np.empty(len(self.unique_cluster_labels_), dtype=int)

        for output_index, label in enumerate(self.unique_cluster_labels_):
            cluster_indices = np.flatnonzero(self.cluster_labels_ == label)
            medoid = self._medoid_index(cluster_indices, dissimilarity)
            medoids[output_index] = medoid

            if self.cluster_reduction == "medoid":
                coefficients[output_index, medoid] = 1.0
                continue

            signs = np.sign(corr[cluster_indices, medoid])
            signs[signs == 0] = 1.0
            coefficients[output_index, cluster_indices] = (signs / cluster_indices.size)

        return coefficients, medoids
    
    def _clear_fitted_attributes(self):
        for name in (
            "coefficient_matrix_",
            "cluster_labels_",
            "consensus_matrix_",
            "medoid_indices_",
            "n_features_out_",
            "transformer_",
        ):
            if hasattr(self, name):
                delattr(self, name)
                
    def _fit_reducer(self, X, y=None):
        self._validate_parameters()
        self._clear_fitted_attributes()
        corr = self._correlation_matrix(X)
        dissimilarity = self._dissimilarity_from_correlation(corr)

        if self.bootstrap:
            labels, consensus = self._bootstrap_consensus(X)
            self.consensus_matrix_ = consensus
        else:
            labels = self._labels_from_dissimilarity(dissimilarity)

        self.cluster_labels_ = labels
        self.unique_cluster_labels_ = np.unique(self.cluster_labels_)
        self.coefficient_matrix_, self.medoid_indices_ = self._build_coefficient_matrix(corr, dissimilarity)
        
    def _transform_reducer(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coefficient_matrix_.T
    
    def _get_feature_names_out_reducer(self, input_features: np.ndarray) -> np.ndarray:
        if self.cluster_reduction == "medoid":
            return input_features[self.medoid_indices_]
        return np.asarray([f"cluster_{cluster_index}" for cluster_index in self.unique_cluster_labels_], dtype=object)