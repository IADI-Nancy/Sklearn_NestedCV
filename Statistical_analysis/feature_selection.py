import numpy as np
from scipy.stats import mannwhitneyu, kruskal, pearsonr, spearmanr, rankdata
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from skfeature.function.information_theoretical_based.MRMR import mrmr
from sklearn.base import BaseEstimator
import numbers
from sklearn.exceptions import NotFittedError


class FeatureSelection(BaseEstimator):
    """A general class to handle feature selection according to a scoring/ranking method. Bootstrap is implemented
        to ensure stability in feature selection process
        Parameters
        ----------
        method: str or callable
            Method used to score/rank features. Either str or callable
            If str inbuild function named as str is called, must be one of following:
                'wlcx_score': score of kruskall wallis test
                'auc_roc': scoring with area under the roc curve
                'pearson_corr': scoring with pearson correlation coefficient between features and labels
                'spearman_corr': scoring with spearman correlation coefficient between features and labels
                'mi': scoring with mutual information between features and labels
                'mrmr': ranking according to Minimum redundancy Maximum relevance algorithm
            If callable method must take (feature_array,label_array) as arguments and return either the score or the rank
            associated with each feature in the same order as features are in feature_array
        bootstrap: boolean (default=False)
            Choose whether feature selection must be done in a bootstraped way
        n_bsamples: int (default=100)
            Number of bootstrap samples generated for bootstraped feature selection. Ignored if bootstrap is False
        n_selected_features: int or None, default = 20
            Number of the best features that must be selected by the class
            If None all the feature are returned (no feature selection)
        ranking_aggregation: str or callable (default=None)
            Method used to aggregate rank of bootstrap samples. Either str or callable
            If str inbuild function named as str is called, must be one of following:'enhanced_borda', 'borda',
            'importance_score', 'mean', 'stability_selection', 'exponential_weighting'
            If callable method must take ((bootstrap_ranks, n_selected_features) as arguments and return the
            aggregate rank associated with each feature in the same order as features are in feature_array
        ranking_done: boolean (default=False)
            Indicate whether the method return a score or directly calculate ranks
        score_indicator_lower: boolean (default=None)
            Choose whether lower score correspond to higher rank for the rank calculation or higher score is better,
            `True` means lower score is better. Determined automatically for inbuild functions
        """
    scoring_methods = {'name': ['auc_roc', 'pearson_corr', 'spearman_corr', 'mi', 'wlcx_score'],
                       'score_indicator_lower': [False, False, False, False, False]}
    ranking_methods = ['mrmr']
    # TODO: implement more inbuild functions like multivariate selection algorithm
    #  (see skfeature : https://github.com/jundongl/scikit-feature)
    # TODO : implement t-test, chisquare
    ranking_aggregation_methods = ['enhanced_borda', 'borda', 'importance_score', 'mean', 'stability_selection',
                                   'exponential_weighting']
    # Importance score method :
    # A comparative study of machine learning methods for time-to-event survival data for radiomics risk modelling
    # Leger et al., 2017, Scientific Reports
    # Other methods:
    # An extensive comparison of feature ranking aggregation techniques in bioinformatics.
    # Randall et al., 2012, IEEE

    def __init__(self, method='mrmr', bootstrap=False, n_bsamples=100, n_selected_features=20, ranking_aggregation=None,
                 ranking_done=False, score_indicator_lower=None):
        self.method = method
        self.ranking_done = ranking_done
        self.score_indicator_lower = score_indicator_lower

        self.bootstrap = bootstrap
        self.n_bsamples = n_bsamples
        self.n_selected_features = n_selected_features
        self.ranking_aggregation = ranking_aggregation
        self.is_fitted = False

    def _get_fs_func(self):
        if callable(self.method):
            return self.method
        elif isinstance(self.method, str):
            method_name = self.method.lower()
            if method_name not in self.scoring_methods['name'] and method_name not in self.ranking_methods:
                raise ValueError('If string method must be one of : {0}. '
                                 '%s was passed'.format(str(self.scoring_methods['name'] + self.ranking_methods), method_name))
            if self.method in FeatureSelection.ranking_methods:
                self.ranking_done = True
            else:
                self.ranking_done = False
                self.score_indicator_lower = self.scoring_methods['score_indicator_lower'][self.scoring_methods['name'].index(self.method)]
            return getattr(self, method_name)
        else:
            raise TypeError('method argument must be a callable or a string')

    def _get_aggregation_method(self):
        if not callable(self.ranking_aggregation) and not isinstance(self.ranking_aggregation, str):
            raise TypeError('ranking_aggregation option must be a callable or a string')
        else:
            if isinstance(self.ranking_aggregation, str):
                ranking_aggregation_name = self.ranking_aggregation.lower()
                if self.ranking_aggregation not in FeatureSelection.ranking_aggregation_methods:
                    raise ValueError('If string ranking_aggregation must be one of : {0}. '
                                     '%s was passed'.format(str(FeatureSelection.ranking_aggregation_methods),
                                                            ranking_aggregation_name))
                return getattr(FeatureSelection, self.ranking_aggregation)

    @staticmethod
    def _check_X_Y(X, y=None):
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
        if y is not None:
            # Check y
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

    @staticmethod
    def _check_n_selected_feature(X, n_selected_features):
        if not isinstance(n_selected_features, numbers.Integral) and n_selected_features is not None:
            raise TypeError('n_selected_feature must be int or None')
        else:
            if n_selected_features is None:
                n_selected_features = X.shape[1]
            else:
                n_selected_features = n_selected_features
            return n_selected_features

    # === Scoring method ===
    @staticmethod
    def wlcx_score(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        labels = np.unique(y)
        for i in range(n_features):
            X_by_label = [X[:, i][y == _] for _ in labels]
            statistic, pvalue = kruskal(*X_by_label)
            score[i] = statistic
        return score

    @staticmethod
    def auc_roc(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        labels = np.unique(y)
        if len(labels) == 2:
            for i in range(n_features):
                # Replicate of roc function from pROC R package to find positive class
                control_median = np.median(X[:, i][y == labels[0]])
                case_median = np.median(X[:, i][y == labels[1]])
                if case_median > control_median:
                    positive_label = 1
                else:
                    positive_label = 0
                fpr, tpr, thresholds = roc_curve(y, X[:, i], pos_label=positive_label)
                roc_auc = auc(fpr, tpr)
                score[i] = roc_auc
        else:
            # Adapted from roc_auc_score for multi_class labels
            # See sklearn.metrics._base_average_multiclass_ovo_score
            # Hand & Till (2001) implementation (ovo)
            n_classes = labels.shape[0]
            n_pairs = n_classes * (n_classes - 1) // 2
            for i in range(n_features):
                pair_scores = np.empty(n_pairs)
                for ix, (a, b) in enumerate(combinations(labels, 2)):
                    a_mask = y == a
                    b_mask = y == b
                    ab_mask = np.logical_or(a_mask, b_mask)

                    # Replicate of roc function from pROC R package to find positive class
                    control_median = np.median(X[:, i][ab_mask][y[ab_mask] == a])
                    case_median = np.median(X[:, i][ab_mask][y[ab_mask] == b])
                    if control_median > case_median:
                        positive_class = y[ab_mask] == a
                    else:
                        positive_class = y[ab_mask] == b
                    fpr, tpr, _ = roc_curve(positive_class, X[:, i][ab_mask])
                    roc_auc = auc(fpr, tpr)
                    pair_scores[ix] = roc_auc
                score[i] = np.average(pair_scores)
        return score

    @staticmethod
    def pearson_corr(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        for i in range(n_features):
            correlation, pvalue = pearsonr(X[:, i], y)
            score[i] = correlation
        return score

    @staticmethod
    def spearman_corr(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        for i in range(n_features):
            correlation, pvalue = spearmanr(X[:, i], y)
            score[i] = correlation
        return score

    @staticmethod
    def mi(X, y):
        score = mutual_info_classif(X, y, random_state=111)
        return score

    @staticmethod
    def mrmr(X, y):
        n_samples, n_features = X.shape
        rank_index, _, _ = mrmr(X, y, n_selected_features=n_features)
        ranks = np.array([list(rank_index).index(_) + 1 for _ in range(len(rank_index))])        
        return ranks

    # === Ranking aggregation method ===
    @staticmethod
    def borda(bootstrap_ranks, n_selected_features):
        return rankdata(np.sum(bootstrap_ranks.shape[1] - bootstrap_ranks, axis=0) * -1, method='ordinal')

    @staticmethod
    def mean(bootstrap_ranks, n_selected_features):
        return rankdata(np.mean(bootstrap_ranks, axis=0), method='ordinal')

    @staticmethod
    def stability_selection(bootstrap_ranks, n_selected_features):
        """
        A.-C. Haury, P. Gestraud, and J.-P. Vert,
        The influence of feature selection methods on accuracy, stability and interpretability of molecular signatures
        PLoS ONE
        """
        return rankdata(np.sum(bootstrap_ranks <= n_selected_features, axis=0) * - 1, method='ordinal')

    @staticmethod
    def exponential_weighting(bootstrap_ranks, n_selected_features):
        """
        A.-C. Haury, P. Gestraud, and J.-P. Vert,
        The influence of feature selection methods on accuracy, stability and interpretability of molecular signatures
        PLoS ONE
        """
        return rankdata(np.sum(np.exp(-bootstrap_ranks / n_selected_features), axis=0)*-1, method='ordinal')

    @staticmethod
    def enhanced_borda(bootstrap_ranks, n_selected_features):
        borda_count = np.sum(bootstrap_ranks.shape[1] - bootstrap_ranks, axis=0)
        stability_selection = np.sum(bootstrap_ranks <= n_selected_features, axis=0)
        return rankdata(borda_count * stability_selection * -1, method='ordinal')

    @staticmethod
    def importance_score(bootstrap_ranks, n_selected_features):
        """
        A comparative study of machine learning methods for time-to-event survival data for
        radiomics risk modelling. Leger et al., Scientific Reports, 2017
        """
        occurence = np.sum(bootstrap_ranks <= n_selected_features, axis=0)**2
        importance_score = np.divide(np.sum(np.sqrt(bootstrap_ranks), axis=0), occurence,
                                     out=np.full(occurence.shape, np.inf), where=occurence != 0)
        return rankdata(importance_score, method='ordinal')

    # === Applying feature selection ===
    def fit(self, X, y):
        """A method to fit feature selection.
            Parameters
            ----------
            X : pandas dataframe or array-like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y : pandas dataframe or array-like of shape (n_samples,)
                Target vector relative to X.
            Returns
            -------
            It will not return directly the values, but it's accessable from the class object it self.
            You should be able to access:
            ranking_index
                 A list of features indexes sorted by ranks. ranking_index[0] returns the index of the best selected
                 feature according to scoring/ranking function
        """
        X, y = self._check_X_Y(X, y)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.fs_func = self._get_fs_func()
        if self.ranking_aggregation is not None:
            aggregation_method = self._get_aggregation_method()
        self.n_selected_features = self._check_n_selected_feature(X, self.n_selected_features)
        if self.bootstrap:
            if self.ranking_aggregation is None:
                raise ValueError('ranking_aggregation option must be given if bootstrap is True')
            bsamples_index = []
            n = 0
            while len(bsamples_index) < self.n_bsamples:
                bootstrap_sample = resample(range(n_samples), random_state=n)
                if len(np.unique(y[bootstrap_sample])) == n_classes:
                    bsamples_index.append(bootstrap_sample)
                n += 1
            bsamples_index = np.array(bsamples_index)
            if self.ranking_done:
                bootstrap_ranks = np.array([self.fs_func(X[_, :], y[_]) for _ in bsamples_index])
            else:
                if self.score_indicator_lower is None:
                    raise ValueError('score_indicator_lower option must be given if a user scoring function is used')
                boostrap_scores = np.array([self.fs_func(X[_, :], y[_]) for _ in bsamples_index])
                if not self.score_indicator_lower:
                    boostrap_scores *= -1
                bootstrap_ranks = np.array([rankdata(_) for _ in boostrap_scores])
            bootstrap_ranks_aggregated = aggregation_method(bootstrap_ranks, self.n_selected_features)
            self.ranking_index = [list(bootstrap_ranks_aggregated).index(_) for _ in sorted(bootstrap_ranks_aggregated)]
        else:
            if self.ranking_done:
                ranks = self.fs_func(X, y)
            else:
                if self.score_indicator_lower is None:
                    raise ValueError('score_indicator_lower option must be given if a user scoring function is used')
                score = self.fs_func(X, y)
                if not self.score_indicator_lower:
                    score *= -1
                ranks = rankdata(score, method='ordinal')
            self.ranking_index = [list(ranks).index(_) for _ in sorted(ranks)]
        self.is_fitted = True

    def transform(self, X):
        """Select the n_selected_features best features to create a new dataset.
            Parameters
            ----------
            X : pandas dataframe or array-like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            Returns
            -------
            n_selected_features
                 array of shape (n_samples, n_selected_features) containing the selected features
        """

        X, _ = self._check_X_Y(X, None)
        if self.is_fitted:
            self.selected_features = X[:, self.ranking_index[:self.n_selected_features]]
            return self.selected_features
        else:
            raise NotFittedError('Fit method must be used before calling transform')

    def fit_transform(self, X, y):
        """A method to fit feature selection and reduce X to selected features.
            Parameters
            ----------
            X : pandas dataframe or array-like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y : pandas dataframe or array-like of shape (n_samples,)
                Target vector relative to X.
            Returns
            -------
            n_selected_features
                array of shape (n_samples, n_selected_features) containing the selected features
            You should be able to access as class attribute to:
            ranking_index
                 A list of features indexes sorted by ranks. ranking_index[0] returns the index of the best selected
                 feature according to scoring/ranking function
        """
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        if self.is_fitted:
            ranks = np.array([list(self.ranking_index).index(_) + 1 for _ in range(len(self.ranking_index))])
            return ranks <= self.n_selected_features
        else:
            raise NotFittedError('Fit method must be used before calling get_support')
