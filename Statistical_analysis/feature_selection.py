import numbers
import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr, rankdata
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from skfeature.function.information_theoretical_based.MRMR import mrmr
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from BorutaShap import BorutaShap
try:
    from stability_selection import StabilitySelection
except ModuleNotFoundError:
    raise ModuleNotFoundError('sklearn.externals.joblib was deprecated in v0.21 of scikit-learn. Must import directly'
                              'joblib. See PR https://github.com/scikit-learn-contrib/stability-selection/pull/34')

# TODO : class FeatureSelection, FilterFeatureSelection, AutomaticFeatureSelection (classe pour boruta etc)

class FeatureSelection(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
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
        n_selected_features: int or None or 'automatic', default = 20
            Number of the best features that must be selected by the class
            If None all the feature are returned (no feature selection)
            If 'automatic', the method used automatically select the number of selected features.
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
        classification: boolean (default=True)
            Define whether the current problem is a classification problem.
        random_state: int, RandomState instance or None (default=None)
            Controls the randomness of the estimator.
        fit_params: dict (default=None)
            Dictionary of additional parameters used to fit methods such as stability selection or boruta shap
        """
    scoring_methods = {'name': ['auc_roc', 'pearson_corr', 'spearman_corr', 'mi', 'wlcx_score'],
                       'score_indicator_lower': [False, False, False, False, False]}
    ranking_methods = ['mrmr']
    # TODO: implement more inbuild functions like multivariate selection algorithm
    #  (see skfeature : https://github.com/jundongl/scikit-feature)
    # TODO : implement t-test, chisquare
    ranking_aggregation_methods = ['enhanced_borda', 'borda', 'importance_score', 'mean',
                                   'stability_selection_aggregation', 'exponential_weighting']

    automatic_methods = ['boruta_shap', 'stability_selection']

    def __init__(self, method='mrmr', bootstrap=False, n_bsamples=100, n_selected_features=20,
                 ranking_aggregation=None, ranking_done=False, score_indicator_lower=None,
                 classification=True, random_state=None, fit_params=None):
        self.method = method
        self.ranking_done = ranking_done
        self.score_indicator_lower = score_indicator_lower

        self.bootstrap = bootstrap
        self.n_bsamples = n_bsamples
        self.n_selected_features = n_selected_features
        self.ranking_aggregation = ranking_aggregation
        self.classification = classification
        self.random_state = random_state
        self.fit_params = fit_params
        self.accepted_features_index = None
        self.is_fitted = False

    def _get_fs_func(self):
        if callable(self.method):
            return self.method
        elif isinstance(self.method, str):
            method_name = self.method.lower()
            if method_name not in (self.scoring_methods['name'] + self.ranking_methods + self.automatic_methods):
                raise ValueError('If string method must be one of : %s. '
                                 '%s was passed' % (str(self.scoring_methods['name'] + self.ranking_methods + self.automatic_methods),
                                                    method_name))
            if method_name not in self.automatic_methods:
                if method_name in FeatureSelection.ranking_methods:
                    self.ranking_done = True
                else:
                    self.ranking_done = False
                    self.score_indicator_lower = self.scoring_methods['score_indicator_lower'][self.scoring_methods['name'].index(self.method)]
            else:
                self.ranking_done = False
                self.score_indicator_lower = None
                self.n_selected_features = 'automatic'
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

    def _get_bsamples_index(self, y):
        bsamples_index = []
        n = 0
        while len(bsamples_index) < self.n_bsamples:
            bootstrap_sample = resample(range(self.n_samples), random_state=n)
            # Ensure all classes are present in bootstrap sample.
            if len(np.unique(y[bootstrap_sample])) == self.n_classes:
                bsamples_index.append(bootstrap_sample)
            n += 1
        bsamples_index = np.array(bsamples_index)
        return bsamples_index

    def _get_support_mask(self):
        mask = np.zeros(self.n_features, dtype=bool)
        mask[self.accepted_features_index] = True
        return mask

    # === Scoring methods ===
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

    def mi(self, X, y):
        score = mutual_info_classif(X, y, random_state=self.random_state)
        return score

    # === Ranking methods ===
    @staticmethod
    def mrmr(X, y):
        n_samples, n_features = X.shape
        rank_index, _, _ = mrmr(X, y, n_selected_features=n_features)
        ranks = np.array([list(rank_index).index(_) + 1 for _ in range(len(rank_index))])        
        return ranks

    # === Automatic methods ===
    def boruta_shap(self, X, y):
        """
        Wrapper around BorutaShap package which is based on Boruta feature selection algorithm [1] with the addition
        of feature importance assessed using SHAP values [2]. Tree based algorithm are used to allow the use of
        TreeExplainer to compute SHAP values that scales linearly with the number of observations [3].
        https://github.com/Ekeany/Boruta-Shap
        [1] Kursa and Rudnicki, Feature Selection with the Boruta Package. Journal of Statistical Software, 2010
        [2] Lundberg and Lee, A Unified Approach to Interpreting Model Predictions. arXiv:1705.07874, 2017
        [3] Lundberg et al., Consistent Individualized Feature Attribution for Tree Ensembles. arXiv:1802.03888, 2019
        """
        init_params_dic = {'model': self.fit_params.get('model', None),
                           'importance_measure': self.fit_params.get('importance_measure', 'shap'),
                           'classification': self.fit_params.get('classification', self.classification),
                           'percentile': self.fit_params.get('percentile', 100),
                           'pvalue': self.fit_params.get('pvalue', 0.05)}
        fit_params = self.fit_params.copy()
        for init_params in init_params_dic:
            if init_params in fit_params:
                fit_params.pop(init_params)
        feature_selector = BorutaShap(**init_params_dic)
        X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        fit_params['random_state'] = fit_params.get('random_state', self.random_state)
        fit_params['n_trials'] = fit_params.get('n_trials', self.n_bsamples)
        fit_params['verbose'] = fit_params.get('verbose', False)
        feature_selector.fit(X, y, **fit_params)
        if feature_selector.tentative:
            # Might get some undecided features after fit
            # Method which compares the median values of the max shadow feature and the undecided features
            feature_selector.TentativeRoughFix()
        self.accepted_features_index = [int(_) for _ in feature_selector.accepted]

    # === Ranking aggregation methods ===
    # Importance score method :
    # A comparative study of machine learning methods for time-to-event survival data for radiomics risk modelling
    # Leger et al., 2017, Scientific Reports
    # Other methods:
    # An extensive comparison of feature ranking aggregation techniques in bioinformatics.
    # Randall et al., 2012, IEEE
    @staticmethod
    def borda(bootstrap_ranks, n_selected_features):
        return rankdata(np.sum(bootstrap_ranks.shape[1] - bootstrap_ranks, axis=0) * -1, method='ordinal')

    @staticmethod
    def mean(bootstrap_ranks, n_selected_features):
        return rankdata(np.mean(bootstrap_ranks, axis=0), method='ordinal')

    @staticmethod
    def stability_selection_aggregation(bootstrap_ranks, n_selected_features):
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
    def fit(self, X, y=None):
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
            accepted_features_index
                 A list of features indexes that are selected by the feature selection process
        """
        X, y = self._check_X_Y(X, y)
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        fs_func = self._get_fs_func()
        if self.ranking_aggregation is not None:
            aggregation_method = self._get_aggregation_method()
        if self.n_selected_features == 'automatic':
            fs_func(X, y)
        else:
            self.n_selected_features = self._check_n_selected_feature(X, self.n_selected_features)
            if self.bootstrap:
                if self.ranking_aggregation is None:
                    raise ValueError('ranking_aggregation option must be given if bootstrap is True')
                bsamples_index = self._get_bsamples_index(y)
                if self.ranking_done:
                    bootstrap_ranks = np.array([fs_func(X[_, :], y[_]) for _ in bsamples_index])
                else:
                    if self.score_indicator_lower is None:
                        raise ValueError('score_indicator_lower option must be given if a user scoring function is used')
                    boostrap_scores = np.array([fs_func(X[_, :], y[_]) for _ in bsamples_index])
                    if not self.score_indicator_lower:
                        boostrap_scores *= -1
                    bootstrap_ranks = np.array([rankdata(_) for _ in boostrap_scores])
                bootstrap_ranks_aggregated = aggregation_method(bootstrap_ranks, self.n_selected_features)
                ranking_index = [list(bootstrap_ranks_aggregated).index(_) for _ in sorted(bootstrap_ranks_aggregated)]
            else:
                if self.ranking_done:
                    ranks = fs_func(X, y)
                else:
                    if self.score_indicator_lower is None:
                        raise ValueError('score_indicator_lower option must be given if a user scoring function is used')
                    score = fs_func(X, y)
                    if not self.score_indicator_lower:
                        score *= -1
                    ranks = rankdata(score, method='ordinal')
                ranking_index = [list(ranks).index(_) for _ in sorted(ranks)]
            self.accepted_features_index = ranking_index[:self.n_selected_features]
        self.is_fitted = True
        return self

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
            self.selected_features = super(FeatureSelection, self).transform(X)
            return self.selected_features
        else:
            raise NotFittedError('Fit method must be used before calling transform')

    def fit_transform(self, X, y=None, **fit_params):
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
        return self.fit(X, y, **fit_params).transform(X)