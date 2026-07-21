import numbers
import numpy as np
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr, rankdata
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import validate_data, column_or_1d, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils import check_random_state
from abc import abstractmethod, ABC
# TODO : other methods such as relief ? 

class FeatureSelection(SelectorMixin, BaseEstimator, ABC):
    """
    Abstract class for feature selection
     """
    @abstractmethod
    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, y):
        """
        A method to fit feature selection.
        Parameters
        ----------
        X : pandas dataframe or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : pandas dataframe or array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        Instance of fitted estimator.
        """
        raise NotImplementedError

class FilterFeatureSelection(FeatureSelection):
    """A general class to handle feature selection according to a scoring/ranking method. Bootstrap is implemented
    to ensure stability in feature selection process. Only works for classification purposes.
    Parameters
    ----------
    method: str or callable
        Method used to score/rank features. Either str or callable
        If str inbuild function named as str is called, must be one of following:
            'wlcx_score' or 'kruskal_score': score of kruskall wallis test
            'auc_roc': scoring with area under the roc curve
            'pearson_corr': scoring with pearson correlation coefficient between features and labels
            'spearman_corr': scoring with spearman correlation coefficient between features and labels
            'mi_classif': scoring with mutual information between features and labels
            'mrmr': ranking according to Minimum redundancy Maximum relevance algorithm
        pearson_corr and spearman_corr require a numerical target.
        For categorical labels, encode y before fitting the selector.
        For multiclass targets, the numerical encoding imposes an ordering
        and should only be used when that ordering is scientifically meaningful.
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
        'importance_score', 'mean', 'stability_selection_aggregation', 'exponential_weighting'
        If callable method must take ((bootstrap_ranks, n_selected_features) as arguments and return the
        aggregate rank associated with each feature in the same order as features are in feature_array
    ranking_done: boolean (default=False)
        Indicate whether the method return a score or directly calculate ranks
    score_indicator_lower: boolean (default=None)
        Choose whether lower score correspond to higher rank for the rank calculation or higher score is better,
        `True` means lower score is better. Determined automatically for inbuild functions
    random_state: int, RandomState instance or None (default=None)
        Controls the randomness of the estimator.
    """
    _SCORING_METHODS = {"pearson_corr": False, "spearman_corr": False, 
                        "auc_roc": False, "mi_classif": False, "wlcx_score": False,
                        "kruskal_score": False}
    _RANKING_METHODS = {'mrmr'}
    # TODO: implement more inbuild functions like multivariate selection algorithm
    #  (see skfeature : https://github.com/jundongl/scikit-feature)
    # TODO : implement t-test, chisquare
    _AGGREGATION_METHODS = {'enhanced_borda', 'borda', 'importance_score', 'mean', 
                            'stability_selection_aggregation', 'exponential_weighting'}

    def __init__(self, method='auc_roc', bootstrap=False, n_bsamples=100, n_selected_features=20,
                 ranking_aggregation=None, ranking_done=False, score_indicator_lower=None, random_state=None):
        self.method = method
        self.ranking_done = ranking_done
        self.score_indicator_lower = score_indicator_lower

        self.bootstrap = bootstrap
        self.n_bsamples = n_bsamples
        self.n_selected_features = n_selected_features
        self.ranking_aggregation = ranking_aggregation
        self.random_state = random_state
        
    def _validate_parameters(self):
        if not isinstance(self.bootstrap, bool):
            raise TypeError("bootstrap must be a boolean.")
        if (not isinstance(self.n_bsamples, numbers.Integral) or isinstance(self.n_bsamples, bool) or self.n_bsamples < 1):
            raise ValueError("n_bsamples must be a positive integer.")
        if not isinstance(self.ranking_done, bool):
            raise TypeError("ranking_done must be a boolean.")
        if (self.score_indicator_lower is not None and not isinstance(self.score_indicator_lower, bool)):
            raise TypeError("score_indicator_lower must be a boolean or None.")
        if (self.bootstrap and self.ranking_aggregation is None):
            raise ValueError("ranking_aggregation must be provided when bootstrap=True.")
        if (not self.bootstrap and self.ranking_aggregation is not None):
            raise ValueError("ranking_aggregation can only be used when bootstrap=True.")
        check_random_state(self.random_state)
    
    def _resolve_selection_method(self):
        if callable(self.method):
            return self.method, self.ranking_done, self.score_indicator_lower
        
        if not isinstance(self.method, str):
            raise TypeError("method must be a string or a callable.")

        method_name = self.method.lower()
        if method_name in self._RANKING_METHODS:
            return getattr(self, method_name), True, None

        if method_name in self._SCORING_METHODS:
            lower_is_better = self._SCORING_METHODS[method_name]
            return getattr(self, method_name), False, lower_is_better
        
        valid_methods = sorted(set(self._SCORING_METHODS) | set(self._RANKING_METHODS))
        raise ValueError(f"Unknown method {self.method!r}. Expected one of {valid_methods}.")

    def _resolve_aggregation_method(self):
        if callable(self.ranking_aggregation):
            return self.ranking_aggregation

        if not isinstance(self.ranking_aggregation, str):
            raise TypeError('ranking_aggregation option must be a callable or a string')

        method_name = self.ranking_aggregation.lower()
        if method_name not in self._AGGREGATION_METHODS:
            raise ValueError(f"Unknown ranking_aggregation {self.ranking_aggregation!r}. "
                             f"Expected one of  {sorted(self._AGGREGATION_METHODS)}.")
        return getattr(self, method_name)
      
    def _resolve_n_selected_features(self, n_features):
        if self.n_selected_features is None:
            return n_features

        if not isinstance(self.n_selected_features, numbers.Integral) or isinstance(self.n_selected_features, bool):
            raise TypeError("n_selected_features must be an integer or None.")

        if not 1 <= self.n_selected_features <= n_features:
            raise ValueError("n_selected_features must be between 1 and the number of input features.")
        return int(self.n_selected_features)

    def _generate_bootstrap_indices(self, y):
        rng = check_random_state(self.random_state)        
        return np.asarray([self._bootstrap_indices(y, rng) for _ in range(self.n_bsamples)], dtype=int)
    
    @staticmethod
    def _bootstrap_indices(y, rng):
        class_indices = [np.flatnonzero(y == label) for label in np.unique(y)]        
        sampled = [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        indices = np.concatenate(sampled)
        rng.shuffle(indices)
        return indices

    def _get_support_mask(self):
        check_is_fitted(self, "accepted_features_index_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.accepted_features_index_] = True
        return mask

    # === Scoring methods ===
    @staticmethod
    def kruskal_score(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        labels = np.unique(y)
        for i in range(n_features):
            X_by_label = [X[:, i][y == _] for _ in labels]
            try:
                statistic, pvalue = kruskal(*X_by_label)
            except ValueError:
                statistic = 0
            score[i] = statistic if np.isfinite(statistic) else 0
        return score
    
    wlcx_score = kruskal_score

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
                    positive_label = labels[1]
                else:
                    positive_label = labels[0]
                fpr, tpr, thresholds = roc_curve(y, X[:, i], pos_label=positive_label)
                roc_auc = auc(fpr, tpr)
                score[i] = roc_auc if np.isfinite(roc_auc) else 0.5
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
            score[i] = abs(correlation) if np.isfinite(correlation) else 0.0
        return score

    @staticmethod
    def spearman_corr(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        for i in range(n_features):
            correlation, pvalue = spearmanr(X[:, i], y)
            score[i] = abs(correlation) if np.isfinite(correlation) else 0.0
        return score

    def mi_classif(self, X, y):
        score = mutual_info_classif(X, y, random_state=self.random_state)
        return score

    # === Ranking methods ===
    @staticmethod
    def mrmr(X, y):
        try:
            # TODO : not maintain anymore. Get mrmr from other package ? https://pypi.org/project/mrmr-selection/ ? 
            from skfeature.function.information_theoretical_based.MRMR import mrmr
        except ImportError as exc:
            raise ImportError("The 'mrmr' method requires scikit-feature.") from exc
        
        n_samples, n_features = X.shape
        rank_index, _, _ = mrmr(X, y, n_selected_features=n_features)
        ranks = np.array([list(rank_index).index(_) + 1 for _ in range(len(rank_index))])
        return ranks

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
        return rankdata(np.sum(np.exp(-bootstrap_ranks / n_selected_features), axis=0) * -1, method='ordinal')

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
        occurence = np.sum(bootstrap_ranks <= n_selected_features, axis=0) ** 2
        importance_score = np.divide(np.sum(np.sqrt(bootstrap_ranks), axis=0), occurence,
                                     out=np.full(occurence.shape, np.inf), where=occurence != 0)
        return rankdata(importance_score, method='ordinal')

    # === Applying feature selection ===
    def fit(self, X, y=None):
        """
        A method to fit feature selection.
        Parameters
        ----------
        X : pandas dataframe or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : pandas dataframe or array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        Instance of fitted estimator.
        """
        self._validate_parameters()
        X, y = validate_data(self, X=X, y=y, reset=True, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        y = column_or_1d(y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("y must contain at least two classes.")
        fs_func, ranking_done, score_indicator_lower = self._resolve_selection_method()            
        self.n_selected_features_ = self._resolve_n_selected_features(self.n_features_in_)
        if self.bootstrap:
            aggregation_method = self._resolve_aggregation_method()
            bsamples_index = self._generate_bootstrap_indices(y)
            if ranking_done:
                bootstrap_ranks = np.array([self._validate_method_output(fs_func(X[_, :], y[_]), self.n_features_in_) for _ in bsamples_index])
            else:
                if score_indicator_lower is None:
                    raise ValueError('score_indicator_lower option must be given if a user scoring function is used')
                bootstrap_scores = np.array([self._validate_method_output(fs_func(X[_, :], y[_]), self.n_features_in_) for _ in bsamples_index])
                if not score_indicator_lower:
                    bootstrap_scores *= -1
                bootstrap_ranks = np.array([rankdata(_, method="ordinal") for _ in bootstrap_scores])
            bootstrap_ranks_aggregated = self._validate_method_output(aggregation_method(bootstrap_ranks, self.n_selected_features_), self.n_features_in_)
            self.feature_ranks_ = bootstrap_ranks_aggregated
            self.ranking_index_ = np.argsort(bootstrap_ranks_aggregated, kind="stable")
        else:
            if ranking_done:
                ranks = self._validate_method_output(fs_func(X, y), self.n_features_in_)
            else:
                if score_indicator_lower is None:
                    raise ValueError('score_indicator_lower option must be given if a user scoring function is used')
                score = self._validate_method_output(fs_func(X, y), self.n_features_in_)
                if not score_indicator_lower:
                    score *= -1
                ranks = rankdata(score, method='ordinal')
            self.feature_ranks_ = ranks
            self.ranking_index_ = np.argsort(ranks, kind="stable")
        self.accepted_features_index_ = np.asarray(self.ranking_index_[:self.n_selected_features_])
        return self
    
    @staticmethod
    def _validate_method_output(values, n_features):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("The feature-selection method must return a one-dimensional array.")
        if values.shape[0] != n_features:
            raise ValueError("The method must return one value per feature.")
        if not np.all(np.isfinite(values)):
            raise ValueError("The method returned non-finite values.")
        return values


class BorutaShapFeatureSelection(FeatureSelection, MetaEstimatorMixin):
    """
    Wrapper around BorutaShap package which is based on Boruta feature selection algorithm [1] with the addition
    of feature importance assessed using SHAP values [2]. Tree based algorithm are used to allow the use of
    TreeExplainer to compute SHAP values that scales linearly with the number of observations [3].
    https://github.com/Ekeany/Boruta-Shap
    [1] Kursa and Rudnicki, Feature Selection with the Boruta Package. Journal of Statistical Software, 2010
    [2] Lundberg and Lee, A Unified Approach to Interpreting Model Predictions. arXiv:1705.07874, 2017
    [3] Lundberg et al., Consistent Individualized Feature Attribution for Tree Ensembles. arXiv:1802.03888, 2019
    Parameters
    ----------
    model: estimator
        If no model specified then a base Random Forest will be returned otherwise the specifed model will
        be returned.

    importance_measure: str (default='Shap')
        Which importance measure too use either Shap or Gini/Gain

    classification: boolean (default=True)
        If true then the problem is either a binary or multiclass problem otherwise if false then it is regression

    percentile: int (default=100)
        An integer ranging from 0-100 it changes the value of the max shadow importance values. Thus, lowering its value
        would make the algorithm more lenient.

    pvalue: float (default=0.05)
        A float used as a significance level again if the p-value is increased the algorithm will be more lenient making
        it smaller would make it more strict also by making the model more strict could impact runtime making it slower.
        As it will be less likley to reject and accept features.

    random_state: int, RandomState instance or None (default=None)
        Controls the randomness of the estimator.

    sample: boolean (default=False)
        If true then a rowise sample of the data will be used to calculate the feature importance values

    train_or_test: string (default='test')
        Decides whether the feature importance should be calculated on out of sample data see the dicussion here.
        https://compstat-lmu.github.io/iml_methods_limitations/pfi-data.html#introduction-to-test-vs.training-data

    normalize: boolean (default=True)
        If true the importance values will be normalized using the z-score formula

    verbose: boolean (default=False)
        A flag indicator to print out all the rejected or accepted features.

    stratify: np.array (default=None)
        allows the train test splits to be stratified based on given values.
    """
    def __init__(self, model=None, importance_measure='Shap', classification=True, percentile=100, pvalue=0.05,
                 n_trials=20, random_state=None, sample=False, train_or_test='test', normalize=True, verbose=False,
                 stratify=None):
        self.model = model
        self.importance_measure = importance_measure
        self.classification = classification
        self.percentile = percentile
        self.pvalue = pvalue
        self.n_trials = n_trials
        self.random_state = random_state
        self.sample = sample
        self.train_or_test = train_or_test
        self.normalize = normalize
        self.verbose = verbose
        self.stratify = stratify
        
    def _validate_parameters(self) -> None:
        if not isinstance(self.classification, bool):
            raise TypeError("classification must be a boolean.")

        if (not isinstance(self.percentile, numbers.Real) or isinstance(self.percentile, bool) or not 0 <= self.percentile <= 100):
            raise ValueError("percentile must be in [0, 100].")

        if (not isinstance(self.pvalue, numbers.Real) or isinstance(self.pvalue, bool) or not 0 < self.pvalue < 1):
            raise ValueError("pvalue must be in (0, 1).")

        if (not isinstance(self.n_trials, numbers.Integral) or isinstance(self.n_trials, bool) or self.n_trials < 1):
            raise ValueError("n_trials must be a positive integer.")

        if not isinstance(self.sample, bool):
            raise TypeError("sample must be a boolean.")

        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a boolean.")

        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean.")

        if self.train_or_test not in {"train", "test"}:
            raise ValueError("train_or_test must be either 'train' or 'test'.")

        check_random_state(self.random_state)

    def _get_support_mask(self):
        check_is_fitted(self, "accepted_features_index_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.accepted_features_index_] = True
        return mask

    def fit(self, X, y):
        """
        A method to fit feature selection.
        Parameters
        ----------
        X : pandas dataframe or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : pandas dataframe or array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        Instance of fitted estimator.
        """
        try:
            from BorutaShap import BorutaShap
        except ImportError as exc:
            raise ImportError("The 'BorutaShapFeatureSelection' class requires BorutaShap.") from exc
        self._validate_parameters()
        X, y = validate_data(self, X=X, y=y, reset=True, ensure_2d=True, dtype="numeric", ensure_all_finite=True)
        y = column_or_1d(y)
        if self.classification:
            check_classification_targets(y)
            self.classes_ = np.unique(y)
            if self.classes_.size < 2:
                raise ValueError("y must contain at least two classes.")
        else:
            if type_of_target(y) not in {"continuous"}:
                raise ValueError("This method requires a continuous regression target.")
        model = None if self.model is None else clone(self.model)
        self.feature_selector_ = BorutaShap(model=model, importance_measure=self.importance_measure,
                                      classification=self.classification, percentile=self.percentile, pvalue=self.pvalue)
        X = pd.DataFrame(X, columns=[str(_) for _ in range(X.shape[1])])
        self.feature_selector_.fit(X, y, n_trials=self.n_trials, random_state=self.random_state, sample=self.sample,
                             train_or_test=self.train_or_test, normalize=self.normalize, verbose=self.verbose,
                             stratify=self.stratify)
        if self.feature_selector_.tentative:
            # Might get some undecided features after fit
            # Method which compares the median values of the max shadow feature and the undecided features
            self.feature_selector_ .TentativeRoughFix()
        self.accepted_features_index_ = [int(_) for _ in self.feature_selector_.accepted]
        return self