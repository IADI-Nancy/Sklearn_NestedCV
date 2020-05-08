import time
import numpy as np
import numbers
from collections import defaultdict
from functools import partial
# from mlxtend.evaluate import BootstrapOutOfBag
from joblib import Parallel, delayed
from scipy.stats import rankdata
from itertools import product
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.utils import _message_with_time
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable, _check_fit_params
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import resample
from sklearn.metrics._scorer import _ProbaScorer, _ThresholdScorer


class Bootstrap_inner():
    def __init__(self, estimator, params_grid, scorers=None, n_jobs=None, refit=True, method=None, n_bsamples=200,
                 verbose=0, pre_dispatch='2*n_jobs', return_train_score=False, random_state=None):
        self.estimator = estimator
        self.params_grid = params_grid
        self.scorers = scorers
        self.n_jobs = n_jobs
        self.refit = refit
        self.method = method
        self.n_bsamples = n_bsamples
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score
        self.random_state = random_state

    def _format_results(self, candidate_params, scorers, n_splits, out):
        # From sklearn.model_selection.BaseSearchCV
        n_candidates = len(candidate_params)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_score_dicts, test_score_dicts, fit_time, score_time) = zip(*out)
        else:
            (test_score_dicts, fit_time, score_time) = zip(*out)

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        results = {}

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name], splits=True, rank=True)
            if self.return_train_score:
                _store('train_%s' % scorer_name, train_scores[scorer_name], splits=True)

        return results


    # @staticmethod
    # def no_information_rate(X, y, estimator, scorer):
    #     score_func = scorer._score_func
    #     score_kwargs = scorer._kwargs
    #     if isinstance(scorer, _ProbaScorer):
    #         predictions = estimator.predict_proba(X)
    #     elif isinstance(scorer, _ThresholdScorer):
    #         predictions = estimator.decision_function(X)
    #     else:
    #         predictions = estimator.predict(X)
    #     combinations = np.array(list(product(y, predictions)))
    #     return score_func(combinations[:, 0], combinations[:, 1], **score_kwargs)

    @staticmethod
    def no_information_rate(y_true, y_pred):
        pk = np.bincount(y_true)/len(y_true)
        qk = np.bincount(y_pred) / len(y_pred)
        return np.sum(pk * (1 - qk))

    def bootstrap_point632_score(self, estimator, parameters, X, y, train, test, method, fit_params):
        """
        Rewriting of : https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/bootstrap_point632.py
        Implementation of the .632 [1] and .632+ [2] bootstrap
        for supervised learning
        References:
        - [1] Efron, Bradley. 1983. "Estimating the Error Rate
          of a Prediction Rule: Improvement on Cross-Validation."
          Journal of the American Statistical Association
          78 (382): 316. doi:10.2307/2288636.
        - [2] Efron, Bradley, and Robert Tibshirani. 1997.
          "Improvements on Cross-Validation: The .632+ Bootstrap Method."
          Journal of the American Statistical Association
          92 (438): 548. doi:10.2307/2965703.
        """
        if not isinstance(self.n_bsamples, int) or self.n_bsamples < 1:
            raise ValueError('Number of bootstrap splits must be greater than 1. Got %s.' % self.n_bsamples)

        if self.verbose > 1:
            if parameters is None:
                msg = ''
            else:
                msg = '%s' % (', '.join('%s=%s' % (k, v)
                                        for k, v in parameters.items()))
            print("[Bootstrap_%s] %s %s" % (method, msg, (64 - len(msg)) * '.'))

        # Adjust length of sample weights
        fit_params = fit_params if fit_params is not None else {}
        fit_params = _check_fit_params(X, fit_params, train)

        if parameters is not None:
            # clone after setting parameters in case any parameters
            # are estimators (like pipeline steps)
            # because pipeline doesn't clone steps in fit
            cloned_parameters = {}
            for k, v in parameters.items():
                cloned_parameters[k] = clone(v, safe=False)
            estimator.set_params(**cloned_parameters)

        start_time = time.time()

        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)

        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

        fit_time = time.time() - start_time
        test_scores = {_: self.scorers[_](estimator, X_test, y_test) for _ in self.scorers}

        if self.return_train_score:
            train_scores = {_: self.scorers[_](estimator, X_train, y_train) for _ in self.scorers}

        if method == 'oob':
            final_test_scores = test_scores
            score_time = time.time() - start_time - fit_time

        else:
            if not self.return_train_score:
                train_scores = {_: self.scorers[_](estimator, X_train, y_train) for _ in self.scorers}
            if method == '.632+':
                weight = {}
                for scorer_name in self.scorers:
                    score_sign = self.scorers[scorer_name]._sign
                    # gamma = self.no_information_rate(X, y, estimator, self.scorers[scorer_name])
                    gamma = self.no_information_rate(y, estimator.predict(X))  # Only work for classifier
                    # Original calculation in mlxtend which got an error in denominator
                    # R = (-(test_scores[scorer_name] - train_scores[scorer_name])) / (gamma - (1 - test_scores[scorer_name]))
                    R = (-score_sign * (test_scores[scorer_name] - train_scores[scorer_name])) / \
                        (gamma - (-score_sign * train_scores[scorer_name]))
                    weight[scorer_name] = 0.632 / (1 - 0.368 * R)

            else:
                weight = {_: 0.632 for _ in self.scorers}

            final_test_scores = {_: weight[_] * test_scores[_] + (1. - weight[_]) * train_scores[_] for _ in self.scorers}
            score_time = time.time() - start_time - fit_time
            
            if self.verbose > 2:
                for scorer_name in sorted(test_scores):
                    msg += ", %s=" % scorer_name
                    if self.return_train_score:
                        msg += "(train=%.3f," % train_scores[scorer_name]
                        msg += " test=%.3f)" % test_scores[scorer_name]
                    else:
                        msg += "%.3f" % test_scores[scorer_name]

            if self.verbose > 1:
                total_time = score_time + fit_time
                print(_message_with_time('Bootstrap_%s' % method, msg, total_time))

        ret = [train_scores, final_test_scores] if self.return_train_score else [final_test_scores]
        ret.extend([fit_time, score_time])

        return ret

    def fit(self, X, y, groups=None, **fit_params):
        # Inspired from sklearn.model_selection.GridsearchCV

        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scorers)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        base_estimator = clone(self.estimator)
        candidate_params = ParameterGrid(self.params_grid)
        oob = BootstrapOutOfBag(n_splits=self.n_bsamples, random_seed=self.random_state)
        n_splits = oob.get_n_splits()
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)

        all_candidate_params = []
        all_out = []

        candidate_params = list(candidate_params)
        n_candidates = len(candidate_params)

        if self.verbose > 0:
            print("Fitting {0} bootstrap samples for each of {1} candidates,"
                  " totalling {2} fits".format(n_splits, n_candidates, n_candidates * n_splits))

        out = parallel(delayed(self.bootstrap_point632_score)(clone(base_estimator), parameters, X, y, train, test,
                                                              self.method, fit_params)
                       for parameters, (train, test) in product(candidate_params, oob.split(X, y)))

        all_candidate_params.extend(candidate_params)
        all_out.extend(out)

        results = self._format_results(all_candidate_params, self.scorers, n_splits, all_out)

        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"  % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class BootstrapOutOfBag(object):
    """
    Rewriting of mlxtend.evaluate.bootstrap_outofbag with sklearn.utils.resample to havec always the same split
    according to a random state
    Parameters
    ----------

    n_splits : int (default=200)
        Number of bootstrap iterations.
        Must be larger than 1.

    random_seed : int (default=None)
        If int, random_seed is the seed used by
        the random number generator.


    Returns
    -------
    train_idx : ndarray
        The training set indices for that split.

    test_idx : ndarray
        The testing set indices for that split.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/BootstrapOutOfBag/

    """

    def __init__(self, n_splits=200, random_seed=None):
        self.random_seed = random_seed

        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError('Number of splits must be greater than 1.')
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """

        y : array-like or None (default: None)
            Argument is not used and only included as parameter
            for compatibility, similar to `KFold` in scikit-learn.

        groups : array-like or None (default: None)
            Argument is not used and only included as parameter
            for compatibility, similar to `KFold` in scikit-learn.


        """
        rng = np.random.RandomState(self.random_seed)
        sample_idx = np.arange(X.shape[0])
        set_idx = set(sample_idx)

        for _ in range(self.n_splits):
            # train_idx = rng.choice(sample_idx,
            #                        size=sample_idx.shape[0],
            #                        replace=True)
            # test_idx = np.array(list(set_idx - set(train_idx)))
            train_idx = resample(sample_idx, random_state=_)
            test_idx = np.array(list(set_idx - set(train_idx)))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility with scikit-learn.

        y : object
            Always ignored, exists for compatibility with scikit-learn.

        groups : object
            Always ignored, exists for compatibility with scikit-learn.

        Returns
        -------

        n_splits : int
            Returns the number of splitting iterations in the cross-validator.

        """
        return self.n_splits
