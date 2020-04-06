import string
import warnings
import numpy as np
import pandas as pd
from collections.abc import Mapping
from sklearn.pipeline import Pipeline as skPipeline
from imblearn.pipeline import Pipeline as imbPipeline
from Statistical_analysis.dimensionality_reduction import DimensionalityReduction
from Statistical_analysis.feature_selection import FeatureSelection
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._scorer import check_scoring
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator


class NestedCV(BaseEstimator):
    """
    A sklearn wrapper to handle Nested Cross Validation
    Parameters
    ----------
    pipeline_dic: dictionary {str: callable}
        Dictionary containing the steps with which the pipeline will be constructed. Pipeline steps names (string) as
        keys and sklearn-like transform object (callable) as value except the last object that must be
        an sklearn-like estimator (callable).
        Steps will be chained in the order in which they are given.
        If some steps include callable from imblearn package
        (https://imbalanced-learn.readthedocs.io/en/stable/index.html) imblearn_pipeline option must be set to True (see
        cv_options argument).
        If key is either 'FeatureSelection' or 'DimensionalityReduction' the value can be either str or callable.
        Keyword options of each callable can be given in cv_options (see cv_options argument)
        Example:
        pipeline_dic = {'scale': sklearn.preprocessing.StandardScaler,
                        'oversampling': imblearn.over_sampling.SMOTE,
                        'DimensionalityReduction': sklearn.decomposition.PCA,
                        'FeatureSelection': 'mw',
                        'classifier': sklearn.linear_model.LogisiticRegression}
    params_grid: dict or list of dictionaries
        Dictionary with step names (string) as keys and dictionary with parameters grid as values,
        or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.
        Parameters grid given as value must be dictionary with parameters names (string) as keys and lists of parameter
        settings to try as values as taken by sklearn.model_selection.GridSearchCV.
        Example with previous pipeline:
        params_grid = [{'DimensionalityReduction': {'n_components': [0.95, 0.99], 'svd_solver': ['full']},
                        'FeatureSelection': {'n_selected_features': [5,10,15,20,n_features]},
                        'classifier': {'penalty': ['l1'], 'C': np.arange(0.001, 1, 0.002), 'solver': ['saga']}},
                       {'DimensionalityReduction': {'n_components': [0.95, 0.99], 'svd_solver': ['full']},
                        'FeatureSelection': {'n_selected_features': [5,10,15,20,n_features]},
                        'classifier': {'penalty': ['elasticnet'], 'C': np.arange(0.001, 1, 0.002), 'solver': ['saga'],
                                       'l1_ratio': np.arange(0.1, 1, 0.1)}}]
    outer_cv: int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, KFold is used.
        See sklearn.model_selection.GridSearchCV for more details.
    inner_cv: int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, KFold is used.
        See sklearn.model_selection.GridSearchCV for more details.
    n_jobs: int or None, optional (default=None)
        Number of jobs to run in parallel in inner loop. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
        See sklearn.model_selection.GridSearchCV for more details.
    imblearn_pipeline: boolean (default=False)
        Indicate whether callable from imblearn package are used in pipeline
    pipeline_options: dict (default={})
        Dictionary with step names (string) as key and dictionary of the associated keywords as values. These
        keywords will be used to construct the object while creating the pipeline. If not specified, the object will
        be constructed with default parameters.
        Example with previous pipeline:
        pipeline_options = {'oversampling': {'sampling_strategy': 'minority_class'},
                            'DimensionalityReduction': {'n_components': 0.95},
                            'FeatureSelection': {'bootstrap': True, 'n_bsamples': 200, 'n_selected_features': 10,
                                                 'ranking_aggregation': 'importance_score'}}
    metric: string, callable or None, (default='roc_auc')
        A single string (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
        or a callable (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) to evaluate the
        predictions on the test set.
        If None, the estimator’s score method is used.
        In contrast to sklearn.model_selection.GridSearhCV it does NOT handle multimetric scoring.
        See sklearn.model_selection.GridSearchCV for more details.
    verbose: int (default=1)
        Controls the verbosity: the higher, the more messages.
    refit: boolean (default=True)
        Refit an estimator using the whole dataset in two steps:
        1. Hyperparameter optimization with a gridsearch cross-validation (same parameter as outer CV).
        2. Refit an estimator using the best found parameters on the whole dataset.
    return_train_score: boolean (default=False)
        If False, the cross_validation results attribute will not include training scores.
        Computing training scores is used to get insights on how different parameter settings impact
        the overfitting/underfitting trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the parameters that yield the best
        generalization performance.
    """
    def __init__(self, pipeline_dic, params_dic, outer_cv=5, inner_cv=5, n_jobs=None, imblearn_pipeline=False,
                 pipeline_options={}, metric='roc_auc', verbose=1, refit=True, return_train_score=False,
                 random_state=None):
        self.imblearn_pipeline = imblearn_pipeline
        self.pipeline_options = pipeline_options
        self.pipeline_dic = pipeline_dic
        self.params_dic = params_dic
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.n_jobs = n_jobs
        self.metric = metric
        self.verbose = verbose
        self.refit = refit
        self.return_train_score = return_train_score
        self.random_state = random_state

    @staticmethod
    def _string_processing(key):
        table = str.maketrans({key: '' for key in string.punctuation})
        key = key.translate(table)
        table = str.maketrans({key: '' for key in string.whitespace})
        key = key.translate(table)
        return key.lower()

    def _check_pipeline_dic(self, pipeline_dic):
        if not isinstance(pipeline_dic, Mapping):
            raise TypeError('pipeline_dic argument must be a dictionary')
        for step in pipeline_dic.keys():
            if self._string_processing(step) == 'dimensionalityreduction' or self._string_processing(step) == 'featureselection':
                if not callable(pipeline_dic[step]) and not isinstance(pipeline_dic[step], str):
                    raise TypeError('Dictionary values must be a callable or a string when the associated key is '
                                    'DimensionalityReduction or FeatureSelection')
            else:
                if not callable(pipeline_dic[step]):
                    raise TypeError('Dictionary value must be a callable if associated key is not '
                                    'DimensionalityReduction or FeatureSelection')

    def _get_parameters_grid(self, parameters_grid):
        if isinstance(parameters_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            parameters_grid = [parameters_grid]
        new_parameters_grid = []
        for grid in parameters_grid:
            parameters_dic = {}
            for step in grid.keys():
                for params in grid[step].keys():
                    if self._string_processing(step) == 'dimensionalityreduction':
                        parameters_dic[step + '__method__' + params] = grid[step][params]
                    else:
                        parameters_dic[step + '__' + params] = grid[step][params]
            new_parameters_grid.append(parameters_dic)
        return new_parameters_grid

    def _get_pipeline(self, pipeline_dic):
        pipeline_steps = []
        for step in pipeline_dic.keys():
            kwargs = self.pipeline_options.get(step, {})
            if not kwargs:
                warnings.warn('Default parameters are loaded for {0} (see corresponding class for detailed kwargs)'.format(step))
            if self._string_processing(step) == 'dimensionalityreduction':
                if callable(pipeline_dic[step]):
                    step_object = DimensionalityReduction(pipeline_dic[step](**kwargs))
                else:
                    step_object = DimensionalityReduction(pipeline_dic[step])
            elif self._string_processing(step) == 'featureselection':
                step_object = FeatureSelection(pipeline_dic[step], **kwargs)
            else:
                step_object = pipeline_dic[step](**kwargs)
            pipeline_steps.append((step, step_object))
        if self.imblearn_pipeline:
            return imbPipeline(pipeline_steps)
        else:
            return skPipeline(pipeline_steps)

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_params_`` '
                                 'attribute'
                                 % (type(self).__name__, method_name))
        else:
            check_is_fitted(self)

    @staticmethod
    def _check_X_Y(X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        X = np.array(X)
        assert len(X.shape) == 2, 'X array must 2D'
        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.to_numpy()
            y = np.array(y)
        return X, y

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Fit Nested CV with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        Returns
            -------
            It will not return directly the values, but it's accessable from the class object it self.
            You should be able to access:
            outer_models
                 A dictionary to access the train indexes, the test indexes and the model  of each outer loop
                 for further post-processing. Keys are respectively train, test and model with values being
                 lists of length outer_cv.get_n_splits().
            outer_results
                A dictionary to access the outer test scores, the best inner scores, the best inner parameters (and
                outer_train_scores if return_train_score == True). Keys are respectively outer_test_score,
                best_inner_score, best_inner_params (and outer_train_score) with values being lists of length
                outer_cv.get_n_splits().
            inner_results
                A list of dictionary of length outer_cv.get_n_splits().
                Each dictionary having params, mean_test_score, std_test_score (and mean_train_score, std_train_score
                if return_train_score == True) as keys and values being the list of params or associated results
                over the inner loops.
            best_estimator_
                Model when refit on the whole dataset with hyperparameter optimized by GridSearch CV.
                Available only if refit == True.
        """
        X, y = self._check_X_Y(X, y)

        self._check_pipeline_dic(self.pipeline_dic)
        self.model = self._get_pipeline(self.pipeline_dic)
        self.params_grid = self._get_parameters_grid(self.params_dic)

        outer_cv = check_cv(self.outer_cv, y, is_classifier(self.model[-1]))  # Last element of pipeline = estimator
        inner_cv = check_cv(self.inner_cv, y, is_classifier(self.model[-1]))  # Last element of pipeline = estimator

        self.outer_models = {'train': [], 'test': [], 'model': []}
        self.outer_results = {'outer_test_score': [], 'best_inner_score': [], 'best_inner_params': []}
        self.inner_results = []
        if self.return_train_score:
            self.outer_results.update({'outer_train_score': []})
        scorer = check_scoring(self.model, scoring=self.metric)
        for k_outer, (train_outer_index, test_outer_index) in enumerate(outer_cv.split(X, y, groups)):
            if self.verbose > 1:
                print('\n-----------------\n{0}/{1} <-- Current outer fold'.format(k_outer + 1, outer_cv.get_n_splits()))
            X_train_outer, X_test_outer = X[train_outer_index], X[test_outer_index]
            y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]
            pipeline_inner = GridSearchCV(self.model, self.params_grid, scoring=scorer, n_jobs=self.n_jobs, cv=inner_cv,
                                          return_train_score=self.return_train_score, verbose=self.verbose - 1)
            pipeline_inner.fit(X_train_outer, y_train_outer, groups=groups, **fit_params)
            self.inner_results.append({'params': pipeline_inner.cv_results_['params'],
                                       'mean_test_score': pipeline_inner.cv_results_['mean_test_score'],
                                       'std_test_score': pipeline_inner.cv_results_['std_test_score']})
            if self.return_train_score:
                self.inner_results[-1].update({'mean_train_score': pipeline_inner.cv_results_['mean_train_score'],
                                               'std_train_score': pipeline_inner.cv_results_['std_train_score']})
            if self.verbose > 2:
                for params_dict in pipeline_inner.cv_results_['params']:
                    mean_test_score = pipeline_inner.cv_results_['mean_test_score']
                    index_params_dic = pipeline_inner.cv_results_['params'].index(params_dict)
                    print('\t\t Params: {0}, Mean inner score: {1}'.format(params_dict, mean_test_score[index_params_dic]))

            self.outer_results['best_inner_score'].append(pipeline_inner.best_score_)
            self.outer_results['best_inner_params'].append(pipeline_inner.best_params_)
            if self.return_train_score:
                self.outer_results['outer_train_score'].append(pipeline_inner.score(X_train_outer, y_train_outer))
            self.outer_results['outer_test_score'].append(pipeline_inner.score(X_test_outer, y_test_outer))
            if self.verbose > 1:
                print('\nResults for outer fold:\nBest inner parameters was: {0}'.format(self.outer_results['best_inner_params'][-1]))
                print('Outer score: {0}'.format(self.outer_results['outer_test_score'][-1]))
                print('Inner score: {0}'.format(self.outer_results['best_inner_score'][-1]))
            self.outer_models['train'].append(train_outer_index)
            self.outer_models['test'].append(test_outer_index)
            self.outer_models['model'].append(pipeline_inner.best_estimator_)
        if self.verbose > 0:
            print('\nOverall outer score (mean +/- std): {0} +/- {1}'.format(np.mean(self.outer_results['outer_test_score']),
                                                                             np.std(self.outer_results['outer_test_score'])))
            print('Best params by outer fold:')
            for i, params_dict in enumerate(self.outer_results['best_inner_params']):
                print('\t Outer fold {0}: {1}'.format(i + 1, params_dict))
            print('\n')

        # If refit is True Hyperparameter optimization on whole dataset and fit with best params
        if self.refit:
            print('=== Refit ===')
            pipeline_refit = GridSearchCV(self.model, self.params_grid, scoring=scorer, n_jobs=self.n_jobs,
                                                cv=outer_cv, verbose=self.verbose - 1)
            pipeline_refit.fit(X, y, groups=groups, **fit_params)
            self.best_estimator_ = pipeline_refit.best_estimator_

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        return self.best_estimator_.score(X, y)

    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.inverse_transform(Xt)

    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_
