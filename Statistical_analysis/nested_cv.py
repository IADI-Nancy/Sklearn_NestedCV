import numpy as np
import pandas as pd
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Mapping
from sklearn.pipeline import Pipeline as skPipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics._scorer import check_scoring, _check_multimetric_scoring
from sklearn.utils.validation import check_is_fitted, check_memory
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, clone
from joblib import Memory
from shutil import rmtree
try:
    from .bayes_search_multiscore import BayesSearchCV
except ImportError:
    BayesSearchCV = None


class NestedCV(BaseEstimator):
    """
    Base class implementing nested cross-validation orchestration.
    
        Subclasses provide the concrete inner and final hyperparameter-search
        objects.
    
        Parameters
        ----------
        pipeline_dic : mapping of str to estimator specification
            Ordered mapping used to construct the pipeline. Keys are pipeline step
            names. Values may be an estimator class, a callable factory returning
            an estimator, an already instantiated scikit-learn compatible
            ``BaseEstimator``, or the string ``"passthrough"``.
    
            Intermediate steps of a standard scikit-learn pipeline must implement
            ``fit`` and ``transform``. When ``imblearn_pipeline=True``, compatible
            imbalanced-learn samplers are also accepted. The final step must
            implement ``fit``. Steps are chained in mapping insertion order.
    
            Step names have no special meaning. In particular,
            ``"DimensionalityReduction"`` and ``"FeatureSelection"`` are ordinary
            step names; their values must directly specify the estimator to use.
        params_dic : mapping or list of mappings
            Hyperparameter search space grouped by pipeline step. Each outer key
            must match a step name from ``pipeline_dic`` and each value must be a
            mapping from unprefixed estimator parameter names to candidate values
            accepted by the selected search method. Parameters are internally
            converted to scikit-learn's ``step_name__parameter_name`` format.
            A list of mappings defines separate search spaces.
        outer_cv : int, CV splitter or iterable, default=5
            Cross-validation strategy used to estimate generalization performance.
            Integer values are passed to :func:`sklearn.model_selection.check_cv`.
        inner_cv : int, CV splitter or iterable, default=5
            Cross-validation strategy used by the hyperparameter search within
            each outer-training set.
        n_jobs : int or None, default=None
            Number of parallel jobs used by the search object. ``None`` means one
            job unless a joblib backend is active; ``-1`` uses all processors.
        pre_dispatch : int or str, default='2*n_jobs'
            Number of jobs dispatched ahead of execution by the search object.
        imblearn_pipeline : bool, default=False
            If True, construct an :class:`imblearn.pipeline.Pipeline`; otherwise
            construct an :class:`sklearn.pipeline.Pipeline`. Set this to True when
            the pipeline contains a sampler such as SMOTE.
        pipeline_options : mapping of str to mapping or None, default=None
            Constructor arguments or parameter overrides for pipeline steps. Each
            key must match a step in ``pipeline_dic``. For estimator classes or
            factories, options are passed at construction. For estimator
            instances, the instance is cloned and options are applied with
            ``set_params``. Options are not allowed for ``"passthrough"`` steps.
        metric : str, callable, mapping or sequence, default='roc_auc'
            Scoring specification accepted by scikit-learn. With multiple metrics,
            ``refit_inner`` determines which metric is used to select the model
            evaluated in the outer loop.
        verbose : int, default=1
            Verbosity level. Larger values print more information.
        refit_outer : bool, default=True
            If True, after nested-CV performance estimation, run an additional
            hyperparameter search on the complete dataset using ``outer_cv`` and
            fit the selected pipeline on all samples. This final search does not
            alter the outer-fold performance estimate. It creates
            ``best_estimator_``, ``best_params_``, ``best_index_`` and
            ``cv_results_`` (and ``best_score_`` when available).
        error_score : 'raise' or numeric, default=np.nan
            Value assigned by the underlying search object when a candidate fit
            fails. ``'raise'`` propagates the exception.
        refit_inner : True, str or callable selection rule, default=True
            Rule used by the inner search to select and refit the best candidate on
            each complete outer-training set. ``False`` is not supported because a
            fitted estimator is required for evaluation on the outer test set.
            For multi-metric scoring, provide a scorer name. Grid and randomized
            search also accept the library's ``{scorer_name: callable}`` form.
        return_train_score : bool, default=False
            Whether inner-search results also contain training scores.
        memory : None, 'temporary', path-like or joblib.Memory, default=None
            Pipeline caching configuration. ``None`` disables caching.
            ``'temporary'`` creates one temporary shared cache for each inner
            search and for the optional final full-data search, then removes it
            when that search ends. A path-like value or ``joblib.Memory`` object is
            normalized with :func:`sklearn.utils.validation.check_memory` and is
            managed by the caller. Only intermediate pipeline transformers are
            cached; the final estimator is never cached
    """
    def __init__(self, pipeline_dic, params_dic, outer_cv=5, inner_cv=5, n_jobs=None, pre_dispatch='2*n_jobs',
                 imblearn_pipeline=False, pipeline_options=None, metric='roc_auc', verbose=1, refit_outer=True,
                 error_score=np.nan, refit_inner=True, return_train_score=False, memory=None):
        self.imblearn_pipeline = imblearn_pipeline
        self.pipeline_options = pipeline_options
        self.pipeline_dic = pipeline_dic
        self.params_dic = params_dic
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.metric = metric
        self.verbose = verbose
        self.refit_outer = refit_outer
        self.error_score = error_score
        self.refit_inner = refit_inner
        self.return_train_score = return_train_score
        self.memory = memory

    def _check_pipeline_dic(self):
        if not isinstance(self.pipeline_dic, Mapping):
            raise TypeError("pipeline_dic must be a mapping from step names "
            "to estimator classes, estimator instances, or "
            "'passthrough'.")
        if len(self.pipeline_dic) == 0:
            raise ValueError("pipeline_dic must contain at least one step.")
        for step in self.pipeline_dic.keys():
            if not isinstance(step, str):
                raise TypeError("Every pipeline step name must be a string.")
            if not step:
                raise ValueError("Pipeline step names cannot be empty.")
            if "__" in step:
                    raise ValueError(f"Pipeline step name {step!r} must not contain '__'.")
            is_estimator = isinstance(self.pipeline_dic[step], BaseEstimator)
            is_factory = callable(self.pipeline_dic[step])
            is_special_value = isinstance(self.pipeline_dic[step], str) and self.pipeline_dic[step] == "passthrough"
            if not (is_estimator or is_factory or is_special_value ):
                raise TypeError(
                    f"Pipeline step {step!r} must be an "
                    "estimator instance, an estimator class or "
                    "factory, or 'passthrough'. Got "
                    f"{type(self.pipeline_dic[step]).__name__}."
                )
    
    def _check_pipeline_options(self):
        unknown_steps = set(self.pipeline_options_) - set(self.pipeline_dic)
        if unknown_steps:
            raise ValueError(f"pipeline_options contains unknown pipeline steps: {sorted(unknown_steps)}.")
        for step_name, options in self.pipeline_options_.items():
            if not isinstance(options, Mapping):
                raise TypeError(f"Options for step {step_name!r} must be provided as a mapping.")

    def _get_parameters_grid(self):
        if isinstance(self.params_dic, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            self.params_dic = [self.params_dic]
        new_parameters_grid = []
        for grid in self.params_dic:
            if not isinstance(grid, Mapping):
                raise TypeError("Each parameter grid must be a mapping.")
            parameters_dic = {}
            for step in grid.keys():
                if not isinstance(grid[step], Mapping):
                    raise TypeError( f"Parameters for step {step!r} must be provided as a mapping.")
                for params in grid[step].keys():
                    parameters_dic[step + '__' + params] = grid[step][params]
            new_parameters_grid.append(parameters_dic)
        return new_parameters_grid

    def _get_pipeline(self):
        pipeline_steps = []
        for step in self.pipeline_dic.keys():
            kwargs = self.pipeline_options_.get(step, {})
            if isinstance(self.pipeline_dic[step], str) and self.pipeline_dic[step] == "passthrough":
                if kwargs:
                    raise ValueError(f"Pipeline options were provided for passthrough step {step!r}.")
                step_object = "passthrough"
            elif isinstance(self.pipeline_dic[step], BaseEstimator):
                step_object = clone(self.pipeline_dic[step])
                if kwargs:
                    step_object.set_params(**kwargs)
            else:
                # Estimator class or user-provided factory.
                step_object = self.pipeline_dic[step](**kwargs)
            pipeline_steps.append((step, step_object))
        if self.imblearn_pipeline:
            # TODO : detect automatically if imblearn pipeline should be used
            return imbPipeline(pipeline_steps, memory=None)
        else:
            return skPipeline(pipeline_steps, memory=None)

    def _check_is_fitted(self, method_name):
        if not self.refit_outer:
            raise NotFittedError(f"This {type(self).__name__} instance was fitted with "
            "refit_outer=False. No final estimator was fitted on the "
            f"full dataset, so {method_name} is unavailable. "
            "The parameter choices from each outer fold are available "
            "in outer_results['best_inner_params']."
        )
        else:
            check_is_fitted(self, attributes=["best_estimator_"])

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

    def _get_inner_param_optimizer(self, inner_model, inner_cv):
        raise NotImplementedError('_get_inner_param_optimizer not implemented')

    def _get_outer_param_optimizer(self, final_model, outer_cv):
        raise NotImplementedError('_get_outer_param_optimizer not implemented')

    def _check_refit_for_multimetric(self):
        if self.refit_inner is False:
            raise ValueError(
                "refit_inner=False is not supported because the best "
                "inner-loop estimator must be refitted on the complete "
                "outer training set before evaluation on the outer test set."
            )
        if self.multimetric_:
            if callable(self.refit_inner):
                raise ValueError('If inner loops use multimetric scoring and the user want to refit according to a '
                                 'callable, the latter must be passed in a dictionary {score: callable} with score '
                                 'being the score name with which the score on different sets will be calculated')
            if self.refit_inner is not False and (not isinstance(self.refit_inner, str) or
                                                  # This will work for both dict / list (tuple)
                                                  self.refit_inner not in self.scorers):
                if isinstance(self.refit_inner, Mapping):
                    if len(self.refit_inner.keys()) > 1:
                        raise ValueError(
                            'refit_inner dict must have only one key, got %d' % len(self.refit_inner.keys()))
                    self.refit_metric = list(self.refit_inner.keys())[0]
                    self.refit_inner = self.refit_inner[self.refit_metric]
                else:
                    raise ValueError("For multi-metric scoring, the parameter "
                                     "refit must be set to a scorer key or a "
                                     "dict with scorer key and callable value to refit an estimator with the "
                                     "best parameter setting on the whole "
                                     "data and make the best_* attributes "
                                     "available for that metric. If this is "
                                     "not needed, refit should be set to "
                                     "False explicitly. %r was passed."
                                     % self.refit_inner)
            else:
                self.refit_metric = self.refit_inner
        else:
            self.refit_metric = 'score'
            if self.refit_inner is True:
                self.refit_inner = 'score'

    @contextmanager
    def _memory_context(self):
        if self.memory is None:
            yield None
            return

        if isinstance(self.memory, str) and self.memory == "temporary":
            with TemporaryDirectory(prefix="nested_cv_cache_") as directory:
                yield Memory(location=directory, verbose=0)
            return

        # String path, pathlib.Path, or Memory-like object.
        yield check_memory(self.memory)
    
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
        self : object
        Instance of fitted estimator.

        Attributes
        ----------
        outer_pred
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

        if self.pipeline_options is None:
            self.pipeline_options_ = {}
        else:
            self.pipeline_options_ = dict(self.pipeline_options)
        self._check_pipeline_dic()
        self._check_pipeline_options()
        self.model = self._get_pipeline()
        self.params_grid = self._get_parameters_grid()

        # Last element of pipeline = estimator
        outer_cv = check_cv(self.outer_cv, y, classifier=is_classifier(self.model[-1]))
        inner_cv = check_cv(self.inner_cv, y, classifier=is_classifier(self.model[-1]))

        self.outer_pred = {'train': [], 'test': [], 'model': [], 'predict_train': [], 'predict_test': []}
        if hasattr(self.model[-1], 'predict_proba'):
            self.outer_pred.update({'predict_proba_train': [], 'predict_proba_test': []})
        if hasattr(self.model[-1], 'decision_function'):
            self.outer_pred.update({'decision_function_train': [], 'decision_function_test': []})

        self.outer_results = {'outer_test_score': [], 'best_inner_score': [], 'best_inner_params': []}
        self.inner_results = []
        if self.return_train_score:
            self.outer_results.update({'outer_train_score': []})

        # From sklearn.model_selection._search.BasesearchCV
        if callable(self.metric) or self.metric is None or isinstance(self.metric, str):
            self.scorers = {"score": check_scoring(self.model, scoring=self.metric)}
            self.multimetric_ = False
        else:
            self.scorers = _check_multimetric_scoring(self.model, scoring=self.metric)
            self.multimetric_ = True
        self._check_refit_for_multimetric()

        for k_outer, (train_outer_index, test_outer_index) in enumerate(outer_cv.split(X, y, groups)):
            if self.verbose > 1:
                print('\n-----------------\n{0}/{1} <-- Current outer fold'.format(k_outer + 1, outer_cv.get_n_splits()))
            X_train_outer, X_test_outer = X[train_outer_index], X[test_outer_index]
            y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]
            groups_train_outer = None if groups is None else np.asarray(groups)[train_outer_index]
            with self._memory_context() as memory:
                inner_model = clone(self.model)
                inner_model.set_params(memory=memory)
                pipeline_inner = self._get_inner_param_optimizer(inner_model, inner_cv)
                pipeline_inner.fit(X_train_outer, y_train_outer, groups=groups_train_outer, **fit_params)
                self.append_scores(pipeline_inner, X_train_outer, X_test_outer, y_train_outer, y_test_outer,
                                   train_outer_index, test_outer_index)
                if self.verbose > 2:
                    for params_dict in pipeline_inner.cv_results_['params']:
                        mean_test_score = pipeline_inner.cv_results_['mean_test_%s' % self.refit_metric]
                        index_params_dic = pipeline_inner.cv_results_['params'].index(params_dict)
                        print('\t\t Params: {0}, Mean inner score: {1}'.format(params_dict, mean_test_score[index_params_dic]))
                if self.verbose > 1:
                    print('\nResults for outer fold:\nBest inner parameters was: {0}'.format(self.outer_results['best_inner_params'][-1]))
                    print('Outer score: {0}'.format(self.outer_results['outer_test_score'][-1]))
                    print('Inner score: {0}'.format(self.outer_results['best_inner_score'][-1]))
        if self.verbose > 0:
            print('\nOverall outer score (mean +/- std): {0} +/- {1}'.format(np.mean(self.outer_results['outer_test_score']),
                                                                             np.std(self.outer_results['outer_test_score'])))
            print('Best params by outer fold:')
            for i, params_dict in enumerate(self.outer_results['best_inner_params']):
                print('\t Outer fold {0}: {1}'.format(i + 1, params_dict))
            print('\n')

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = self.scorers if self.multimetric_ else self.scorers['score']

        # If refit is True Hyperparameter optimization on whole dataset and fit with best params
        if self.refit_outer:
            if self.verbose > 0:
                print('=== Refit ===')
            with self._memory_context() as memory:
                final_model = clone(self.model)
                final_model.set_params(memory=memory)
                pipeline_refit = self._get_outer_param_optimizer(final_model, outer_cv)
                pipeline_refit.fit(X, y, groups=groups, **fit_params)
                best_estimator = pipeline_refit.best_estimator_
                best_estimator.set_params(memory=None)
                self.best_estimator_ = best_estimator
                self.best_params_ = pipeline_refit.best_params_
                self.best_index_ = pipeline_refit.best_index_
                self.cv_results_ = pipeline_refit.cv_results_
                if hasattr(pipeline_refit, "best_score_"):
                    self.best_score_ = pipeline_refit.best_score_
        return self

    def append_scores(self, pipeline_inner, X_train_outer, X_test_outer, y_train_outer, y_test_outer, train_outer_index,
                      test_outer_index):
        self.inner_results.append({'params': pipeline_inner.cv_results_['params'],
                                   'mean_test_score': pipeline_inner.cv_results_['mean_test_%s' % self.refit_metric],
                                   'std_test_score': pipeline_inner.cv_results_['std_test_%s' % self.refit_metric]})
        if self.return_train_score:
            self.inner_results[-1].update({'mean_train_score': pipeline_inner.cv_results_['mean_train_%s' % self.refit_metric],
                                           'std_train_score': pipeline_inner.cv_results_['std_train_%s' % self.refit_metric]})
        self.outer_results['best_inner_score'].append(pipeline_inner.cv_results_['mean_test_%s' % self.refit_metric][pipeline_inner.best_index_])  # Because best_score doesn't exist if refit_inner is a callable
        self.outer_results['best_inner_params'].append(pipeline_inner.best_params_)
        if self.return_train_score:
            self.outer_results['outer_train_score'].append(self.scorers[self.refit_metric](pipeline_inner.best_estimator_, X_train_outer, y_train_outer))
        self.outer_results['outer_test_score'].append(self.scorers[self.refit_metric](pipeline_inner.best_estimator_, X_test_outer, y_test_outer))
        self.outer_pred['train'].append(train_outer_index)
        self.outer_pred['test'].append(test_outer_index)
        fitted_model = pipeline_inner.best_estimator_
        fitted_model.set_params(memory=None)
        self.outer_pred['model'].append(fitted_model)
        self.outer_pred['predict_train'].append(fitted_model.predict(X_train_outer))
        self.outer_pred['predict_test'].append(fitted_model.predict(X_test_outer))
        if hasattr(fitted_model[-1], 'predict_proba'):
            self.outer_pred['predict_proba_train'].append(fitted_model.predict_proba(X_train_outer))
            self.outer_pred['predict_proba_test'].append(fitted_model.predict_proba(X_test_outer))
        if hasattr(fitted_model[-1], 'decision_function'):
            self.outer_pred['decision_function_train'].append(fitted_model.decision_function(X_train_outer))
            self.outer_pred['decision_function_test'].append(fitted_model.decision_function(X_test_outer))


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
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        score = self.scorer_[self.refit_metric] if self.multimetric_ else self.scorer_
        return score(self.best_estimator_, X, y)

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

    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_


class GridSearchNestedCV(NestedCV):
    """
    Nested Cross Validation  with grid search hyperparameter optimization in inner loop
    Parameters
    ----------
    pipeline_dic : mapping of str to estimator specification
        Ordered mapping used to construct the pipeline. Keys are pipeline step
        names. Values may be an estimator class, a callable factory returning
        an estimator, an already instantiated scikit-learn compatible
        ``BaseEstimator``, or the string ``"passthrough"``.

        Intermediate steps of a standard scikit-learn pipeline must implement
        ``fit`` and ``transform``. When ``imblearn_pipeline=True``, compatible
        imbalanced-learn samplers are also accepted. The final step must
        implement ``fit``. Steps are chained in mapping insertion order.
    params_dic : mapping or list of mappings
        Hyperparameter search space grouped by pipeline step. Each outer key
        must match a step name from ``pipeline_dic`` and each value must be a
        mapping from unprefixed estimator parameter names to candidate values
        accepted by the selected search method. Parameters are internally
        converted to scikit-learn's ``step_name__parameter_name`` format.
        A list of mappings defines separate search spaces.
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
    pre_dispatch: int, or string, optional
        Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful
        to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
            -None, in which case all the jobs are immediately created and spawned. Use this for lightweight and
            fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            -An int, giving the exact number of total jobs that are spawned
            -A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
        See sklearn.model_selection.GridSearchCV for more details
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
        Multimetric scoring is available for inner loops. If multimetric, outer loop will only be score with metric
        specified by refit_inner
        See sklearn.model_selection.GridSearchCV for more details.
    verbose: int (default=1)
        Controls the verbosity: the higher, the more messages.
    error_score: 'raise' or numeric (default=np.nan)
        Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which
        will always raise the error.
    refit_inner: True, string or callable (default=True)
        Refit an estimator using the best found parameters on the whole outer training set
        Argument will be given to GridsearchCV that select hyperparameters in the inner loop :
            For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the
            best parameters for refitting the estimator at the end. Where there are considerations other than maximum
            score in choosing a best estimator, refit can be set to a function which returns the selected best_index_
            given cv_results_. In that case, the best_estimator_ and best_parameters_ will be set according to the
            returned best_index_. The refitted estimator is made available at the best_estimator_ attribute and permits
            using predict directly on this GridSearchCV instance. If inner loops use multimetric scoring and the user
            want to refit according to a callable, the latter must be passed in a dictionary {score: callable} with
            score being the score name with which the score on different sets will be calculated
    refit_outer: boolean (default=True)
        Refit an estimator using the whole dataset in two steps:
        1. Hyperparameter optimization with a gridsearch cross-validation (same parameter as outer CV).
        2. Refit an estimator using the best found parameters on the whole dataset.
    return_train_score: boolean (default=False)
        If False, the cross_validation results attribute will not include training scores.
        Computing training scores is used to get insights on how different parameter settings impact
        the overfitting/underfitting trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the parameters that yield the best
        generalization performance.
    memory : None, 'temporary', path-like or joblib.Memory, default=None
            Pipeline caching configuration. ``None`` disables caching.
            ``'temporary'`` creates one temporary shared cache for each inner
            search and for the optional final full-data search, then removes it
            when that search ends. A path-like value or ``joblib.Memory`` object is
            normalized with :func:`sklearn.utils.validation.check_memory` and is
            managed by the caller. Only intermediate pipeline transformers are
            cached; the final estimator is never cached.
    """
    def __init__(self, pipeline_dic, params_dic, outer_cv=5, inner_cv=5, n_jobs=None, pre_dispatch='2*n_jobs',
                 imblearn_pipeline=False, pipeline_options=None, metric='roc_auc', verbose=1, refit_outer=True,
                 error_score=np.nan, refit_inner=True, return_train_score=False, memory=None):
        super().__init__(pipeline_dic, params_dic, outer_cv=outer_cv, inner_cv=inner_cv, n_jobs=n_jobs,
                         pre_dispatch=pre_dispatch, imblearn_pipeline=imblearn_pipeline, pipeline_options=pipeline_options,
                         metric=metric, verbose=verbose, refit_outer=refit_outer, error_score=error_score,
                         refit_inner=refit_inner, return_train_score=return_train_score, memory=memory)

    def _get_inner_param_optimizer(self, inner_model, inner_cv):
        return GridSearchCV(inner_model, self.params_grid, scoring=self.scorers, n_jobs=self.n_jobs, cv=inner_cv,
                            return_train_score=self.return_train_score, verbose=max(0, self.verbose - 1),
                            pre_dispatch=self.pre_dispatch, refit=self.refit_inner, error_score=self.error_score)

    def _get_outer_param_optimizer(self, final_model, outer_cv):
        return GridSearchCV(final_model, self.params_grid, scoring=self.scorers[self.refit_metric], n_jobs=self.n_jobs,
                            cv=outer_cv, verbose=max(0, self.verbose - 1), pre_dispatch=self.pre_dispatch,
                            error_score=self.error_score)


class RandomSearchNestedCV(NestedCV):
    """
    Nested Cross Validation with random search hyperparameter optimization in inner loop
    Parameters
    ----------
    pipeline_dic : mapping of str to estimator specification
        Ordered mapping used to construct the pipeline. Keys are pipeline step
        names. Values may be an estimator class, a callable factory returning
        an estimator, an already instantiated scikit-learn compatible
        ``BaseEstimator``, or the string ``"passthrough"``.

        Intermediate steps of a standard scikit-learn pipeline must implement
        ``fit`` and ``transform``. When ``imblearn_pipeline=True``, compatible
        imbalanced-learn samplers are also accepted. The final step must
        implement ``fit``. Steps are chained in mapping insertion order.
    params_dic : mapping or list of mappings
        Hyperparameter search space grouped by pipeline step. Each outer key
        must match a step name from ``pipeline_dic`` and each value must be a
        mapping from unprefixed estimator parameter names to candidate values
        accepted by the selected search method. Parameters are internally
        converted to scikit-learn's ``step_name__parameter_name`` format.
        A list of mappings defines separate search spaces.
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
    pre_dispatch: int, or string, optional
        Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful
        to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
            -None, in which case all the jobs are immediately created and spawned. Use this for lightweight and
            fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            -An int, giving the exact number of total jobs that are spawned
            -A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
        See sklearn.model_selection.GridSearchCV for more details
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
        Multimetric scoring is available for inner loops. If multimetric, outer loop will only be score with metric
        specified by refit_inner
        See sklearn.model_selection.GridSearchCV for more details.
    verbose: int (default=1)
        Controls the verbosity: the higher, the more messages.
    error_score: 'raise' or numeric (default=np.nan)
        Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which
        will always raise the error.
    refit_inner: True, string or callable (default=True)
        Refit an estimator using the best found parameters on the whole outer training set
        Argument will be given to search method that select hyperparameters in the inner loop :
            For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the
            best parameters for refitting the estimator at the end. Where there are considerations other than maximum
            score in choosing a best estimator, refit can be set to a function which returns the selected best_index_
            given cv_results_. In that case, the best_estimator_ and best_parameters_ will be set according to the
            returned best_index_.The refitted estimator is made available at the best_estimator_ attribute and permits
            using predict directly on this RandomSearchCV instance. If inner loops use multimetric scoring and the user
            want to refit according to a callable, the latter must be passed in a dictionary {score: callable} with
            score being the score name with which the score on different sets will be calculated
    refit_outer: boolean (default=True)
        Refit an estimator using the whole dataset in two steps:
        1. Hyperparameter optimization with a random search cross-validation (same parameter as outer CV).
        2. Refit an estimator using the best found parameters on the whole dataset.
    return_train_score: boolean (default=False)
        If False, the cross_validation results attribute will not include training scores.
        Computing training scores is used to get insights on how different parameter settings impact
        the overfitting/underfitting trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the parameters that yield the best
        generalization performance.
    memory : None, 'temporary', path-like or joblib.Memory, default=None
        Pipeline caching configuration. ``None`` disables caching.
        ``'temporary'`` creates one temporary shared cache for each inner
        search and for the optional final full-data search, then removes it
        when that search ends. A path-like value or ``joblib.Memory`` object is
        normalized with :func:`sklearn.utils.validation.check_memory` and is
        managed by the caller. Only intermediate pipeline transformers are
        cached; the final estimator is never cached.
    n_iter: int (default=10)
        Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    random_state: int, RandomState instance or None (default=None)
        Pseudo random number generator state used for random uniform sampling from lists of possible values instead
        of scipy.stats distributions. Pass an int for reproducible output across multiple function calls.
    """
    def __init__(self, pipeline_dic, params_dic, outer_cv=5, inner_cv=5, n_jobs=None, pre_dispatch='2*n_jobs',
                 imblearn_pipeline=False, pipeline_options=None, metric='roc_auc', verbose=1, refit_outer=True,
                 error_score=np.nan, refit_inner=True, return_train_score=False, memory=None, n_iter=10, random_state=None):
        super().__init__(pipeline_dic, params_dic, outer_cv=outer_cv, inner_cv=inner_cv, n_jobs=n_jobs,
                         pre_dispatch=pre_dispatch, imblearn_pipeline=imblearn_pipeline, pipeline_options=pipeline_options,
                         metric=metric, verbose=verbose, refit_outer=refit_outer, error_score=error_score,
                         refit_inner=refit_inner, return_train_score=return_train_score, memory=self.memory)
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_inner_param_optimizer(self, inner_model, inner_cv):
        return RandomizedSearchCV(inner_model, self.params_grid, scoring=self.scorers,
                                  n_jobs=self.n_jobs, cv=inner_cv, n_iter=self.n_iter,
                                  return_train_score=self.return_train_score, verbose=max(0, self.verbose - 1),
                                  pre_dispatch=self.pre_dispatch, refit=self.refit_inner,
                                  random_state=self.random_state, error_score=self.error_score)

    def _get_outer_param_optimizer(self, final_model, outer_cv):
        return RandomizedSearchCV(final_model, self.params_grid, scoring=self.scorers[self.refit_metric],
                                  n_jobs=self.n_jobs,  cv=outer_cv, n_iter=self.n_iter,
                                  verbose=max(0, self.verbose - 1), pre_dispatch=self.pre_dispatch,
                                  error_score=self.error_score, random_state=self.random_state)


class BayesianSearchNestedCV(NestedCV):
    """
    Nested Cross Validation with bayesian search hyperparameter optimization in inner loop
    Parameters
    ----------
    pipeline_dic : mapping of str to estimator specification
        Ordered mapping used to construct the pipeline. Keys are pipeline step
        names. Values may be an estimator class, a callable factory returning
        an estimator, an already instantiated scikit-learn compatible
        ``BaseEstimator``, or the string ``"passthrough"``.

        Intermediate steps of a standard scikit-learn pipeline must implement
        ``fit`` and ``transform``. When ``imblearn_pipeline=True``, compatible
        imbalanced-learn samplers are also accepted. The final step must
        implement ``fit``. Steps are chained in mapping insertion order.
    params_dic : mapping or list of mappings
        Hyperparameter search space grouped by pipeline step. Each outer key
        must match a step name from ``pipeline_dic`` and each value must be a
        mapping from unprefixed estimator parameter names to candidate values
        accepted by the selected search method. Parameters are internally
        converted to scikit-learn's ``step_name__parameter_name`` format.
        A list of mappings defines separate search spaces.
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
    pre_dispatch: int, or string, optional
        Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful
        to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
            -None, in which case all the jobs are immediately created and spawned. Use this for lightweight and
            fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            -An int, giving the exact number of total jobs that are spawned
            -A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
        See sklearn.model_selection.GridSearchCV for more details
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
        Multimetric scoring is available for inner loops. If multimetric, outer loop will only be score with metric
        specified by refit_inner
        See sklearn.model_selection.GridSearchCV for more details.
    verbose: int (default=1)
        Controls the verbosity: the higher, the more messages.
    error_score: 'raise' or numeric (default=np.nan)
        Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which
        will always raise the error.
    refit_inner: True, string (default=True)
        Refit an estimator using the best found parameters on the whole outer training set
        Argument will be given to search method that select hyperparameters in the inner loop :
            For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the
            best parameters for refitting the estimator at the end. Callable is not supported with BayesSearch.
            The refitted estimator is made available at the best_estimator_ attribute and permits
            using predict directly on this GridSearchCV instance.
    refit_outer: boolean (default=True)
        Refit an estimator using the whole dataset in two steps:
        1. Hyperparameter optimization with a Bayesian search cross-validation (same parameter as outer CV).
        2. Refit an estimator using the best found parameters on the whole dataset.
    return_train_score: boolean (default=False)
        If False, the cross_validation results attribute will not include training scores.
        Computing training scores is used to get insights on how different parameter settings impact
        the overfitting/underfitting trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the parameters that yield the best
        generalization performance.
    memory : None, 'temporary', path-like or joblib.Memory, default=None
        Pipeline caching configuration. ``None`` disables caching.
        ``'temporary'`` creates one temporary shared cache for each inner
        search and for the optional final full-data search, then removes it
        when that search ends. A path-like value or ``joblib.Memory`` object is
        normalized with :func:`sklearn.utils.validation.check_memory` and is
        managed by the caller. Only intermediate pipeline transformers are
        cached; the final estimator is never cached.
    n_iter: int (default=10)
        Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    random_state: int, RandomState instance or None (default=None)
        Pseudo random number generator state used for random uniform sampling from lists of possible values instead
        of scipy.stats distributions. Pass an int for reproducible output across multiple function calls.
    optimizer_kwargs: dict (default=None)
        Dict of arguments passed to :class:`Optimizer`.  For example, ``{'base_estimator': 'RF'}`` would use a
        Random Forest surrogate instead of the default Gaussian Process.
    n_points : int (default=1)
        Number of parameter settings to sample in parallel. If this does not align with ``n_iter``, the last iteration
        will sample less points. See also :func:`~Optimizer.ask`
    """
    def __init__(self, pipeline_dic, params_dic, outer_cv=5, inner_cv=5, n_jobs=None, pre_dispatch='2*n_jobs',
                 imblearn_pipeline=False, pipeline_options=None, metric='roc_auc', verbose=1, refit_outer=True,
                 error_score=np.nan, refit_inner=True, return_train_score=False, memory=None, n_iter=50, random_state=None,
                 optimizer_kwargs=None, n_points=1):
        super().__init__(pipeline_dic, params_dic, outer_cv=outer_cv, inner_cv=inner_cv, n_jobs=n_jobs,
                         pre_dispatch=pre_dispatch, imblearn_pipeline=imblearn_pipeline, pipeline_options=pipeline_options,
                         metric=metric, verbose=verbose, refit_outer=refit_outer, error_score=error_score,
                         refit_inner=refit_inner, return_train_score=return_train_score, memory=memory)
        self.n_iter = n_iter
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self.n_points = n_points
        
    def _check_bayes_search_available(self):
        if BayesSearchCV is None:
            raise ImportError(
                "BayesianSearchNestedCV requires the local "
                "'bayes_search_multiscore.py' module to be available "
                "inside the same package."
            )

    def _get_inner_param_optimizer(self, inner_model, inner_cv):
        self._check_bayes_search_available()
        return BayesSearchCV(inner_model, self.params_grid, scoring=self.scorers,
                             optimizer_kwargs=self.optimizer_kwargs, n_points=self.n_points, n_jobs=self.n_jobs,
                             cv=inner_cv, n_iter=self.n_iter, return_train_score=self.return_train_score,
                             verbose=max(0, self.verbose - 1), pre_dispatch=self.pre_dispatch, refit=self.refit_inner,
                             random_state=self.random_state, error_score=self.error_score)

    def _get_outer_param_optimizer(self, final_model, outer_cv):
        self._check_bayes_search_available()
        return BayesSearchCV(final_model, self.params_grid, scoring=self.scorers[self.refit_metric],
                             optimizer_kwargs=self.optimizer_kwargs, n_points=self.n_points, n_jobs=self.n_jobs,
                             cv=outer_cv, n_iter=self.n_iter, verbose=max(0, self.verbose - 1), pre_dispatch=self.pre_dispatch,
                             error_score=self.error_score, random_state=self.random_state)