"""Example 2: imbalanced-learning pipeline with custom module estimators.

Pipeline:
    standardization -> SMOTE -> correlation clustering -> bootstrapped
    filter feature selection -> linear SVM
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Statistical_analysis.dimensionality_reduction import HierarchicalClusteringLeger
from Statistical_analysis.feature_selection import FilterFeatureSelection
from Statistical_analysis.nested_cv import GridSearchNestedCV


def statistical_pipeline(X, y, save_dir=None, seed=111):
    """Run a custom radiomics-style pipeline in nested CV."""

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)

    pipeline_dic = {
        "scale": StandardScaler,
        "oversampling": SMOTE,
        "correlation_reduction": HierarchicalClusteringLeger,
        "selector": FilterFeatureSelection,
        "classifier": SVC
    }

    params_dic = {
        "selector": {"n_selected_features": [10, 20, None]},
        "classifier": {"C": np.logspace(0, 1, 5)}
    }

    pipeline_options = {
        "oversampling": {"sampling_strategy": "minority", "random_state": seed},
        "correlation_reduction": {"corr_metric": "spearman", "correlation_threshold": 0.9,
                                  "cluster_reduction": "medoid", "bootstrap": False, "random_state": seed},
        "selector": {"method": "kruskal_score", "bootstrap": True, "n_bsamples": 100, 
                     "ranking_aggregation": "importance_score", "random_state": seed},
        "classifier": {"kernel": "linear", "random_state": seed}
    }

    nested_cv = GridSearchNestedCV(
        pipeline_dic=pipeline_dic,
        params_dic=params_dic,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        n_jobs=10,
        pipeline_options=pipeline_options,
        metric="roc_auc",
        verbose=1,
        refit_inner=True,
        return_train_score=False,
        refit_outer=True,
        imblearn_pipeline=True,
        memory='temporary'
    )
    nested_cv.fit(X, y)

    print("Best full-data parameters:", nested_cv.best_params_)
    print("Best full-data CV score:", nested_cv.best_score_)

    if save_dir is not None:
        save_path = Path(save_dir) / "nested_cv_example2_results.xlsx"
        pd.DataFrame(nested_cv.outer_results).to_excel(save_path, index=False)

    return nested_cv


if __name__ == "__main__":
    breast_cancer = load_breast_cancer()
    s = time.time()
    statistical_pipeline(breast_cancer.data, breast_cancer.target)
    print(time.time() - s)
