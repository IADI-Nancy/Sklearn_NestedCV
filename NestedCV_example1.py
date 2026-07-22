"""Example 1: multiclass nested cross-validation.

The example uses only scikit-learn estimators inside ``GridSearchNestedCV``.
The pipeline is fitted independently within every inner and outer fold.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from Statistical_analysis.nested_cv import GridSearchNestedCV


def statistical_pipeline(X, y, save_dir=None, seed=111):
    """Run PCA + logistic regression in nested cross-validation."""

    pipeline_dic = {
        "scale": StandardScaler(),
        "pca": PCA(svd_solver="full", random_state=seed),
        "classifier": LogisticRegression(penalty='l2', solver="saga", max_iter=int(1e5), random_state=seed)
    }

    params_dic = {
        "pca": {"n_components": [0.95, 0.99]},
        "classifier": {"C": np.logspace(-1, 1, 100)}
    }

    nested_cv = GridSearchNestedCV(
        pipeline_dic=pipeline_dic,
        params_dic=params_dic,
        outer_cv=10,
        inner_cv=10,
        n_jobs=10,
        metric="roc_auc_ovo",
        verbose=1,
        refit_inner=True,
        return_train_score=True,
        refit_outer=True
    )
    nested_cv.fit(X, y)

    print("Best full-data parameters:", nested_cv.best_params_)
    print("Best full-data CV score:", nested_cv.best_score_)

    if save_dir is not None:
        save_path = Path(save_dir) / "nested_cv_example1_results.xlsx"
        pd.DataFrame(nested_cv.outer_results).to_excel(save_path, index=False)

    return nested_cv


if __name__ == "__main__":
    iris = load_iris()
    statistical_pipeline(iris.data, iris.target)
