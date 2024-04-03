import os
import numpy as np
import pandas as pd
from Statistical_analysis.nested_cv import GridSearchNestedCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def statistical_pipeline(X, y, save_dir=None, seed=111):
    # NestedCV with outer loop and inner loop being 10Fold Stratified cross validation
    # Pipeline = Z-score normalization + PCA + L2 Logistic Regression
    pipeline_dic = {'scale': StandardScaler, 'DimensionalityReduction': PCA, 'classifier': LogisticRegression}
    params_dic = {'DimensionalityReduction': {'n_components': [0.95, 0.99]},
                  'classifier': {'C': 1 / np.arange(0.1, 10.1, 0.1)}}
    pipeline_options = {'DimensionalityReduction': {'svd_solver': 'full', 'random_state': seed},
                        'classifier': {'penalty': 'l2', 'random_state': seed, 'solver': 'saga', 'max_iter': 1e5}}

    clf = GridSearchNestedCV(pipeline_dic, params_dic, outer_cv=10, inner_cv=10, n_jobs=-1,
                             pipeline_options=pipeline_options, metric='roc_auc_ovo', verbose=2, refit_inner=True,
                             return_train_score=True, refit_outer=False)
    clf.fit(X, y)
    print(clf.best_estimator_, clf.best_score_)
    # Save outer results
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'NestedCV_results.xlsx')
        df = pd.DataFrame(clf.outer_results)
        df.to_excel(save_path)


# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
statistical_pipeline(X, y)
