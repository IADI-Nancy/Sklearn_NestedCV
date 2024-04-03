import imblearn
import os
import numpy as np
import pandas as pd
from Statistical_analysis.nested_cv import NestedCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer


def statistical_pipeline(X, y, save_dir=None, seed=111):
    # NestedCV with outer loop and inner loop being 5Fold Stratified cross validation repeated 5 times
    # Pipeline = Z-score normalization + SMOTE + Dimensionality reduction with Hierarchical clustering +
    #            Feature selection with Wilcoxon score + SVM classifier

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    pipeline_dic = {'scale': StandardScaler,
                    'oversampling': imblearn.over_sampling.SMOTE,
                    'DimensionalityReduction': 'hierarchical_clust_leger',
                    'FeatureSelection': 'wlcx_score',
                    'classifier': SVC}
    params_dic = {'classifier': {'C': 1 / np.arange(0.1, 1.1, 0.2)},
                  'FeatureSelection': {'n_selected_features': [10, 20, None]}}
    pipeline_options = {'oversampling': {'sampling_strategy': 'minority'},
                        'FeatureSelection': {'bootstrap': True, 'ranking_aggregation': 'importance_score'},
                        'classifier': {'kernel': 'linear', 'random_state': seed}}

    clf = NestedCV(pipeline_dic, params_dic, outer_cv=outer_cv, inner_cv=inner_cv, n_jobs=-1,
                   pipeline_options=pipeline_options,
                   metric='roc_auc', verbose=2, refit_inner=True, return_train_score=True, imblearn_pipeline=True)
    clf.fit(X, y)

    # Save outer results
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'NestedCV_restuls.xlsx')
        df = pd.DataFrame(clf.outer_results)
        df.to_excel(save_path)


# Load dataset
breast = load_breast_cancer()
X = breast.data
y = breast.target
statistical_pipeline(X, y)
