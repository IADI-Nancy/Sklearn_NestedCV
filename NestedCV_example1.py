import pandas as pd
from Statistical_analysis.nested_cv import NestedCV
from Statistical_analysis.univariate_statistical_analysis import univariate_analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

def statistical_pipeline(X, y, save_dir=None, seed=111):
    # Univariate analysis
    univariate_results = univariate_analysis(X, y, save_dir=None)
    
    # NestedCV with outer loop and inner loop being 10Fold Stratified cross validation
    # Pipeline = Z-score normalization + PCA + L2 Logistic Regression
    pipeline_dic = {'scale': StandardScaler, 'DimensionalityReduction': PCA, 'classifier': LogisticRegression}
    params_dic = {'classifier': {'C': 1/np.arange(0.1, 10.1, 0.1)}}
    pipeline_options = {'DimensionalityReduction': {'sklearn_kwargs': {'n_components': 0.95, 'random_state': seed}},
                        'classifier': {'penalty': 'l2', 'random_state': seed, 'solver': 'saga', 'max_iter': 1e5}}

    clf = NestedCV(pipeline_dic, params_dic, outer_cv=10, inner_cv=10, n_jobs=-1, pipeline_options=pipeline_options,
                   metric='roc_auc', verbose=2, refit=True, return_train_score=True, imblearn_pipeline=False)
    clf.fit(X, y)
    
    # Save outer results
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'NestedCV_restuls.xlsx')
        df = pd.DataFrame(clf.outer_results)
        df.to_excel(save_path)

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
statistical_pipeline(X, y)
