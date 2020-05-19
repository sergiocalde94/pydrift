import shap
import warnings
import pandas as pd

from typing import List
from typing_extensions import Protocol
from shap.common import SHAPError
from sklearn.pipeline import Pipeline


class ScikitModel(Protocol):
    """Scikit model typing
    """
    def fit(self, X, y, verbose): ...
    def predict_proba(self, X) -> object: ...


def cat_features_fillna(df: pd.DataFrame,
                        cat_features: List[str]) -> pd.DataFrame:
    """Fills NA values for each column in `cat_features` for
    `df` dataframe
    """
    df_copy = df.copy()

    for cat in cat_features:
        try:
            df_copy[cat] = (
                df_copy[cat].cat.add_categories('UNKNOWN').fillna('UNKNOWN')
            )

        except AttributeError:
            # The dtype is object instead of category
            df_copy[cat] = df_copy[cat].fillna('UNKNOWN')

    return df_copy


def explainer_plots(model: ScikitModel,
                    X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    column_names: List[str]) -> None:
    """Shap plots depending on what model is passed
    in `model` parameter

    `shap.TreeExplainer` by default and if not it uses
    `KernelExplainer`

    Also provides compatibility with sklearn pipelines
    """
    if isinstance(model, Pipeline):
        X_train_to_shap = model[:-1].transform(X_train)
        X_test_to_shap = model[:-1].transform(X_test)
        model = model.steps[-1][1]
    else:
        X_train_to_shap = X_train.copy()
        X_test_to_shap = X_test.copy()

    with warnings.catch_warnings():
        # Some `shap` warnings are not useful for this implementation
        warnings.simplefilter("ignore")
        try:
            explainer = shap.TreeExplainer(
                model=model,
                feature_perturbation='tree_path_dependent'
            )

            shap_values_arguments = dict(X=X_test_to_shap)
        except SHAPError:
            def model_predict(data_array):
                data_frame = pd.DataFrame(data_array, columns=column_names)
                return model.predict_proba(data_frame)[:, 1]

            explainer = shap.KernelExplainer(model=model_predict,
                                             data=shap.sample(X_train_to_shap,
                                                              100),
                                             link='logit')

            shap_values_arguments = dict(X=X_test_to_shap, l1_reg='aic')

    shap_values = explainer.shap_values(**shap_values_arguments)

    shap.summary_plot(shap_values,
                      X_train,
                      plot_type='bar',
                      title='Most Discriminative Features')
