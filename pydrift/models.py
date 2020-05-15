import shap
import pandas as pd

from typing import List
from typing_extensions import Protocol


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
                    minimal: bool = False) -> None:
    if not minimal:
        explainer = shap.TreeExplainer(
            model=model,
            feature_perturbation='tree_path_dependent'
        )

        shap_values = explainer.shap_values(X_train)

        shap.summary_plot(shap_values,
                          X_train,
                          plot_type='bar',
                          title='Most Discriminative Features')
