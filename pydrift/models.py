import pandas as pd

from typing import List
from typing_extensions import Protocol


class ScikitModel(Protocol):
    """Scikit model typing
    """
    def fit(self, X, y, verbose): ...
    def predict(self, X) -> object: ...
    def predict_proba(self, X) -> object: ...


def cat_features_fillna(df: pd.DataFrame,
                        cat_features: List[str]) -> pd.DataFrame:
    """Fills NA values for each column in `cat_features` for
    `df` dataframe

    Needed for catboost default model for checking
    `DriftChecker.ml_model_can_discriminate`
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
