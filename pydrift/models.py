import pandas as pd

from typing import List


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
