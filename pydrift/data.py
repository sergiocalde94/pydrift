import pandas as pd


def compute_levels_count_and_pct(df: pd.DataFrame,
                                 categorical_column: str) -> pd.DataFrame:
    """Computes categorical levels count and percentage and
    returns a `pd.DataFrame`
    """
    categorical_levels_count = (
        df[categorical_column]
        .value_counts(dropna=False, sort=False)
        .reset_index()
        .rename(columns={categorical_column: f'{categorical_column}_count'})
    )

    categorical_levels_pct = (
        df[categorical_column]
        .value_counts(normalize=True, dropna=False, sort=False)
        .reset_index()
        .rename(columns={categorical_column: f'{categorical_column}_norm'})
    )

    return (
        pd
        .concat([categorical_levels_count,
                 categorical_levels_pct.drop(columns='index')], axis=1)
    )
