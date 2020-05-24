import shap
import warnings
import pandas as pd
import numpy as np
import plotly_express as px

from typing import List, Union
from shap.common import SHAPError
from sklearn.pipeline import Pipeline
from plotly.graph_objects import Figure

from ..models import ScikitModel


class InterpretableDrift:
    def __init__(self,
                 model: ScikitModel,
                 X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.DataFrame,
                 y_test: pd.DataFrame,
                 column_names: List[str]):
        """Inits `InterpretableDrift` for a given `model`,
        `X_train` and `X_test` datasets and `column_names
        """
        if isinstance(model, Pipeline):
            X_train_to_shap = model[:-1].transform(X_train)
            X_test_to_shap = model[:-1].transform(X_test)
            model_to_shap = model.steps[-1][1]
        else:
            X_train_to_shap = X_train.copy()
            X_test_to_shap = X_test.copy()
            model_to_shap = model

        self.model = model_to_shap
        self.X_train_to_shap = pd.DataFrame(X_train_to_shap,
                                            columns=column_names)
        self.X_test_to_shap = pd.DataFrame(X_test_to_shap,
                                           columns=column_names)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.column_names = column_names
        self.shap_values = np.empty(0)

    def compute_shap_values(self) -> None:
        """Shap values depending on what model we are using

        `shap.TreeExplainer` by default and if not it uses
        `KernelExplainer`

        Also provides compatibility with sklearn pipelines

        `shap_values` are stored in `self.shap_values`
        """
        with warnings.catch_warnings():
            # Some `shap` warnings are not useful for this implementation
            warnings.simplefilter("ignore")
            try:
                explainer = shap.TreeExplainer(
                    model=self.model,
                    feature_perturbation='tree_path_dependent'
                )

                shap_values_arguments = dict(X=self.X_test_to_shap)
            except SHAPError:
                def model_predict(data_array):
                    data_frame = pd.DataFrame(data_array,
                                              columns=self.column_names)
                    return self.model.predict_proba(data_frame)[:, 1]

                explainer = shap.KernelExplainer(model=model_predict,
                                                 data=shap.sample(
                                                     self.X_train_to_shap,
                                                     100
                                                 ),
                                                 link='logit')

                shap_values_arguments = dict(X=self.X_test_to_shap,
                                             l1_reg='aic')

        self.shap_values = explainer.shap_values(**shap_values_arguments)

    def most_discriminative_features_plot(self) -> None:
        """Plots most discriminative features with its
        shap values
        """
        if self.shap_values.size == 0:
            self.compute_shap_values()

        shap.summary_plot(self.shap_values,
                          self.X_test_to_shap,
                          plot_type='bar',
                          title='Most Discriminative Features')

    def both_histogram_plot(self,
                            column: str,
                            fillna_value: Union[str, float, int] = None,
                            nbins: int = None) -> Figure:
        """Plots histogram for the column passed
        in `column`

        You can set `nbins` to any number that makes
        your plot better
        """
        X_train_column = self.X_train.loc[:, [column]]
        X_test_column = self.X_test.loc[:, [column]]

        if fillna_value:
            X_train_column.fillna(fillna_value, inplace=True)
            X_test_column.fillna(fillna_value, inplace=True)

        X_train_total_nans = X_train_column[column].isna().sum()
        X_test_total_nans = X_test_column[column].isna().sum()

        if X_train_total_nans or X_test_total_nans:
            warnings.warn(
                f'Column {column} has '
                f'{X_train_total_nans + X_test_total_nans} nan values, '
                f'you can use `fillna_value` if you need it'
            )

        X_train_column['is_left'] = self.y_train.to_numpy()
        X_test_column['is_left'] = self.y_test.to_numpy()

        X_train_and_test = pd.concat([X_train_column, X_test_column])

        fig = px.histogram(X_train_and_test,
                           title=f'Both Histogram Normalized For {column}',
                           x=column,
                           color='is_left',
                           facet_row='is_left',
                           nbins=nbins,
                           histnorm='probability density')

        return fig
