import shap
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    _has_plotly = False
    _plotly_exception_message = (
        'Plotly is required to run this pydrift functionality.'
    )
else:
    _has_plotly = True
    _plotly_exception_message = None

from typing import List, Union, Dict, Tuple
from sklearn.pipeline import Pipeline
from pathlib import Path

from ..models import ScikitModel
from ..decorators import check_optional_module


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
            except Exception:
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

    def most_discriminative_features_plot(self,
                                          save_plot_path: Path = None) -> None:
        """Plots most discriminative features with its
        shap values

        You can save the plot in `save_plot_path` path
        """
        if self.shap_values.size == 0:
            self.compute_shap_values()

        shap.summary_plot(self.shap_values,
                          self.X_test_to_shap,
                          plot_type='bar',
                          title='Most Discriminative Features',
                          show=True if not save_plot_path else False)

        if save_plot_path:
            plt.savefig(save_plot_path, bbox_inches='tight')

    @check_optional_module(has_module=_has_plotly,
                           exception_message=_plotly_exception_message)
    def both_histogram_plot(self,
                            column: str,
                            fillna_value: Union[str, float, int] = None,
                            nbins: int = None,
                            save_plot_path: Path = None) -> None:
        """Plots histogram for the column passed
        in `column`

        You can set `nbins` to any number that makes
        your plot better

        You can save the plot as html file in `save_plot_path` path

        Requires `plotly`
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
                           barmode='group',
                           nbins=nbins,
                           histnorm='probability density')

        fig.update_layout(bargroupgap=.1)

        if save_plot_path:
            fig.write_html(save_plot_path)
        else:
            fig.show()

    @check_optional_module(has_module=_has_plotly,
                           exception_message=_plotly_exception_message)
    def feature_importance_vs_drift_map_plot(
            self,
            dict_each_column_drift_coefficient: Dict[str, float],
            top: int = 10,
            save_plot_path: Path = None) -> None:
        """Feature importance versus drift coefficient map,
        with this plot you can visualize the most critical
        features involved in your model drift process

        By default shows you the top 10 most important features
        but you can customize it with `top` parameter

        You can save the plot as html file in `save_plot_path` path
        """
        df_feature_importance = pd.DataFrame(
            zip(self.column_names,
                np.abs(self.shap_values).mean(axis=0)),
            columns=['Feature Name', 'Feature Importance']
        )

        df_feature_importance['Drift Coefficient'] = (
            (df_feature_importance['Feature Name']
             .map(dict_each_column_drift_coefficient))
        )

        value_min = df_feature_importance['Feature Importance'].min()
        value_max = df_feature_importance['Feature Importance'].max()

        df_feature_importance['Feature Importance Scaled'] = (
                (df_feature_importance['Feature Importance'] - value_min)
                / (value_max - value_min)
        )

        df_feature_importance_to_plot = (
            df_feature_importance
            .sort_values('Feature Importance Scaled', ascending=False)
            .nlargest(top, columns='Feature Importance Scaled')
        )

        fig = px.scatter(df_feature_importance_to_plot,
                         x='Feature Importance Scaled',
                         y='Drift Coefficient',
                         text='Feature Name',
                         hover_name='Feature Name',
                         hover_data={'Feature Importance Scaled': ':.2f',
                                     'Drift Coefficient': ':.2f',
                                     'Feature Importance': False,
                                     'Feature Name': False},
                         title='Feature Importance vs Drift Map')

        fig.update_traces(marker=dict(size=10, opacity=.75))

        axis_value_min, axis_value_medium, axis_value_max = 0, .5, 1

        fig.add_trace(
            go.Scatter(
                x=[axis_value_min + .15, axis_value_max - .15,
                   axis_value_max - .15, axis_value_min + .15],
                y=[axis_value_max + .05, axis_value_max + .05,
                   axis_value_min - .05, axis_value_min - .05],
                text=['NON-IMPORTANT FEATURES DRIFTED',
                      'IMPORTANT FEATURES AND DRIFTED',
                      'IMPORTANT FEATURES NON-DRIFTED',
                      'NON-IMPORTANT FEATURES NON-DRIFTED'],
                mode="text",
                showlegend=False
            )
        )

        fig.add_shape(
            type="rect",
            x0=axis_value_min,
            y0=axis_value_min,
            x1=axis_value_medium,
            y1=axis_value_medium,
            fillcolor="khaki",
            opacity=.25
        )

        fig.add_shape(
            type="rect",
            x0=axis_value_min,
            y0=axis_value_medium,
            x1=axis_value_medium,
            y1=axis_value_max,
            fillcolor="coral",
            opacity=.25
        )

        fig.add_shape(
            type="rect",
            x0=axis_value_medium,
            y0=axis_value_min,
            x1=axis_value_max,
            y1=axis_value_medium,
            fillcolor="limegreen",
            opacity=.25
        )

        fig.add_shape(
            type="rect",
            x0=axis_value_medium,
            y0=axis_value_medium,
            x1=axis_value_max,
            y1=axis_value_max,
            fillcolor="crimson",
            opacity=.25
        )

        fig.update_layout(
            xaxis=dict(range=[axis_value_min - .05, axis_value_max + .05]),
            yaxis=dict(range=[axis_value_min - .1, axis_value_max + .1])
        )

        if save_plot_path:
            fig.write_html(save_plot_path)
        else:
            fig.show()

    @staticmethod
    @check_optional_module(has_module=_has_plotly,
                           exception_message=_plotly_exception_message)
    def weights_plot(weights: np.array, save_plot_path: Path = None) -> None:
        """Weights plot, the higher the weight, the more
        similar the train data is to the test data

        This will be used to retrain the model

        You can save the plot as html file in `save_plot_path` path
        """
        fig = px.histogram(weights,
                           title='Weights From The Discriminative Model')

        fig.update_layout(showlegend=False)

        if save_plot_path:
            fig.write_html(save_plot_path)
        else:
            fig.show()

    @staticmethod
    def _drop_outliers_between(
            df: pd.DataFrame,
            feature: str,
            percentiles: Tuple[float,
                               float] = (.05, .95)) -> pd.DataFrame:
        """Drop outliers for column `feature` of
        `df` between `percentiles`
        """
        lower, upper = percentiles

        return df[df[feature].between(df[feature].quantile(lower),
                                      df[feature].quantile(upper))]

    @staticmethod
    def _convert_to_mid_interval_with_max_bins(
            serie: pd.Series,
            bins: int = 25) -> pd.DataFrame:
        """Convert `series` values to a binned version
        of it in `bins` as number of
        intervals
        """
        return (pd
                .cut(serie, bins=bins)
                .apply(lambda x: x.mid)
                .astype(float))

    @check_optional_module(has_module=_has_plotly,
                           exception_message=_plotly_exception_message)
    def partial_dependence_comparison_plot(
            self,
            feature: str,
            percentiles: Tuple[float, float] = (.05, .95),
            max_bins: int = 25,
            save_plot_path: Path = None) -> None:
        """Partial dependence plot for `feature` in
        both datasets predictions

        You can save the plot as html file in `save_plot_path` path
        """
        X_train_copy = self.X_train.copy()
        X_test_copy = self.X_test.copy()

        X_train_copy['is_left'] = '1'
        X_test_copy['is_left'] = '0'

        X_train_copy['Prediction'] = (
            self.model.predict_proba(X_train_copy)[:, 1]
        )

        X_test_copy['Prediction'] = (
            self.model.predict_proba(X_test_copy)[:, 1]
        )

        is_numeric = pd.api.types.is_numeric_dtype(
            X_train_copy[feature]
        )

        if is_numeric:
            X_train_copy = (
                self._drop_outliers_between(X_train_copy,
                                            feature=feature,
                                            percentiles=percentiles)
            )

            X_test_copy = (
                self._drop_outliers_between(X_test_copy,
                                            feature=feature,
                                            percentiles=percentiles)
            )

            bins = min(X_train_copy[feature].nunique(),
                       max_bins)

            X_train_copy[feature] = (
                self._convert_to_mid_interval_with_max_bins(
                    X_train_copy[feature],
                    bins
                )
            )

            X_test_copy[feature] = (
                self._convert_to_mid_interval_with_max_bins(
                    X_test_copy[feature],
                    bins
                )
            )

        X_both = pd.concat([X_train_copy, X_test_copy])

        data_to_plot = (
            X_both
            .groupby(['is_left', feature])
            .Prediction
            .mean()
            .reset_index()
        )

        if is_numeric:
            fig = px.scatter(data_to_plot,
                             x=feature,
                             y='Prediction',
                             color='is_left',
                             trendline="ols")
        else:
            fig = px.bar(data_to_plot,
                         x=feature,
                         y='Prediction',
                         color='is_left',
                         barmode='group')

        fig.update_layout(title=f'Partial Dependence For {feature}',
                          bargroupgap=.1)

        if save_plot_path:
            fig.write_html(save_plot_path)
        else:
            fig.show()

    @check_optional_module(has_module=_has_plotly,
                           exception_message=_plotly_exception_message)
    def drift_by_sorted_bins_plot(self,
                                  feature: str,
                                  bins: int = 10,
                                  save_plot_path: Path = None) -> None:
        """Concat all the data in both dataframes and
        sort it by `feature`, then it cuts in `bins`
        number of bins and computes quantity of registers
        in each bin

        You can save the plot as html file in `save_plot_path` path
        """
        X_train_copy = self.X_train.copy()
        X_test_copy = self.X_test.copy()

        X_train_copy['is_left'] = '1'
        X_test_copy['is_left'] = '0'

        X_both = (
            pd
            .concat([X_train_copy[[feature, 'is_left']],
                     X_test_copy[[feature, 'is_left']]])
            .sample(frac=1)
            .reset_index()
        )

        is_categorical = not pd.api.types.is_numeric_dtype(
            X_both[feature]
        )

        if is_categorical:
            X_both[feature] = X_both[feature].astype('category')

        X_both['rank'] = (
            X_both[feature].cat.codes .rank(method='first') if is_categorical
            else X_both[feature].rank(method='first')
        )

        X_both['Bin Number'] = pd.qcut(X_both['rank'],
                                       q=bins,
                                       labels=range(1, bins + 1))

        fig = px.histogram(X_both,
                           x='Bin Number',
                           color='is_left',
                           nbins=bins,
                           barmode='group')

        fig.update_layout(title=f'Drift By Bin For {feature}',
                          bargroupgap=.1,
                          xaxis=dict(tickmode='linear'))

        if save_plot_path:
            fig.write_html(save_plot_path)
        else:
            fig.show()
