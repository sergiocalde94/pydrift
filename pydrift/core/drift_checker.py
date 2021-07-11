import abc
import warnings
import pandas as pd
import numpy as np

from sklearn.base import is_classifier, clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from scipy import stats
from collections import defaultdict, namedtuple
from typing import List
from functools import partial
from pathlib import Path

from .interpretable_drift import InterpretableDrift
from ..constants import RANDOM_STATE
from ..exceptions import ColumnsNotMatchException
from ..models import ScikitModel, cat_features_fillna
from ..decorators import track_calls


Coordinates = namedtuple('Coordinates', 'x0 x1 y0 y1')


class DriftChecker(abc.ABC):
    """
    Parent class for drift checks
    """
    def __init__(self,
                 df_left_data: pd.DataFrame,
                 df_right_data: pd.DataFrame,
                 verbose: bool = True,
                 minimal: bool = False):
        """Inits `DriftChecker` with `left_data` and `right_data`
        to compare

        Both needed in `pandas.DataFrame` type

        If `verbose` is set to False it won't print nothing

        When `minimal` set to True some calculations are ignored
        """
        is_dataframe_left_data = isinstance(df_left_data, pd.DataFrame)
        is_dataframe_right_data = isinstance(df_right_data, pd.DataFrame)

        if not (is_dataframe_left_data and is_dataframe_right_data):
            raise TypeError(
                'Both `left_data` and `right_data` '
                'needed in pandas.DataFrame type, '
                f'current types are {type(df_left_data)} '
                f'and {type(df_right_data)}'
            )

        set_right_data_column_names = set(df_right_data.columns)
        set_left_data_column_names = set(df_left_data.columns)

        if set_right_data_column_names != set_left_data_column_names:
            column_name_right_not_in_left = (
                set_right_data_column_names
                .difference(set_left_data_column_names)
            )

            column_name_left_not_in_right = (
                set_left_data_column_names
                .difference(set_right_data_column_names)
            )

            raise ColumnsNotMatchException(
                'Different columns for left and right dataframes\n\n'
                f'Columns in right dataframe but not in left one: '
                f'{", ".join(column_name_right_not_in_left) or "None"}\n'
                f'Columns in left dataframe but not in right one: '
                f'{", ".join(column_name_left_not_in_right) or "None"}'
            )

        self.df_left_data = df_left_data
        self.df_right_data = df_right_data
        self.verbose = verbose
        self.minimal = minimal

        self.cat_features = (df_left_data
                             .select_dtypes(include=['category', 'object'])
                             .columns)

        self.num_features = (df_left_data
                             .select_dtypes(include='number')
                             .columns)

        self.ml_discriminate_model = None
        self.auc_discriminate_model = None
        self.drift = False
        self.interpretable_drift = None

    @track_calls
    def ml_model_can_discriminate(self,
                                  ml_discriminate_model: ScikitModel = None,
                                  column_names: List[str] = None,
                                  auc_threshold: float = .1,
                                  new_target_column: str = 'is_left',
                                  save_plot_path: Path = None) -> bool:
        """Creates a machine learning model based in `sklearn`,
        this model will be a classification model that will try
        to predict if a register is from `left_data` or `right_data`

        `CatBoostClassifier` is used by default because it takes categorical
        data natively and is a state of the art algorithm. Parameters are
        not too high to avoid overfitting. It is within the function instead
        of having it in the parameters because `self.cat_features` is needed

        You can change `ml_discriminate_model` to any sklearn model or
        pipeline

        Parameter `column_names` is only needed when the model output column
        names are not the same as its input, for example in a pipeline with
        one hot encoding step

        If the model gets an auc higher than `auc_threshold` it means
        that it can discriminate between `left_data` and `right_data`
        so there is a drift in the data

        By default the new target name (only used within the function)
        is provided by `new_target_column`

        In minimal mode, this method doesn't neither compute nor show
        the shap values (explainability)
        """
        def symmetric_auc(auc: float) -> float:
            """Inner function to compute symmetric AUC
            (45 is as bad as 55)
            """
            return abs(auc - .5)

        df_all_data_with_target = pd.concat(
            [self.df_left_data.assign(**{new_target_column: 1}),
             self.df_right_data.assign(**{new_target_column: 0})]
        )

        self.X_all_data_with_target = (
            df_all_data_with_target.drop(columns=new_target_column)
        )

        self.y_all_data_with_target = (
            df_all_data_with_target[new_target_column]
        )

        if not ml_discriminate_model:
            self.ml_discriminate_model = CatBoostClassifier(
                num_trees=3,
                max_depth=3,
                cat_features=self.cat_features,
                random_state=RANDOM_STATE,
                verbose=False
            )

            self.X_all_data_with_target = cat_features_fillna(
                self.X_all_data_with_target, self.cat_features
            )
        else:
            self.ml_discriminate_model = ml_discriminate_model

        if not is_classifier(self.ml_discriminate_model):
            raise TypeError(
                'Model `ml_discriminate_model` '
                'has to be a classification model'
            )

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_all_data_with_target,
            self.y_all_data_with_target,
            test_size=.5,
            random_state=RANDOM_STATE
        )

        self.ml_discriminate_model.fit(X_train, y_train)

        y_score_left = (
            self.ml_discriminate_model.predict_proba(X_test)[:, 1]
        )

        self.auc_discriminate_model = roc_auc_score(
            y_true=y_test, y_score=y_score_left
        )

        self.drift = (symmetric_auc(self.auc_discriminate_model)
                      < symmetric_auc(auc_threshold))

        self.interpretable_drift = InterpretableDrift(
            model=self.ml_discriminate_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            column_names=(column_names if column_names
                          else self.df_left_data.columns.tolist())
        )

        if not self.minimal:
            self.interpretable_drift.most_discriminative_features_plot(
                save_plot_path=save_plot_path
            )

        is_there_drift = (
                symmetric_auc(self.auc_discriminate_model) > auc_threshold
        )

        if self.verbose:
            print(
                'Drift found in discriminative model step, '
                'take a look on the most discriminative features '
                '(plots when minimal is set to False)' if is_there_drift
                else 'No drift found in discriminative model step',
                end='\n\n'
            )

            print(f'AUC drift check model: {self.auc_discriminate_model:.2f}')
            print(f'AUC threshold: .5 ± {auc_threshold:.2f}')

        return is_there_drift


class DataDriftChecker(DriftChecker):
    """
    Parent class for drift checks
    """
    def __init__(self,
                 df_left_data: pd.DataFrame,
                 df_right_data: pd.DataFrame,
                 verbose: bool = True,
                 minimal: bool = False,
                 pvalue_threshold_numerical: float = .05,
                 pvalue_threshold_categorical: float = .05,
                 cardinality_threshold: int = 20):
        """Inits `DriftChecker` with `left_data` and `right_data`
        to compare

        Both needed in `pandas.DataFrame` type

        `pvalue_threshold` is set to .05 by default, `cardinality_threshold`
        is set to 20, and `pct_level_threshold` is set to .05, you can modify
        them if your problem fits better with other thresholds

        This class refers to data drift checks, which what it compares
        is the distribution of the features (one by one)
        """
        super().__init__(df_left_data, df_right_data, verbose, minimal)

        self.pvalue_threshold_numerical = pvalue_threshold_numerical
        self.pvalue_threshold_categorical = pvalue_threshold_categorical
        self.cardinality_threshold = cardinality_threshold
        self.dict_each_column_pvalues_numerical = defaultdict(float)
        self.dict_each_column_pvalues_categorical = defaultdict(float)
        self.dict_each_column_cardinality = defaultdict(float)
        self.dict_each_column_drift_coefficients = defaultdict(float)
        self.high_cardinality_features = []
        self.drifted_features = set()
        self._convert_pvalues_to_drift_coefficients_numerical = partial(
            self._convert_pvalues_to_drift_coefficients,
            pvalue_threshold=self.pvalue_threshold_numerical
        )
        self._convert_pvalues_to_drift_coefficients_categorical = partial(
            self._convert_pvalues_to_drift_coefficients,
            pvalue_threshold=self.pvalue_threshold_categorical
        )

    @staticmethod
    def _line_equation(x: float, points_coordinates: Coordinates) -> float:
        """Computes the line equation given two points
        `points_coordinates` and returns `y` value for
        specific `x`
        """
        def slope_from_coordinates(
                _points_coordinates: Coordinates) -> float:
            """Slope equation:

                (y0 - y1) / (x0 - x1)
            """
            return ((_points_coordinates.y0 - _points_coordinates.y1)
                    / (_points_coordinates.x0 - _points_coordinates.x1))

        def intercept_from_coordinates(
                _points_coordinates: Coordinates) -> float:
            """Intercept equation:

                (x0 * y1 - x1 * y0) / (x0 - x1)
            """
            return ((_points_coordinates.x0 * _points_coordinates.y1
                     - _points_coordinates.x1 * _points_coordinates.y0)
                    / (_points_coordinates.x0 - _points_coordinates.x1))

        return (slope_from_coordinates(points_coordinates) * x
                + intercept_from_coordinates(points_coordinates))

    def _convert_pvalues_to_drift_coefficients(
            self, pvalue: float, pvalue_threshold: float) -> float:
        """
        Takes pvalues and return the drift coefficient
        """

        dict_coordinates = (
            dict(x0=0,
                 x1=pvalue_threshold,
                 y0=0,
                 y1=0.5) if pvalue < pvalue_threshold
            else dict(x0=pvalue_threshold,
                      x1=1,
                      y0=.5,
                      y1=1)
        )

        coordinates = Coordinates(**dict_coordinates)

        return 1 - self._line_equation(pvalue, coordinates)

    def _update_each_column_drift_coefficient(self) -> None:
        """Updates `self.dict_each_column_pvalues` when
        all of the pvalues are calculated
        """
        are_all_features_pvalues_filled = (
                self.dict_each_column_pvalues_numerical
                and self.dict_each_column_pvalues_categorical
        )

        empty_num_features_and_cat_features_pvalues_filled = (
                not self.num_features.tolist()
                and self.dict_each_column_pvalues_categorical
        )

        empty_cat_features_and_num_features_pvalues_filled = (
                not self.cat_features.tolist()
                and self.dict_each_column_pvalues_numerical
        )

        if (are_all_features_pvalues_filled
                or empty_num_features_and_cat_features_pvalues_filled
                or empty_cat_features_and_num_features_pvalues_filled):
            dict_each_column_drift_coefficients_numerical = {
                feature: (
                    self
                    ._convert_pvalues_to_drift_coefficients_numerical(
                        pvalue
                    )
                )
                for feature, pvalue in (self
                                        .dict_each_column_pvalues_numerical
                                        .items())
            }

            dict_each_column_drift_coefficients_categorical = {
                feature: (
                    self
                    ._convert_pvalues_to_drift_coefficients_categorical(
                        pvalue
                    )
                )
                for feature, pvalue in (self
                                        .dict_each_column_pvalues_categorical
                                        .items())
            }

            self.dict_each_column_drift_coefficients = {
                **dict_each_column_drift_coefficients_numerical,
                **dict_each_column_drift_coefficients_categorical
            }

    def check_numerical_columns(self) -> bool:
        """Given `numerical_columns` check all drifts

        Kolmogorov-Smirnov test:

            This is a two-sided test for the null hypothesis that 2
            independent samples are drawn from the same continuous
            distribution.
        """
        for numerical_column in self.num_features:
            _, pvalue = stats.ks_2samp(self.df_left_data[numerical_column],
                                       self.df_right_data[numerical_column])
            self.dict_each_column_pvalues_numerical[numerical_column] = pvalue

        self._update_each_column_drift_coefficient()

        drifted_features = [
            column
            for column, pvalue in (self
                                   .dict_each_column_pvalues_numerical
                                   .items())
            if pvalue < self.pvalue_threshold_numerical
        ]

        self.drifted_features = self.drifted_features.union(drifted_features)

        is_there_drift = len(self.drifted_features) > 0

        if self.verbose:
            if is_there_drift:
                print('Drift found in numerical columns check step, '
                      'take a look on the variables that are drifted, '
                      'if one is not important you could simply delete it, '
                      'otherwise check the data source',
                      end='\n\n')

                warnings.warn(f'Features drifted (numerical): '
                              f'{", ".join(drifted_features)}')
            else:
                print('No drift found in numerical columns check step',
                      end='\n\n')

        return is_there_drift

    def check_categorical_columns(self) -> bool:
        """Given `categorical_columns` check all drifts

        Calculate a one-way chi-square test:

            The chi-square test tests the null hypothesis
            that the categorical data has the given frequencies.
        """
        sample_size = min(len(self.df_left_data), len(self.df_right_data))

        for categorical_column in self.cat_features:
            df_left_data_to_compare = (self
                                       .df_left_data[categorical_column]
                                       .sample(sample_size))

            df_right_data_to_compare = (self
                                        .df_right_data[categorical_column]
                                        .sample(sample_size))

            # .1 is just to avoid zero-division in some cases
            dict_left_frequencies = (
                df_left_data_to_compare
                .value_counts()
                .to_dict(defaultdict(lambda: .01))
            )

            dict_right_frequencies = (
                df_right_data_to_compare
                .value_counts()
                .to_dict(defaultdict(lambda: .01))
            )

            frequencies_zipped = [
                (dict_left_frequencies[key], dict_right_frequencies[key])
                for key in (set(dict_left_frequencies)
                            .union(set(dict_right_frequencies)))
            ]

            _, pvalue = stats.chisquare(
                list(map(lambda x: x[0], frequencies_zipped)),
                f_exp=list(map(lambda x: x[1], frequencies_zipped))
            )

            self.dict_each_column_cardinality[categorical_column] = (
                len(frequencies_zipped)
            )

            self.dict_each_column_pvalues_categorical[
                categorical_column
            ] = pvalue

        self._update_each_column_drift_coefficient()

        self.high_cardinality_features = [
            column
            for column, cardinality in (self
                                        .dict_each_column_cardinality
                                        .items())
            if cardinality > self.cardinality_threshold
        ]

        drifted_features = [
            column
            for column, pvalue in (self
                                   .dict_each_column_pvalues_categorical
                                   .items())
            if pvalue < self.pvalue_threshold_categorical
        ]

        self.drifted_features = self.drifted_features.union(drifted_features)

        is_there_drift = len(drifted_features) > 0
        is_there_high_cardinality = len(self.high_cardinality_features)

        if self.verbose:
            if is_there_drift:
                print('Drift found in categorical columns check step, '
                      'take a look on the variables that are drifted, '
                      'if one is not important you could simply delete it, '
                      'otherwise check the data source',
                      end='\n\n')

                warnings.warn(f'Features drifted (categorical): '
                              f'{", ".join(drifted_features)}')
            else:
                print('No drift found in categorical columns check step',
                      end='\n\n')

            if is_there_high_cardinality:
                warnings.warn(f'Features cardinality warning: '
                              f'{", ".join(self.high_cardinality_features)}')

        return is_there_drift


class ModelDriftChecker(DriftChecker):
    """
    Parent class for drift checks
    """
    def __init__(self,
                 df_left_data: pd.DataFrame,
                 df_right_data: pd.DataFrame,
                 ml_classifier_model: ScikitModel,
                 target_column_name: str,
                 verbose: bool = True,
                 minimal: bool = False,
                 auc_threshold: float = .03):
        """Inits `ModelDriftChecker` with `left_data` and `right_data`
        to compare

        Both needed in `pandas.DataFrame` type

        `ml_classifier_model` and `target_column_name` has to be also
        provided to check your model

        `auc_threshold` is set to .03 by default, you can modify if your
        problem fits better with other threshold

        This class refers to model drift checks, which what it compares
        is the relationship of the variables with the target (univariate)
        """
        super().__init__(df_left_data, df_right_data, verbose, minimal)

        self.ml_classifier_model = ml_classifier_model
        self.target_column_name = target_column_name
        self.auc_threshold = auc_threshold
        self.interpretable_drift_classifier_model = None

    def check_model(self,
                    column_names: List[str] = None,
                    new_target_column: str = 'is_left',
                    save_plot_path: Path = None) -> bool:
        """Checks if features relations with target are the same
        for `self.df_left_data` and `self.df_right_data`

        Parameter `column_names` is only needed when the model output column
        names are not the same as its input, for example in a pipeline with
        one hot encoding step
        """
        X_left = self.df_left_data.drop(columns=self.target_column_name)
        y_left = self.df_left_data[self.target_column_name]
        X_right = self.df_right_data.drop(columns=self.target_column_name)
        y_right = self.df_right_data[self.target_column_name]

        y_score_left = (self
                        .ml_classifier_model
                        .predict_proba(X_left)[:, 1])

        y_score_right = (self
                         .ml_classifier_model
                         .predict_proba(X_right)[:, 1])

        auc_left = roc_auc_score(y_true=y_left, y_score=y_score_left)
        auc_right = roc_auc_score(y_true=y_right, y_score=y_score_right)

        if not self.minimal:
            self.interpretable_drift_classifier_model = InterpretableDrift(
                model=self.ml_classifier_model,
                X_train=X_left,
                X_test=X_right,
                y_train=(
                    pd
                    .Series(1, name=new_target_column)
                    .repeat(len(X_left))
                    .to_frame()
                ),
                y_test=(
                    pd
                    .Series(0, name=new_target_column)
                    .repeat(len(X_right))
                    .to_frame()
                ),
                column_names=(column_names if column_names
                              else (X_left.columns.tolist()))
            )

            (self
             .interpretable_drift_classifier_model
             .most_discriminative_features_plot(save_plot_path=save_plot_path))

        is_there_drift = abs(auc_left - auc_right) > self.auc_threshold

        if self.verbose:
            print(
                'Drift found in your model, '
                'take a look on the most discriminative features '
                '(plots when minimal is set to False), '
                'DataDriftChecker can help you with changes in features '
                'distribution and also look at your hyperparameters'
                if is_there_drift
                else 'No drift found in your model',
                end='\n\n'
            )

            print(f'AUC left data: {auc_left:.2f}')
            print(f'AUC right data: {auc_right:.2f}')

        return is_there_drift

    def show_feature_importance_vs_drift_map_plot(
            self, top: int = 10, save_plot_path: Path = None) -> None:
        """Shows feature importance versus drift coefficient
        map

        By default shows you the top 10 most important features
        but you can customize it with `top` parameter

        You can found more details for this function in its module:

            `pydrift.InterpretableDrift.feature_importance_vs_drift_map_plot`
        """
        if self.minimal:
            raise ValueError(
                'To plot drift map, set minimal argument to False when '
                'instantiating ModelDriftChecker'
            )

        data_drift_checker = DataDriftChecker(self.df_left_data,
                                              self.df_right_data,
                                              verbose=False,
                                              minimal=True)

        data_drift_checker.check_numerical_columns()
        data_drift_checker.check_categorical_columns()

        (self
         .interpretable_drift_classifier_model
         .feature_importance_vs_drift_map_plot(
            dict_each_column_drift_coefficient=(
                data_drift_checker.dict_each_column_drift_coefficients
            ),
            top=top,
            save_plot_path=save_plot_path))

    def sample_weight_for_retrain(self,
                                  save_plot_path: Path = None) -> np.array:
        """If you need to retrain your model maybe
        it's better applying this weights when you
        do it

        From https://bit.ly/2Xf39ks (thanks!):

            We can use this w as sample weights in any of our
            classifier to increase the weight of these observation
            which seems similar to our test data. Intuitively this
            makes sense as our model will focus more on capturing
            patterns from the observations which seems similar to our test.

        An example with random forest:

            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train, y_train, sample_weight=weights)
        """
        # Temporary change `self.minimal` to avoid confusing plots
        actual_self_minimal, self.minimal = self.minimal, True
        # Temporary change `self.verbose` to avoid confusing plots
        actual_verbose, self.verbose = self.verbose, False

        if not self.ml_model_can_discriminate.has_been_called:
            self.ml_model_can_discriminate()

        skf = StratifiedKFold(n_splits=5,
                              shuffle=True,
                              random_state=RANDOM_STATE)

        ml_discriminator = clone(self.ml_discriminate_model)

        df_predictions_all_folds = pd.DataFrame(columns=['prediction'])

        for train_idx, test_idx in skf.split(self.X_all_data_with_target,
                                             self.y_all_data_with_target):
            X_fold_train, X_fold_test = (
                self.X_all_data_with_target.iloc[train_idx],
                self.X_all_data_with_target.iloc[test_idx]
            )

            y_fold_train = self.y_all_data_with_target.iloc[train_idx]

            ml_discriminator.fit(X_fold_train, y_fold_train)

            df_predictions_all_folds = (
                df_predictions_all_folds
                .append(
                    pd.DataFrame(
                        ml_discriminator.predict_proba(X_fold_test)[:, 1],
                        index=X_fold_test.index,
                        columns=['prediction'])
                )
            )

        # Only left data scores are needed
        y_score_left = df_predictions_all_folds.loc[self.df_left_data.index]

        # Reset verbose and minimal values
        self.minimal = actual_self_minimal
        self.verbose = actual_verbose

        weights = (1 / y_score_left) - 1
        weights /= np.mean(weights)

        if not self.minimal:
            (self
             .interpretable_drift
             .weights_plot(weights, save_plot_path=save_plot_path))

            print('Higher the weight for the observation, '
                  'more is it similar to the test data')

        return weights
