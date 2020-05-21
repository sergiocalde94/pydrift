import abc
import warnings
import pandas as pd

from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from scipy import stats
from collections import defaultdict
from typing import List

from .interpretable_drift import InterpretableDrift
from ..constants import RANDOM_STATE
from ..exceptions import ColumnsNotMatchException
from ..models import ScikitModel, cat_features_fillna
from ..data import compute_levels_count_and_pct


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
        self.drift = False
        self.interpretable_drift = None

    def ml_model_can_discriminate(self,
                                  ml_discriminate_model: ScikitModel = None,
                                  column_names: List[str] = None,
                                  auc_threshold: float = .1,
                                  new_target_column: str = 'is_left') -> bool:
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

        X = df_all_data_with_target.drop(columns=new_target_column)
        y = df_all_data_with_target[new_target_column]

        if not ml_discriminate_model:
            self.ml_discriminate_model = CatBoostClassifier(
                num_trees=3,
                max_depth=3,
                cat_features=self.cat_features,
                random_state=RANDOM_STATE,
                verbose=False
            )

            X = cat_features_fillna(X, self.cat_features)
        else:
            self.ml_discriminate_model = ml_discriminate_model

        if not is_classifier(self.ml_discriminate_model):
            raise TypeError(
                'Model `ml_discriminate_model` '
                'has to be a classification model'
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.5, random_state=RANDOM_STATE
        )

        self.ml_discriminate_model.fit(X_train, y_train)

        y_score = self.ml_discriminate_model.predict_proba(X_test)[:, 1]

        auc_drift_check_model = roc_auc_score(y_true=y_test, y_score=y_score)

        self.drift = (symmetric_auc(auc_drift_check_model)
                      < symmetric_auc(auc_threshold))

        if not self.minimal:
            self.interpretable_drift = InterpretableDrift(
                model=self.ml_discriminate_model,
                X_train=X_train,
                X_test=X_test,
                column_names=(column_names if column_names
                              else self.df_left_data.columns.tolist())
            )

            self.interpretable_drift.most_discriminative_features_plot()

        is_there_drift = symmetric_auc(auc_drift_check_model) > auc_threshold

        if self.verbose:
            print(
                'Drift found in discriminative model step, '
                'take a look on the most discriminative features '
                '(plots when minimal is set to False)' if is_there_drift
                else 'No drift found in discriminative model step',
                end='\n\n'
            )

            print(f'AUC drift check model: {auc_drift_check_model:.2f}')
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
                 pvalue_threshold: float = .05,
                 cardinality_threshold: int = 20,
                 pct_level_threshold: float = .05,
                 pct_change_level_threshold: float = .05):
        """Inits `DriftChecker` with `left_data` and `right_data`
        to compare

        Both needed in `pandas.DataFrame` type

        `pvalue_threshold` is set to .01 by default, `cardinality_threshold`
        is set to 20 , `pct_level_threshold` is set to .05 and
        `pct_change_level_threshold` is set to .05, you can modify
        them if your problem fits better with other thresholds

        This class refers to data drift checks, which what it compares
        is the distribution of the features (one by one)
        """
        super().__init__(df_left_data, df_right_data, verbose, minimal)

        self.pvalue_threshold = pvalue_threshold
        self.cardinality_threshold = cardinality_threshold
        self.pct_level_threshold = pct_level_threshold
        self.pct_change_level_threshold = pct_change_level_threshold
        self.high_cardinality_features = []
        self.drifted_features = []

    def check_numerical_columns(self) -> bool:
        """Given `numerical_columns` check all drifts
        """
        dict_each_column_pvalues = defaultdict(float)
        for numerical_column in self.num_features:
            _, pvalue = stats.ks_2samp(self.df_left_data[numerical_column],
                                       self.df_right_data[numerical_column])
            dict_each_column_pvalues[numerical_column] = pvalue

        self.drifted_features = (
            self.drifted_features
            + [column
               for column, pvalue in dict_each_column_pvalues.items()
               if pvalue < self.pvalue_threshold]
        )

        is_there_drift = len(self.drifted_features) > 0

        if self.verbose:
            if is_there_drift:
                print('Drift found in numerical columns check step, '
                      'take a look on the variables that are drifted, '
                      'if one is not important you could simply delete it, '
                      'otherwise check the data source',
                      end='\n\n')

                print(f'Features drifted: {", ".join(self.drifted_features)}')
            else:
                print('No drift found in numerical columns check step',
                      end='\n\n')

        return is_there_drift

    def check_categorical_columns(self) -> bool:
        """Given `categorical_columns` check all drifts
        """
        dict_each_column_cardinality = defaultdict(float)
        dict_each_column_level_difference = defaultdict(float)
        for categorical_column in self.cat_features:
            categorical_levels_left = compute_levels_count_and_pct(
                self.df_left_data, categorical_column=categorical_column
            )

            categorical_levels_right = compute_levels_count_and_pct(
                self.df_right_data, categorical_column=categorical_column
            )

            categorical_levels_joined = (
                categorical_levels_left
                .merge(categorical_levels_right,
                       on='index',
                       suffixes=('_left', '_right'),
                       how='outer')
            )

            dict_each_column_cardinality[categorical_column] = (
                len(categorical_levels_joined)
            )

            pct_category_level_left = f'{categorical_column}_norm_left'
            pct_category_level_right = f'{categorical_column}_norm_right'

            dict_each_column_level_difference[categorical_column] = (
                abs(categorical_levels_joined[pct_category_level_left]
                    - categorical_levels_joined[pct_category_level_right])
                .iloc[0]
            )

        self.high_cardinality_features = [
            column
            for column, cardinality in dict_each_column_cardinality.items()
            if cardinality > self.cardinality_threshold
        ]

        self.drifted_features = (
            self.drifted_features
            + [column
               for column, difference in (dict_each_column_level_difference
                                          .items())
               if difference > self.pct_change_level_threshold]
        )

        is_there_drift = len(self.drifted_features) > 0
        is_there_high_cardinality = len(self.high_cardinality_features)

        if self.verbose:
            if is_there_drift:
                print('Drift found in categorical columns check step, '
                      'take a look on the variables that are drifted, '
                      'if one is not important you could simply delete it, '
                      'otherwise check the data source',
                      end='\n\n')

                warnings.warn(f'Features drifted: '
                              f'{", ".join(self.drifted_features)}')
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

    def check_model(self, column_names: List[str] = None) -> bool:
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

        y_score_left = self.ml_classifier_model.predict_proba(X_left)[:, 1]
        y_score_right = self.ml_classifier_model.predict_proba(X_right)[:, 1]

        auc_left = roc_auc_score(y_true=y_left, y_score=y_score_left)
        auc_right = roc_auc_score(y_true=y_right, y_score=y_score_right)

        if not self.minimal:
            self.interpretable_drift_classifier_model = InterpretableDrift(
                model=self.ml_classifier_model,
                X_train=X_left,
                X_test=X_right,
                column_names=(column_names if column_names
                              else self.df_left_data.columns.tolist())
            )

            (self
             .interpretable_drift_classifier_model
             .most_discriminative_features_plot())

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
