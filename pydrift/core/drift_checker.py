import abc
import shap
import pandas as pd

from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from catboost import CatBoostClassifier
from typing_extensions import Protocol

from ..constants import RANDOM_STATE
from ..models import cat_features_fillna


class ScikitModel(Protocol):
    """Scikit model typing
    """
    def fit(self, X, y, verbose): ...
    def predict_proba(self, X) -> object: ...


class DriftChecker(abc.ABC):
    """
    Parent class for drift checks
    """
    def __init__(self,
                 df_left_data: pd.DataFrame,
                 df_right_data: pd.DataFrame,
                 minimal: bool = False):
        """Inits `DriftChecker` with `left_data` and `right_data`
        to compare

        Both needed in `pandas.DataFrame` type

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

        self.df_left_data = df_left_data
        self.df_right_data = df_right_data
        self.minimal = minimal

        self.cat_features = (df_left_data
                             .select_dtypes(include=['category', 'object'])
                             .columns)

        self.num_features = (df_left_data
                             .select_dtypes(include='number')
                             .columns)

        self.ml_classifier_model = None
        self.drift = False

    @abc.abstractmethod
    def check_numerical_columns(self, numerical_columns: [str]):
        """Given `numerical_columns` check all drifts
        """
        pass

    @abc.abstractmethod
    def check_categorical_columns(self, categorical_columns: [str]):
        """Given `categorical_columns` check all drifts
        """
        pass

    def ml_model_can_discriminate(self,
                                  ml_classifier_model: ScikitModel = None,
                                  auc_threshold: float = .55,
                                  new_target_column: str = 'is_left',
                                  verbose: bool = True) -> bool:
        """Creates a machine learning model based in `sklearn`,
        this model will be a classification model that will try
        to predict if a register is from `left_data` or `right_data`

        `CatBoostClassifier` is used by default because it takes categorical
        data natively and is a state of the art algorithm. Parameters are
        not too high to avoid overfitting. It is within the function instead
        of having it in the parameters because `self.cat_features` is needed

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

        if not ml_classifier_model:
            ml_classifier_model = CatBoostClassifier(
                num_trees=10,
                max_depth=3,
                cat_features=self.cat_features,
                random_state=RANDOM_STATE
            )

            X = cat_features_fillna(X, self.cat_features)

        if not is_classifier(ml_classifier_model):
            raise TypeError(
                'Model `ml_classifier_model` has to be a classification model'
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=RANDOM_STATE
        )

        self.ml_classifier_model = clone(ml_classifier_model)

        self.ml_classifier_model.fit(X_train, y_train, verbose=False)

        y_score = self.ml_classifier_model.predict_proba(X_test)[:, 1]

        auc_drift_check_model = roc_auc_score(y_true=y_test, y_score=y_score)

        self.drift = (symmetric_auc(auc_drift_check_model)
                      < symmetric_auc(auc_threshold))

        if not self.minimal:
            explainer = shap.TreeExplainer(
                model=self.ml_classifier_model,
                feature_perturbation='tree_path_dependent'
            )

            shap_values = explainer.shap_values(X_train)

            shap.summary_plot(shap_values,
                              X_train,
                              plot_type='bar',
                              title='Most Discriminative Features')

        is_there_drift = (symmetric_auc(auc_drift_check_model)
                          < symmetric_auc(auc_threshold))

        if verbose:
            print(
                'No drift found in discriminative model step' if is_there_drift
                else 'Drift found in discriminative model step, '
                     'take a look on the most discriminative features '
                     '(plots when minimal is set to False)',
                end='\n\n'
            )

            print(f'AUC drift check model: {auc_drift_check_model}')
            print(f'AUC threshold: {auc_threshold}')

        return is_there_drift


class DataDriftChecker(DriftChecker):
    """
    Parent class for drift checks
    """
    def __init__(self,
                 df_left_data: pd.DataFrame,
                 df_right_data: pd.DataFrame):
        """Inits `DriftChecker` with `left_data` and `right_data`
        to compare

        Both needed in `pandas.DataFrame` type

        This class refers to data drift checks, which what it compares
        is the distribution of the features (one by one)
        """
        super().__init__(df_left_data, df_right_data)

    def check_numerical_columns(self, numerical_columns: [str]):
        """Given `numerical_columns` check all drifts
        """
        pass

    def check_categorical_columns(self, categorical_columns: [str]):
        """Given `categorical_columns` check all drifts
        """
        pass


class ModelDriftChecker(DriftChecker):
    """
    Parent class for drift checks
    """
    def __init__(self,
                 df_left_data: pd.DataFrame,
                 df_right_data: pd.DataFrame,
                 target_column_name: str):
        """Inits `ModelDriftChecker` with `left_data` and `right_data`
        to compare

        Both needed in `pandas.DataFrame` type

        This class refers to model drift checks, which what it compares
        is the relationship of the variables with the target (univariate)
        """
        super().__init__(df_left_data, df_right_data)

        self.target_column_name = target_column_name

    def check_numerical_columns(self, numerical_columns: [str]):
        """Given `numerical_columns` check all drifts
        """
        pass

    def check_categorical_columns(self, categorical_columns: [str]):
        """Given `categorical_columns` check all drifts
        """
        pass
