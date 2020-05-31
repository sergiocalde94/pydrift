import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List

from .drift_checker import DataDriftChecker
from ..models import ScikitModel
from ..exceptions import DriftEstimatorException


class DriftCheckerEstimator(BaseEstimator, ClassifierMixin):
    """Creates a sklearn estimator ready to use it or
    within a pipeline

    Parameter `column_names` is needed because sklearn
    transform data to numpy (no columns)

    Check `DataDriftChecker` to change the rest of
    parameters if you need
    """
    def __init__(self,
                 ml_classifier_model: ScikitModel,
                 column_names: List[str],
                 verbose: bool = False,
                 minimal: bool = True,
                 pvalue_threshold_numerical: float = .05,
                 pvalue_threshold_categorical: float = .05,
                 cardinality_threshold: int = 20):
        self.ml_classifier_model = ml_classifier_model
        self.column_names = column_names
        self.df_left_data = None
        self.df_right_data = None
        self.verbose = verbose
        self.minimal = minimal
        self.pvalue_threshold_numerical = pvalue_threshold_numerical
        self.pvalue_threshold_categorical = pvalue_threshold_categorical
        self.cardinality_threshold = cardinality_threshold
        self.data_drift_checker = None
        self.is_drift_in_numerical_columns = False
        self.is_drift_in_categorical_columns = False
        self.is_drift_in_ml_model_can_discriminate = False

    def fit(self, X: np.array, y: np.array = None):
        """Fits estimator in `self.ml_classifier_model`
        and assigns X to `self.df_left_data`
        """
        self.df_left_data = X

        self.ml_classifier_model.fit(X, y)

        return self

    def fill_data_drift_checker(self, X: np.array):
        """Fill data drift checker object and stores it
        in `self.data_drift_checker`
        """
        self.df_right_data = X

        self.data_drift_checker = DataDriftChecker(
            df_left_data=pd.DataFrame(self.df_left_data,
                                      columns=self.column_names),
            df_right_data=pd.DataFrame(self.df_right_data,
                                       columns=self.column_names),
            verbose=self.verbose,
            minimal=self.minimal,
            pvalue_threshold_numerical=self.pvalue_threshold_numerical,
            pvalue_threshold_categorical=self.pvalue_threshold_categorical,
            cardinality_threshold=self.cardinality_threshold
        )

    def check_drift(self):
        """Checks data drift for numerical and categorical
        data and the discriminative model
        """
        self.is_drift_in_numerical_columns = (
            self.data_drift_checker.check_numerical_columns()
        )

        self.is_drift_in_categorical_columns = (
            self.data_drift_checker.check_categorical_columns()
        )

        self.is_drift_in_ml_model_can_discriminate = (
            self.data_drift_checker.ml_model_can_discriminate()
        )

        is_there_drift = (
                self.is_drift_in_numerical_columns
                or self.is_drift_in_categorical_columns
                or self.is_drift_in_ml_model_can_discriminate
        )

        if is_there_drift:
            if self.is_drift_in_numerical_columns:
                print('Drift found in numerical columns check step')

            if self.is_drift_in_categorical_columns:
                print('Drift found in categorical columns check step')

            if self.is_drift_in_ml_model_can_discriminate:
                print('Drift found in discriminative model step')

            raise DriftEstimatorException(
                'Drift found in your estimation process'
            )

    def get_drifted_features(self):
        """Alias to self.data_drift_checker.drifted_features
        """
        return ', '.join(self.data_drift_checker.drifted_features)

    def get_high_cardinality_features(self):
        """Alias to self.data_drift_checker.high_cardinality_features
        """
        return ', '.join(self.data_drift_checker.high_cardinality_features)

    def predict(self, X: np.array):
        """Checks if there is a data drift and makes a prediction
        with `predict` method of sklearn model
        """
        self.fill_data_drift_checker(X)
        self.check_drift()

        return self.ml_classifier_model.predict(X)

    def predict_proba(self, X: np.array):
        """Checks if there is a data drift and makes a prediction
        with `predict_proba` method of sklearn model
        """
        self.fill_data_drift_checker(X)
        self.check_drift()

        return self.ml_classifier_model.predict_proba(X)
