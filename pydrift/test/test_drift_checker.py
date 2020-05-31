import pytest
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier

from pydrift import DataDriftChecker, ModelDriftChecker, DriftCheckerEstimator
from pydrift.exceptions import (ColumnsNotMatchException,
                                DriftEstimatorException)
from pydrift.models import cat_features_fillna
from pydrift.constants import PATH_DATA, RANDOM_STATE

TARGET = 'Survived'

df_titanic = pd.read_csv(PATH_DATA / 'titanic.csv')

X = df_titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', TARGET])
y = df_titanic[TARGET]

X_women = X[X['Sex'] == 'female']
X_men = X[X['Sex'] == 'male']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.5, random_state=RANDOM_STATE, stratify=y
)

cat_features = (X
                .select_dtypes(include=['category', 'object'])
                .columns)

X_filled = cat_features_fillna(X, cat_features)

X_filled_train, X_filled_test, y_filled_train, y_filled_test = (
    train_test_split(
        X_filled, y, test_size=.5, random_state=RANDOM_STATE, stratify=y
    )
)

df_left_data = pd.concat([X_filled_train, y_filled_train], axis=1)
df_right_data = pd.concat([X_filled_test, y_filled_test], axis=1)


def test_columns_not_match_exception():
    """Tests if correctly raised columns not match
     custom exception"""
    with pytest.raises(ColumnsNotMatchException):
        DataDriftChecker(
            X_train.drop(columns='Sex'), X_test, minimal=True, verbose=False
        )

        DataDriftChecker(
            X_train, X_test.drop(columns='Cabin'), minimal=True, verbose=False
        )


def test_estimator_drift_ko():
    """Tests if correctly check drifted data
    in a pipeline
    """
    with pytest.raises(DriftEstimatorException):
        df_train_filled = pd.concat([X_filled_train, y_train], axis=1)
        df_train_filled_drifted = df_train_filled[
            (df_train_filled['Pclass'] > 1) & (
                    df_train_filled['Fare'] > 10)].copy()

        X_train_filled_drifted = df_train_filled_drifted.drop(columns=TARGET)
        y_train_filled_drifted = df_train_filled_drifted[TARGET]

        df_test_filled = pd.concat([X_filled_test, y_test], axis=1)
        df_test_filled_drifted = df_test_filled[
            ~(df_test_filled['Pclass'] > 1) & (
                    df_test_filled['Fare'] > 10)].copy()

        X_test_filled_drifted = df_test_filled_drifted.drop(columns=TARGET)

        ml_classifier_model = CatBoostClassifier(
            num_trees=5,
            max_depth=3,
            cat_features=cat_features,
            random_state=RANDOM_STATE,
            verbose=False
        )

        pipeline_catboost_drift_checker = make_pipeline(
            DriftCheckerEstimator(ml_classifier_model=ml_classifier_model,
                                  column_names=X.columns,
                                  minimal=True)
        )

        pipeline_catboost_drift_checker.fit(X_train_filled_drifted,
                                            y_train_filled_drifted)

        pipeline_catboost_drift_checker.predict_proba(X_test_filled_drifted)


def test_data_drift_ok():
    """Tests if correctly check non-drifted data"""
    data_drift_checker_ok = DataDriftChecker(
        X_train, X_test, minimal=True, verbose=False
    )

    data_drift_checker_ok.check_categorical_columns()

    print(data_drift_checker_ok.dict_each_column_pvalues_categorical)
    assert not data_drift_checker_ok.ml_model_can_discriminate()
    assert not data_drift_checker_ok.check_numerical_columns()
    assert not data_drift_checker_ok.check_categorical_columns()


def test_data_drift_ko():
    """Tests if correctly check drifted data"""
    data_drift_checker_ok = DataDriftChecker(
        X_women, X_men, minimal=True, verbose=False
    )

    assert data_drift_checker_ok.ml_model_can_discriminate()
    assert data_drift_checker_ok.check_numerical_columns()
    assert data_drift_checker_ok.check_categorical_columns()


def test_model_drift_ok():
    """Tests if correctly check non-drifted model"""
    ml_classifier_model = CatBoostClassifier(
        num_trees=5,
        max_depth=3,
        cat_features=cat_features,
        random_state=RANDOM_STATE,
        verbose=False
    )

    ml_classifier_model.fit(X_filled_train, y_filled_train)

    model_drift_checker_ok = ModelDriftChecker(
        df_left_data, df_right_data, ml_classifier_model,
        target_column_name=TARGET, minimal=True, verbose=False
    )

    assert not model_drift_checker_ok.check_model()


def test_model_drift_ko():
    """Tests if correctly check drifted model"""
    ml_classifier_model_drifted = CatBoostClassifier(
        num_trees=10,
        max_depth=6,
        cat_features=cat_features,
        random_state=RANDOM_STATE,
        verbose=False
    )

    ml_classifier_model_drifted.fit(X_filled_train, y_filled_train)

    model_drift_checker_ko = ModelDriftChecker(
        df_left_data, df_right_data, ml_classifier_model_drifted,
        target_column_name=TARGET, minimal=True, verbose=False
    )

    assert model_drift_checker_ko.check_model()


def test_estimator_drift_ok():
    """Tests if correctly check non-drifted data
    in a pipeline
    """
    ml_classifier_model = CatBoostClassifier(
        num_trees=5,
        max_depth=3,
        cat_features=cat_features,
        random_state=RANDOM_STATE,
        verbose=False
    )

    pipeline_catboost_drift_checker = make_pipeline(
        DriftCheckerEstimator(ml_classifier_model=ml_classifier_model,
                              column_names=X.columns,
                              minimal=True)
    )

    pipeline_catboost_drift_checker.fit(X_filled_train, y_filled_train)
    pipeline_catboost_drift_checker.predict_proba(X_filled_test)
