import pandas as pd

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from pydrift import DataDriftChecker, ModelDriftChecker
from pydrift.models import cat_features_fillna
from pydrift.constants import PATH_DATA, RANDOM_STATE

TARGET = 'Survived'

df_titanic = pd.read_csv(PATH_DATA / 'titanic.csv')

X = df_titanic.drop(columns=['PassengerId', 'Name', TARGET])
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


def test_data_drift_ok():
    """Tests if correctly check non-drifted data"""
    data_drift_checker_ok = DataDriftChecker(
        X_train, X_test, minimal=True, verbose=False
    )

    assert not data_drift_checker_ok.ml_model_can_discriminate()
    assert not data_drift_checker_ok.check_numerical_columns()
    assert not data_drift_checker_ok.check_categorical_columns()


def test_data_drift_ko():
    """Tests if correctly check non-drifted data"""
    data_drift_checker_ok = DataDriftChecker(
        X_women, X_men, minimal=True, verbose=False
    )

    assert data_drift_checker_ok.ml_model_can_discriminate()
    assert data_drift_checker_ok.check_numerical_columns()
    assert data_drift_checker_ok.check_categorical_columns()


def test_model_drift_ok():
    """Tests if correctly check non-drifted data"""
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
    """Tests if correctly check non-drifted data"""
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
