import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, compute_model_metrics

#This is a helper function to make a sample dataFrame
def _make_sample_data():
    df = pd.DataFrame({
        "workclass": ["Private", "State-gov"],
        "education": ["Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Craft-repair"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "United-States"],
        "age": [39, 50],
        "fnlwgt": [77516, 234721],
        "education-num": [13, 9],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "salary": ["<=50K", ">50K"],
    })

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return df, cat_features



def test_process_data_shapes_and_encoder():
    """
    Ensure process_data returns X, y, encoder, lb with correct sizes and types.
    """
    df, cat_features = _make_sample_data()

    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    # Two rows should remain
    assert X.shape[0] == 2
    assert len(y) == 2

    #The encoder and label binarizer should be fitted objects
    assert encoder is not None
    assert lb is not None



def test_train_model_returns_random_forest():
    """
    Ensure train_model trains and returns a RandomForestClassifier instance.
    """
    df, cat_features = _make_sample_data()

    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_values():
    """
    Ensure compute_model_metrics returns correct precision, recall, and F1
    on a simple, known example.
    """
    # True labels and predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    p, r, f1 = compute_model_metrics(y_true, y_pred)

    assert np.isclose(p, 0.5)
    assert np.isclose(r, 0.5)
    assert np.isclose(f1, 0.5)
