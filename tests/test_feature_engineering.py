import numpy as np
import pandas as pd
import pytest


# Funktion zur Log-Transformation (wie in deiner Pipeline)
def compute_log_price(df):
    if (df["price"] <= 0).any():
        raise ValueError("Price must be > 0 for log transformation.")
    df["log_price"] = np.log(df["price"])
    return df


# TEST 1: Log-Transformation korrekt berechnet
def test_log_transformation_correct():
    df = pd.DataFrame({"price": [100, 200, 400]})

    df = compute_log_price(df)

    expected = np.log(np.array([100, 200, 400]))
    assert np.allclose(df["log_price"].values, expected)


# TEST 2: Price <= 0 soll einen Fehler werfen
def test_log_transformation_invalid_values():
    df = pd.DataFrame({"price": [100, 0, -10]})

    with pytest.raises(ValueError):
        compute_log_price(df)


# TEST 3: Keine NaNs nach der Transformation
def test_log_transformation_no_nans():
    df = pd.DataFrame({"price": [50, 80, 120]})
    df = compute_log_price(df)

    assert not df["log_price"].isna().any()


# TEST 4: Case mit float-Werten
def test_log_transformation_floats():
    df = pd.DataFrame({"price": [99.99, 150.50, 300.10]})
    df = compute_log_price(df)

    expected = np.log(df["price"])
    assert np.allclose(df["log_price"], expected)