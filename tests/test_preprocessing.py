import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer



# Preprocessing-Pipeline
def build_preprocessing():
    numeric_features = ["beds", "bathrooms"]
    categorical_features = ["room_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# TEST 1: Numerische Imputation funktioniert korrekt
def test_numeric_imputation():
    df = pd.DataFrame({
        "beds": [1, np.nan, 3],
        "bathrooms": [np.nan, 2, 3],
        "room_type": ["Entire home", "Private room", "Entire home"]
    })

    pre = build_preprocessing()
    transformed = pre.fit_transform(df)

    # Beds median = 2
    # Bathrooms median = 2.5
    median_beds = 2
    median_bathrooms = 2.5

    # Zeile 1 (index 1) hatte NaN bei beds
    assert not np.isnan(transformed[1, 0])
    # Zeile 0 (index 0) hatte NaN bei bathrooms → gescalter Wert != NaN
    assert not np.isnan(transformed[0, 1])



# TEST 2: StandardScaler erzeugt Mittelwert ~0 und Varianz ~1
def test_scaling_properties():
    df = pd.DataFrame({
        "beds": [1, 2, 3],
        "bathrooms": [1, 2, 3],
        "room_type": ["A", "A", "B"]
    })

    pre = build_preprocessing()
    transformed = pre.fit_transform(df)

    # Extrahiere die ersten beiden Spalten = numerische Features
    numeric_data = transformed[:, :2]

    # Mittelwert sollte ~0 sein
    assert np.allclose(np.mean(numeric_data, axis=0), 0, atol=1e-7)
    # Varianz sollte ~1 sein
    assert np.allclose(np.var(numeric_data, axis=0), 1, atol=1e-7)


# TEST 3: One-Hot-Encoding funktioniert und ignoriert unknown categories
def test_one_hot_encoding_unknown_categories():
    train_df = pd.DataFrame({
        "beds": [1, 2],
        "bathrooms": [1, 2],
        "room_type": ["Entire home", "Private room"]
    })

    test_df = pd.DataFrame({
        "beds": [3],
        "bathrooms": [3],
        "room_type": ["Igloo"]   # unbekannte Kategorie
    })

    pre = build_preprocessing()
    pre.fit(train_df)
    transformed_test = pre.transform(test_df)

    # Check: für unknown category muss der OHE-Block ein Nullvektor sein
    ohe_vector = transformed_test[0, 2:]  # ab Spalte 2 = OHE Spalten
    assert np.all(ohe_vector == 0)


# TEST 4: Pipeline gibt array ohne NaN zurück
def test_no_nans_in_output():
    df = pd.DataFrame({
        "beds": [np.nan, 2],
        "bathrooms": [1, np.nan],
        "room_type": [np.nan, "Private room"]
    })

    pre = build_preprocessing()
    transformed = pre.fit_transform(df)

    assert not np.isnan(transformed).any()