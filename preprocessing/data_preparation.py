"""
MODUL: DATA PREPARATION & PIPELINE SETUP

Beschreibung:
    Dieses Modul lädt die Airbnb-Rohdaten, führt eine umfassende Reinigung durch
    und erstellt die Preprocessing-Pipeline für das Machine Learning.

Author: Burak Ugurelli
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Windows support
try:
    import colorama

    colorama.init(autoreset=True)
except ImportError:
    pass


# Design config

class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    GREY = '\033[90m'


def block(title: str) -> None:
    """
    Erstellt einen visuell hervorgehobenen Hauptblock im Terminal.
    """
    print("\n" + C.BLUE + "╔" + "═" * 60 + "╗" + C.ENDC)
    print(f"{C.BLUE}║{C.ENDC} {C.BOLD}{title.center(58)}{C.ENDC} {C.BLUE}║{C.ENDC}")
    print(C.BLUE + "╚" + "═" * 60 + "╝" + C.ENDC)


def subblock(title: str) -> None:
    """
    Erstellt einen Unterabschnitt mit Titel und Trennlinie.
    """
    print(f"\n{C.YELLOW}➤ {C.BOLD}{title}{C.ENDC}")
    print(f"{C.GREY}" + "─" * 40 + f"{C.ENDC}")


def print_status(icon: str, key: str, value: str, color: str = C.ENDC) -> None:
    """
    Gibt eine formatierte Statuszeile aus.
    """
    print(f"  {icon} {C.GREY}{key:<20}{C.ENDC} : {color}{value}{C.ENDC}")


# Core Logic

def load_and_prepare_data(path: Optional[Path] = None) -> Tuple[
    ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Lädt den Datensatz, reinigt ihn und bereitet die ML-Pipeline vor.
    """

    block("PREPROCESSING PIPELINE")

    # 1. PFAD & LOADING
    subblock("Data Loading")

    if path is None:
        base_dir = Path(__file__).resolve().parents[1]
        path = base_dir / "data" / "train.csv"

    print_status("📂", "Source Path", str(path))

    try:
        df = pd.read_csv(path, sep=",", engine="python", on_bad_lines="skip")
    except FileNotFoundError:
        print(f"\n{C.RED}❌ CRITICAL ERROR: Datei nicht gefunden!{C.ENDC}")
        raise

    df = df.drop_duplicates().copy()

    print_status("📊", "Rows Loaded", f"{df.shape[0]:,}", C.BOLD)
    print_status("📊", "Columns Loaded", f"{df.shape[1]}", C.BOLD)

    # 2. DATA CLEANING
    subblock("Data Cleaning & Engineering")

    # Drop Columns
    drop_cols = ["id", "name", "description", "thumbnail_url"]
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)

    # Zielvariable sicherstellen: log_price
    if "log_price" not in df.columns:
        if "price" not in df.columns:
            raise KeyError("Weder 'log_price' noch 'price' im Datensatz vorhanden.")
        df["log_price"] = np.log(df["price"].astype(float))
    target = "log_price"

    # Imputation Strategien
    num_cols = ["bathrooms", "bedrooms", "beds", "review_scores_rating"]
    cat_cols = ["neighbourhood", "zipcode", "property_type"]

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Boolean Mapping
    bool_map = {"t": True, "f": False}
    for col in ["host_has_profile_pic", "host_identity_verified", "instant_bookable"]:
        if col in df.columns:
            df[col] = df[col].map(bool_map)

    # Parsing Prozente
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = (
            df["host_response_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )
        df["host_response_rate"] = df["host_response_rate"].fillna(df["host_response_rate"].median())

    # Parsing Dates & Feature Engineering
    date_cols = ["first_review", "last_review", "host_since"]
    for date_col in date_cols:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Timestamp today kann Ergebnisse verändern
    if "last_review" in df.columns:
        df["days_since_last_review"] = (pd.Timestamp("today") - df["last_review"]).dt.days
        df["days_since_last_review"] = df["days_since_last_review"].fillna(df["days_since_last_review"].median())

    if "host_since" in df.columns:
        df["host_duration_days"] = (pd.Timestamp("today") - df["host_since"]).dt.days
        df["host_duration_days"] = df["host_duration_days"].fillna(df["host_duration_days"].median())

    # Amenities Parsing
    if "amenities" in df.columns:
        df["n_amenities"] = df["amenities"].apply(lambda x: len(str(x).split(",")))
        df["has_wifi"] = df["amenities"].str.contains("Wireless Internet", case=False, na=False)
        df = df.drop(columns=["amenities"])

    print_status("🛠 ", "Feature Engineering", "Amenities geparsed & Zeitdifferenzen berechnet")

    # Filterung
    if "cancellation_policy" in df.columns:
        blacklist = ["super_strict_30", "super_strict_60"]
        before = len(df)
        df = df[~df["cancellation_policy"].isin(blacklist)].copy()
        removed = before - len(df)

        if removed > 0:
            pct = removed / before * 100
            print_status(
                "✂️ ",
                "Filtered Rows",
                f"-{removed} ({pct:.2f}%)",
                C.YELLOW
            )

    # 3. FEATURE SELECTION
    feature_candidates = [
        "accommodates", "bathrooms", "bedrooms", "beds",
        "property_type", "room_type", "cancellation_policy",
        "cleaning_fee", "host_response_rate",
        "host_has_profile_pic", "host_identity_verified",
        "instant_bookable", "n_amenities", "has_wifi",
        "number_of_reviews", "review_scores_rating",
        "latitude", "longitude", "days_since_last_review",
        "host_duration_days", "city"
    ]

    features = [f for f in feature_candidates if f in df.columns]

    X = df[features].copy()
    y = df[target]

    # 4. PIPELINE SETUP
    subblock("Pipeline Construction")

    # Identifikation von Typen
    cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Sicherstellung String-Typ für OneHot
    for col in cat_features:
        X[col] = X[col].astype(str)

    # Globale Kategorien sammeln (für Konsistenz zwischen Train/Test/Prod)
    global_categories = {
        col: sorted(X[col].unique().tolist())
        for col in cat_features
    }

    # Pipeline Definitionen
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[global_categories[col] for col in cat_features],
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    print_status("🔢", "Numeric Features", str(len(num_features)))
    print_status("🔠", "Categorical Feats", str(len(cat_features)))

    total_encoded = sum(len(v) for v in global_categories.values())
    print_status("🧩", "OneHot Dimensions", str(total_encoded), C.CYAN)

    # 5. SPLITTING
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print_status("✂️ ", "Train Split", f"{X_train.shape}", C.GREEN)
    print_status("✂️ ", "Test Split", f"{X_test.shape}", C.GREEN)

    # Trennlinie für sauberen Übergang zum nächsten Modul
    print("\n" + C.BLUE + "═" * 60 + C.ENDC)

    return preprocessor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Testlauf wenn direkt ausgeführt
    preprocessor, X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"\n{C.GREEN}✔ Preprocessing erfolgreich.{C.ENDC}")