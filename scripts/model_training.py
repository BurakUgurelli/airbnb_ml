"""
MODUL: MODELLEVALUATION UND TRAININGSPIPELINE

Beschreibung:
    Zentrales Steuermodul für den Machine Learning Workflow.
    Fügt das Training verschiedener Modellarchitekturen (Linear, Tree-based, Deep Learning),
    führt eine hybride Validierung durch und generiert umfassende Reports.

Features:
    - Hybride Validierung: K-Fold CV für Standardmodelle, Hold-out mit Early Stopping für Boosting/TabNet.
    - Automatische Hardware-Erkennung (CUDA/CPU).
    - Integrierte SHAP-Analyse für Interpretierbarkeit.
    - Error-Handling und Logging.

Autor: Burak Ugurelli
Projekt: Airbnb ML Price Prediction (RWU)
"""

import os
import sys
import time
import warnings
import datetime
from pathlib import Path

# SYSTEM KONFIGURATION (MACOS/MP)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

# Bibliotheken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

# Scikit-Learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Boosting & Deep Learning
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback

# Projekt-Imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts.shap_summary import run_global_shap_summary
from preprocessing.data_preparation import load_and_prepare_data

# Warnungen
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Colorama Setup (Fallback falls nicht installiert)
try:
    import colorama

    colorama.init(autoreset=True)


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
except ImportError:
    class C:
        HEADER = BLUE = CYAN = GREEN = YELLOW = RED = ENDC = BOLD = GREY = ""


# Utils
def block(title):
    print(f"\n{C.BLUE}╔{'═' * 60}╗{C.ENDC}")
    print(f"{C.BLUE}║{C.ENDC} {C.BOLD}{title.center(58)}{C.ENDC} {C.BLUE}║{C.ENDC}")
    print(f"{C.BLUE}╚{'═' * 60}╝{C.ENDC}")


def subblock(title):
    print(f"\n{C.YELLOW}➤ {C.BOLD}{title}{C.ENDC}")
    print(f"{C.GREY}{'─' * 40}{C.ENDC}")


def print_kpi(label, value, unit="", color=C.CYAN):
    print(f"  {C.GREY}• {label:<15}{C.ENDC} : {color}{value}{unit}{C.ENDC}")


# Hardware check
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{C.BOLD}⚡ Hardware-Beschleunigung:{C.ENDC} TabNet läuft auf {C.GREEN}{device_name.upper()}{C.ENDC}")


# Callbacks

class EpochLogger(Callback):
    """
    Custom Callback für TabNet, um den Trainingsfortschritt
    sauber formatiert in der Konsole auszugeben.
    """

    def __init__(self):
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.start_time = time.time()

        tr_loss = logs.get("loss", 0.0)
        val_rmse = logs.get("val_0_rmse", None)

        ep_str = f"{C.BOLD}Ep {epoch + 1:>3}{C.ENDC}"
        loss_str = f"Loss: {tr_loss:.4f}"

        if val_rmse is not None:
            rmse_str = f"Val RMSE: {C.CYAN}{val_rmse:.4f}{C.ENDC}"
            print(f"   ↳ {ep_str} | {loss_str} | {rmse_str} | ⏱ {epoch_time:.2f}s")
        else:
            print(f"   ↳ {ep_str} | {loss_str} | ⏱ {epoch_time:.2f}s")


# Data prep

def prepare_tabnet_inputs(X_train_full, y_train_full, X_test, validation_split=0.2):
    """
    Bereitet Daten spezifisch für TabNet vor.

    Achtung: TabNet benötigt numerische Matrizen (np.ndarray) und verträgt keine
    Pandas DataFrames oder fehlende Werte (NaNs) im Input-Stream.
    """
    # 1. Split in Train/Val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=validation_split, random_state=42
    )

    num_cols = X_train_full.select_dtypes(include=[np.number]).columns
    cat_cols = X_train_full.select_dtypes(exclude=[np.number]).columns

    # 2. Imputation & Scaling Pipeline
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    scaler = MinMaxScaler()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # Numerische Verarbeitung
    X_tr_num = scaler.fit_transform(num_imputer.fit_transform(X_tr[num_cols]))
    X_val_num = scaler.transform(num_imputer.transform(X_val[num_cols]))
    X_test_num = scaler.transform(num_imputer.transform(X_test[num_cols]))

    # Kategorische Verarbeitung
    X_tr_cat = encoder.fit_transform(cat_imputer.fit_transform(X_tr[cat_cols]))
    X_val_cat = encoder.transform(cat_imputer.transform(X_val[cat_cols]))
    X_test_cat = encoder.transform(cat_imputer.transform(X_test[cat_cols]))

    # 3. Merge & Type Casting (float32 ist essentiell für PyTorch)
    X_tr_final = np.hstack([X_tr_num, X_tr_cat]).astype(np.float32)
    X_val_final = np.hstack([X_val_num, X_val_cat]).astype(np.float32)
    X_test_final = np.hstack([X_test_num, X_test_cat]).astype(np.float32)

    y_tr_final = y_tr.values.reshape(-1, 1).astype(np.float32)
    y_val_final = y_val.values.reshape(-1, 1).astype(np.float32)

    return X_tr_final, y_tr_final, X_val_final, y_val_final, X_test_final


# Evaluation

def cross_validation_evaluation(model, X_train, y_train, preprocessor, n_splits=3):
    """
    Führt eine K-Fold Cross Validation für Standard-Modelle durch.
    TabNet wird hier explizit ausgeschlossen.
    """
    if isinstance(model, TabNetRegressor):
        print(f"  {C.YELLOW}⚠ Info: TabNet überspringt Standard-CV.{C.ENDC}")
        return float("nan"), float("nan"), float("nan")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model_clone = clone(model)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model_clone)])

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        r2 = r2_score(y_val, preds)
        fold_metrics.append((rmse, mae, r2))

    return (np.mean([m[0] for m in fold_metrics]),
            np.mean([m[1] for m in fold_metrics]),
            np.mean([m[2] for m in fold_metrics]))


def train_and_evaluate_final(model, X_train, X_test, y_train, y_test,
                             preprocessor, name, results_path, shap_path, csv_path):
    """
    Hauptfunktion für Training und Evaluation eines einzelnen Modells.
    Unterscheidet Logik zwischen TabNet, Boosting-Modellen und Standard-Modellen.
    """
    block(f"MODEL: {name.upper()}")
    start_time = time.time()

    # Modell-Gruppen Identifikation
    boosting_models = (XGBRegressor, LGBMRegressor, CatBoostRegressor)
    tree_models = (RandomForestRegressor, GradientBoostingRegressor,
                   XGBRegressor, LGBMRegressor, CatBoostRegressor)

    is_booster = isinstance(model, boosting_models)
    is_tabnet = isinstance(model, TabNetRegressor)

    # 1. Cross Validation Phase
    if is_tabnet:
        cv_rmse, cv_mae, cv_r2 = float("nan"), float("nan"), float("nan")
    else:
        cv_rmse, cv_mae, cv_r2 = cross_validation_evaluation(model, X_train, y_train, preprocessor)

    if not np.isnan(cv_rmse):
        subblock("Cross Validation Results")
        print_kpi("CV RMSE", f"{cv_rmse:.4f}")
        print_kpi("CV R²", f"{cv_r2:.4f}")

    preds = None
    fitted_model = None

    # 2. Training Phase

    # TabNet Special Handling
    if is_tabnet:
        X_tr_tab, y_tr_tab, X_val_tab, y_val_tab, X_test_tab = prepare_tabnet_inputs(
            X_train, y_train, X_test
        )
        subblock("Training Phase (TabNet)")
        model.fit(
            X_tr_tab, y_tr_tab,
            eval_set=[(X_val_tab, y_val_tab)],
            eval_metric=["rmse"],
            max_epochs=30, patience=7,
            batch_size=512, virtual_batch_size=128,
            num_workers=0, pin_memory=False,
            callbacks=[EpochLogger()]
        )
        preds = model.predict(X_test_tab).flatten()
        fitted_model = model

    # Boosting mit Early Stopping
    elif is_booster:
        subblock("Training Phase (Boosting w/ Early Stopping)")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        preprocessor.fit(X_tr)
        X_tr_trans = preprocessor.transform(X_tr)
        X_val_trans = preprocessor.transform(X_val)
        X_test_trans = preprocessor.transform(X_test)

        if isinstance(model, LGBMRegressor):
            cb = [lgb.early_stopping(50, verbose=False)]
            model.fit(X_tr_trans, y_tr, eval_set=[(X_val_trans, y_val)], eval_metric="rmse", callbacks=cb)
        elif isinstance(model, XGBRegressor):
            model.set_params(early_stopping_rounds=50)
            model.fit(X_tr_trans, y_tr, eval_set=[(X_val_trans, y_val)], verbose=False)
        elif isinstance(model, CatBoostRegressor):
            model.fit(X_tr_trans, y_tr, eval_set=[(X_val_trans, y_val)], early_stopping_rounds=50, verbose=False)

        preds = model.predict(X_test_trans)
        fitted_model = model

    # Standard Modelle
    else:
        subblock("Training Phase (Standard)")
        print(f"  {C.GREY}Starte Train auf Trainingsdaten...{C.ENDC}")
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        fitted_model = pipeline.named_steps["model"]

    # 3. Interpretability (SHAP)

    try:
        if isinstance(fitted_model, tree_models):
            feature_names = preprocessor.get_feature_names_out()
            X_test_trans = preprocessor.transform(X_test)
            if hasattr(X_test_trans, "toarray"):
                X_test_trans = X_test_trans.toarray()

            explainer = shap.Explainer(fitted_model)
            shap_values = explainer(X_test_trans[:500])

            path_summary = shap_path / f"{name}_shap_summary.png"
            np.save(shap_path / f"{name}_shap_values.npy", shap_values.values)
            np.save(shap_path / f"{name}_feature_names.npy", np.array(feature_names))

            plt.figure()
            shap.summary_plot(shap_values, X_test_trans[:500], feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(path_summary, dpi=300)
            plt.close()

            subblock("Explainability (SHAP)")
            print_status = f"{C.GREEN}✔ Erstellt{C.ENDC}"
            print_kpi("Summary Plot", print_status)
    except Exception as e:
        print(f"  {C.RED}✘ SHAP Fehler: {e}{C.ENDC}")

    # 4. Final Metrics
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    runtime = time.time() - start_time

    subblock("Final Test Evaluation")
    print_kpi("RMSE", f"{rmse:.4f}", "", C.BOLD + C.CYAN)
    print_kpi("MAE", f"{mae:.4f}", "", C.CYAN)
    print_kpi("R² Score", f"{r2:.4f}", "", C.BOLD + C.GREEN)
    print_kpi("Runtime", f"{runtime:.2f}", "s", C.YELLOW)

    # Export Fehleranalyse
    pd.DataFrame({"true": y_test, "pred": preds, "error": y_test - preds}) \
        .to_csv(csv_path / f"{name}_errors.csv", index=False)

    # Export Feature Importance
    try:
        if hasattr(fitted_model, "feature_importances_"):
            try:
                f_names = preprocessor.get_feature_names_out()
            except:
                f_names = [f"Feature_{i}" for i in range(len(fitted_model.feature_importances_))]

            fi_path = csv_path / f"{name}_feature_importances.csv"
            pd.DataFrame({"feature": f_names, "importance": fitted_model.feature_importances_}) \
                .to_csv(fi_path, index=False)
            print(f"  {C.GREY}Feature Importance gespeichert.{C.ENDC}")
    except:
        pass

    return {
        "model": name, "cv_rmse": cv_rmse, "cv_mae": cv_mae, "cv_r2": cv_r2,
        "test_rmse": rmse, "test_mae": mae, "test_r2": r2, "runtime": runtime
    }


def get_models():
    """Definiert die Liste der zu evaluierenden Modelle und deren Hyperparameter."""

    # Prüfen, ob GPU verfügbar ist, um Parameter dynamisch zu setzen
    gpu_available = torch.cuda.is_available()

    # XGBoost Konfiguration
    xgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1
    }

    if gpu_available:
        # GPU Einstellungen für XGBoost
        xgb_params.update({
            "device": "cuda",
            "tree_method": "hist"
        })
        print(f"{C.GREEN}✔ XGBoost nutzt GPU{C.ENDC}")
    else:
        xgb_params["tree_method"] = "hist"  # CPU Fall

    # CatBoost Konfiguration
    cat_params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "l2_leaf_reg": 3.0,
        "allow_writing_files": False,
        "verbose": False
    }

    if gpu_available:
        cat_params["task_type"] = "GPU"
        cat_params["devices"] = "0"
        print(f"{C.GREEN}✔ CatBoost nutzt GPU{C.ENDC}")
    else:
        cat_params["task_type"] = "CPU"

    # Definitions
    return {
        "LinearRegression": LinearRegression(),

        "Ridge": Ridge(alpha=1.0, max_iter=2000),

        "Lasso": Lasso(alpha=0.001, max_iter=2000),

        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000),

        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ),

        "XGBoost": XGBRegressor(**xgb_params),

        "LightGBM": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=45,
            lambda_l1=0.1,
            lambda_l2=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            device='cpu'  # Explizit CPU, da GPU-Installation oft fehleranfällig
        ),

        "CatBoost": CatBoostRegressor(**cat_params),

        "TabNet": TabNetRegressor(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3),
            mask_type="entmax",
            seed=42,
            verbose=0,
            device_name='cuda' if gpu_available else 'cpu'
        )
    }


def main():
    """Haupteinstiegspunkt für die Pipeline."""

    # Reproduzierbarkeit
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    block("START: DATA PREPARATION")
    preprocessor, X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Output Setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_results_path = Path(__file__).resolve().parents[1] / "results"
    results_path = base_results_path / timestamp

    for p in [results_path, results_path / "SHAP", results_path / "CSV"]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"\n{C.GREEN}📂 Output Ordner erstellt:{C.ENDC} {results_path}")

    # Training Loop
    models = get_models()
    all_results = []

    for name, model in models.items():
        try:
            res = train_and_evaluate_final(
                model, X_train, X_test, y_train, y_test,
                preprocessor, name, results_path,
                results_path / "SHAP", results_path / "CSV"
            )
            all_results.append(res)
        except Exception as e:
            print(f"\n{C.RED}‼ FEHLER bei {name}: {e}{C.ENDC}")
            import traceback
            traceback.print_exc()

    # Reporting
    run_global_shap_summary(results_path / "SHAP")

    df = pd.DataFrame(all_results)
    df.to_csv(results_path / "CSV" / "model_comparison_cv.csv", index=False)

    block("FINAL RESULTS COMPARISON")
    pd.set_option("display.max_rows", None)
    print(df[['model', 'test_rmse', 'test_mae', 'test_r2', 'runtime']])

    subblock("Speed Analysis")
    if not df.empty:
        fastest = df["runtime"].min()
        slowest = df["runtime"].max()
        for _, row in df.iterrows():
            rt = row["runtime"]
            icon = "⚡" if rt == fastest else ("🐢" if rt == slowest else "·")
            color = C.GREEN if rt == fastest else (C.RED if rt == slowest else C.YELLOW)
            print(f"  {icon} {row['model']:<20} {color}{rt:.2f} s{C.ENDC}")

    print(f"\n{C.BLUE}{'═' * 60}{C.ENDC}\n")


if __name__ == "__main__":
    main()
