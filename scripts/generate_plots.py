import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np  

# Basisverzeichnis setzen
base_dir = Path(__file__).resolve().parents[1]
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

preprocess_path = base_dir / "preprocessing"
if str(preprocess_path) not in sys.path:
    sys.path.append(str(preprocess_path))

from preprocessing.data_preparation import load_and_prepare_data


def generate_all_plots():
    # Daten laden
    preprocessor, X_train, X_test, y_train, y_test = load_and_prepare_data()
    train_df = X_train.copy()
    train_df["log_price"] = y_train

    # Preis aus log(Preis) rekonstruieren
    train_df["price"] = np.exp(train_df["log_price"])

    # Plot-Ordner
    plot_base = os.path.join(base_dir, "plots")
    plot_eda = os.path.join(plot_base, "eda")
    os.makedirs(plot_eda, exist_ok=True)

    # 1. Verteilung log(Preis)
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df["log_price"], bins=40, kde=True)
    plt.title("Verteilung der logarithmierten Airbnb-Preise")
    plt.xlabel("log(Preis)")
    plt.ylabel("Anzahl Angebote")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_eda, "price_distribution.png"), dpi=300)
    plt.close()

    # 2. Preis vs. Log-Preis (NEU, jetzt garantiert vorhanden)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(train_df["price"], bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Verteilung der Preise (Original)")
    axes[0].set_xlabel("Preis (€)")
    axes[0].set_ylabel("Häufigkeit")

    sns.histplot(train_df["log_price"], bins=50, kde=True, color="orange", ax=axes[1])
    axes[1].set_title("Verteilung der Preise (log-transformiert)")
    axes[1].set_xlabel("log(Preis)")
    axes[1].set_ylabel("Häufigkeit")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_eda, "price_vs_logprice.png"), dpi=300)
    plt.close()

    # 3. Scatter: Betten vs Preis
    if "beds" in train_df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x="beds", y="log_price", data=train_df, alpha=0.5)
        plt.title("Zusammenhang zwischen Anzahl Betten und Preis")
        plt.xlabel("Betten")
        plt.ylabel("log(Preis)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "beds_vs_price.png"), dpi=300)
        plt.close()

    # 4. Preis nach Stadt
    if "city" in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="city", y="log_price", data=train_df)
        plt.xticks(rotation=45)
        plt.title("Preisverteilung nach Stadt")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "price_by_city.png"), dpi=300)
        plt.close()

    # 5. Mindestnächte
    if "minimum_nights" in train_df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(train_df["minimum_nights"], bins=30)
        plt.title("Verteilung der Mindestaufenthaltsdauer")
        plt.xlabel("Mindestnächte")
        plt.ylabel("Anzahl Angebote")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "minimum_nights_distribution.png"), dpi=300)
        plt.close()

    # 6. Inserate pro Stadt
    if "city" in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x="city", data=train_df)
        plt.xticks(rotation=45)
        plt.title("Anzahl der Inserate pro Stadt")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "listings_per_city.png"), dpi=300)
        plt.close()

    # 7. Bewertung vs Preis
    if "review_scores_rating" in train_df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x="review_scores_rating", y="log_price", data=train_df, alpha=0.5)
        plt.title("Preis in Abhängigkeit der Bewertung")
        plt.xlabel("Bewertung (Score)")
        plt.ylabel("log(Preis)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "rating_vs_price.png"), dpi=300)
        plt.close()

    # 8. WLAN vs Preis
    if "has_wifi" in train_df.columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(x="has_wifi", y="log_price", data=train_df)
        plt.title("Einfluss von WLAN-Verfügbarkeit auf den Preis")
        plt.xlabel("WLAN vorhanden")
        plt.ylabel("log(Preis)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "wifi_vs_price.png"), dpi=300)
        plt.close()

    # 9. Preis nach Unterkunftstyp
    if "property_type" in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="property_type", y="log_price", data=train_df)
        plt.title("Durchschnittlicher Preis nach Unterkunftstyp")
        plt.xticks(rotation=45)
        plt.xlabel("Unterkunftstyp")
        plt.ylabel("Durchschnittlicher log(Preis)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "price_by_property_type.png"), dpi=300)
        plt.close()

    # 10. Korrelationsmatrix
    plt.figure(figsize=(10, 8))
    corr = train_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Korrelationsmatrix der numerischen Variablen")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_eda, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # 11. Geoplot
    if "latitude" in train_df.columns and "longitude" in train_df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="longitude", y="latitude",
            hue="log_price", data=train_df,
            palette="viridis", alpha=0.7, edgecolor=None
        )
        plt.title("Geografische Preisverteilung")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="log(Preis)", loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_eda, "geographical_price_distribution.png"), dpi=300)
        plt.close()

    # 12. Korrelation wichtiger Features ausgeben
    important_features = [
        "accommodates", "beds", "bathrooms", "review_scores_rating",
        "n_amenities", "host_response_rate", "days_since_last_review"
    ]

    def interpret_corr(value):
        abs_val = abs(value)
        if abs_val >= 0.7:
            return "starke"
        elif abs_val >= 0.4:
            return "mittlere"
        elif abs_val >= 0.2:
            return "schwache"
        else:
            return "keine"

    print("\nKorrelationseinschätzung wichtiger Features mit log_price:")
    for feature in important_features:
        if feature in corr.columns:
            corr_value = corr.at[feature, "log_price"]
            corr_type = interpret_corr(corr_value)
            print(f"- {feature}: Korrelationswert = {corr_value:.3f} ({corr_type} Korrelation)")
        else:
            print(f"- {feature}: Feature nicht im Datensatz vorhanden.")


if __name__ == "__main__":
    generate_all_plots()