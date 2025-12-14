"""
visualize_results.py

Dieses Modul dient der Visualisierung der Trainingsergebnisse.
Es analysiert die `model_comparison_cv.csv` eines Runs und erstellt:
1. Vergleichsdiagramme (Barplots) für die Metriken R², RMSE und MAE.
2. Einen zusammenfassenden PDF-Report.
3. Eine detaillierte Fehleranalyse (Scatterplot & Residual-Histogramm) für das beste Modell.

Die generierten Grafiken werden im Unterordner `model_plots/` des jeweiligen Runs gespeichert.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


def get_latest_run_directory(base_results_path: Path) -> Path:
    """
    Ermittelt das zeitlich aktuellste Unterverzeichnis im angegebenen Basispfad.

    Args:
        base_results_path (Path): Der Pfad zum 'results'-Ordner.

    Returns:
        Path: Pfad zum neuesten Run-Verzeichnis.
    """
    if not base_results_path.exists():
        print(f"❌ Basis-Ordner nicht gefunden: {base_results_path}")
        sys.exit(1)

    subdirs = [d for d in base_results_path.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"❌ Keine Run-Ordner in {base_results_path} gefunden.")
        sys.exit(1)

    return sorted(subdirs)[-1]


def generate_model_plots(run_folder: Path, best_model: str = None) -> None:
    """
    Erstellt Vergleichsplots der Modellmetriken und Fehleranalysen.

    Liest die CSV-Ergebnisse ein, generiert Balkendiagramme für R², RMSE und MAE
    und erstellt Detailplots (Residuals) für das leistungsstärkste Modell.

    Args:
        run_folder (Path): Pfad zum Run-Verzeichnis, das analysiert werden soll.
        best_model (str, optional): Name des Modells für die Detailanalyse.
                                    Falls None, wird das Modell mit dem höchsten R² gewählt.
    """
    print(f"📂 Erstelle Plots für Run: {run_folder.name}")

    csv_path = run_folder / "CSV" / "model_comparison_cv.csv"

    if not csv_path.exists():
        print(f"❌ Datei nicht gefunden: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Mapping für Achsenbeschriftungen
    metric_map = {
        "test_r2": "R²",
        "test_rmse": "RMSE",
        "test_mae": "MAE"
    }

    available_metrics = [m for m in metric_map.keys() if m in df.columns]

    if not available_metrics:
        print(f"❌ Keine bekannten Metriken in CSV gefunden. Spalten: {df.columns}")
        return

    plots_path = run_folder / "model_plots"
    plots_path.mkdir(exist_ok=True)

    # Automatisches Ermitteln des besten Modells
    if best_model is None:
        if "test_r2" in df.columns:
            best_model = df.loc[df["test_r2"].idxmax(), "model"]
            print(f"   🏆 Bestes Modell automatisch erkannt: {best_model}")
        else:
            best_model = df.iloc[0]["model"]

    sns.set_style("whitegrid")

    # Generierung der Einzelplots
    for metric_col in available_metrics:
        pretty_name = metric_map[metric_col]

        plt.figure(figsize=(10, 5))
        # R2: größer ist besser (descending), Fehler: kleiner ist besser (ascending)
        ascending = False if "r2" in metric_col else True
        df_sorted = df.sort_values(metric_col, ascending=ascending)

        sns.barplot(
            x=metric_col, y="model", hue="model",
            data=df_sorted, orient="h", palette="viridis", legend=False
        )

        plt.title(f"Modellvergleich: {pretty_name}")
        plt.xlabel(pretty_name)
        plt.ylabel("Modell")
        plt.tight_layout()

        save_path = plots_path / f"comparison_{pretty_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   📊 {save_path.name} erstellt")

    # Generierung der Übersichtsgrafik
    if len(available_metrics) >= 2:
        fig, axes = plt.subplots(
            1, len(available_metrics),
            figsize=(6 * len(available_metrics), 5)
        )

        for i, metric_col in enumerate(available_metrics):
            pretty_name = metric_map[metric_col]
            ascending = False if "r2" in metric_col else True
            df_sorted = df.sort_values(metric_col, ascending=ascending)

            ax = axes[i] if len(available_metrics) > 1 else axes

            sns.barplot(
                x=metric_col, y="model", hue="model",
                data=df_sorted, orient="h", ax=ax, palette="viridis", legend=False
            )
            ax.set_title(pretty_name)
            ax.set_xlabel("")
            ax.set_ylabel("")

        plt.tight_layout()
        overview_path = plots_path / "comparison_overview.png"
        plt.savefig(overview_path, dpi=300)
        plt.savefig(plots_path / "comparison_overview.pdf", format="pdf")
        plt.close()
        print(f"   📊 Übersichtsgrafik erstellt: {overview_path.name}")

        # PDF Export via PdfPages
        pdf_path = plots_path / "comparison_overview.pdf"
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(6 * len(available_metrics), 5))
            img = plt.imread(overview_path)
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()

        print(f"   📄 PDF erstellt: {pdf_path.name}")

    # Residuals für das beste Modell
    error_file = run_folder / "CSV" / f"{best_model}_errors.csv"

    if error_file.exists():
        err = pd.read_csv(error_file)

        # Plot 1: Prediction vs True Value Scatter
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x="true", y="pred", data=err, alpha=0.3, color="steelblue")

        min_val = min(err["true"].min(), err["pred"].min())
        max_val = max(err["true"].max(), err["pred"].max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

        plt.title(f"Wahrheit vs. Vorhersage: {best_model}")
        plt.xlabel("Tatsächlicher Log-Preis")
        plt.ylabel("Vorhergesagter Log-Preis")
        plt.tight_layout()
        plt.savefig(plots_path / f"{best_model}_scatter.png", dpi=300)
        plt.close()

        # Plot 2: Residual Histogramm
        plt.figure(figsize=(8, 5))
        sns.histplot(err["error"], bins=50, kde=True, color="purple")
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f"Fehlerverteilung (Residuals): {best_model}")
        plt.xlabel("Fehler (Wahr - Vorhersage)")
        plt.ylabel("Anzahl")
        plt.tight_layout()
        plt.savefig(plots_path / f"{best_model}_residuals.png", dpi=300)
        plt.close()

        print(f"   ✅ Detail-Plots für {best_model} erstellt.")
    else:
        print(f"   ⚠️ Keine Fehlerdatei für {best_model} gefunden.")

    print(f"\n[DONE] Alle Plots gespeichert in:\n{plots_path}")


def main():
    """
    Haupteinstiegspunkt. Bestimmt den neuesten Run und startet die Visualisierung.
    """
    # Pfad relativ zum Skript ermitteln
    base_path = Path(__file__).resolve().parents[1] / "results/"
    latest_run = get_latest_run_directory(base_path)

    generate_model_plots(latest_run)


if __name__ == "__main__":
    main()