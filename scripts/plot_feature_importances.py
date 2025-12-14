"""
visualize_importances.py

Dieses Modul visualisiert die Feature-Importance-Werte trainierter Modelle.
Es durchsucht das angegebene (oder aktuellste) Ergebnisverzeichnis nach CSV-Dateien
mit dem Suffix `_feature_importances.csv` und generiert für jedes Modell
ein horizontales Balkendiagramm der 15 wichtigsten Features.

Die erzeugten Grafiken werden im Unterordner `feature_importance_plots/`
des jeweiligen Run-Verzeichnisses gespeichert.

Verwendung:
    python scripts/plot_feature_importances.py
    python scripts/plot_feature_importances.py --run results/2023-10-27_14-00
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def get_latest_run_directory(base_results_path: Path) -> Path:
    """
    Ermittelt das zeitlich aktuellste Verzeichnis im Results-Ordner.

    Args:
        base_results_path (Path): Pfad zum Basis-Ergebnisverzeichnis.

    Returns:
        Path: Pfad zum neuesten Run-Verzeichnis.
    """
    if not base_results_path.exists():
        print(f"CRITICAL: Basis-Ordner nicht gefunden: {base_results_path}")
        sys.exit(1)

    subdirs = [d for d in base_results_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"CRITICAL: Keine Run-Ordner in {base_results_path} gefunden.")
        sys.exit(1)

    return sorted(subdirs)[-1]


def generate_importance_plots(run_folder: Path) -> None:
    """
    Sucht Feature-Importance-CSVs, erstellt Visualisierungen und speichert diese.

    Args:
        run_folder (Path): Pfad zum Run-Verzeichnis, das analysiert werden soll.
    """
    print(f"Analysiere Feature Importances für: {run_folder.name}")

    csv_dir = run_folder / "CSV"
    if not csv_dir.exists():
        print(f"ERROR: CSV-Ordner nicht gefunden in {run_folder}")
        return

    csv_files = list(csv_dir.glob("*_feature_importances.csv"))

    if not csv_files:
        print("   Warning: Keine Feature-Importance-Dateien gefunden.")
        return

    # Ausgabeordner erstellen
    plots_dir = run_folder / "feature_importance_plots"
    plots_dir.mkdir(exist_ok=True)

    # Globales Styling
    sns.set_style("whitegrid")

    for file_path in csv_files:
        try:
            model_name = file_path.name.replace("_feature_importances.csv", "")
            df = pd.read_csv(file_path)

            if "feature" not in df.columns or "importance" not in df.columns:
                print(f"   Skipping {file_path.name}: Erwartete Spalten 'feature' und 'importance' fehlen.")
                continue

            # Top 15 Features filtern
            df_sorted = df.sort_values(by="importance", ascending=False).head(15)

            plt.figure(figsize=(10, 6))

            # Erstellung des Barplots
            sns.barplot(
                x="importance",
                y="feature",
                hue="feature",
                data=df_sorted,
                palette="viridis",
                legend=False
            )

            plt.title(f"Top 15 Feature Importance: {model_name}")
            plt.xlabel("Relative Wichtigkeit")
            plt.ylabel("")
            plt.tight_layout()

            save_path = plots_dir / f"{model_name}_importance.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"   Plot erstellt: {save_path.name}")

        except Exception as e:
            print(f"   ERROR bei {file_path.name}: {e}")

    print(f"Fertig. Plots gespeichert in: {plots_dir}")


def main():
    """
    Haupteinstiegspunkt. Verarbeitet Kommandozeilenargumente und startet die Plot-Generierung.
    """
    parser = argparse.ArgumentParser(description="Erstellt Feature-Importance-Plots für ML-Runs.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Pfad zum Run-Ordner (Standard: neuester Run)"
    )

    # Fallback für Jupyter Notebook
    if "ipykernel" in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # Robuste Pfad-Ermittlung (funktioniert in Skripten und Notebooks)
    try:
        current_path = Path(__file__).resolve()
        project_root = current_path.parents[1]
    except NameError:
        current_path = Path.cwd()
        if current_path.name == "scripts":
            project_root = current_path.parent
        else:
            project_root = current_path

    base_results_path = project_root / "results"

    # Zielordner bestimmen
    if args.run:
        target_run = Path(args.run).resolve()
        if not target_run.exists():
            print(f"CRITICAL: Ordner '{target_run}' existiert nicht.")
            sys.exit(1)
    else:
        target_run = get_latest_run_directory(base_results_path)

    generate_importance_plots(target_run)


if __name__ == "__main__":
    main()