"""
MODUL: DATA PIPELINE (KAGGLE)

Beschreibung:
    Dieses Skript lädt den Airbnb-Datensatz automatisch von Kaggle herunter
    und platziert ihn in der korrekten Ordnerstruktur für das Projekt.

    Ablauf:
    1. Erstellt den Ordner '../data', falls nicht vorhanden.
    2. Nutzt die Kaggle API (via kagglehub) für den Download.
    3. Extrahiert und verschiebt die 'train.csv' an den Zielort.

Autor: Burak Ugurelli
Projekt: Airbnb ML Price Prediction (RWU)
"""

import os
import shutil
import sys
import time
from pathlib import Path

# 3rd Party
import kagglehub

# Windows Support für ANSI-Farben
try:
    import colorama

    colorama.init(autoreset=True)
except ImportError:
    pass


# OUTPUT CONFIG
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
    print("\n" + C.BLUE + "╔" + "═" * 60 + "╗" + C.ENDC)
    print(f"{C.BLUE}║{C.ENDC} {C.BOLD}{title.center(58)}{C.ENDC} {C.BLUE}║{C.ENDC}")
    print(C.BLUE + "╚" + "═" * 60 + "╝" + C.ENDC)


def subblock(title: str) -> None:
    print(f"\n{C.YELLOW}➤ {C.BOLD}{title}{C.ENDC}")
    print(f"{C.GREY}" + "─" * 40 + f"{C.ENDC}")


def print_status(icon: str, key: str, value: str, color: str = C.ENDC) -> None:
    print(f"  {icon} {C.GREY}{key:<15}{C.ENDC} : {color}{value}{C.ENDC}")


def main():
    # Header
    block("DATA PIPELINE")
    start_time = time.time()

    # 1. PFAD SETUP
    subblock("Verzeichnis Setup")

    try:
        # Pfade definieren
        current_dir = Path(__file__).resolve().parent
        project_dir = current_dir.parent
        target_dir = project_dir / "data"

        # Ordner erstellen
        target_dir.mkdir(parents=True, exist_ok=True)

        print_status("📂", "Projekt Root", str(project_dir))
        print_status("📂", "Zielordner", str(target_dir), C.CYAN)

    except Exception as e:
        print(f"\n{C.RED}❌ Fehler beim Erstellen der Pfade: {e}{C.ENDC}")
        sys.exit(1)

    # 2. DOWNLOAD
    subblock("Kaggle Download")
    dataset_name = "stevezhenghp/airbnb-price-prediction"
    file_name = "train.csv"

    print_status("⬇️ ", "Dataset", dataset_name)
    print(f"  {C.GREY}Verbindung zu Kaggle Hub wird aufgebaut...{C.ENDC}\n")

    try:
        # Der eigentliche Download
        dataset_path = kagglehub.dataset_download(dataset_name)

        print(f"\n  {C.GREEN}✔ Download abgeschlossen.{C.ENDC}")
        print_status("💾", "Temp. Cache", dataset_path, C.GREY)

    except Exception as e:
        print(f"\n{C.RED}❌ Download fehlgeschlagen!{C.ENDC}")
        print(f"   Grund: {e}")
        print(f"   {C.YELLOW}Hinweis: Prüfe Internetverbindung & Kaggle Credentials.{C.ENDC}")
        sys.exit(1)

    # 3. DATEI TRANSFER
    subblock("Datei Verarbeitung")

    source_file = Path(dataset_path) / file_name
    target_file = target_dir / file_name

    if not source_file.exists():
        print(f"{C.RED}❌ Fehler: '{file_name}' nicht im heruntergeladenen Paket gefunden!{C.ENDC}")
        print(f"   Inhalt des Ordners: {[f.name for f in Path(dataset_path).iterdir()]}")
        sys.exit(1)

    try:
        shutil.copy2(source_file, target_file)
        print_status("📄", "Datei", file_name)
        print_status("✅", "Status", "Erfolgreich kopiert", C.BOLD + C.GREEN)
        print_status("📍", "Gespeichert in", str(target_dir), C.CYAN)

    except Exception as e:
        print(f"{C.RED}❌ Fehler beim Kopieren der Datei: {e}{C.ENDC}")
        sys.exit(1)

    # Abschluss
    runtime = time.time() - start_time
    print(f"\n{C.BLUE}" + C.ENDC)
    print(f"{C.GREY}⏱  Fertig in {runtime:.2f} Sekunden.{C.ENDC}\n")


if __name__ == "__main__":
    main()