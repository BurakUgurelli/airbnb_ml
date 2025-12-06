"""
MODUL: DATA DOWNLOAD (KAGGLE)

Beschreibung:
    Lädt den Datensatz von Kaggle herunter und verschiebt ihn in den data-Ordner.
    Dieses Modul ist so strukturiert, dass es importiert werden kann.
"""

import os
import shutil
import sys
import time
from pathlib import Path
import kagglehub

# Windows support
try:
    import colorama

    colorama.init(autoreset=True)
except ImportError:
    pass


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


def print_status(icon, key, value, color=C.ENDC):
    print(f"  {icon} {C.GREY}{key:<15}{C.ENDC} : {color}{value}{C.ENDC}")


def block(title):
    print("\n" + C.BLUE + "╔" + "═" * 60 + "╗" + C.ENDC)
    print(f"{C.BLUE}║{C.ENDC} {C.BOLD}{title.center(58)}{C.ENDC} {C.BLUE}║{C.ENDC}")
    print(C.BLUE + "╚" + "═" * 60 + "╝" + C.ENDC)


def subblock(title):
    print(f"\n{C.YELLOW}➤ {C.BOLD}{title}{C.ENDC}")
    print(f"{C.GREY}" + "─" * 40 + f"{C.ENDC}")


# Main
def main():
    block("DATA INGESTION PIPELINE")
    start_time = time.time()

    # 1. Pfade setzen
    subblock("Verzeichnis Setup")

    # Pfad finden
    current_dir = Path(__file__).resolve().parent
    target_dir = current_dir

    # Falls das Skript woanders liegt, hier anpassen:
    if target_dir.name != "data":
        target_dir = current_dir.parent / "data"

    target_dir.mkdir(parents=True, exist_ok=True)
    print_status("📂", "Zielordner", str(target_dir), C.CYAN)

    # 2. Download
    subblock("Kaggle Download")
    dataset_name = "stevezhenghp/airbnb-price-prediction"
    file_name = "train.csv"

    try:
        print(f"  {C.GREY}Starte Download via kagglehub...{C.ENDC}")
        dataset_path = kagglehub.dataset_download(dataset_name)
        print_status("⬇️ ", "Download Pfad", dataset_path)
    except Exception as e:
        print(f"{C.RED}❌ Download fehlgeschlagen: {e}{C.ENDC}")
        return

    # 3. Kopieren
    subblock("Datei Transfer")
    source_file = Path(dataset_path) / file_name
    target_file = target_dir / file_name

    if not source_file.exists():
        print(f"{C.RED}❌ Datei '{file_name}' nicht im Download gefunden.{C.ENDC}")
        return

    try:
        shutil.copy2(source_file, target_file)
        print_status("✅", "Status", "Kopieren erfolgreich", C.GREEN)
        print_status("📄", "Datei", str(target_file))
    except Exception as e:
        print(f"{C.RED}❌ Fehler beim Kopieren: {e}{C.ENDC}")

    runtime = time.time() - start_time
    print("\n" + C.BLUE + "═" * 60 + C.ENDC)
    print(f"{C.GREY}⏱  Fertig in {runtime:.2f}s{C.ENDC}\n")


if __name__ == "__main__":
    main()