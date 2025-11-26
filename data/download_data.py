import kagglehub
import os
import shutil

# Absoluter Pfad
# Ordner der aktuellen Datei (z. B. airbnb_ml/data/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Projektordner ist 1 Ebene darüber (airbnb_ml/)
project_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Zielordner "data" im Projekt
target_dir = os.path.join(project_dir, "data")
os.makedirs(target_dir, exist_ok=True)

print("Zielordner:", target_dir)

# Kaggle Dataset Name
dataset = "stevezhenghp/airbnb-price-prediction"

print("Downloading dataset...")
dataset_path = kagglehub.dataset_download(dataset)
print("Downloaded to:", dataset_path)

# Dateiname im Kaggle-Dataset
file_name = "train.csv"

source_file = os.path.join(dataset_path, file_name)
target_file = os.path.join(target_dir, file_name)

# Prüfen ob die Datei existiert
if not os.path.exists(source_file):
    raise FileNotFoundError(f"'{file_name}' wurde im Kaggle-Dataset nicht gefunden!")

# Datei kopieren
shutil.copy2(source_file, target_file)

print(f"'{file_name}' wurde erfolgreich nach '{target_dir}' kopiert.")