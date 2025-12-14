# Airbnb Preisvorhersage
Autor: Burak Ugurelli

Dieses Projekt trainiert ein Modell zur Vorhersage von Airbnb-Preisen basierend auf dem Datensatz `train.csv`.

# Voraussetzungen
- Python 3.10+
- Jupyter Lab (optional, empfohlen)

```bash
jupyter lab LAUNCH.ipynb

### Installation

### 1) Abhängigkeiten installieren
```bash
pip install -r requirements.txt

python data/download_data.py

├── notebooks/
│   └── airbnb_analysis.ipynb
├── scripts/
│   └── (Hilfsfunktionen & Training-Skripte)
├── results/
├── ├── [2025-12-10_14-13-01]
├──     ├── CSV, feature importance, model_plots, SHAP
├── data/
│   ├── download_data.py
│   └── train.csv
├── requirements.txt
└── LAUNCH.ipynb