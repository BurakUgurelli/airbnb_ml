# Airbnb Preisvorhersage
Autor: Burak Ugurelli

Dieses Projekt trainiert ein Modell zur Vorhersage von Airbnb-Preisen basierend auf dem Datensatz `train.csv`.

# Voraussetzungen
- Python 3.10+
- Jupyter Lab (optional, empfohlen)

### Installation & Start

Schritt 1: Abhängigkeiten installieren
```bash
pip install -r requirements.txt
python data/download_data.py
pip install jupyterlab
jupyter lab LAUNCH.ipynb
```

```text
├── notebooks/
│   └── airbnb_analysis.ipynb
├── scripts/
│   └── (Hilfsfunktionen & Training-Skripte)
├── results/
│   └── [TIMESTAMP]/          
│       ├── model_plots/
│       ├── feature_importance_plots/
│       ├── SHAP/
│       └── CSV/
├── data/
│   ├── download_data.py
│   └── train.csv
├── requirements.txt
└── LAUNCH.ipynb