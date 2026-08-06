<img width="5400" height="1500" alt="comparison_overview" src="https://github.com/user-attachments/assets/cff7835e-7c4e-4d7f-9179-0b409949c7a5" />

# Short-Term Rental Price Prediction (Airbnb)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange" alt="Jupyter">
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-green" alt="Machine Learning">
</p>

A machine learning project for predicting Airbnb listing prices from tabular data.

The project covers data preprocessing, exploratory analysis, feature engineering, model training, evaluation, and SHAP-based model interpretation.

## Features

- Comparison of multiple regression models
- Reproducible preprocessing and training pipeline
- Evaluation using R², RMSE, and MAE
- Feature importance and SHAP analysis
- Timestamped experiment outputs

## Getting Started

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python data/download_data.py
```

Install JupyterLab if necessary:

```bash
pip install jupyterlab
```

Launch the project:

```bash
jupyter lab LAUNCH.ipynb
```

## Project Structure

```text
├── data/
│   ├── download_data.py
│   └── train.csv
├── notebooks/
│   └── airbnb_analysis.ipynb
├── scripts/
│   └── training scripts and helper functions
├── results/
│   └── timestamped experiment outputs
├── LAUNCH.ipynb
└── requirements.txt
```

Developed as part of an academic machine learning project.
