def run_global_shap_summary(shap_path):
    """
    Lädt alle gespeicherten SHAP Werte und erzeugt eine globale Summary-Grafik.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import shap
    import os

    shap_files = sorted([f for f in os.listdir(shap_path) if f.endswith("_shap_values.npy")])

    if len(shap_files) == 0:
        print("⚠️ Keine SHAP Werte gefunden. Globale Summary wird übersprungen.")
        return

    all_shap_abs_values = []
    feature_names = None

    # Werte Laden
    for f in shap_files:
        model = f.replace("_shap_values.npy", "")

        shap_values = np.load(shap_path / f)
        shap_abs = np.abs(shap_values)

        all_shap_abs_values.append(shap_abs)

        # Feature-Namen laden
        fn_file = shap_path / f"{model}_feature_names.npy"
        if fn_file.exists():
            feature_names = np.load(fn_file, allow_pickle=True)

    # Form: (total_samples, n_features)
    merged = np.vstack(all_shap_abs_values)

    # globaler Mittelwert: (n_features,)
    global_shap_mean = merged.mean(axis=0)

    # Für SHAP Barplot wird eine Matrix benötigt
    global_shap_matrix = global_shap_mean.reshape(1, -1)

    # Plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        global_shap_matrix,
        features=None,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("Globale SHAP Summary über alle Modelle")
    plt.tight_layout()
    plt.savefig(shap_path / "SHAP_GLOBAL_SUMMARY.png", dpi=300)
    plt.close()

    print(f"🌍 Globale SHAP Summary gespeichert unter: {shap_path / 'SHAP_GLOBAL_SUMMARY.png'}")
