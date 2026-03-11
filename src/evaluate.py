"""
evaluate.py
-----------
Loads the trained LSTM model and test set,
computes regression metrics, and saves visualisation plots.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from preprocess import (
    FEATURE_COLS, TARGET_COL, SEQ_LENGTH,
    run_preprocessing_pipeline,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH_KERAS = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.keras')
MODEL_PATH_H5    = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.h5')
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), '..', 'results')

# Select the existing model path
if os.path.exists(MODEL_PATH_KERAS):
    MODEL_PATH = MODEL_PATH_KERAS
else:
    MODEL_PATH = MODEL_PATH_H5
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')


# ── 1. Compute Metrics ────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    metrics = {
        'MAE':  round(mae,  6),
        'MSE':  round(mse,  8),
        'RMSE': round(rmse, 6),
        'R2':   round(r2,   4),
        'MAPE': round(mape, 4),
    }

    print("\n[evaluate] ── Test-Set Metrics ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v}")
    print("─────────────────────────────────────────────────────")
    return metrics


# ── 2. Plots ──────────────────────────────────────────────────────────────────
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 2a. Actual vs Predicted (first 300 samples) ──
    fig, ax = plt.subplots(figsize=(14, 5))
    n = min(300, len(y_true))
    ax.plot(y_true[:n], label='Actual Capacity',    color='steelblue',  linewidth=1.5)
    ax.plot(y_pred[:n], label='Predicted Capacity', color='darkorange',
            linestyle='--', linewidth=1.5)
    ax.set_title('EcoCharge — Actual vs Predicted Battery Capacity', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Capacity (Ah)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'), dpi=150)
    plt.close()
    print("[evaluate] Saved → actual_vs_predicted.png")

    # ── 2b. Scatter Plot ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color='royalblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
    ax.set_xlabel('Actual Capacity (Ah)')
    ax.set_ylabel('Predicted Capacity (Ah)')
    ax.set_title('Predicted vs Actual — Scatter', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'scatter_plot.png'), dpi=150)
    plt.close()
    print("[evaluate] Saved → scatter_plot.png")

    # ── 2c. Residuals Distribution ──
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(residuals[:300], color='purple', linewidth=0.8)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals Over Samples')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Residual (Ah)')
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=60, color='purple', edgecolor='white', alpha=0.8)
    axes[1].set_title('Residuals Distribution')
    axes[1].set_xlabel('Residual (Ah)')
    axes[1].set_ylabel('Count')
    axes[1].grid(alpha=0.3)

    plt.suptitle('EcoCharge — Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residuals.png'), dpi=150)
    plt.close()
    print("[evaluate] Saved → residuals.png")


def plot_soh_degradation_curve(y_true: np.ndarray, y_pred: np.ndarray,
                                nominal_capacity: float = 1.9):
    """Plot SoH (%) curves for actual vs predicted."""
    soh_true = (y_true / nominal_capacity) * 100
    soh_pred = (y_pred / nominal_capacity) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    n = min(500, len(soh_true))
    ax.plot(soh_true[:n], label='Actual SoH (%)',    color='mediumseagreen', linewidth=1.5)
    ax.plot(soh_pred[:n], label='Predicted SoH (%)', color='tomato',
            linestyle='--', linewidth=1.5)
    ax.axhline(80, color='gray', linestyle=':', linewidth=1.2, label='EOL Threshold (80%)')
    ax.fill_between(range(n), 0, 80, alpha=0.05, color='red')
    ax.set_title('EcoCharge — State-of-Health Degradation Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('State of Health (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'soh_curve.png'), dpi=150)
    plt.close()
    print("[evaluate] Saved → soh_curve.png")


# ── 3. Entry Point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  EcoCharge — Model Evaluation")
    print("=" * 60)

    # Rebuild test set
    _, _, X_test, _, _, y_test = run_preprocessing_pipeline()

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Metrics
    metrics = compute_metrics(y_test, y_pred)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {METRICS_PATH}")

    # Plots
    plot_predictions(y_test, y_pred)
    plot_soh_degradation_curve(y_test, y_pred)

    print("\n[evaluate] Evaluation complete. Check /results/ folder.")