"""
predict.py
----------
Inference utilities for EcoCharge:
  - Load trained LSTM model
  - Predict State-of-Health (SoH) from a sequence of sensor readings
  - Estimate Remaining Useful Life (RUL)
  - Classify battery reuse/repurpose/recycle recommendation
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import keras
from typing import Dict, List, Tuple, Union

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH_KERAS = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.keras')
MODEL_PATH_H5    = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.h5')
SCALER_PATH      = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

# Select the existing model path
if os.path.exists(MODEL_PATH_KERAS):
    MODEL_PATH = MODEL_PATH_KERAS
else:
    MODEL_PATH = MODEL_PATH_H5

# ── Constants ─────────────────────────────────────────────────────────────────
NOMINAL_CAPACITY   = 1.9       # Ah  (fresh cell)
EOL_THRESHOLD      = 0.80      # 80 % SoH → End-of-Life
REPURPOSE_THRESHOLD= 0.60      # 60 % SoH → Repurpose (2nd-life energy storage)
FEATURE_COLS       = [
    'cycle', 'voltage_measured', 'current_measured',
    'temperature_measured', 'current_charge', 'voltage_charge', 'time',
]

# Average CO₂ saved by reusing one battery instead of manufacturing a new one
CO2_PER_BATTERY_KG = 74.0     # kg CO₂ eq. (literature estimate)


# ── 1. Load Artefacts ─────────────────────────────────────────────────────────
_model  = None
_scaler = None


def _load_artefacts():
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _model = keras.models.load_model(MODEL_PATH)
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. Run preprocess.py first."
            )
        _scaler = joblib.load(SCALER_PATH)


# ── 2. Core Prediction ────────────────────────────────────────────────────────
def predict_capacity(X: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    X : np.ndarray  shape (n_samples, seq_length, n_features)

    Returns
    -------
    predictions : np.ndarray  shape (n_samples,)  — capacity in Ah
    """
    _load_artefacts()
    preds = _model.predict(X, verbose=0).flatten()
    return preds


def capacity_to_soh(capacity: float, nominal: float = NOMINAL_CAPACITY) -> float:
    """Convert raw capacity (Ah) → State-of-Health (%)."""
    return min(100.0, max(0.0, (capacity / nominal) * 100.0))


# ── 3. Remaining Useful Life Estimation ───────────────────────────────────────
def estimate_rul(
    soh_series: List[float],
    cycle_series: List[int],
    eol_soh: float = EOL_THRESHOLD * 100,
) -> int:
    """
    Simple linear extrapolation of SoH curve to estimate RUL.

    Parameters
    ----------
    soh_series   : list of recent SoH values (%)
    cycle_series : corresponding cycle numbers
    eol_soh      : SoH at End-of-Life (default 80 %)

    Returns
    -------
    rul : estimated remaining cycles until EOL (0 if already past EOL)
    """
    if len(soh_series) < 2:
        return 0

    # Fit a linear trend to the SoH series
    coeffs = np.polyfit(cycle_series, soh_series, deg=1)   # slope, intercept
    slope, intercept = coeffs

    if slope >= 0:
        # SoH not declining — return a large number
        return 9999

    # Cycle at which SoH hits EOL threshold
    eol_cycle = (eol_soh - intercept) / slope
    current_cycle = cycle_series[-1]
    rul = max(0, int(eol_cycle - current_cycle))
    return rul


# ── 4. Lifecycle Recommendation ───────────────────────────────────────────────
def get_recommendation(soh: float) -> Dict[str, str]:
    """
    Returns a lifecycle recommendation dict based on SoH.

    SoH > 80%  → Continue using in EV
    60–80%     → Second-life use (stationary storage, e-bikes, etc.)
    < 60%      → Recycle / material recovery
    """
    if soh >= EOL_THRESHOLD * 100:
        return {
            'action':      '✅  Continue Use',
            'label':       'Healthy — Keep in EV',
            'color':       'green',
            'description': (
                'Battery is in good health. Continue standard EV operation. '
                'Schedule next inspection in 50 cycles.'
            ),
        }
    elif soh >= REPURPOSE_THRESHOLD * 100:
        return {
            'action':      '⚠️  Repurpose',
            'label':       'Second-Life — Energy Storage',
            'color':       'orange',
            'description': (
                'Battery is past EV-grade SoH but still viable for stationary '
                'energy storage (solar buffers, e-bikes, UPS systems).'
            ),
        }
    else:
        return {
            'action':      '♻️  Recycle',
            'label':       'End-of-Life — Material Recovery',
            'color':       'red',
            'description': (
                'Battery has reached end-of-life. Send to certified recycler '
                'for lithium, cobalt, and nickel recovery.'
            ),
        }


# ── 5. CO₂ Impact Calculator ──────────────────────────────────────────────────
def compute_co2_savings(n_batteries_reused: int) -> Dict[str, float]:
    """Estimate CO₂ saved by reusing batteries instead of buying new ones."""
    saved_kg  = n_batteries_reused * CO2_PER_BATTERY_KG
    trees_eq  = saved_kg / 21.0          # avg tree absorbs ~21 kg CO₂/year
    car_km_eq = saved_kg / 0.21          # avg car emits ~210 g CO₂/km
    return {
        'co2_saved_kg':   round(saved_kg,  1),
        'trees_equivalent': round(trees_eq, 1),
        'car_km_avoided':   round(car_km_eq, 0),
    }


# ── 6. Full Inference Pipeline (for Streamlit) ────────────────────────────────
def run_inference_from_df(df: pd.DataFrame, seq_length: int = 30) -> Dict:
    """
    Given a DataFrame with FEATURE_COLS columns (one row per cycle),
    run the full inference pipeline and return a results dict.

    Designed for use in the Streamlit app's CSV-upload feature.
    """
    _load_artefacts()

    # Scale
    feat = df[FEATURE_COLS].copy()
    feat_scaled = _scaler.transform(feat)

    # Build sequences
    X_list = []
    for i in range(len(feat_scaled) - seq_length):
        X_list.append(feat_scaled[i: i + seq_length])
    X = np.array(X_list, dtype=np.float32)

    if len(X) == 0:
        return {'error': f'Need at least {seq_length + 1} rows of data.'}

    # Predict
    cap_preds = predict_capacity(X)
    soh_preds = [capacity_to_soh(c) for c in cap_preds]

    # Use last predicted SoH
    latest_soh = soh_preds[-1]
    cycles     = df['cycle'].values[seq_length:].tolist()

    rul = estimate_rul(soh_preds[-50:], cycles[-50:])
    rec = get_recommendation(latest_soh)

    return {
        'soh_series':  soh_preds,
        'cycle_series': cycles,
        'latest_soh':  round(latest_soh, 2),
        'rul':         rul,
        'recommendation': rec,
    }


# ── 7. Single-point inference (for manual input in Streamlit) ─────────────────
def predict_from_inputs(
    cycle: int,
    voltage: float,
    current: float,
    temperature: float,
    current_charge: float,
    voltage_charge: float,
    time: float,
    seq_length: int = 30,
) -> Dict:
    """
    Builds a synthetic sequence from a single set of sensor readings
    (repeats the reading seq_length times) and returns a prediction.
    Used for the manual-input demo in the Streamlit sidebar.
    """
    _load_artefacts()

    single = np.array([[
        cycle, voltage, current, temperature,
        current_charge, voltage_charge, time,
    ]])
    single_scaled = _scaler.transform(single)

    # Repeat to form a sequence
    seq = np.tile(single_scaled, (seq_length, 1))[np.newaxis]  # (1, seq, feat)

    cap_pred  = float(predict_capacity(seq)[0])
    soh       = capacity_to_soh(cap_pred)
    rul       = max(0, int((soh - EOL_THRESHOLD * 100) / 0.08))
    rec       = get_recommendation(soh)

    return {
        'predicted_capacity': round(cap_pred, 4),
        'soh':  round(soh, 2),
        'rul':  rul,
        'recommendation': rec,
    }


if __name__ == '__main__':
    # Quick smoke-test with random input
    result = predict_from_inputs(
        cycle=200, voltage=3.7, current=-1.8, temperature=27.0,
        current_charge=1.5, voltage_charge=4.2, time=3400,
    )
    print("\n[predict] Smoke-test result:")
    for k, v in result.items():
        print(f"  {k}: {v}")