"""
preprocess.py
-------------
Handles all data cleaning, feature engineering, scaling, and
sequence creation for the EcoCharge LSTM pipeline.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',       'battery_data.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'battery_processed.csv')
SCALER_PATH         = os.path.join(os.path.dirname(__file__), '..', 'models',            'scaler.pkl')

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'cycle',
    'voltage_measured',
    'current_measured',
    'temperature_measured',
    'current_charge',
    'voltage_charge',
    'time',
]
TARGET_COL   = 'capacity'
SEQ_LENGTH   = 30          # cycles to look back
EOL_CAPACITY = 0.8 * 1.9  # 80 % of nominal capacity → End-of-Life threshold


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded  {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


# ── 2. Clean ──────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows where capacity is physically impossible
    df = df[df['capacity'] > 0]
    df = df[df['voltage_measured'] > 0]

    # Clip temperature to realistic range (−20 °C to 80 °C)
    df['temperature_measured'] = df['temperature_measured'].clip(-20, 80)

    # Forward-fill any NaNs within each battery group
    df = df.sort_values(['battery_id', 'cycle'])
    df = df.groupby('battery_id', group_keys=False).apply(lambda g: g.ffill().bfill())

    after = len(df)
    print(f"[preprocess] Cleaned {before - after:,} bad rows.  Remaining: {after:,}")
    return df.reset_index(drop=True)


# ── 3. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # State-of-Health (SoH) as a percentage of initial capacity per battery
    df['initial_capacity'] = df.groupby('battery_id')['capacity'].transform('first')
    df['soh']              = (df['capacity'] / df['initial_capacity']) * 100.0

    # Capacity fade rate (rolling 10-cycle mean of Δcapacity)
    df['capacity_fade']    = df.groupby('battery_id')['capacity'].transform(
        lambda x: x.diff().rolling(10, min_periods=1).mean()
    )

    # Internal-resistance proxy: ΔV / ΔI  (absolute value)
    df['resistance_proxy'] = (
        (df['voltage_charge'] - df['voltage_measured']).abs() /
        (df['current_charge']  - df['current_measured'].abs()).replace(0, np.nan)
    ).fillna(0)

    print(f"[preprocess] Features engineered. New cols: soh, capacity_fade, resistance_proxy")
    return df


# ── 4. Scale ──────────────────────────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    fit: bool = True
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocess] Scaler fitted and saved → {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[feature_cols] = scaler.transform(df[feature_cols])
        print("[preprocess] Loaded existing scaler.")

    return df, scaler


# ── 5. Sequence Creation (for LSTM) ──────────────────────────────────────────
def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    target_col:   str        = TARGET_COL,
    seq_length:   int        = SEQ_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of `seq_length` cycles over each battery's data.
    X : (samples, seq_length, n_features)
    y : (samples,)  — capacity at cycle t+1
    """
    X_list, y_list = [], []

    for _, group in df.groupby('battery_id'):
        group = group.sort_values('cycle').reset_index(drop=True)
        feat_arr = group[feature_cols].values
        targ_arr = group[target_col].values

        for i in range(len(group) - seq_length):
            X_list.append(feat_arr[i: i + seq_length])
            y_list.append(targ_arr[i + seq_length])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"[preprocess] Sequences created — X: {X.shape}, y: {y.shape}")
    return X, y


# ── 6. Train / Validation / Test Split ───────────────────────────────────────
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    seed:      int   = 42,
) -> Tuple[np.ndarray, ...]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=seed, shuffle=True
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=seed
    )
    print(f"[preprocess] Split — Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 7. Master Pipeline ────────────────────────────────────────────────────────
def run_preprocessing_pipeline() -> Tuple[np.ndarray, ...]:
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = scale_features(df, fit=True)

    # Save processed CSV
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[preprocess] Processed data saved → {PROCESSED_DATA_PATH}")

    X, y = create_sequences(df)
    return split_data(X, y)


if __name__ == '__main__':
    splits = run_preprocessing_pipeline()
    print("\n[preprocess] Pipeline complete.")
    for name, arr in zip(['X_train','X_val','X_test','y_train','y_val','y_test'], splits):
        print(f"  {name}: {arr.shape}")