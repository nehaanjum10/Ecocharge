"""
train.py
--------
Defines, compiles, and trains the LSTM model for EV battery
degradation prediction. Saves the best model to /models/.
"""

import os
import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks as keras_callbacks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import (
    FEATURE_COLS, SEQ_LENGTH,
    run_preprocessing_pipeline,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'lstm_model.keras')  # Recommended format
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── Hyper-parameters ──────────────────────────────────────────────────────────
HPARAMS = {
    'lstm_units_1':   128,
    'lstm_units_2':   64,
    'dense_units':    32,
    'dropout_rate':   0.2,
    'learning_rate':  1e-3,
    'batch_size':     64,
    'epochs':         150,
    'patience':       15,        # early stopping patience
}


# ── 1. Build Model ────────────────────────────────────────────────────────────
def build_model(
    seq_length:   int = SEQ_LENGTH,
    n_features:   int = len(FEATURE_COLS),
    hparams: dict     = HPARAMS,
) -> keras.Model:
    """
    Stacked LSTM → Dropout → Dense → Output

    Input shape : (batch, seq_length, n_features)
    Output      : scalar capacity prediction
    """
    inp = keras.Input(shape=(seq_length, n_features), name='battery_sequence')

    x = layers.LSTM(
        hparams['lstm_units_1'],
        return_sequences=True,
        name='lstm_1',
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(inp)
    x = layers.Dropout(hparams['dropout_rate'], name='drop_1')(x)

    x = layers.LSTM(
        hparams['lstm_units_2'],
        return_sequences=False,
        name='lstm_2',
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(hparams['dropout_rate'], name='drop_2')(x)

    x = layers.Dense(hparams['dense_units'], activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)

    out = layers.Dense(1, name='output')(x)

    model = keras.Model(inputs=inp, outputs=out, name='EcoCharge_LSTM')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hparams['learning_rate']),
        loss='mse',
        metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')],
    )
    model.summary()
    return model


# ── 2. Callbacks ──────────────────────────────────────────────────────────────
def get_callbacks(model_path: str = MODEL_PATH) -> list:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        keras_callbacks.EarlyStopping(
            monitor='val_loss',
            patience=HPARAMS['patience'],
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        keras_callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, 'logs'),
            histogram_freq=0,
        ),
    ]


# ── 3. Train ──────────────────────────────────────────────────────────────────
def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> keras.callbacks.History:
    model = build_model()
    cb    = get_callbacks()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=HPARAMS['epochs'],
        batch_size=HPARAMS['batch_size'],
        callbacks=cb,
        verbose=1,
    )

    # Save training history as JSON
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"[train] History saved → {HISTORY_PATH}")

    return history, model


# ── 4. Plot Training Curves ───────────────────────────────────────────────────
def plot_training_history(history: keras.callbacks.History):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'],     label='Train Loss', color='royalblue')
    axes[0].plot(history.history['val_loss'], label='Val Loss',   color='tomato', linestyle='--')
    axes[0].set_title('MSE Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # MAE
    axes[1].plot(history.history['mae'],     label='Train MAE', color='royalblue')
    axes[1].plot(history.history['val_mae'], label='Val MAE',   color='tomato', linestyle='--')
    axes[1].set_title('Mean Absolute Error', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (Ah)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('EcoCharge — LSTM Training Curves', fontsize=15, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[train] Training curves saved → {out}")


# ── 5. Entry Point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  EcoCharge — LSTM Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing_pipeline()

    # Step 2: Train
    history, model = train(X_train, y_train, X_val, y_val)

    # Step 3: Plot
    plot_training_history(history)

    print(f"\n[train] Model saved → {MODEL_PATH}")
    print("[train] Done! Run evaluate.py next.")