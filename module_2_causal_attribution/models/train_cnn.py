"""
Module 2: 1D Convolutional Neural Network (CNN) Classifier
Trains, validates, evaluates, and serializes the 1D Tabular CNN for customer churn prediction.
Integrates with TelcoDataPipeline from Module 1 for strict schema alignment.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, roc_curve
)
from typing import Dict, Any, Union, Optional

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from module_1_data_ingestion.pipeline import TelcoDataPipeline

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def build_1d_cnn(input_features: int) -> models.Sequential:
    """
    Constructs a 1D Convolutional Neural Network tailored for tabular feature interaction.
    Input shape: (batch_size, input_features, 1)
    """
    model = models.Sequential([
        layers.Input(shape=(input_features, 1)),

        # Conv Block 1: Local feature correlation extraction
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),

        # Conv Block 2: Higher-order feature motif representation
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),

        # Dense Classification Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Binary risk probability [0, 1]
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


class ChurnModelWrapper:
    """
    Unified inference wrapper that bridges raw customer tabular data / numpy arrays
    and the trained 1D CNN model. Used by Causal Attribution, Recourse, and API.
    """
    def __init__(self, model_path: str, preprocessor_path: str):
        self.pipeline = TelcoDataPipeline.load(preprocessor_path)
        self.model = tf.keras.models.load_model(model_path)
        self.n_features = len(self.pipeline.feature_names_)

    def predict_proba(self, data: Union[pd.DataFrame, Dict[str, Any], np.ndarray]) -> np.ndarray:
        """Computes churn probability P(Churn = 1 | X)."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        if isinstance(data, pd.DataFrame):
            X_proc = self.pipeline.transform(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(1, -1)
            X_proc = data
        else:
            raise TypeError("Unsupported data type for inference.")

        # Ensure shape (batch, n_features, 1) for 1D CNN
        if X_proc.ndim == 2:
            X_cnn = X_proc.reshape(X_proc.shape[0], X_proc.shape[1], 1)
        else:
            X_cnn = X_proc

        probs = self.model.predict(X_cnn, verbose=0).ravel()
        return probs

    def predict(self, data: Union[pd.DataFrame, Dict[str, Any], np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions (0 = No Churn, 1 = Churn)."""
        probs = self.predict_proba(data)
        return (probs >= threshold).astype(int)


def train_and_evaluate_cnn(
    dataset_path: str = "data/Telco_Customer_Churn_Augmented.csv",
    preprocessor_path: str = "module_1_data_ingestion/preprocessor.joblib",
    model_save_path: str = "module_2_causal_attribution/models/saved_models/cnn_churn_model.keras",
    epochs: int = 35,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Trains 1D CNN on augmented dataset using Module 1 preprocessor and saves best model."""
    print("=" * 65)
    print("MODULE 2A: 1D TABULAR CNN CLASSIFIER TRAINING")
    print("=" * 65)

    # 1. Load Preprocessor and Data
    if not os.path.exists(preprocessor_path):
        from module_1_data_ingestion.pipeline import run_module_1_pipeline
        run_module_1_pipeline(dataset_path=dataset_path, save_path=preprocessor_path)

    pipeline = TelcoDataPipeline.load(preprocessor_path)
    df = pd.read_csv(dataset_path)
    X, y = pipeline.transform(df), pipeline.clean(df)[pipeline.target_col].values

    # Stratified 70 / 15 / 15 Split
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    n_features = X_train.shape[1]
    X_train_cnn = X_train.reshape(-1, n_features, 1)
    X_val_cnn = X_val.reshape(-1, n_features, 1)
    X_test_cnn = X_test.reshape(-1, n_features, 1)

    print(f"\nFeature dimensionality : {n_features} features")
    print(f"Training partition     : {X_train.shape[0]:,} samples")
    print(f"Validation partition   : {X_val.shape[0]:,} samples")
    print(f"Testing partition      : {X_test.shape[0]:,} samples")

    # 2. Build & Compile Model
    model = build_1d_cnn(n_features)
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)

    checkpoint_cb = callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=0
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=8,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    print("\nTraining 1D CNN Architecture...")
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        verbose=1
    )

    # 3. Comprehensive Evaluation
    best_model = tf.keras.models.load_model(model_save_path)
    y_prob = best_model.predict(X_test_cnn, verbose=0).ravel()
    y_pred = (y_prob >= 0.50).astype(int)

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print("1D CNN TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Test ROC-AUC  : {test_auc:.4f}")
    print(f"Test Precision: {test_prec * 100:.2f}%")
    print(f"Test Recall   : {test_rec * 100:.2f}%")
    print(f"Test F1-Score : {test_f1:.4f}")
    print("=" * 65)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"[✓] Best model saved to: {model_save_path}")

    return {
        "accuracy": float(test_acc),
        "roc_auc": float(test_auc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1": float(test_f1),
        "model_path": model_save_path
    }


if __name__ == "__main__":
    train_and_evaluate_cnn()
