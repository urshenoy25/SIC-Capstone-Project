"""
1D Convolutional Neural Network for Telco Customer Churn Prediction
Trained on Augmented Dataset (18,000 samples)
"""
# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
import os
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_PATH = "Telco_Customer_Churn_Augmented.csv"

# ─────────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────────
print(f"Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop('Churn', axis=1).values
y = df['Churn'].values

print(f"Dataset shape  : {X.shape}")
print(f"Class counts   : No Churn = {(y==0).sum():,}, Churn = {(y==1).sum():,}\n")

# ─────────────────────────────────────────────
# 2. Train / Val / Test Split (70% train | 15% val | 15% test)
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# ─────────────────────────────────────────────
# 3. Standard Scaling
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Reshape for 1D CNN: (batch_size, n_features, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn   = X_val.reshape(X_val.shape[0],     X_val.shape[1],   1)
X_test_cnn  = X_test.reshape(X_test.shape[0],   X_test.shape[1],  1)

n_features = X_train_cnn.shape[1]

# ─────────────────────────────────────────────
# 4. Build 1D CNN Architecture
# ─────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(n_features, 1)),

    layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# ─────────────────────────────────────────────
# 5. Train Model
# ─────────────────────────────────────────────
history = model.fit(
    X_train_cnn, y_train,
    epochs=40,
    batch_size=64,
    validation_data=(X_val_cnn, y_val),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=3,
            factor=0.5,
            min_lr=1e-5,
            verbose=1
        )
    ]
)

# ─────────────────────────────────────────────
# 6. Evaluate on Test Set
# ─────────────────────────────────────────────
y_prob_test = model.predict(X_test_cnn).ravel()
y_pred = (y_prob_test >= 0.50).astype(int)
acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
test_auc = auc(fpr, tpr)

print("\n" + "=" * 55)
print(f"CNN TEST RESULTS ON AUGMENTED DATASET")
print("=" * 55)
print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"Test AUC      : {test_auc:.4f}")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
