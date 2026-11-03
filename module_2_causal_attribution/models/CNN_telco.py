# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore[missing-import]
from imblearn.over_sampling import SMOTE

# --- 1. Load & preprocess ---
df = pd.read_csv('Telco_Customer_Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

cat_cols = df.select_dtypes('object').columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop('Churn', axis=1).values
y = df['Churn'].values

# --- 2. Scale ---
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- 3. Train/test split + SMOTE ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# --- 4. Reshape for 1D CNN: (samples, features, 1) ---
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

n_features = X_train.shape[1]

# --- 5. Build 1D CNN ---
model = models.Sequential([
    layers.Input(shape=(n_features, 1)),

    layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')   # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# --- 6. Train ---
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)

# --- 7. Evaluate ---
from sklearn.metrics import classification_report
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred))
