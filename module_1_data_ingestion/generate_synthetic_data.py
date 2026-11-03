"""
Synthetic Data Generator for Telco Customer Churn Dataset
=========================================================
Strategy:
  1. Load the original CSV and understand distributions.
  2. Use SMOTE-style oversampling on the minority (Churn=Yes) class to reduce class imbalance.
  3. Use conditional bootstrapping + Gaussian noise on continuous vars to generate realistic rows.
  4. Keep all categorical relationships consistent (e.g., No PhoneService → MultipleLines = 'No phone service').
  5. Combine original + synthetic → ~15,000 rows, save to Telco_Customer_Churn_Augmented.csv.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1.  Load original data
# ─────────────────────────────────────────────
df_orig = pd.read_csv("/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn.csv")
print(f"Original shape : {df_orig.shape}")
print(f"Churn distribution:\n{df_orig['Churn'].value_counts()}\n")

# Fix TotalCharges (some blank strings)
df_orig["TotalCharges"] = pd.to_numeric(df_orig["TotalCharges"], errors="coerce")
df_orig["TotalCharges"].fillna(df_orig["MonthlyCharges"], inplace=True)

# ─────────────────────────────────────────────
# 2.  Categorical metadata
# ─────────────────────────────────────────────
CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COL = "SeniorCitizen"  # already 0/1

# ─────────────────────────────────────────────
# 3.  Encode for SMOTE
# ─────────────────────────────────────────────
df_enc = df_orig.copy()
encoders = {}
for col in CAT_COLS + ["Churn"]:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

X = df_enc.drop(columns=["Churn"])
y = df_enc["Churn"]

smote = SMOTE(sampling_strategy=0.75, random_state=SEED, k_neighbors=5)
X_res, y_res = smote.fit_resample(X, y)

df_smote = pd.DataFrame(X_res, columns=X.columns)
df_smote["Churn"] = y_res

# Decode SMOTE output back to original labels
for col in CAT_COLS:
    le = encoders[col]
    # SMOTE may produce float indices for encoded categoricals → round & clip
    df_smote[col] = df_smote[col].round().astype(int).clip(0, len(le.classes_) - 1)
    df_smote[col] = le.inverse_transform(df_smote[col])

df_smote["Churn"] = encoders["Churn"].inverse_transform(df_smote["Churn"].round().astype(int))
df_smote["SeniorCitizen"] = df_smote["SeniorCitizen"].round().astype(int).clip(0, 1)
df_smote["tenure"] = df_smote["tenure"].round().astype(int).clip(0, 72)

print(f"After SMOTE : {df_smote.shape}")
print(f"Churn distribution:\n{df_smote['Churn'].value_counts()}\n")


# ─────────────────────────────────────────────
# 4.  Rule-based repair of categorical consistency
#     (ensures business logic is not violated)
# ─────────────────────────────────────────────
def repair_row(row):
    # If no phone service, MultipleLines must reflect that
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "'No phone service'"
    # If no internet service, all internet add-ons must reflect that
    if row["InternetService"] == "No":
        for c in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                  "TechSupport","StreamingTV","StreamingMovies"]:
            row[c] = "'No internet service'"
    else:
        # Remove 'No internet service' values if internet exists
        for c in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                  "TechSupport","StreamingTV","StreamingMovies"]:
            if row[c] == "'No internet service'":
                row[c] = np.random.choice(["Yes", "No"])
    return row

df_smote = df_smote.apply(repair_row, axis=1)

# ─────────────────────────────────────────────
# 5.  Recalculate TotalCharges realistically
# ─────────────────────────────────────────────
df_smote["TotalCharges"] = (
    df_smote["MonthlyCharges"] * df_smote["tenure"]
    + np.random.normal(0, 5, size=len(df_smote))
).clip(lower=0).round(2)

# ─────────────────────────────────────────────
# 6.  Pure bootstrap synthetic rows (additional diversity)
# ─────────────────────────────────────────────
TARGET_TOTAL = 15000
n_bootstrap = TARGET_TOTAL - len(df_smote)
print(f"Generating {n_bootstrap} additional bootstrap rows …")

def generate_bootstrap_rows(df_source, n, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    # Sample with replacement from original
    sampled = df_source.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    for col in NUM_COLS:
        std = df_source[col].std() * 0.08   # 8 % noise
        noise = rng.normal(0, std, size=n)
        sampled[col] = (sampled[col] + noise).clip(
            lower=df_source[col].min(),
            upper=df_source[col].max()
        ).round(2)

    sampled["tenure"] = sampled["tenure"].round().astype(int).clip(0, 72)
    sampled = sampled.apply(repair_row, axis=1)
    # Recalc TotalCharges
    sampled["TotalCharges"] = (
        sampled["MonthlyCharges"] * sampled["tenure"]
        + rng.normal(0, 5, size=n)
    ).clip(lower=0).round(2)
    return sampled

df_bootstrap = generate_bootstrap_rows(df_orig, n_bootstrap)

# ─────────────────────────────────────────────
# 7.  Combine all three sets
# ─────────────────────────────────────────────
df_combined = pd.concat([df_smote, df_bootstrap], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

print(f"\nFinal combined shape : {df_combined.shape}")
print(f"Churn distribution:\n{df_combined['Churn'].value_counts()}")
print(f"Churn %:\n{(df_combined['Churn'].value_counts(normalize=True)*100).round(2)}\n")

# ─────────────────────────────────────────────
# 8.  Save
# ─────────────────────────────────────────────
out_path = "/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn_Augmented.csv"
df_combined.to_csv(out_path, index=False)
print(f"✅  Saved augmented dataset → {out_path}")
print(f"    Rows: {len(df_combined):,}  |  Columns: {df_combined.shape[1]}")
