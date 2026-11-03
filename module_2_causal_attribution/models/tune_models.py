"""
Tune Random Forest + try XGBoost on augmented dataset to clear 90%.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

SEED = 42

def load_and_encode(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "Churn"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

X_aug, y_aug = load_and_encode(
    "/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn_Augmented.csv"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("Trying Random Forest configurations…")
configs = [
    dict(n_estimators=300, max_depth=None, min_samples_split=2,  class_weight=None),
    dict(n_estimators=500, max_depth=20,   min_samples_split=3,  class_weight="balanced"),
    dict(n_estimators=300, max_depth=15,   min_samples_split=4,  class_weight="balanced"),
    dict(n_estimators=400, max_depth=None, min_samples_split=2,  class_weight="balanced"),
]
for cfg in configs:
    rf = RandomForestClassifier(**cfg, random_state=SEED, n_jobs=-1)
    scores = cross_val_score(rf, X_aug, y_aug, cv=cv, scoring="accuracy")
    print(f"  RF {cfg}  →  {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# Try XGBoost if available
try:
    # pyrefly: ignore[missing-import]
    from xgboost import XGBClassifier
    print("\nTrying XGBoost…")
    xgb_configs = [
        dict(n_estimators=300, max_depth=6, learning_rate=0.1,  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1),
        dict(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1),
    ]
    for cfg in xgb_configs:
        xgb = XGBClassifier(**cfg, random_state=SEED, use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
        scores = cross_val_score(xgb, X_aug, y_aug, cv=cv, scoring="accuracy")
        print(f"  XGB {cfg}  →  {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
except ImportError:
    print("XGBoost not installed, skipping.")
