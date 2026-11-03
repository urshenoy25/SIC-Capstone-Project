"""
Final approach: Feature Engineering + Stacking Ensemble on Augmented Data
Target: > 90% accuracy
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

SEED = 42

def load_and_feature_engineer(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # ── Feature Engineering ──────────────────────────────────────────
    # 1. Monthly charge tier
    df["ChargePerTenure"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    # 2. Is long-term customer?
    df["LongTenure"] = (df["tenure"] >= 24).astype(int)
    # 3. Number of internet add-on services
    internet_svcs = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                     "TechSupport","StreamingTV","StreamingMovies"]
    # We'll compute after encoding, use raw strings
    df["NumInternetSvcs"] = df[internet_svcs].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    # 4. Has any contract commitment?
    df["HasContract"] = (df["Contract"] != "Month-to-month").astype(int)
    # 5. Monthly charge bucket
    df["HighMonthlyCharge"] = (df["MonthlyCharges"] > 70).astype(int)
    # 6. Auto payment
    df["AutoPay"] = df["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)

    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "Churn"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

print("Loading augmented dataset with feature engineering…")
X_aug, y_aug = load_and_feature_engineer(
    "/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn_Augmented.csv"
)
print(f"Feature count: {X_aug.shape[1]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── 1. Tuned Random Forest ────────────────────────────────────────────
print("\n[1] Tuned Random Forest + Feature Engineering")
rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)
scores = cross_val_score(rf, X_aug, y_aug, cv=cv, scoring="accuracy")
print(f"    Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# ── 2. Gradient Boosting ──────────────────────────────────────────────
print("\n[2] Gradient Boosting + Feature Engineering")
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=SEED
)
scores = cross_val_score(gb, X_aug, y_aug, cv=cv, scoring="accuracy")
print(f"    Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# ── 3. Stacking Ensemble ──────────────────────────────────────────────
print("\n[3] Stacking Ensemble (RF + GB → Logistic Regression)")
estimators = [
    ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                       learning_rate=0.1, random_state=SEED)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500, random_state=SEED),
    cv=5,
    passthrough=True,  # pass original features too
    n_jobs=-1,
)
scores = cross_val_score(stack, X_aug, y_aug, cv=cv, scoring="accuracy")
print(f"    Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# ── Original dataset baseline (with FE) ──────────────────────────────
print("\n[Baseline] Original dataset + Feature Engineering")
X_orig, y_orig = load_and_feature_engineer(
    "/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn.csv"
)
rf_orig = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)
scores_orig = cross_val_score(rf_orig, X_orig, y_orig, cv=cv, scoring="accuracy")
print(f"    RF Accuracy: {scores_orig.mean()*100:.2f}% ± {scores_orig.std()*100:.2f}%")
