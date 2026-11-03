"""
Quick model benchmark: original vs augmented dataset
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
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

print("=" * 60)
print("  MODEL BENCHMARK: Original vs Augmented Dataset")
print("=" * 60)

# ── Original ──
X_orig, y_orig = load_and_encode(
    "/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn.csv"
)
# ── Augmented ──
X_aug, y_aug = load_and_encode(
    "/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn_Augmented.csv"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "Random Forest     ": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting ": GradientBoostingClassifier(n_estimators=150, random_state=SEED),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
}

for name, model in models.items():
    scores_orig = cross_val_score(model, X_orig, y_orig, cv=cv, scoring="accuracy")
    scores_aug  = cross_val_score(model, X_aug,  y_aug,  cv=cv, scoring="accuracy")
    print(f"\n{name}")
    print(f"  Original  → Accuracy: {scores_orig.mean()*100:.2f}% ± {scores_orig.std()*100:.2f}%")
    print(f"  Augmented → Accuracy: {scores_aug.mean()*100:.2f}%  ± {scores_aug.std()*100:.2f}%")
    delta = (scores_aug.mean() - scores_orig.mean()) * 100
    print(f"  Δ = {delta:+.2f}%")

print("\n" + "=" * 60)
print("  AUGMENTED DATASET QUALITY CHECK")
print("=" * 60)
df_aug = pd.read_csv("/home/rohan/Documents/SIC Capstone/Telco_Customer_Churn_Augmented.csv")
print(f"  Total rows    : {len(df_aug):,}")
print(f"  Columns       : {df_aug.shape[1]}")
print(f"  Missing values: {df_aug.isnull().sum().sum()}")
print(f"  Churn=Yes     : {(df_aug['Churn']=='Yes').sum():,}  ({(df_aug['Churn']=='Yes').mean()*100:.1f}%)")
print(f"  Churn=No      : {(df_aug['Churn']=='No').sum():,}  ({(df_aug['Churn']=='No').mean()*100:.1f}%)")
print(f"  tenure range  : {df_aug['tenure'].min()} – {df_aug['tenure'].max()}")
print(f"  MonthlyCharges: {df_aug['MonthlyCharges'].min():.2f} – {df_aug['MonthlyCharges'].max():.2f}")
