"""
Module 1: Data Ingestion & Preprocessing Pipeline
Provides end-to-end schema validation, missing-value imputation, feature stratification,
categorical encoding, standard scaling, and serialization for high-stakes tabular decisions.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TelcoDataPipeline(BaseEstimator, TransformerMixin):
    """
    Standardized tabular data ingestion & preprocessing pipeline for Telco Churn.
    Guarantees strict schema enforcement, consistent encoding, and inverse transformations
    required by downstream Causal Discovery (Module 2) and Recourse Engine (Module 3).
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "feature_config.json")
        self.config_path = config_path
        self.config = self._load_config()

        self.target_col = self.config["metadata"]["target_column"]
        self.positive_label = self.config["metadata"]["positive_label"]
        self.negative_label = self.config["metadata"]["negative_label"]

        self.immutable_features = [f["name"] for f in self.config["stratification"]["immutable_features"]]
        self.mutable_features = [f["name"] for f in self.config["stratification"]["mutable_features"]]
        self.derived_features = [f["name"] for f in self.config["stratification"]["derived_features"]]

        self.all_input_features = self.immutable_features + self.mutable_features + self.derived_features

        # Preprocessor states
        self.scaler = StandardScaler()
        self.numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.categorical_cols = [c for c in self.all_input_features if c not in self.numerical_cols]

        self.cat_categories_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_ = False

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans strings, removes quoting artifacts, and handles missing/coerced values."""
        df_clean = df.copy()

        # Strip whitespace and quotes from all string columns
        for col in df_clean.columns:
            if col not in self.numerical_cols and col not in ["SeniorCitizen", self.target_col]:
                df_clean[col] = df_clean[col].astype(str).str.strip("'\"").str.strip()

        # Coerce SeniorCitizen to integer binary (0 or 1)
        if "SeniorCitizen" in df_clean.columns:
            df_clean["SeniorCitizen"] = pd.to_numeric(df_clean["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

        # Handle TotalCharges numeric coercion and imputation
        if "TotalCharges" in df_clean.columns:
            df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
            # If missing, impute with tenure * MonthlyCharges if tenure > 0, else MonthlyCharges
            if "tenure" in df_clean.columns and "MonthlyCharges" in df_clean.columns:
                fallback = df_clean["tenure"] * df_clean["MonthlyCharges"]
                df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(fallback)
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

        # Encode target if present
        if self.target_col in df_clean.columns:
            target_str = df_clean[self.target_col].astype(str).str.strip("'\"").str.strip()
            df_clean[self.target_col] = target_str.map(
                {self.positive_label: 1, self.negative_label: 0, "1": 1, "0": 0, 1: 1, 0: 0}
            ).fillna(0).astype(int)

        return df_clean

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates dataframe against schema definitions in feature_config.json."""
        errors = []
        df_clean = self.clean(df)

        for col in self.all_input_features:
            if col not in df_clean.columns:
                errors.append(f"Missing required feature: '{col}'")

        # Validate categorical domains
        for f_meta in self.config["stratification"]["immutable_features"] + self.config["stratification"]["mutable_features"]:
            name = f_meta["name"]
            if name in df_clean.columns and "allowed_values" in f_meta:
                allowed = set(f_meta["allowed_values"])
                actual = set(df_clean[name].unique())
                invalid = actual - allowed
                if invalid:
                    errors.append(f"Invalid values in '{name}': {invalid}. Expected subset of {allowed}")

        # Validate numerical ranges
        if "tenure" in df_clean.columns:
            if (df_clean["tenure"] < 0).any() or (df_clean["tenure"] > 100).any():
                errors.append("Feature 'tenure' contains out-of-range values outside [0, 100]")

        return (len(errors) == 0, errors)

    def fit(self, df: pd.DataFrame, y=None):
        """Fits categorical levels and standard scaler on training distribution."""
        df_clean = self.clean(df)

        # Record exact category levels for deterministic one-hot dummy encoding
        for col in self.categorical_cols:
            if col in df_clean.columns:
                unique_vals = sorted(list(df_clean[col].dropna().unique()))
                self.cat_categories_[col] = unique_vals

        # Determine feature names after dummy encoding
        dummy_feature_names = []
        for col in self.all_input_features:
            if col in self.numerical_cols:
                dummy_feature_names.append(col)
            elif col in self.categorical_cols:
                for val in self.cat_categories_[col]:
                    dummy_feature_names.append(f"{col}_{val}")

        self.feature_names_ = dummy_feature_names

        # Fit numerical standard scaler
        num_matrix = df_clean[self.numerical_cols].values
        self.scaler.fit(num_matrix)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms raw dataframe into model-ready numerical feature matrix X."""
        if not self.is_fitted_:
            raise ValueError("TelcoDataPipeline must be fitted before calling transform().")

        df_clean = self.clean(df)
        n_samples = len(df_clean)

        # 1. Scale numerical features
        num_matrix = df_clean[self.numerical_cols].values
        num_scaled = self.scaler.transform(num_matrix)

        # 2. Encode categorical features into aligned one-hot arrays
        cat_encoded_list = []
        for col in self.categorical_cols:
            categories = self.cat_categories_[col]
            # Create zeros matrix of shape (n_samples, len(categories))
            cat_arr = np.zeros((n_samples, len(categories)), dtype=np.float32)
            series = df_clean[col].values
            for idx, val in enumerate(categories):
                cat_arr[series == val, idx] = 1.0
            cat_encoded_list.append(cat_arr)

        # Assemble features according to self.feature_names_ order
        feature_blocks = []
        cat_idx_map = {col: i for i, col in enumerate(self.categorical_cols)}

        for col in self.all_input_features:
            if col in self.numerical_cols:
                num_idx = self.numerical_cols.index(col)
                feature_blocks.append(num_scaled[:, num_idx:num_idx+1])
            else:
                cat_pos = cat_idx_map[col]
                feature_blocks.append(cat_encoded_list[cat_pos])

        X_transformed = np.hstack(feature_blocks).astype(np.float32)
        return X_transformed

    def fit_transform(self, df: pd.DataFrame, y=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fits preprocessor and returns transformed (X, y) tuple."""
        df_clean = self.clean(df)
        self.fit(df_clean)
        X = self.transform(df_clean)

        y_arr = None
        if self.target_col in df_clean.columns:
            y_arr = df_clean[self.target_col].values.astype(int)

        return X, y_arr

    def transform_single(self, instance_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single raw customer record dictionary into a 2D feature vector (1, n_features)."""
        df_single = pd.DataFrame([instance_dict])
        return self.transform(df_single)

    def inverse_transform_numerics(self, num_scaled: np.ndarray) -> np.ndarray:
        """Inverses scaling for tenure, MonthlyCharges, TotalCharges."""
        return self.scaler.inverse_transform(num_scaled)

    def save(self, output_path: str):
        """Serializes fitted pipeline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        joblib.dump(self, output_path)
        print(f"Pipeline successfully saved to: {output_path}")

    @classmethod
    def load(cls, model_path: str) -> "TelcoDataPipeline":
        """Loads serialized pipeline from disk."""
        return joblib.load(model_path)


def run_module_1_pipeline(
    dataset_path: str = "data/Telco_Customer_Churn_Augmented.csv",
    save_path: str = "module_1_data_ingestion/preprocessor.joblib",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    """
    Executes Module 1 end-to-end:
    Loads augmented data, validates schema, fits transformations, splits sets, and exports preprocessor.
    """
    print("=" * 65)
    print("MODULE 1: DATA INGESTION & FEATURE STRATIFICATION PIPELINE")
    print("=" * 65)

    print(f"\n[1] Ingesting dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"    Raw records loaded : {len(df):,} samples | {df.shape[1]} columns")

    print("\n[2] Initializing TelcoDataPipeline with feature_config.json")
    pipeline = TelcoDataPipeline()
    print(f"    Immutable features ({len(pipeline.immutable_features)}) : {pipeline.immutable_features}")
    print(f"    Mutable features   ({len(pipeline.mutable_features)}) : {pipeline.mutable_features}")
    print(f"    Derived features   ({len(pipeline.derived_features)}) : {pipeline.derived_features}")

    print("\n[3] Validating schema constraints...")
    is_valid, errors = pipeline.validate_schema(df)
    if is_valid:
        print("    [✓] Schema validation passed with 0 errors!")
    else:
        print(f"    [!] Schema validation warnings/errors ({len(errors)}):")
        for err in errors[:5]:
            print(f"        - {err}")

    print("\n[4] Fitting transformations and encoding feature matrix...")
    X, y = pipeline.fit_transform(df)
    print(f"    Transformed feature matrix shape : {X.shape}")
    print(f"    Total engineered columns         : {len(pipeline.feature_names_)}")
    print(f"    Target class distribution        : No Churn={(y==0).sum():,} | Churn={(y==1).sum():,}")

    print("\n[5] Generating stratified Train/Val/Test splits (70% / 15% / 15%)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio), stratify=y_temp, random_state=random_state
    )
    print(f"    Train partition : {X_train.shape[0]:,} samples")
    print(f"    Val partition   : {X_val.shape[0]:,} samples")
    print(f"    Test partition  : {X_test.shape[0]:,} samples")

    print(f"\n[6] Serializing fitted preprocessor to: {save_path}")
    pipeline.save(save_path)

    # Verification test
    loaded = TelcoDataPipeline.load(save_path)
    test_sample = df.iloc[0:1].to_dict(orient="records")[0]
    x_test_vec = loaded.transform_single(test_sample)
    print(f"    [✓] Verification test: single-sample transformed shape = {x_test_vec.shape}")

    print("\n" + "=" * 65)
    print("MODULE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return pipeline, (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    run_module_1_pipeline()
