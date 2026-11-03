"""
Module 2D: Causal Feature Attribution & SHAP Benchmark Engine
Computes Individual Causal Effect (ICE) and Average Causal Effect (ACE) using Pearl's do-calculus.
Benchmarks causal attributions against conventional correlational SHAP values.
"""

import os
import sys
import json
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from module_2_causal_attribution.models.train_cnn import ChurnModelWrapper
from module_2_causal_attribution.scm_engine import TelcoSCMEngine
from module_1_data_ingestion.pipeline import TelcoDataPipeline


class CausalAttributionEngine:
    """
    Computes interventional feature attribution using Structural Causal Models (SCM).
    Contrasts interventional impact Δ = E[Y|do(X=x')] - E[Y|do(X=x)] against correlational SHAP.
    """

    def __init__(
        self,
        model_path: str = "module_2_causal_attribution/models/saved_models/cnn_churn_model.keras",
        preprocessor_path: str = "module_1_data_ingestion/preprocessor.joblib",
        config_path: str = "module_1_data_ingestion/feature_config.json"
    ):
        self.model = ChurnModelWrapper(model_path, preprocessor_path)
        self.pipeline = self.model.pipeline
        self.scm = TelcoSCMEngine()

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.immutable_features = [f["name"] for f in self.config["stratification"]["immutable_features"]]
        self.mutable_features = [f["name"] for f in self.config["stratification"]["mutable_features"]]

    def fit_scm(self, df_raw: pd.DataFrame):
        """Fits structural causal equations on observational data."""
        self.scm.fit(df_raw)
        return self

    def compute_ice(self, instance: Dict[str, Any]) -> pd.DataFrame:
        """
        Computes Individual Causal Effect (ICE) for an individual customer.
        Evaluates the causal delta in churn risk when intervening on each actionable feature.
        """
        # Baseline probability P(Churn = 1 | X = x)
        baseline_prob = float(self.model.predict_proba(instance)[0])

        results = []

        # Candidate interventions for each actionable feature
        candidate_actions: Dict[str, List[Any]] = {
            "Contract": ["Two year", "One year"],
            "TechSupport": ["Yes"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["Yes"],
            "InternetService": ["DSL", "No"],
            "PaperlessBilling": ["No"],
            "PaymentMethod": ["Credit card (automatic)", "Bank transfer (automatic)"],
            "MonthlyCharges": [max(18.25, float(instance.get("MonthlyCharges", 70.0)) - 25.0)]
        }

        for feature in self.mutable_features:
            current_val = instance.get(feature)
            actions = candidate_actions.get(feature, [])

            for target_val in actions:
                if str(current_val).strip("'\"") == str(target_val).strip("'\""):
                    continue

                # Execute Pearl's do-calculus intervention with downstream propagation
                intervened_instance = self.scm.intervene(instance, {feature: target_val})
                new_prob = float(self.model.predict_proba(intervened_instance)[0])

                # Risk Reduction Δ = P(Factual) - P(Intervened)
                risk_reduction = baseline_prob - new_prob

                results.append({
                    "Feature": feature,
                    "Current_Value": str(current_val).strip("'\""),
                    "Suggested_Action": f"Set to {target_val}",
                    "Baseline_Risk": round(baseline_prob, 4),
                    "Intervened_Risk": round(new_prob, 4),
                    "Causal_Impact_Delta": round(risk_reduction, 4),
                    "Percentage_Points_Improvement": round(risk_reduction * 100, 2)
                })

        df_ice = pd.DataFrame(results)
        if not df_ice.empty:
            df_ice = df_ice.sort_values(by="Causal_Impact_Delta", ascending=False).reset_index(drop=True)
        return df_ice

    def compute_shap_values(
        self,
        instance: Dict[str, Any],
        background_samples: Optional[np.ndarray] = None,
        nsamples: int = 100
    ) -> Dict[str, float]:
        """
        Computes correlational SHAP feature attributions on transformed feature space.
        """
        # Create background baseline if not passed
        if background_samples is None:
            background_samples = np.zeros((10, self.model.n_features), dtype=np.float32)

        def predict_func(x):
            return self.model.predict_proba(x)

        explainer = shap.KernelExplainer(predict_func, background_samples)
        x_target = self.pipeline.transform(pd.DataFrame([instance]))

        shap_vals = explainer.shap_values(x_target, nsamples=nsamples, silent=True)
        if isinstance(shap_vals, list):
            shap_arr = shap_vals[0] if len(shap_vals) == 1 else shap_vals[1]
        else:
            shap_arr = shap_vals

        feature_names = self.pipeline.feature_names_
        return {name: float(shap_arr[0][i]) for i, name in enumerate(feature_names)}

    def compare_causal_vs_shap(self, instance: Dict[str, Any]) -> pd.DataFrame:
        """
        Generates comparative attribution table contrasting Causal Interventional Effect vs. Correlational SHAP.
        Demonstrates the fundamental scientific divergence between Correlation and Causation.
        """
        df_ice = self.compute_ice(instance)
        shap_dict = self.compute_shap_values(instance)

        # Map dummy SHAP values back to high-level domain features
        aggregated_shap = {}
        for feat in self.immutable_features + self.mutable_features + ["TotalCharges"]:
            total_shap = 0.0
            for col, val in shap_dict.items():
                if col == feat or col.startswith(f"{feat}_"):
                    total_shap += abs(val)
            aggregated_shap[feat] = total_shap

        comparison = []
        for feat in self.mutable_features:
            match = df_ice[df_ice["Feature"] == feat]
            causal_delta = match["Causal_Impact_Delta"].max() if not match.empty else 0.0
            action_desc = match["Suggested_Action"].iloc[0] if not match.empty else "No action"

            comparison.append({
                "Feature": feat,
                "Type": "Mutable (Actionable)",
                "Action": action_desc,
                "Causal_Impact": round(causal_delta, 4),
                "SHAP_Association": round(aggregated_shap.get(feat, 0.0), 4)
            })

        for feat in self.immutable_features:
            comparison.append({
                "Feature": feat,
                "Type": "Immutable (Protected)",
                "Action": "[NON-ACTIONABLE]",
                "Causal_Impact": 0.0000,
                "SHAP_Association": round(aggregated_shap.get(feat, 0.0), 4)
            })

        df_comp = pd.DataFrame(comparison).sort_values(by="Causal_Impact", ascending=False).reset_index(drop=True)
        return df_comp


if __name__ == "__main__":
    df_raw = pd.read_csv("data/Telco_Customer_Churn_Augmented.csv")
    engine = CausalAttributionEngine()
    engine.fit_scm(df_raw)

    print("=" * 65)
    print("MODULE 2D: CAUSAL FEATURE ATTRIBUTION (ICE & DO-CALCULUS)")
    print("=" * 65)

    # Pick an adverse customer case (high churn risk)
    high_risk_customer = df_raw[df_raw["Churn"].astype(str).str.contains("Yes")].iloc[5].to_dict()
    prob = engine.model.predict_proba(high_risk_customer)[0]
    print(f"Customer Baseline Churn Risk: {prob * 100:.2f}% (Adverse Prediction)")
    print(f"Contract: {high_risk_customer.get('Contract')} | Internet: {high_risk_customer.get('InternetService')} | Monthly: ${float(high_risk_customer.get('MonthlyCharges')):.2f}\n")

    print("[1] Top Individual Causal Effects (ICE Interventions):")
    ice_results = engine.compute_ice(high_risk_customer)
    print(ice_results.head(6)[["Feature", "Current_Value", "Suggested_Action", "Intervened_Risk", "Percentage_Points_Improvement"]].to_string(index=False))

    print("\n[2] Comparative Attribution (Causal Effect vs. SHAP Association):")
    comp = engine.compare_causal_vs_shap(high_risk_customer)
    print(comp.head(10).to_string(index=False))
    print("=" * 65)
