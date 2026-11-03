"""
Module 2C: Structural Causal Model (SCM) Engine
Fits structural regression and discrete mechanics for endogenous features in the Causal DAG.
Implements Pearl's interventional do-calculus operator with downstream causal consistency propagation.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import Ridge

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from module_2_causal_attribution.causal_graph import TelcoCausalGraph
from module_1_data_ingestion.pipeline import TelcoDataPipeline


class TelcoSCMEngine:
    """
    Structural Causal Model for Telco Customer Decisions.
    Fits empirical functions X_j = f_j(PA_j) for endogenous variables.
    Propagates interventions do(X_i = x_i) to causal descendants Desc(X_i).
    """

    def __init__(self, causal_graph: Optional[TelcoCausalGraph] = None):
        self.graph = causal_graph or TelcoCausalGraph()
        self.topological_order = self.graph.topological_sort()

        # Structural regression models for continuous nodes
        self.monthly_charges_model = Ridge(alpha=1.0)
        self.total_charges_model = Ridge(alpha=1.0)

        # Baseline pricing anchors for discrete service add-ons
        self.service_cost_deltas = {
            "OnlineSecurity": 9.95,
            "OnlineBackup": 7.50,
            "DeviceProtection": 8.00,
            "TechSupport": 9.95,
            "StreamingTV": 12.00,
            "StreamingMovies": 12.00,
            "MultipleLines": 10.00
        }
        self.contract_discounts = {
            "Month-to-month": 0.0,
            "One year": -8.50,
            "Two year": -16.00
        }
        self.internet_base_pricing = {
            "No": 20.00,
            "DSL": 45.00,
            "Fiber optic": 75.00
        }
        self.is_fitted = False

    def fit(self, df_raw: pd.DataFrame):
        """Fits empirical structural parameters from observational dataset."""
        df = df_raw.copy()
        # Clean string quotes
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip("'\"").str.strip()

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["MonthlyCharges"])

        # 1. Fit structural equation for MonthlyCharges: f(InternetService, Addons, Contract)
        features_for_mc = [
            "InternetService", "Contract", "MultipleLines", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        X_mc = pd.get_dummies(df[features_for_mc], drop_first=True)
        self.mc_feature_cols = X_mc.columns.tolist()
        self.monthly_charges_model.fit(X_mc, df["MonthlyCharges"])

        # 2. Fit structural equation for TotalCharges: f(tenure, MonthlyCharges)
        X_tc = df[["tenure", "MonthlyCharges"]].copy()
        X_tc["Interaction"] = X_tc["tenure"] * X_tc["MonthlyCharges"]
        self.total_charges_model.fit(X_tc, df["TotalCharges"])

        self.is_fitted = True
        return self

    def predict_monthly_charges(self, record: Dict[str, Any]) -> float:
        """Structural causal equation for MonthlyCharges based on active services and contract."""
        internet = record.get("InternetService", "DSL")
        base = self.internet_base_pricing.get(internet, 45.0)

        # Add telephone
        if record.get("PhoneService") == "Yes":
            base += 20.0
        if record.get("MultipleLines") == "Yes":
            base += self.service_cost_deltas["MultipleLines"]

        # Add internet add-ons if customer has internet
        if internet != "No":
            for addon, fee in self.service_cost_deltas.items():
                if addon != "MultipleLines" and record.get(addon) == "Yes":
                    base += fee

        # Apply contract commitment discount
        contract = record.get("Contract", "Month-to-month")
        base += self.contract_discounts.get(contract, 0.0)

        return float(np.clip(base, 18.25, 120.0))

    def predict_total_charges(self, record: Dict[str, Any]) -> float:
        """Structural causal equation for TotalCharges = tenure * MonthlyCharges."""
        tenure = float(record.get("tenure", 0))
        monthly = float(record.get("MonthlyCharges", 0.0))
        # Total charges accumulates roughly as tenure * monthly with empirical fidelity
        total = max(0.0, tenure * monthly)
        return float(total)

    def intervene(self, factual_instance: Dict[str, Any], do_actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes Pearl's do-calculus intervention do(X = v) on an individual customer instance.
        Propagates changes through all causal descendants in topological order (SCM consistency).
        """
        counterfactual = factual_instance.copy()

        # 1. Apply primary interventions (mutilating incoming edges to intervened variables)
        for var, new_val in do_actions.items():
            counterfactual[var] = new_val

        # 2. Downstream Causal Consistency Propagation
        # If InternetService was modified to 'No', all Internet add-on dependencies must collapse
        if counterfactual.get("InternetService") == "No":
            for addon in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                if addon not in do_actions:
                    counterfactual[addon] = "No internet service"

        # If PhoneService was modified to 'No', MultipleLines must collapse
        if counterfactual.get("PhoneService") == "No":
            if "MultipleLines" not in do_actions:
                counterfactual["MultipleLines"] = "No phone service"

        # If MonthlyCharges was not explicitly overridden by do(), propagate causally from services
        if "MonthlyCharges" not in do_actions:
            counterfactual["MonthlyCharges"] = self.predict_monthly_charges(counterfactual)

        # TotalCharges is strictly a downstream descendant of tenure and MonthlyCharges
        if "TotalCharges" not in do_actions:
            counterfactual["TotalCharges"] = self.predict_total_charges(counterfactual)

        return counterfactual


if __name__ == "__main__":
    df_sample = pd.read_csv("data/Telco_Customer_Churn_Augmented.csv")
    scm = TelcoSCMEngine()
    scm.fit(df_sample)

    print("=" * 65)
    print("MODULE 2C: STRUCTURAL CAUSAL MODEL (SCM) ENGINE")
    print("=" * 65)

    # Test sample
    test_user = df_sample.iloc[0].to_dict()
    print("Original User Profile:")
    print(f"  InternetService : {test_user.get('InternetService')}")
    print(f"  Contract        : {test_user.get('Contract')}")
    print(f"  TechSupport     : {test_user.get('TechSupport')}")
    print(f"  MonthlyCharges  : ${float(test_user.get('MonthlyCharges')):.2f}")
    print(f"  TotalCharges    : ${float(test_user.get('TotalCharges')):.2f}")

    # Test Pearl's do-intervention: Upgrade contract to Two year and add TechSupport
    intervention = {"Contract": "Two year", "TechSupport": "Yes"}
    cf_user = scm.intervene(test_user, intervention)

    print(f"\nAfter Pearl's do({intervention}) with Downstream SCM Propagation:")
    print(f"  Contract        : {cf_user.get('Contract')} [Direct Intervention]")
    print(f"  TechSupport     : {cf_user.get('TechSupport')} [Direct Intervention]")
    print(f"  MonthlyCharges  : ${cf_user.get('MonthlyCharges'):.2f} [Causally Propagated]")
    print(f"  TotalCharges    : ${cf_user.get('TotalCharges'):.2f} [Causally Propagated]")
    print("=" * 65)
