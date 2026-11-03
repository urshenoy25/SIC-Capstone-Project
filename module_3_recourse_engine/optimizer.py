"""
Module 3C: Constrained Causal Recourse Optimizer
Solves the non-linear constrained optimization problem to identify the minimum-cost
actionable intervention vector theta* that flips an adverse decision to favorable.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from module_2_causal_attribution.models.train_cnn import ChurnModelWrapper
from module_2_causal_attribution.scm_engine import TelcoSCMEngine
from module_3_recourse_engine.actionability_filter import ActionabilityFilter
from module_3_recourse_engine.causal_propagator import CausalRecoursePropagator


class ConstrainedRecourseOptimizer:
    """
    Constrained optimization solver for algorithmic recourse.
    Explores actionable intervention space, enforces zero immutable violations,
    propagates downstream causal effects, and selects candidates flipping adverse decisions.
    """

    def __init__(
        self,
        model_wrapper: Optional[ChurnModelWrapper] = None,
        scm: Optional[TelcoSCMEngine] = None,
        config_path: str = "module_1_data_ingestion/feature_config.json"
    ):
        self.model = model_wrapper or ChurnModelWrapper(
            "module_2_causal_attribution/models/saved_models/cnn_churn_model.keras",
            "module_1_data_ingestion/preprocessor.joblib"
        )
        self.filter = ActionabilityFilter(config_path)
        self.propagator = CausalRecoursePropagator(scm, self.filter)

        # Domain Action Space for Mutable Levers
        self.action_space = {
            "Contract": ["One year", "Two year"],
            "TechSupport": ["Yes"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["Yes"],
            "InternetService": ["DSL", "No"],
            "PaperlessBilling": ["No"],
            "PaymentMethod": ["Credit card (automatic)", "Bank transfer (automatic)"]
        }

    def compute_cost(self, factual: Dict[str, Any], interventions: Dict[str, Any]) -> float:
        """Computes aggregate weighted intervention effort: sum(w_j * cost_j)."""
        total_cost = 0.0
        for feat, target_val in interventions.items():
            base_w = self.filter.get_action_cost(feat)
            cur_val = factual.get(feat)
            if str(cur_val).strip("'\"") == str(target_val).strip("'\""):
                continue

            # Tier-specific cost modifiers
            if feat == "Contract":
                multiplier = 1.0 if target_val == "One year" else 1.5
            elif feat == "InternetService" and target_val == "No":
                multiplier = 1.8  # Terminating broadband is high friction
            else:
                multiplier = 1.0

            total_cost += base_w * multiplier

        # Sparsity penalty: encourage changing <= 3 features
        sparsity_penalty = max(0, len(interventions) - 2) * 1.5
        return float(round(total_cost + sparsity_penalty, 2))

    def solve(
        self,
        factual_instance: Dict[str, Any],
        target_prob_threshold: float = 0.45,
        max_feature_changes: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generates actionable counterfactual solutions flipping decision below threshold.
        Returns candidate solutions ranked by minimal intervention cost.
        """
        baseline_prob = float(self.model.predict_proba(factual_instance)[0])
        if baseline_prob < target_prob_threshold:
            return []  # Already favorable prediction (No recourse needed)

        candidates = []

        # 1. Single-Feature Interventions (Rank-1 Sparsity)
        for feat, options in self.action_space.items():
            cur_v = factual_instance.get(feat)
            for opt in options:
                if str(cur_v).strip("'\"") == str(opt).strip("'\""):
                    continue

                action = {feat: opt}
                cf_state = self.propagator.propagate(factual_instance, action)
                cf_prob = float(self.model.predict_proba(cf_state)[0])

                if cf_prob < target_prob_threshold:
                    cost = self.compute_cost(factual_instance, action)
                    candidates.append({
                        "interventions": action,
                        "counterfactual_state": cf_state,
                        "baseline_prob": baseline_prob,
                        "achieved_prob": round(cf_prob, 4),
                        "cost": cost,
                        "sparsity": 1,
                        "verdict_flip": True
                    })

        # 2. Dual-Feature Interventions (Rank-2 Sparsity)
        action_items = list(self.action_space.items())
        for i in range(len(action_items)):
            feat1, opts1 = action_items[i]
            cur1 = factual_instance.get(feat1)
            for opt1 in opts1:
                if str(cur1).strip("'\"") == str(opt1).strip("'\""):
                    continue

                for j in range(i + 1, len(action_items)):
                    feat2, opts2 = action_items[j]
                    cur2 = factual_instance.get(feat2)
                    for opt2 in opts2:
                        if str(cur2).strip("'\"") == str(opt2).strip("'\""):
                            continue

                        action = {feat1: opt1, feat2: opt2}
                        cf_state = self.propagator.propagate(factual_instance, action)
                        cf_prob = float(self.model.predict_proba(cf_state)[0])

                        if cf_prob < target_prob_threshold:
                            cost = self.compute_cost(factual_instance, action)
                            candidates.append({
                                "interventions": action,
                                "counterfactual_state": cf_state,
                                "baseline_prob": baseline_prob,
                                "achieved_prob": round(cf_prob, 4),
                                "cost": cost,
                                "sparsity": 2,
                                "verdict_flip": True
                            })

        # 3. Triple-Feature Interventions (Rank-3 Sparsity for difficult cases)
        if len(candidates) < 3 and max_feature_changes >= 3:
            # Pair contract change with high-impact services
            for opt_c in ["One year", "Two year"]:
                for opt_s in ["Yes"]:
                    for opt_b in ["Yes"]:
                        action = {"Contract": opt_c, "OnlineSecurity": opt_s, "TechSupport": opt_b}
                        cf_state = self.propagator.propagate(factual_instance, action)
                        cf_prob = float(self.model.predict_proba(cf_state)[0])
                        if cf_prob < target_prob_threshold:
                            cost = self.compute_cost(factual_instance, action)
                            candidates.append({
                                "interventions": action,
                                "counterfactual_state": cf_state,
                                "baseline_prob": baseline_prob,
                                "achieved_prob": round(cf_prob, 4),
                                "cost": cost,
                                "sparsity": 3,
                                "verdict_flip": True
                            })

        # Sort candidates by minimum cost, then lowest achieved risk
        candidates.sort(key=lambda x: (x["cost"], x["achieved_prob"]))
        return candidates
