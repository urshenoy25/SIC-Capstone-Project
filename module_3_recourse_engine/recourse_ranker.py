"""
Module 3D: Recourse Ranker & Pathway Generator
Filters, scores, and ranks recourse suggestions into human-readable action pathways.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from module_3_recourse_engine.optimizer import ConstrainedRecourseOptimizer
from module_2_causal_attribution.scm_engine import TelcoSCMEngine


class RecourseRanker:
    """
    Ranks viable counterfactual candidates and generates human-interpretable action checklists.
    """

    def __init__(self, optimizer: Optional[ConstrainedRecourseOptimizer] = None):
        self.optimizer = optimizer or ConstrainedRecourseOptimizer()

    def generate_top_recourse(
        self,
        factual_instance: Dict[str, Any],
        top_k: int = 3,
        threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Solves for feasible interventions and produces the top-k diverse, cost-effective recourse pathways.
        """
        raw_candidates = self.optimizer.solve(factual_instance, target_prob_threshold=threshold)

        if not raw_candidates:
            baseline_prob = float(self.optimizer.model.predict_proba(factual_instance)[0])
            if baseline_prob < threshold:
                return [{
                    "rank": 1,
                    "title": "Decision Already Favorable",
                    "summary": f"Current churn risk is {baseline_prob * 100:.1f}%. No intervention necessary.",
                    "cost_score": 0.0,
                    "risk_drop": 0.0,
                    "actions": []
                }]
            return []

        # Deduplicate and extract top diverse pathways
        seen_keys = set()
        ranked_pathways = []
        rank_idx = 1

        for cand in raw_candidates:
            key = tuple(sorted(cand["interventions"].items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            actions_list = []
            for feat, val in cand["interventions"].items():
                old_v = str(factual_instance.get(feat)).strip("'\"")
                actions_list.append({
                    "feature": feat,
                    "from_value": old_v,
                    "to_value": val,
                    "description": f"Change {feat} from '{old_v}' to '{val}'"
                })

            old_risk = cand["baseline_prob"]
            new_risk = cand["achieved_prob"]
            risk_drop = (old_risk - new_risk) * 100

            # Title categorization based on primary intervention
            interv_keys = list(cand["interventions"].keys())
            if "Contract" in interv_keys:
                title = f"Path {rank_idx}: Contract Commitment & Stability"
            elif "InternetService" in interv_keys:
                title = f"Path {rank_idx}: Plan Tier Rebalancing"
            else:
                title = f"Path {rank_idx}: Value Add-on & Service Enhancement"

            ranked_pathways.append({
                "rank": rank_idx,
                "title": title,
                "cost_score": cand["cost"],
                "sparsity": cand["sparsity"],
                "baseline_risk": f"{old_risk * 100:.1f}%",
                "achieved_risk": f"{new_risk * 100:.1f}%",
                "risk_drop_pct": f"{risk_drop:.1f}%",
                "actions": actions_list,
                "interventions": cand["interventions"],
                "causal_consequences": {
                    "New_MonthlyCharges": f"${cand['counterfactual_state'].get('MonthlyCharges'):.2f}",
                    "New_TotalCharges": f"${cand['counterfactual_state'].get('TotalCharges'):.2f}"
                }
            })

            rank_idx += 1
            if len(ranked_pathways) >= top_k:
                break

        return ranked_pathways


if __name__ == "__main__":
    df_raw = pd.read_csv("data/Telco_Customer_Churn_Augmented.csv")
    scm = TelcoSCMEngine().fit(df_raw)
    opt = ConstrainedRecourseOptimizer(scm=scm)
    ranker = RecourseRanker(opt)

    print("=" * 65)
    print("MODULE 3: CONSTRAINED RECOURSE OPTIMIZER & RANKER")
    print("=" * 65)

    high_risk_user = df_raw[df_raw["Churn"].astype(str).str.contains("Yes")].iloc[5].to_dict()
    prob = opt.model.predict_proba(high_risk_user)[0]
    print(f"Target User Baseline Churn Probability: {prob * 100:.2f}%\n")

    pathways = ranker.generate_top_recourse(high_risk_user, top_k=3)
    for p in pathways:
        print(f"[{p['title']}] (Effort Cost: {p['cost_score']}, Churn Risk: {p['baseline_risk']} -> {p['achieved_risk']})")
        for act in p["actions"]:
            print(f"  • {act['description']}")
        print(f"  Causal Consequence: Monthly = {p['causal_consequences']['New_MonthlyCharges']}, Total = {p['causal_consequences']['New_TotalCharges']}\n")
    print("=" * 65)
