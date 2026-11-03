"""
Module 3A: Actionability & Immutability Filter
Strictly enforces domain boundaries and guarantees zero changes to protected immutable features.
"""

import json
import os
from typing import Dict, Any, List, Tuple


class ActionabilityFilter:
    """
    Guarantees strict adherence to feature actionability.
    Enforces that protected features cannot be modified during recourse generation.
    """

    def __init__(self, config_path: str = "module_1_data_ingestion/feature_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.immutable_names = {f["name"] for f in self.config["stratification"]["immutable_features"]}
        self.derived_names = {f["name"] for f in self.config["stratification"]["derived_features"]}
        self.forbidden_keys = self.immutable_names.union(self.derived_names)

        # Map mutable features to metadata and cost weights
        self.mutable_meta = {
            f["name"]: f for f in self.config["stratification"]["mutable_features"]
        }

    def validate_action(self, feature: str, new_value: Any) -> Tuple[bool, str]:
        """Validates if an individual feature modification is actionable and within valid bounds."""
        if feature in self.forbidden_keys:
            return False, f"Violation: Feature '{feature}' is immutable or derived; modification strictly forbidden."

        if feature not in self.mutable_meta:
            return False, f"Violation: Feature '{feature}' is not recognized in mutable schema."

        meta = self.mutable_meta[feature]
        f_type = meta.get("type")

        if f_type == "categorical":
            allowed = meta.get("allowed_values", [])
            val_clean = str(new_value).strip("'\"")
            if val_clean not in allowed:
                return False, f"Violation: Value '{val_clean}' not in allowed categories {allowed} for '{feature}'."

        elif f_type == "continuous":
            min_v = meta.get("min_value", 0.0)
            max_v = meta.get("max_value", 1e6)
            try:
                v_num = float(new_value)
                if v_num < min_v or v_num > max_v:
                    return False, f"Violation: Value {v_num} out of bounds [{min_v}, {max_v}] for '{feature}'."
            except (ValueError, TypeError):
                return False, f"Violation: Non-numeric value for continuous feature '{feature}'."

        return True, "Valid"

    def filter_intervention_vector(self, proposed_vector: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Filters a proposed recourse intervention vector, stripping forbidden features
        and returning validated actionable changes along with violation logs.
        """
        valid_actions = {}
        violations = []

        for feat, val in proposed_vector.items():
            is_valid, msg = self.validate_action(feat, val)
            if is_valid:
                valid_actions[feat] = val
            else:
                violations.append(msg)

        return valid_actions, violations

    def get_action_cost(self, feature: str) -> float:
        """Returns the relative implementation cost weight w_j for a feature."""
        return self.mutable_meta.get(feature, {}).get("actionability_cost", 2.0)
