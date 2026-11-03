"""
Module 3B: Causal Recourse Propagator
Simulates counterfactual customer states by propagating interventions through the SCM.
Ensures that secondary dependencies (e.g., plan downgrades altering billing) stay physically consistent.
"""

import os
import sys
from typing import Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from module_2_causal_attribution.scm_engine import TelcoSCMEngine
from module_3_recourse_engine.actionability_filter import ActionabilityFilter


class CausalRecoursePropagator:
    """
    Applies validated interventions and re-evaluates all downstream causal descendants
    to maintain strict SCM consistency.
    """

    def __init__(self, scm: Optional[TelcoSCMEngine] = None, filter_module: Optional[ActionabilityFilter] = None):
        self.scm = scm or TelcoSCMEngine()
        self.filter = filter_module or ActionabilityFilter()

    def propagate(self, factual_instance: Dict[str, Any], raw_interventions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filters out forbidden modifications, then executes Pearl's do-calculus intervention
        with downstream structural propagation.
        """
        valid_actions, violations = self.filter.filter_intervention_vector(raw_interventions)
        if violations:
            print(f"[Recourse Warning] Stripped {len(violations)} invalid actions: {violations}")

        counterfactual = self.scm.intervene(factual_instance, valid_actions)
        return counterfactual
