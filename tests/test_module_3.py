"""
Unit Tests for Module 3: Actionable Causal Recourse Engine & Optimizer
"""

import os
import pytest
from module_3_recourse_engine.actionability_filter import ActionabilityFilter
from module_3_recourse_engine.optimizer import ConstrainedRecourseOptimizer
from module_3_recourse_engine.recourse_ranker import RecourseRanker
from module_2_causal_attribution.models.train_cnn import ChurnModelWrapper
from module_2_causal_attribution.causal_graph import TelcoCausalGraph
from module_2_causal_attribution.scm_engine import TelcoSCMEngine


def test_actionability_filter():
    """Verify mutability enforcement and immutable rejection."""
    filt = ActionabilityFilter("module_1_data_ingestion/feature_config.json")
    
    valid_c, _ = filt.validate_action("Contract", "One year")
    assert valid_c
    
    valid_g, msg_g = filt.validate_action("gender", "Male")
    assert not valid_g
    assert "immutable" in msg_g.lower()
    
    valid_t, msg_t = filt.validate_action("tenure", 50)
    assert not valid_t
    assert "immutable" in msg_t.lower()
    
    # Filter intervention vector
    filtered, violations = filt.filter_intervention_vector({"gender": "Male", "Contract": "Two year"})
    assert "gender" not in filtered
    assert filtered.get("Contract") == "Two year"
    assert len(violations) > 0


def test_recourse_ranker_generation(sample_customer_record):
    """Test generating ranked recourse recommendations."""
    model_path = "module_2_causal_attribution/models/saved_models/cnn_churn_model.keras"
    preproc_path = "module_1_data_ingestion/preprocessor.joblib"
    
    if not (os.path.exists(model_path) and os.path.exists(preproc_path)):
        pytest.skip("Model artifacts not present")
        
    wrapper = ChurnModelWrapper(model_path, preproc_path)
    cg = TelcoCausalGraph()
    scm = TelcoSCMEngine(cg)
    optimizer = ConstrainedRecourseOptimizer(model_wrapper=wrapper, scm=scm)
    ranker = RecourseRanker(optimizer)
    
    pathways = ranker.generate_top_recourse(sample_customer_record, top_k=3)
    assert isinstance(pathways, list)
    if len(pathways) > 0:
        top_pathway = pathways[0]
        assert "rank" in top_pathway
        assert "cost_score" in top_pathway
        assert "achieved_risk" in top_pathway
        assert "actions" in top_pathway
        if "interventions" in top_pathway:
            # Verify no immutable features were mutated
            for feat in top_pathway["interventions"].keys():
                assert feat not in ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]
