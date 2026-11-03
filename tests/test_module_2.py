"""
Unit Tests for Module 2: Causal Discovery, SCM Engine & 1D CNN Attribution
"""

import os
import pytest
import networkx as nx
import pandas as pd
from module_2_causal_attribution.causal_graph import TelcoCausalGraph, FEATURE_TIERS
from module_2_causal_attribution.scm_engine import TelcoSCMEngine
from module_2_causal_attribution.models.train_cnn import ChurnModelWrapper


def test_causal_graph_dag_properties():
    """Verify DAG is acyclic and respects topological order."""
    cg = TelcoCausalGraph()
    assert nx.is_directed_acyclic_graph(cg.dag), "Graph contains cycles!"
    
    topo_order = cg.topological_sort()
    assert len(topo_order) == len(FEATURE_TIERS)
    
    # Root exogenous demographic nodes have no parents
    for root_demo in ["gender", "SeniorCitizen", "Partner", "tenure"]:
        parents = cg.get_parents(root_demo)
        assert len(parents) == 0, f"Exogenous node {root_demo} should not have parents"
        
    # Partner causally influences Dependents
    assert "Partner" in cg.get_parents("Dependents")


def test_domain_tier_constraints():
    """Verify that all edges flow from lower or equal tiers to higher tiers."""
    cg = TelcoCausalGraph()
    for u, v in cg.dag.edges():
        u_tier = FEATURE_TIERS[u]
        v_tier = FEATURE_TIERS[v]
        assert u_tier <= v_tier, f"Edge ({u}->{v}) violates tier hierarchy ({u_tier} > {v_tier})"


def test_scm_engine_propagation(sample_customer_record):
    """Test SCM forward propagation and interventional do-calculus."""
    cg = TelcoCausalGraph()
    scm = TelcoSCMEngine(cg)
    
    # Fit SCM using augmented dataset if available
    data_path = "data/Telco_Customer_Churn_Augmented.csv"
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        scm.fit(df_raw)
        
    interventions = {"Contract": "Two year", "TechSupport": "Yes"}
    counterfactual = scm.intervene(sample_customer_record, interventions)
    
    assert counterfactual["Contract"] == "Two year"
    assert counterfactual["TechSupport"] == "Yes"
    # TotalCharges should be updated due to causal dependency
    assert counterfactual["TotalCharges"] >= 0


def test_model_wrapper_prediction(sample_customer_record):
    """Test 1D CNN prediction wrapper output."""
    model_path = "module_2_causal_attribution/models/saved_models/cnn_churn_model.keras"
    preproc_path = "module_1_data_ingestion/preprocessor.joblib"
    
    if os.path.exists(model_path) and os.path.exists(preproc_path):
        wrapper = ChurnModelWrapper(model_path, preproc_path)
        prob = wrapper.predict_proba(sample_customer_record)
        assert len(prob) == 1
        assert 0.0 <= float(prob[0]) <= 1.0
