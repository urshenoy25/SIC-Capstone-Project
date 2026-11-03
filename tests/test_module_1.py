"""
Unit Tests for Module 1: Tabular Ingestion & Preprocessing Pipeline
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from module_1_data_ingestion.pipeline import TelcoDataPipeline


def test_feature_config_structure():
    """Verify feature_config.json contains required metadata and stratification rules."""
    config_path = "module_1_data_ingestion/feature_config.json"
    assert os.path.exists(config_path), "feature_config.json missing"
    
    with open(config_path, "r") as f:
        cfg = json.load(f)
        
    assert "metadata" in cfg
    assert "stratification" in cfg
    assert "immutable_features" in cfg["stratification"]
    assert "mutable_features" in cfg["stratification"]
    
    immutable_names = [f["name"] for f in cfg["stratification"]["immutable_features"]]
    assert "gender" in immutable_names
    assert "SeniorCitizen" in immutable_names
    assert "tenure" in immutable_names


def test_pipeline_fit_transform(sample_customer_record):
    """Test fitting and transforming tabular data via TelcoDataPipeline."""
    pipeline = TelcoDataPipeline()
    df = pd.DataFrame([sample_customer_record, sample_customer_record])
    df["Churn"] = ["Yes", "No"]
    
    X, y = pipeline.fit_transform(df)
    assert X.shape[0] == 2
    assert X.shape[1] == 19
    assert len(y) == 2
    assert pipeline.is_fitted_


def test_pipeline_single_record_transform(sample_customer_record):
    """Test transforming a single record DataFrame for model inference."""
    pipeline = TelcoDataPipeline()
    df = pd.DataFrame([sample_customer_record])
    df["Churn"] = ["Yes"]
    pipeline.fit(df)
    
    df_single = pd.DataFrame([sample_customer_record])
    x_transformed = pipeline.transform(df_single)
    assert isinstance(x_transformed, np.ndarray)
    assert x_transformed.shape == (1, 19)
