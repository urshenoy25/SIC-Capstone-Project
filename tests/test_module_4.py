"""
Integration & End-to-End Tests for Module 4: FastAPI REST API & Audit Database
"""

import os
import pytest
from fastapi.testclient import TestClient
from module_4_decision_dashboard.api.main import app
from module_4_decision_dashboard.audit_store.audit_db import AuditStore


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify system health and status reporting."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["causal_graph_nodes"] > 0
    assert data["causal_graph_edges"] > 0


def test_graph_endpoint(client):
    """Verify DAG topology endpoint returns nodes and directed edges."""
    response = client.get("/graph")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert "topological_order" in data


def test_predict_endpoint(client, sample_customer_record):
    """Verify predictive scoring endpoint."""
    response = client.post("/predict", json=sample_customer_record)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "audit_request_id" in data
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_explain_causal_endpoint(client, sample_customer_record):
    """Verify causal attribution calculation endpoint."""
    response = client.post("/explain/causal", json=sample_customer_record)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "top_causal_interventions" in data
    assert "comparative_attribution" in data


def test_recourse_generate_endpoint(client, sample_customer_record):
    """Verify actionable recourse pathway generation endpoint."""
    response = client.post("/recourse/generate", json=sample_customer_record)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "ranked_recourse_pathways" in data


def test_audit_store_logging():
    """Verify audit persistence in SQLite."""
    audit = AuditStore(db_path=":memory:")
    req_id = audit.log_decision(
        raw_features={"gender": "Male", "tenure": 10},
        prediction=1,
        probability=0.85,
        latency_ms=12.5
    )
    assert req_id is not None
    recent = audit.get_recent_audits(limit=5)
    assert len(recent) == 1
    assert recent[0]["request_id"] == req_id
    assert recent[0]["prediction"] == "Churn"
