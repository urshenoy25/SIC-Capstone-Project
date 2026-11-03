"""
Module 4B: FastAPI RESTful Microservice
Exposes prediction, causal attribution, recourse generation, and DAG topology via asynchronous REST endpoints.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from module_2_causal_attribution.models.train_cnn import ChurnModelWrapper
from module_2_causal_attribution.causal_graph import TelcoCausalGraph
from module_2_causal_attribution.scm_engine import TelcoSCMEngine
from module_2_causal_attribution.attribution import CausalAttributionEngine
from module_3_recourse_engine.optimizer import ConstrainedRecourseOptimizer
from module_3_recourse_engine.recourse_ranker import RecourseRanker
from module_4_decision_dashboard.audit_store.audit_db import AuditStore

app = FastAPI(
    title="Causal Feature Attribution & Actionable Recourse API",
    description="High-Stakes Decision Support System integrating SCMs, Pearl's do-calculus, and Constrained Recourse Optimization.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Singletons
MODEL_PATH = os.path.join(BASE_DIR, "module_2_causal_attribution/models/saved_models/cnn_churn_model.keras")
PREPROC_PATH = os.path.join(BASE_DIR, "module_1_data_ingestion/preprocessor.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "data/Telco_Customer_Churn_Augmented.csv")

print("Initializing Causal Engine Singletons...")
model_wrapper = ChurnModelWrapper(MODEL_PATH, PREPROC_PATH)
causal_graph = TelcoCausalGraph()
scm_engine = TelcoSCMEngine(causal_graph)

# Fit SCM parameters from data
import pandas as pd
if os.path.exists(DATASET_PATH):
    df_raw = pd.read_csv(DATASET_PATH)
    scm_engine.fit(df_raw)

attribution_engine = CausalAttributionEngine(MODEL_PATH, PREPROC_PATH)
attribution_engine.scm = scm_engine

recourse_optimizer = ConstrainedRecourseOptimizer(model_wrapper=model_wrapper, scm=scm_engine)
recourse_ranker = RecourseRanker(recourse_optimizer)
audit_store = AuditStore()
print("All Engine Services Ready.")


# ── Pydantic Request & Response Schemas ──────────────────────────

class CustomerRecord(BaseModel):
    gender: str = Field(default="Female", description="Demographic gender ('Female', 'Male')")
    SeniorCitizen: int = Field(default=0, description="0 or 1")
    Partner: str = Field(default="No", description="'Yes' or 'No'")
    Dependents: str = Field(default="No", description="'Yes' or 'No'")
    tenure: int = Field(default=12, description="Months as subscriber [0-72]")
    PhoneService: str = Field(default="Yes", description="'Yes' or 'No'")
    MultipleLines: str = Field(default="No", description="'No', 'Yes', 'No phone service'")
    InternetService: str = Field(default="Fiber optic", description="'DSL', 'Fiber optic', 'No'")
    OnlineSecurity: str = Field(default="No", description="'No', 'Yes', 'No internet service'")
    OnlineBackup: str = Field(default="No", description="'No', 'Yes', 'No internet service'")
    DeviceProtection: str = Field(default="No", description="'No', 'Yes', 'No internet service'")
    TechSupport: str = Field(default="No", description="'No', 'Yes', 'No internet service'")
    StreamingTV: str = Field(default="No", description="'No', 'Yes', 'No internet service'")
    StreamingMovies: str = Field(default="No", description="'No', 'Yes', 'No internet service'")
    Contract: str = Field(default="Month-to-month", description="'Month-to-month', 'One year', 'Two year'")
    PaperlessBilling: str = Field(default="Yes", description="'Yes' or 'No'")
    PaymentMethod: str = Field(default="Electronic check", description="Billing payment channel")
    MonthlyCharges: float = Field(default=75.0, description="Monthly charges in USD")
    TotalCharges: Optional[float] = Field(default=None, description="Optional; derived if omitted")


class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: int
    churn_probability: float
    confidence: float
    latency_ms: float
    audit_request_id: str


# ── REST API Endpoints ──────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Causal Decision Support Framework",
        "model_architecture": "1D Tabular Convolutional Neural Network",
        "causal_graph_nodes": causal_graph.dag.number_of_nodes(),
        "causal_graph_edges": causal_graph.dag.number_of_edges()
    }


@app.get("/graph")
def get_causal_graph():
    """Returns causal DAG nodes, directed edges, and topological order."""
    return causal_graph.export_graph_dict()


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(record: CustomerRecord):
    """Computes model risk prediction and logs event to audit database."""
    t0 = time.time()
    data = record.model_dump()
    if data.get("TotalCharges") is None:
        data["TotalCharges"] = data["tenure"] * data["MonthlyCharges"]

    prob = float(model_wrapper.predict_proba(data)[0])
    pred_code = int(prob >= 0.50)
    pred_label = "Churn (Adverse)" if pred_code == 1 else "No Churn (Favorable)"
    confidence = round(prob if pred_code == 1 else (1.0 - prob), 4)
    latency = round((time.time() - t0) * 1000, 2)

    req_id = audit_store.log_decision(
        raw_features=data,
        prediction=pred_code,
        probability=prob,
        latency_ms=latency
    )

    return PredictionResponse(
        prediction=pred_label,
        prediction_code=pred_code,
        churn_probability=round(prob, 4),
        confidence=confidence,
        latency_ms=latency,
        audit_request_id=req_id
    )


@app.post("/explain/causal")
def explain_causal(record: CustomerRecord):
    """Computes Individual Causal Effect (ICE) attributions and contrasts against SHAP."""
    data = record.model_dump()
    if data.get("TotalCharges") is None:
        data["TotalCharges"] = data["tenure"] * data["MonthlyCharges"]

    try:
        comparison_df = attribution_engine.compare_causal_vs_shap(data)
        ice_df = attribution_engine.compute_ice(data)
        return {
            "status": "success",
            "top_causal_interventions": ice_df.head(6).to_dict(orient="records"),
            "comparative_attribution": comparison_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recourse/generate")
def generate_recourse(record: CustomerRecord, top_k: int = 3):
    """Generates minimum-cost, actionable recourse pathways flipping adverse decisions."""
    data = record.model_dump()
    if data.get("TotalCharges") is None:
        data["TotalCharges"] = data["tenure"] * data["MonthlyCharges"]

    try:
        pathways = recourse_ranker.generate_top_recourse(data, top_k=top_k)
        return {
            "status": "success",
            "customer_baseline": {
                "tenure": data["tenure"],
                "Contract": data["Contract"],
                "MonthlyCharges": data["MonthlyCharges"]
            },
            "pathways_count": len(pathways),
            "ranked_recourse_pathways": pathways
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audits")
def get_audit_trail(limit: int = 20):
    """Retrieves decision audit trail for compliance verification."""
    return {
        "count": limit,
        "audits": audit_store.get_recent_audits(limit=limit)
    }
