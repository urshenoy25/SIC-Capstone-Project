# Causal Feature Attribution for High-Stakes Tabular Decisions with Actionable Recourse

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)]()

> **Samsung Innovation Campus (SIC) Capstone Project**  
> An end-to-end Explainable AI (XAI) and Decision-Support Framework integrating Structural Causal Models (SCMs), Pearl's *do*-calculus, and constrained non-linear mathematical optimization for high-stakes tabular decision systems.

---

## 1. Executive Summary

Standard post-hoc explainability methods—such as **SHAP** (Shapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations)—rely purely on statistical associations and conditional expectations:

$$\mathbb{E}[Y \mid X = x]$$

In the presence of multicollinearity or latent confounding, correlational methods routinely misattribute importance to non-causal features. Furthermore, standard counterfactual frameworks (e.g., vanilla DiCE) generate recommendations by minimizing mathematical $L_1$/$L_2$ distance in unstructured feature space, recommending impossible actions (e.g., reducing age) or ignoring cascading downstream dependencies (e.g., modifying service tiers without updating billing charges).

This framework resolves these fundamental flaws through:
1. **Structural Causal Models (SCMs):** Modeling data-generating mechanisms as Directed Acyclic Graphs (DAGs) with strict domain tier hierarchies.
2. **Pearl's *do*-calculus Interventions:** Computing true Individual and Average Causal Effects (ICE/ACE) under $P(Y \mid \text{do}(X = x'))$.
3. **Constrained Actionable Recourse Optimization:** Finding minimum-cost, actionable intervention vectors $\delta^*$ that guarantee zero mutations to protected immutable features and propagate cascading downstream effects.
4. **Production Microservice & Dashboard:** Providing an asynchronous FastAPI REST backend, an SQLite regulatory audit trail, and an interactive Streamlit analyst interface.

---

## 2. System Architecture

```
                                [ Client / Compliance Officer ]
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │   FastAPI Service Layer (REST)  │
                             │        (Port :8000)             │
                             └────────────────┬────────────────┘
                                              │
                     ┌────────────────────────┴───────────────────────┐
                     ▼                                                ▼
          ┌────────────────────────┐                       ┌────────────────────────┐
          │  Predictive Inference  │                       │   Causal Attribution   │
          │  (1D ConvNet Model)    │                       │    (Pearl's do-calc)   │
          └──────────┬─────────────┘                       └──────────┬─────────────┘
                     │                                                │
                     └────────────────────────┬───────────────────────┘
                                              ▼
                             ┌─────────────────────────────────┐
                             │   Constrained Recourse Engine   │
                             │  (Actionability + Propagation)  │
                             └────────────────┬────────────────┘
                                              ▼
                             ┌─────────────────────────────────┐
                             │    Streamlit Analyst Portal     │
                             │        (Port :8501)             │
                             └─────────────────────────────────┘
```

---

## 3. Repository Organization

```
SIC Capstone/
├── module_1_data_ingestion/            # Module 1: Ingestion, Validation & Stratification
│   ├── pipeline.py                     # TelcoDataPipeline with schema validation & scaling
│   ├── feature_config.json             # Stratification config (immutable, mutable, derived)
│   ├── preprocessor.joblib             # Fitted scikit-learn preprocessor artifact
│   └── generate_coherent_dataset.py    # Synthetic & augmented data generator
│
├── module_2_causal_attribution/        # Module 2: Causal Discovery, SCM & Deep Model
│   ├── causal_graph.py                 # TelcoCausalGraph DAG definition & tier rules
│   ├── scm_engine.py                   # Structural Causal Model & Pearl's do-calculus
│   ├── attribution.py                  # CausalAttributionEngine (ICE/ACE vs SHAP)
│   └── models/                         # 1D Tabular CNN architecture & trained weights
│       ├── CNN_telco.py
│       ├── train_cnn.py                # Model training routine and inference wrapper
│       └── saved_models/
│           └── cnn_churn_model.keras   # Serialized deep learning checkpoint
│
├── module_3_recourse_engine/           # Module 3: Constrained Actionable Recourse
│   ├── actionability_filter.py         # Enforces zero protected trait mutations
│   ├── causal_propagator.py            # Propagates downstream causal perturbations
│   ├── optimizer.py                    # Non-linear constrained recourse optimizer
│   └── recourse_ranker.py              # Diverse top-k pathway generator & cost ranker
│
├── module_4_decision_dashboard/        # Module 4: Production Microservice & UI
│   ├── api/
│   │   └── main.py                     # Asynchronous FastAPI REST microservice
│   ├── dashboard/
│   │   └── app.py                      # Interactive Streamlit compliance dashboard
│   └── audit_store/
│       ├── audit_db.py                 # SQLite regulatory persistence layer
│       └── audit_trail.db              # Timestamped decision audit database
│
├── data/                               # Canonical and augmented tabular datasets
│   ├── Telco_Customer_Churn.csv
│   └── Telco_Customer_Churn_Augmented.csv
│
├── tests/                              # Automated Pytest Suite (15/15 passing)
│   ├── conftest.py                     # Shared fixtures & mock records
│   ├── test_module_1.py                # Schema validation & preprocessor tests
│   ├── test_module_2.py                # DAG acyclicity, SCM equations & CNN inference
│   ├── test_module_3.py                # Actionability filter & recourse optimizer tests
│   └── test_module_4.py                # REST API endpoints & audit trail tests
│
├── docs/                               # Presentations, technical reports & literature
├── Dockerfile                          # Multi-stage production container definition
├── docker-compose.yml                  # Service orchestration (API + Dashboard)
├── pyproject.toml                      # PEP 517/621 packaging metadata
├── requirements.txt                    # Pinned project dependencies
├── run.sh                              # Unified command-line runner script
└── README.md                           # Master project documentation
```

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.14)
- (Optional) Docker & Docker Compose

### Option A: Unified CLI Runner (Recommended)

The project includes an executable script `run.sh` to manage all tasks:

```bash
# Display help and available commands
./run.sh help

# Run complete automated test suite
./run.sh test

# Launch the FastAPI REST service on port 8000
./run.sh api

# Launch the Streamlit Interactive Analyst UI on port 8501
./run.sh dashboard

# Launch both FastAPI backend and Streamlit dashboard concurrently
./run.sh all
```

### Option B: Docker Compose Deployment

Spin up the entire architecture (FastAPI backend + Streamlit dashboard) with health checks:

```bash
# Build and run containers
docker-compose up --build

# In background mode
docker-compose up -d

# Stop services
docker-compose down
```
- **FastAPI Documentation:** `http://localhost:8000/docs`
- **Streamlit Dashboard:** `http://localhost:8501`

---

## 5. REST API Specification

The FastAPI microservice exposes full decision support and auditability:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check, node/edge counts, and model metadata. |
| `GET` | `/graph` | Causal DAG topology (nodes, directed edges, topological order). |
| `POST` | `/predict` | Evaluates churn probability using the 1D Tabular CNN and logs event to audit DB. |
| `POST` | `/explain/causal` | Calculates Individual Causal Effect (ICE) attributions contrasted against SHAP. |
| `POST` | `/recourse/generate`| Solves constrained optimization for minimum-cost actionable recourse pathways. |
| `GET` | `/audits` | Retrieves timestamped decision audits for regulatory compliance. |

### Sample Prediction Payload

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "No",
       "Dependents": "No",
       "tenure": 6,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 85.0
     }'
```

---

## 6. Verification & Automated Testing

The project is covered by a 15-test automated validation suite checking:
- **Module 1:** Strict schema compliance, categorical encoding, standard scaling, and missing-value imputation.
- **Module 2:** Directed acyclic graph verification (`nx.is_directed_acyclic_graph`), domain tier hierarchy constraints, SCM interventional equation updates, and CNN inference.
- **Module 3:** Zero protected-trait mutation enforcement (rejecting illegal edits to `gender`, `SeniorCitizen`, `tenure`), and cost-optimal recourse ranking.
- **Module 4:** All FastAPI HTTP endpoints (`/health`, `/graph`, `/predict`, `/explain/causal`, `/recourse/generate`, `/audits`) and SQLite persistence.

```bash
./run.sh test
```

---

## 7. Key Literature & Theoretical References

1. **Wachter, S., Mittelstadt, B., & Russell, C. (2017).** Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR. *Harvard Journal of Law & Technology*.
2. **Karimi, A. H., von Kügelgen, J., Schölkopf, B., & Valera, I. (2021).** Algorithmic Recourse Under Imperfect Causal Knowledge. *NeurIPS 2021*.
3. **Ustun, B., Spangher, A., & Liu, Y. (2019).** Actionable Recourse in Linear Classification. *ACM FAT\**.
4. **Lundberg, S. M., & Lee, S. I. (2017).** A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017 (SHAP)*.
5. **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

---

## 8. Authors & Acknowledgments

- **Program:** Samsung Innovation Campus (SIC) Capstone Project  
- **Domain:** Explainable AI (XAI) & Causal Decision Support Systems  
- **Year:** 2026
