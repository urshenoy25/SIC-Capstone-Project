# Causal Feature Attribution for High-Stakes Tabular Decisions with Actionable Recourse

## Technical Overview & System Requirements Specification

---

### Document Control & Metadata

| Field | Detail |
| :--- | :--- |
| **Project Title** | Causal Feature Attribution for High-Stakes Tabular Decisions with Actionable Recourse |
| **Domain** | Explainable AI (XAI), Causal Inference, Decision Support Systems (DSS) |
| **Target Application** | High-Stakes Tabular Decisions (Credit Risk, Healthcare, Admissions) |
| **Document Type** | Project Overview & Technical Requirements Document (TRD) |
| **Version** | 1.0.0 |
| **Date** | November 2026 |

---

## 1. Executive Summary

Modern deep learning and ensemble machine learning models (e.g., XGBoost, LightGBM, Random Forests) are extensively deployed in high-stakes socio-technical domains including financial lending, medical diagnostics, university admissions, and employment screening. Despite achieving high predictive accuracy, these models function as black boxes. 

Current post-hoc explainability standards—predominantly **SHAP (Shapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**—derive feature attributions based on statistical associations (correlations) rather than causal mechanisms. In the presence of confounding variables or multicollinearity, correlational methods routinely misattribute importance, leading to ungrounded explanations. Furthermore, when an adverse decision is rendered (e.g., loan or admission denial), existing counterfactual explanation frameworks generate recommendations by minimizing mathematical distance in feature space without enforcing real-world constraints. Consequently, they recommend physically impossible actions (e.g., reducing age) or ignore cascading causal relationships between features (e.g., changing income without propagating changes to savings and debt ratio).

This project designs and implements an end-to-end **Explainable AI and Causal Decision-Support Framework**. The system integrates **Structural Causal Models (SCMs)**, **interventional do-calculus**, and **constrained optimization** to:
1. Learn and represent data-generating mechanisms as Directed Acyclic Graphs (DAGs).
2. Compute causally grounded feature attributions that reflect the true interventional impact on decisions.
3. Generate minimum-cost, actionable, and causally feasible recourse paths for unfavorable predictions.
4. Deliver these capabilities through a production-ready, asynchronous API and an interactive audit dashboard.

---

## 2. Problem Statement & Motivation

### 2.1 The Core Gaps

```
+----------------------------------------------------------------------------------------------------+
|                                    CURRENT STATE VS. CHALLENGES                                    |
+------------------------------+------------------------------------+--------------------------------+
| Dynamic Aspect               | Current Industry Practice          | Failure / Consequence          |
+------------------------------+------------------------------------+--------------------------------+
| Decision Transparency        | High-capacity black-box models     | Complete lack of operational   |
|                              | without mechanistic transparency.  | trust and auditability.        |
+------------------------------+------------------------------------+--------------------------------+
| Causal Validity              | Post-hoc correlational methods     | Identifies symptoms rather     |
|                              | (SHAP, LIME).                      | than actual causal drivers.    |
+------------------------------+------------------------------------+--------------------------------+
| Recourse Feasibility         | Unconstrained nearest-neighbor     | Recommends unrealistic or      |
|                              | counterfactuals (e.g., DiCE).      | impossible adjustments.        |
+------------------------------+------------------------------------+--------------------------------+
| Feature Constraints          | Features treated as independent    | Violates domain laws; suggests |
|                              | and uniformly mutable.             | changes to immutable traits.   |
+------------------------------+------------------------------------+--------------------------------+
```

### 2.2 Formal Justification for Causality
Statistical prediction answers observational queries of the form:
$$P(Y = 1 \mid X = x)$$
However, decision-making and recourse require answering interventional queries under Pearl's causal hierarchy:
$$P(Y = 1 \mid \text{do}(X_i = x_i'))$$
Evaluating an intervention requires knowing how intervening on variable $X_i$ propagates through its causal descendants in the feature graph $\mathcal{G}$. A purely predictive model cannot distinguish whether $X_i \to Y$ or whether an unobserved confounder $U \to \{X_i, Y\}$ induces a spurious statistical correlation.

---

## 3. Project Aim & Measurable Objectives

### 3.1 Primary Aim
To architect, evaluate, and deploy an Explainable AI framework utilizing Structural Causal Models and constrained mathematical optimization to derive causally valid feature attributions and generate actionable, realistic recourse recommendations for high-stakes tabular decision systems.

### 3.2 Specific Objectives
1. **Data Ingestion & Feature Stratification Pipeline:**
   Build an automated tabular pipeline that validates schema constraints, imputes missing records, normalizes features, and stratifies variables into mutable (actionable) and immutable (protected/fixed) categories.
2. **Causal Graph Discovery & SCM Formulation:**
   Construct a Directed Acyclic Graph (DAG) using constraint-based (PC algorithm) and score-based/continuous optimization algorithms (NOTEARS) combined with expert domain prior constraints.
3. **Causal Attribution Engine:**
   Implement interventional attribution metrics using Pearl's do-calculus and Average/Individual Causal Effect (ACE/ICE) formulations, benchmarking outputs against conventional SHAP and LIME values.
4. **Constrained Actionable Recourse Generation:**
   Formulate and solve a constrained non-linear optimization task that computes the minimum-cost feature intervention vector $\delta^*$ required to flip an adverse decision, ensuring causal consistency and physical feasibility.
5. **REST API & Interactive Analyst Dashboard:**
   Develop an asynchronous FastAPI microservice backed by PostgreSQL for persistence and auditability, integrated with an interactive Streamlit/React decision-support user interface.

---

## 4. System Architecture & End-to-End Workflow

### 4.1 Architectural Block Diagram (Aligned with PPT Presentation)

```
                               [ User / Client Application ]
                                             │
                                             ▼
                            ┌─────────────────────────────────┐
                            │   FastAPI Service Layer (REST)  │
                            └────────────────┬────────────────┘
                                             │
                    ┌────────────────────────┴───────────────────────┐
                    ▼                                                ▼
         ┌────────────────────────┐                       ┌────────────────────────┐
         │ PostgreSQL Audit Store │                       │ Streamlit UI Dashboard │
         └────────────────────────┘                       └────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                  CORE ENGINE PIPELINE                                    │
│                                                                                          │
│  ┌─────────────────────────┐         ┌────────────────────────────────────────────────┐  │
│  │        MODULE 1         │         │                    MODULE 2                    │  │
│  │ Data Ingestion &        │ ──────> │ Causal Discovery & Attribution Engine (SCM)    │  │
│  │ Preprocessing           │         │ (Classifier + DAG Learning + do-calculus)      │  │
│  └─────────────────────────┘         └───────────────────────┬────────────────────────┘  │
│                                                              │                           │
│                                                              ▼                           │
│  ┌─────────────────────────┐         ┌────────────────────────────────────────────────┐  │
│  │        MODULE 4         │         │                    MODULE 3                    │  │
│  │ Decision-Support        │ <────── │ Recourse Engine & Ranking                      │  │
│  │ Dashboard (UI & API)    │         │ (Constrained Optimization + Actionability)     │  │
│  └─────────────────────────┘         └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module-by-Module Technical Breakdown (Aligned with PPT.pdf)

#### Module 1: Data Ingestion & Preprocessing
- **Role:** End-to-end ingestion, validation, imputation, and feature stratification.
- **Input:** Raw tabular application records ($\mathbf{x} \in \mathbb{R}^d$).
- **Functionality:**
  - Automated schema enforcement, data type checking, and range validation.
  - Distribution-preserving missing-value imputation.
  - Robust standardization and categorical encoding.
  - Data augmentation and synthesis for high-fidelity class balance.
  - Explicit Feature Stratification:
    - **Immutable Variables ($\mathcal{I}$):** Age, gender, ethnicity, marital status, historical delinquency (strictly protected from intervention).
    - **Mutable Actionable Variables ($\mathcal{M}$):** Payment terms, add-on services, monthly commitments, credit utilization (open to intervention).

#### Module 2: Causal Discovery & Attribution Engine (SCM)
- **Role:** Train high-performance predictive classifiers, discover the underlying causal DAG, fit structural equations, and compute interventional attributions.
- **Predictive Engine:** 1D Tabular Convolutional Neural Network (CNN) / Ensemble Classifiers (XGBoost, Random Forest) trained to maximize class-discriminative power ($\text{AUC-ROC} \ge 0.85$).
- **Causal Discovery & SCM:**
  - Employs the **PC Algorithm** / **NOTEARS** augmented with domain prior constraints (tiered causal ordering, forbidden reverse edges).
  - Fits Structural Causal Model (SCM) parameters:  = f_j(\text{PA}_j, U_j)$, where $\text{PA}_j$ denotes causal parents of node $ in DAG $\mathcal{G}$.
- **Causal Feature Attribution:**
  - Computes Average Causal Effect (ACE) and Individual Causal Effect (ICE) via Pearl's do-calculus:
    30615\Delta_j = \mathbb{E}\left[Y \mid \text{do}(X_j = x_j + \delta)\right] - \mathbb{E}\left[Y \mid \text{do}(X_j = x_j)\right]30615
  - Contrasts against correlational SHAP and LIME values to expose spurious associations and confounding.

#### Module 3: Recourse Engine & Ranking
- **Role:** Generate feasible, realistic, minimum-cost counterfactual interventions that flip adverse decisions.
- **Actionability Filtering:** Strictly guarantees zero recommendations demanding changes to immutable attributes (^{\text{new}} = x_i, \forall i \in \mathcal{I}$).
- **Causal Consistency & Downstream Propagation:** When an actionable variable  \in \mathcal{M}$ is intervened upon, its causal descendants $\text{Desc}(X_j)$ automatically propagate through the fitted SCM equations.
- **Constrained Recourse Formulation:** Solves the non-linear optimization program:
  30615\min_{\boldsymbol{\theta}} \sum_{j \in \mathcal{M}} w_j \cdot \text{cost}_j(\theta_j) + \lambda \cdot \text{Plausibility}(\mathbf{x}^{\text{new}})30615
  subject to:
  30615\begin{cases} 
  f(\mathbf{x}^{\text{new}}) = 1 & \text{(Decision flips to favorable)} \\
  x_i^{\text{new}} = x_i, \quad \forall i \in \mathcal{I} & \text{(Immutable variables remain fixed)} \\
  \mathbf{x}^{\text{new}} = \text{SCM}(\mathbf{x}, \text{do}(\mathbf{X}_{\mathcal{M}} = \boldsymbol{\theta})) & \text{(Causal consistency holds)} \\
  \mathbf{l}_j \le x_j^{\text{new}} \le \mathbf{u}_j, \quad \forall j \in \mathcal{M} & \text{(Real-world bounding limits)}
  \end{cases}30615
- **Solver & Ranking:** Solved using Sequential Least Squares Programming (**SLSQP**) via `scipy.optimize`, ranking viable recourse pathways by aggregate implementation effort.

#### Module 4: Decision-Support Dashboard & API (Analyst/End-User UI)
- **Backend Service:** FastAPI exposes asynchronous REST endpoints:
  - `/predict`: Model prediction and calibrated probabilities.
  - `/explain/causal`: Dual attribution rankings (Causal vs. SHAP) and DAG path extraction.
  - `/recourse/generate`: Top-$ feasible, cost-ranked recourse recommendations.
- **Persistence Store:** PostgreSQL / SQLite logs each decision event, explanation vector, and audit trail with UUIDs and timestamps.
- **Interactive Dashboard:** Streamlit decision-support UI providing:
  - Verdict overview (decision probability, risk score).
  - Interactive DAG visualizer displaying causal parent-child relationships.
  - Comparative attribution bar plots (Causal ACE vs. SHAP).
  - Step-by-step actionable recourse checklist with what-if sliders.

---

### 4.3 Architecture-Aligned Repository Hierarchy

To strictly maintain the modular design defined in `PPT.pdf`, the repository follows a 4-module directory structure:

```
SIC Capstone/
│
├── data/                                      # Datasets (Raw, Augmented, and Processed)
│   ├── Telco_Customer_Churn.csv               # Baseline customer churn dataset
│   ├── Telco_Customer_Churn_Augmented.csv     # Augmented balanced training set (18k samples)
│   └── fico_heloc.csv                         # FICO HELOC benchmark dataset
│
├── docs/                                      # Research & Presentation Documentation
│   ├── PPT.pdf                                # Project Presentation & Architecture Deck
│   ├── Capstone Project.pdf                   # Simplified Concept & Viva Guide
│   ├── Literature Survey.pdf                  # Related works and comparative analysis
│   └── RS Papers/                             # Core research reference papers
│
├── module_1_data_ingestion/                   # MODULE 1: Data Ingestion & Preprocessing
│   ├── __init__.py
│   ├── download_dataset.py                    # Automated dataset fetchers (OpenML / UCI)
│   ├── generate_coherent_dataset.py           # Domain-consistent synthetic data generator
│   ├── generate_synthetic_v2.py               # Statistical distribution-preserving augmentation
│   └── feature_config.json                    # Mutable vs. Immutable stratification schema
│
├── module_2_causal_attribution/               # MODULE 2: Causal Discovery & Attribution Engine
│   ├── __init__.py
│   ├── models/                                # Predictive Classifiers & Benchmarks
│   │   ├── CNN_telco_2.py                     # 1D Convolutional Neural Network
│   │   ├── benchmark_models.py                # Logistic Regression, Random Forest, XGBoost
│   │   ├── final_benchmark.py                 # Stacking ensemble and feature engineering
│   │   └── saved_models/                      # Serialized weights (.keras) & scalers (.joblib)
│   ├── causal_graph.py                        # DAG learning (PC / NOTEARS / Expert prior DAG)
│   ├── scm_engine.py                          # Structural Equation fitting & do-calculus
│   └── attribution.py                         # ACE/ICE causal attributions vs. SHAP benchmarks
│
├── module_3_recourse_engine/                  # MODULE 3: Recourse Engine & Ranking
│   ├── __init__.py
│   ├── optimizer.py                           # SciPy SLSQP constrained recourse solver
│   ├── actionability_filter.py                # Strict immutability constraint validator
│   ├── causal_propagator.py                   # Downstream effect propagation across DAG
│   └── recourse_ranker.py                     # Feasibility scoring & cost-based ranking
│
├── module_4_decision_dashboard/               # MODULE 4: Decision-Support Dashboard & API
│   ├── __init__.py
│   ├── api/                                   # FastAPI REST microservice
│   │   └── main.py                            # Endpoints (/predict, /explain, /recourse)
│   ├── dashboard/                             # Streamlit Interactive Web Application
│   │   └── app.py                             # Decision audit console & interactive sliders
│   └── audit_store/                           # Audit logging database models & SQLite/Postgres
│
├── Project_Overview_and_Technical_Requirements.md
└── requirements.txt
```

---

## 5. Technical Requirements Specification

### 5.1 Hardware Requirements

| Parameter | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 / AMD Ryzen 5 (4 Cores, 8 Threads) | Intel Core i7 / AMD Ryzen 7 (8 Cores, 16 Threads) |
| **RAM** | 8 GB DDR4 | 16 GB DDR4/DDR5 |
| **Storage** | 20 GB free SSD storage | 50 GB NVMe M.2 SSD |
| **GPU** | Not mandatory (CPU inference supported) | NVIDIA RTX 3060 (6GB VRAM) for accelerated graph training |
| **Network** | Broadband internet for dependency downloads | Stable connection for API endpoints |

### 5.2 Software Requirements & Environments

| Component | Standard / Specification |
| :--- | :--- |
| **Operating System** | Linux (Ubuntu 22.04 LTS / Debian 12) or Windows 10/11 (64-bit with WSL2) |
| **Python Environment** | Python 3.10 or 3.11 (Managed via `venv` or `conda`) |
| **Database Engine** | PostgreSQL 14+ / SQLite 3 (for standalone development) |
| **Containerization** | Docker Engine 24.0+ & Docker Compose v2 (Optional for deployment) |
| **Web Browser** | Google Chrome 110+, Mozilla Firefox 110+, or Chromium-based browsers |

### 5.3 Technology Stack & Library Matrix

```
+───────────────────────────────────────────────────────────────────────────────────────+
|                                    SOFTWARE STACK                                     |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Layer / Category          | Technology Selected       | Role / Technical Purpose      |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Core Programming Language | Python 3.10+              | System implementation         |
| Data Manipulation         | NumPy (>=1.23),           | Vector computation & tabular  |
|                           | Pandas (>=2.0)            | preprocessing pipelines       |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Predictive ML Engine      | Scikit-Learn (>=1.2),     | Base classification models,   |
|                           | XGBoost (>=1.7),          | cross-validation, and metrics |
|                           | LightGBM (>=3.3)          | evaluation                    |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Causal Discovery & SCM    | causal-learn (>=0.1.3),   | Constraint/score-based DAG    |
|                           | DoWhy (>=0.10)            | learning & interventional SCM |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Recourse Optimization     | SciPy (>=1.10)            | Constrained non-linear        |
|                           | (`optimize.minimize`)     | optimization (SLSQP solver)   |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Comparative XAI           | SHAP (>=0.42),            | Correlational benchmark       |
|                           | Matplotlib / Seaborn      | generation & static plotting  |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Web API Framework         | FastAPI (>=0.100),        | Asynchronous REST endpoint    |
|                           | Uvicorn (>=0.22)          | hosting and request routing   |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Interactive Dashboard     | Streamlit (>=1.25),       | Web UI, real-time sliders,    |
|                           | Graphviz / PyVis          | and dynamic DAG rendering     |
+───────────────────────────+───────────────────────────+───────────────────────────────+
| Database & Audit Logging  | PostgreSQL / SQLite,      | Relational storage for logs,  |
|                           | SQLAlchemy (>=2.0)        | predictions, and audit trail  |
+───────────────────────────+───────────────────────────+───────────────────────────────+
```

---

## 6. Functional & Non-Functional Requirements

### 6.1 Functional Requirements (FR)

- **FR-1: Tabular Ingestion & Validation**
  The system must ingest structured CSV/JSON payloads, enforce explicit data typing, validate against schema ranges, and reject malformed records with descriptive error codes.
- **FR-2: Automated Causal DAG Discovery**
  The system must generate a valid Directed Acyclic Graph (DAG) from training data, allowing domain experts to introduce forbidden/mandatory directed edges.
- **FR-3: Dual Attribution Generation**
  For any single inference request, the system must produce both conventional correlational attribution scores (SHAP) and interventional causal attribution scores (DoWhy SCM) to allow direct comparison.
- **FR-4: Constrained Actionable Recourse Optimization**
  When a prediction falls below the approval threshold, the system must output at least 2 distinct, causally validated intervention vectors, modifying only variables marked as mutable.
- **FR-5: Downstream Consequence Propagation**
  When an actionable variable $X_j$ is modified, all its descendants $\text{Desc}(X_j)$ in the causal graph must automatically recompute based on structural equations, preventing inconsistent state vectors.
- **FR-6: Audit Trail & Compliance Logging**
  Every decision event (raw features, model probability, DAG version, generated recourse vectors) must be persisted with a UUID and UTC timestamp into the database.

### 6.2 Non-Functional Requirements (NFR)

- **NFR-1: Inference Latency**
  - Base model prediction: $\le 50 \text{ ms}$.
  - Causal attribution generation: $\le 1.5 \text{ seconds}$.
  - Constrained recourse optimization: $\le 3.0 \text{ seconds}$ per query instance.
- **NFR-2: Scalability & Memory Footprint**
  The core pipeline must execute within 4 GB of active RAM during steady-state serving without memory leakage over continuous invocations.
- **NFR-3: Actionability Fidelity**
  Zero recommendations may be generated that demand changes to immutable features (0% violation rate across test benchmarks).
- **NFR-4: Extensibility & Modularity**
  The architecture must be decoupled into standalone modules (Pipelines, Causal Engines, Solvers, APIs), allowing independent swapping of the base classifier (e.g., replacing XGBoost with a Deep Neural Network) without modifying the recourse solver.

---

## 7. Supported Datasets & Feature Breakdown

To ensure domain validity and reproducibility without third-party authentication barriers, the framework supports the following numerical tabular benchmarks:

### Benchmark 1: FICO HELOC (Home Equity Line of Credit) — *Primary Benchmark*
- **Source:** FICO Explainable Machine Learning Challenge
- **Profile:** 10,459 rows $\times$ 24 numerical attributes.
- **Target:** `RiskPerformance` (Binary: 1 = Good / Non-delinquent, 0 = Bad / 90+ days delinquent).
- **Feature Breakdown:**
  - *Mutable Features:* Number of Satisfactory Trades, Net Fraction Revolving Burden, Number of Inquiries in Last 6 Months, Months Since Most Recent Inactivity.
  - *Immutable Features:* Months Since Oldest Trade Opened, Number of Delinquent Trades.

### Benchmark 2: Default of Credit Card Clients (Taiwan Benchmark)
- **Source:** UCI Machine Learning Repository / OpenML (ID: 42477)
- **Profile:** 30,000 rows $\times$ 24 numerical attributes.
- **Target:** `default.payment.next.month` (Binary: 0 = Non-default, 1 = Default).
- **Feature Breakdown:**
  - *Mutable Features:* Payment History Status across past 6 months (`PAY_0` to `PAY_6`), Bill Statement Amounts, Previous Payment Amounts.
  - *Immutable Features:* `AGE`, `SEX`, `EDUCATION`, `MARRIAGE`.

---

## 8. Implementation Roadmap & Project Lifecycle

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PROJECT PHASES TIMELINE                       │
└────────────────────────────────────────────────────────────────────────┘

Phase 1: Environment Setup & Data Pipeline (Weeks 1 - 3)
 ├── Virtual environment configuration & library installation
 ├── Dataset collection (FICO HELOC / Credit Card Default)
 ├── Exploratory data analysis (EDA) & missing value imputation
 └── Feature categorization schema (Mutable vs. Immutable JSON definitions)

Phase 2: Predictive Baseline & Causal Graph Discovery (Weeks 4 - 6)
 ├── Baseline classifier training (Logistic Regression, Random Forest, XGBoost)
 ├── Model performance evaluation (ROC-AUC, Precision, Recall, F1)
 ├── Causal discovery execution using causal-learn (PC Algorithm / NOTEARS)
 └── Expert constraint integration and DAG validation

Phase 3: Causal Attribution & Recourse Optimization Engine (Weeks 7 - 10)
 ├── Interventional do-calculus module development (DoWhy integration)
 ├── Comparative XAI analysis (Causal Attribution vs. SHAP/LIME)
 ├── Constrained recourse formulation using SciPy SLSQP optimization
 └── Downstream causal dependency propagation implementation

Phase 4: API Microservice & Dashboard Development (Weeks 11 - 13)
 ├── FastAPI asynchronous endpoint architecture development
 ├── PostgreSQL relational schema design and SQLAlchemy ORM setup
 ├── Interactive Streamlit frontend development
 └── End-to-end integration (Data Ingestion -> Prediction -> Recourse -> UI)

Phase 5: Benchmarking, Validation & Documentation (Weeks 14 - 16)
 ├── Recourse cost and feasibility benchmarking across test instances
 ├── Latency stress testing and resource profiling
 ├── Comprehensive code documentation, unit testing, and Git versioning
 └── Final project report and defense presentation preparation
```

---

## 9. Verification & Evaluation Framework

| Evaluation Dimension | Metric / Criterion | Target Objective |
| :--- | :--- | :--- |
| **Predictive Accuracy** | ROC-AUC / PR-AUC | $\text{ROC-AUC} \ge 0.82$ on held-out test splits |
| **Recourse Success Rate** | Percentage of adverse cases successfully flipped to favorable | $\ge 98\%$ of unfavorable predictions |
| **Recourse Feasibility** | Violation rate of immutable attributes | $0.0\%$ (Strict constraint enforcement) |
| **Intervention Sparsity** | Average number of features modified per recourse suggestion | $\le 3$ actionable features changed |
| **System Latency** | End-to-end execution time per applicant review | $\le 3.0 \text{ seconds}$ total pipeline response |
| **Attribution Fidelity** | Causal refutation test p-value (Random Common Cause / Placebo) | $p \ge 0.05$ (Validates causal invariance) |

---

## 10. Summary & Deliverables Checklist

- [x] Complete system architecture and pipeline definition.
- [x] Clear specification of mutable vs. immutable feature partitioning.
- [x] Mathematical formulation of causal interventional recourse.
- [x] Hardware and software dependencies specified with zero-login dataset options.
- [x] Concrete implementation roadmap covering end-to-end engineering lifecycle.
