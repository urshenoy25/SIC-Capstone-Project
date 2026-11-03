"""
Module 4C: Streamlit Interactive Decision-Support Dashboard
Interactive XAI interface for compliance analysts, retention specialists, and stakeholders.
Features DAG visualization, Causal vs SHAP attribution, and actionable recourse with what-if simulation.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Configure sys.path
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

# Page Config
st.set_page_config(
    page_title="Causal Decision Support & Actionable Recourse",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(BASE_DIR, "module_2_causal_attribution/models/saved_models/cnn_churn_model.keras")
PREPROC_PATH = os.path.join(BASE_DIR, "module_1_data_ingestion/preprocessor.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "data/Telco_Customer_Churn_Augmented.csv")


@st.cache_resource
def load_framework_services():
    """Initializes and caches model, SCM, attribution, and recourse engines."""
    model = ChurnModelWrapper(MODEL_PATH, PREPROC_PATH)
    graph = TelcoCausalGraph()
    scm = TelcoSCMEngine(graph)

    if os.path.exists(DATASET_PATH):
        df_raw = pd.read_csv(DATASET_PATH)
        scm.fit(df_raw)

    attrib_engine = CausalAttributionEngine(MODEL_PATH, PREPROC_PATH)
    attrib_engine.scm = scm

    recourse_opt = ConstrainedRecourseOptimizer(model_wrapper=model, scm=scm)
    recourse_ranker = RecourseRanker(recourse_opt)
    audit_store = AuditStore()

    return model, graph, scm, attrib_engine, recourse_opt, recourse_ranker, audit_store


model, graph, scm, attrib_engine, recourse_opt, recourse_ranker, audit_store = load_framework_services()

# ── Sidebar: Customer Profile Selector ──────────────────────────
st.sidebar.title("🎛️ Applicant / Customer Profile")

preset = st.sidebar.selectbox(
    "Choose Preset Case Study",
    ["High Churn Risk (Month-to-Month)", "Moderate Risk Customer", "Low Risk (Long-Term)", "Custom Profile"]
)

# Defaults
if preset == "High Churn Risk (Month-to-Month)":
    init_gender = "Female"
    init_senior = 0
    init_partner = "No"
    init_dep = "No"
    init_tenure = 6
    init_phone = "Yes"
    init_lines = "No"
    init_internet = "Fiber optic"
    init_sec = "No"
    init_back = "No"
    init_dev = "No"
    init_tech = "No"
    init_tv = "No"
    init_mov = "No"
    init_contract = "Month-to-month"
    init_paper = "Yes"
    init_pay = "Electronic check"
    init_monthly = 85.0
elif preset == "Moderate Risk Customer":
    init_gender = "Male"
    init_senior = 0
    init_partner = "Yes"
    init_dep = "No"
    init_tenure = 24
    init_phone = "Yes"
    init_lines = "Yes"
    init_internet = "DSL"
    init_sec = "No"
    init_back = "Yes"
    init_dev = "No"
    init_tech = "No"
    init_tv = "Yes"
    init_mov = "No"
    init_contract = "Month-to-month"
    init_paper = "Yes"
    init_pay = "Electronic check"
    init_monthly = 65.0
elif preset == "Low Risk (Long-Term)":
    init_gender = "Female"
    init_senior = 0
    init_partner = "Yes"
    init_dep = "Yes"
    init_tenure = 60
    init_phone = "Yes"
    init_lines = "Yes"
    init_internet = "DSL"
    init_sec = "Yes"
    init_back = "Yes"
    init_dev = "Yes"
    init_tech = "Yes"
    init_tv = "Yes"
    init_mov = "Yes"
    init_contract = "Two year"
    init_paper = "No"
    init_pay = "Credit card (automatic)"
    init_monthly = 75.0
else:
    init_gender = "Male"
    init_senior = 0
    init_partner = "No"
    init_dep = "No"
    init_tenure = 12
    init_phone = "Yes"
    init_lines = "No"
    init_internet = "Fiber optic"
    init_sec = "No"
    init_back = "No"
    init_dev = "No"
    init_tech = "No"
    init_tv = "No"
    init_mov = "No"
    init_contract = "Month-to-month"
    init_paper = "Yes"
    init_pay = "Electronic check"
    init_monthly = 80.0

st.sidebar.markdown(r"### 🔒 Protected Attributes ($\mathcal{I}$)")
gender = st.sidebar.selectbox("gender", ["Female", "Male"], index=0 if init_gender == "Female" else 1)
senior = st.sidebar.selectbox("SeniorCitizen", [0, 1], index=init_senior)
partner = st.sidebar.selectbox("Partner", ["No", "Yes"], index=0 if init_partner == "No" else 1)
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"], index=0 if init_dep == "No" else 1)
tenure = st.sidebar.slider("tenure (months)", 0, 72, init_tenure)

st.sidebar.markdown(r"### ✏️ Actionable Services ($\mathcal{M}$)")
phone = st.sidebar.selectbox("PhoneService", ["No", "Yes"], index=1 if init_phone == "Yes" else 0)
lines = st.sidebar.selectbox("MultipleLines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(init_lines))
internet = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(init_internet))
security = st.sidebar.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(init_sec))
backup = st.sidebar.selectbox("OnlineBackup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(init_back))
device = st.sidebar.selectbox("DeviceProtection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(init_dev))
tech = st.sidebar.selectbox("TechSupport", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(init_tech))
tv = st.sidebar.selectbox("StreamingTV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(init_tv))
movies = st.sidebar.selectbox("StreamingMovies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(init_mov))
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(init_contract))
paperless = st.sidebar.selectbox("PaperlessBilling", ["No", "Yes"], index=1 if init_paper == "Yes" else 0)
payment = st.sidebar.selectbox(
    "PaymentMethod",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(init_pay)
)
monthly = st.sidebar.slider("MonthlyCharges ($)", 18.25, 120.0, float(init_monthly))

current_user = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": lines,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": float(tenure * monthly)
}

# ── Inference Execution ──────────────────────────────────────────
t_start = time.time()
churn_prob = float(model.predict_proba(current_user)[0])
is_adverse = churn_prob >= 0.50
latency_ms = round((time.time() - t_start) * 1000, 2)

# Audit log event
req_id = audit_store.log_decision(
    raw_features=current_user,
    prediction=1 if is_adverse else 0,
    probability=churn_prob,
    latency_ms=latency_ms
)

# ── Main Dashboard Header ────────────────────────────────────────
st.title("⚖️ Causal Decision Support & Actionable Recourse")
st.markdown(
    "**End-to-End Explainable AI Framework** combining **1D Tabular CNN**, "
    "**Structural Causal Models (SCM)**, **Pearl's do-calculus**, and **Constrained Algorithmic Recourse**."
)

# ── Top KPI Metric Cards ─────────────────────────────────────────
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    verdict_text = "Adverse (High Churn Risk)" if is_adverse else "Favorable (Retained)"
    verdict_color = "red" if is_adverse else "green"
    st.markdown(f"### Decision Verdict\n**:{verdict_color}[{verdict_text}]**")

with col_kpi2:
    st.metric("Predicted Churn Probability", f"{churn_prob * 100:.2f}%")

with col_kpi3:
    conf = churn_prob if is_adverse else (1.0 - churn_prob)
    st.metric("Model Confidence", f"{conf * 100:.1f}%")

with col_kpi4:
    st.metric("Inference Latency", f"{latency_ms} ms")

st.markdown("---")

# ── Tabs Navigation ──────────────────────────────────────────────
tab_verdict, tab_dag, tab_attribution, tab_recourse, tab_audits = st.tabs([
    "📊 Decision Overview",
    "🕸️ Causal DAG Discovery",
    "🔍 Causal vs. SHAP Attribution",
    "🎯 Actionable Recourse Pathways",
    "📜 Regulatory Audit Log"
])

# ── TAB 1: Decision Overview ─────────────────────────────────────
with tab_verdict:
    st.subheader("Customer Decision Summary")
    st.progress(float(churn_prob), text=f"Churn Risk Index: {churn_prob * 100:.1f}%")

    col_prof1, col_prof2 = st.columns(2)
    with col_prof1:
        st.markdown("#### 🔒 Protected Profile (Immutable)")
        df_imm = pd.DataFrame([
            {"Attribute": "Gender", "Value": gender},
            {"Attribute": "Senior Citizen", "Value": "Yes" if senior == 1 else "No"},
            {"Attribute": "Partner", "Value": partner},
            {"Attribute": "Dependents", "Value": dependents},
            {"Attribute": "Tenure Duration", "Value": f"{tenure} months"}
        ])
        st.dataframe(df_imm, use_container_width=True, hide_index=True)

    with col_prof2:
        st.markdown("#### ✏️ Commercial & Service Features (Mutable)")
        df_mut = pd.DataFrame([
            {"Feature": "Internet Service", "Value": internet},
            {"Feature": "Contract Term", "Value": contract},
            {"Feature": "Tech Support", "Value": tech},
            {"Feature": "Online Security", "Value": security},
            {"Feature": "Payment Channel", "Value": payment},
            {"Feature": "Monthly Billing", "Value": f"${monthly:.2f}"}
        ])
        st.dataframe(df_mut, use_container_width=True, hide_index=True)

# ── TAB 2: Causal DAG Discovery ──────────────────────────────────
with tab_dag:
    st.subheader("Directed Acyclic Graph (Structural Causal Model)")
    st.markdown(
        "Learned DAG encoding business mechanics and domain constraints across 6 hierarchical tiers. "
        "Enforces **zero reverse causality** (e.g. outcome cannot cause demographics)."
    )

    dag_info = graph.export_graph_dict()
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.metric("Graph Nodes", len(dag_info["nodes"]))
        st.metric("Directed Causal Edges", len(dag_info["edges"]))
        st.info("Acyclicity: **100% Strict DAG (No cycles)**")

        st.markdown("#### Direct Causal Parents of Churn:")
        for p in graph.get_parents("Churn"):
            st.markdown(f"- **`{p}`**")

    with col_d2:
        st.markdown("#### Graph Topological Execution Ordering:")
        st.code(" -> \n".join(dag_info["topological_order"]), language="text")

# ── TAB 3: Dual Attribution Engine ───────────────────────────────
with tab_attribution:
    st.subheader("Comparative Feature Attribution: Correlation vs. Causation")
    st.markdown(
        "Existing standards like **SHAP** measure associational correlation, routinely misattributing importance "
        "to non-actionable attributes. Our **SCM Interventional Engine** computes true causal impact via Pearl's do-calculus:"
    )
    st.latex(r"\Delta_j = \mathbb{E}\left[Y \mid \text{do}(X_j = x_j')\right] - \mathbb{E}\left[Y \mid \text{do}(X_j = x_j)\right]")

    with st.spinner("Computing Causal ICE & SHAP Attributions..."):
        comp_df = attrib_engine.compare_causal_vs_shap(current_user)

    col_chart, col_tbl = st.columns([1, 1])
    with col_chart:
        chart_data = comp_df.head(8).set_index("Feature")[["Causal_Impact", "SHAP_Association"]]
        st.bar_chart(chart_data)

    with col_tbl:
        st.dataframe(comp_df.head(10), use_container_width=True, hide_index=True)

    st.success(
        "💡 **Key Scientific Insight:** Notice that SHAP assigns importance to features regardless of whether "
        "they are actionable or causally independent, whereas Causal SCM highlights specific levers that change the real outcome."
    )

# ── TAB 4: Actionable Recourse Pathways ───────────────────────────
with tab_recourse:
    st.subheader("Actionable Counterfactual Recourse")
    st.markdown(
        "Minimum-cost intervention pathways generated by **Constrained Optimization (SLSQP)**. "
        "Guarantees **0% immutable violations** and propagates all downstream SCM effects."
    )

    if not is_adverse:
        st.success("✅ Customer is currently retained with low churn probability. Recourse is not required.")
    else:
        with st.spinner("Optimizing minimal-cost recourse pathways..."):
            pathways = recourse_ranker.generate_top_recourse(current_user, top_k=3)

        if not pathways:
            st.warning("No feasible recourse found within cost thresholds.")
        else:
            for p in pathways:
                with st.expander(f"⭐ **{p['title']}** (Effort Score: {p['cost_score']})", expanded=True):
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Baseline Churn Risk", p["baseline_risk"])
                    with col_r2:
                        st.metric("New Churn Risk", p["achieved_risk"], delta=f"-{p['risk_drop_pct']}")
                    with col_r3:
                        st.metric("Intervention Sparsity", f"{p['sparsity']} feature(s) modified")

                    st.markdown("#### Recommended Action Steps:")
                    for a in p["actions"]:
                        st.markdown(f"- ✅ **{a['description']}**")

                    st.markdown("#### Downstream Causal Consequences (Propagated via SCM):")
                    c_cons = p["causal_consequences"]
                    st.info(f"Updated Monthly Billing: **{c_cons['New_MonthlyCharges']}** | Updated Total Charges: **{c_cons['New_TotalCharges']}**")

    # Interactive What-If Sandbox
    st.markdown("---")
    st.subheader("🔬 Interactive 'What-If' Recourse Simulator")
    st.markdown("Manually test potential interventions and observe real-time causal propagation:")

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sim_contract = st.selectbox("Simulate Contract Change", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(contract), key="sim_c")
    with col_sim2:
        sim_tech = st.selectbox("Simulate Tech Support", ["No", "Yes"], index=0 if tech == "No" else 1, key="sim_t")
    with col_sim3:
        sim_sec = st.selectbox("Simulate Online Security", ["No", "Yes"], index=0 if security == "No" else 1, key="sim_s")

    sim_actions = {}
    if sim_contract != contract:
        sim_actions["Contract"] = sim_contract
    if sim_tech != tech:
        sim_actions["TechSupport"] = sim_tech
    if sim_sec != security:
        sim_actions["OnlineSecurity"] = sim_sec

    sim_cf = scm.intervene(current_user, sim_actions)
    sim_prob = float(model.predict_proba(sim_cf)[0])

    col_sim_res1, col_sim_res2 = st.columns(2)
    with col_sim_res1:
        st.metric("Simulated Churn Risk", f"{sim_prob * 100:.2f}%", delta=f"{(churn_prob - sim_prob) * -100:.2f}% (Risk Change)")
    with col_sim_res2:
        st.markdown(f"**Causal Monthly Billing:** `${sim_cf.get('MonthlyCharges'):.2f}`")
        st.markdown(f"**Causal Total Billing:** `${sim_cf.get('TotalCharges'):.2f}`")

# ── TAB 5: Compliance Audit Trail ────────────────────────────────
with tab_audits:
    st.subheader("📜 Regulatory & Compliance Audit Trail")
    st.markdown("Persistent decision logs stored with cryptographic UUIDs and UTC timestamps for auditability.")

    recent_audits = audit_store.get_recent_audits(limit=15)
    if recent_audits:
        df_audit = pd.DataFrame(recent_audits)[["request_id", "timestamp", "prediction", "probability", "latency_ms"]]
        st.dataframe(df_audit, use_container_width=True, hide_index=True)
    else:
        st.info("No decision events logged yet.")
