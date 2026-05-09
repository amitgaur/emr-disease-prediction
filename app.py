"""
EMR Disease Prediction Demo — Streamlit App
Production-quality demo for leaders: patient risk prediction, model comparison,
what-if analysis, and clinical reasoning.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="EMR Disease Prediction Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark theme styling
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .main .block-container { background-color: #0f172a; padding-top: 2rem; }
    h1, h2, h3 { color: #f1f5f9; }
    .stText, .stMarkdown { color: #cbd5e1; }
    .css-1d3w5jw { border: 1px solid #334155; }
    .risk-low { color: #22c55e; font-weight: bold; }
    .risk-med { color: #f59e0b; font-weight: bold; }
    .risk-high { color: #ef4444; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #06b6d4; font-size: 2.5rem; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; }
    .risk-gauge { text-align: center; padding: 2rem; }
    .footer { text-align: center; color: #64748b; padding: 2rem; font-size: 0.8rem; }
    .feature-importance { margin-top: 1rem; }
    .outcome-table { margin-top: 1rem; }
    .stButton > button { background-color: #06b6d4; color: white; border: none; }
    .stButton > button:hover { background-color: #0891b2; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Mock data
# =============================================================================

MOCK_PATIENT = {
    "name": "John D.",
    "age": 67,
    "gender": "Male",
    "bmi": 29.4,
    "hba1c": 7.8,
    "heart_rate": 82,
    "systolic_bp": 138,
    "creatinine": 1.2,
    "wbc": 11.2,
    "glucose": 156,
    "ldl": 142,
    "hdl": 38,
    "triglycerides": 210,
    "albumin": 3.4,
    "alt": 32,
    "ast": 28,
}

MOCK_RISK = 0.52  # 52% — medium-high risk

MOCK_SHAP_VALUES = [
    {"feature": "HbA1c", "value": 0.18, "direction": "risk"},
    {"feature": "Age", "value": 0.14, "direction": "risk"},
    {"feature": "Creatinine", "value": 0.11, "direction": "risk"},
    {"feature": "WBC", "value": 0.09, "direction": "risk"},
    {"feature": "Heart Rate", "value": 0.07, "direction": "risk"},
]

MOCK_SIMILAR_PATIENTS = [
    {"id": "P-1042", "age": 65, "risk": 0.48, "outcome": "Stable, discharged day 8"},
    {"id": "P-1087", "age": 69, "risk": 0.55, "outcome": "Readmitted day 12"},
    {"id": "P-1123", "age": 66, "risk": 0.51, "outcome": "Stable, discharged day 6"},
    {"id": "P-1156", "age": 68, "risk": 0.58, "outcome": "Readmitted day 21"},
]

MOCK_MODEL_RESULTS = {
    "XGBoost Baseline": {"auroc": 0.76, "auprc": 0.62, "f1": 0.58, "precision": 0.61, "recall": 0.55},
    "FT-Transformer": {"auroc": 0.81, "auprc": 0.71, "f1": 0.66, "precision": 0.68, "recall": 0.64},
    "BioMistral Fusion": {"auroc": 0.83, "auprc": 0.75, "f1": 0.71, "precision": 0.72, "recall": 0.70},
}

MOCK_TEMPORAL = {
    "months": ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb"],
    "xgb_auc": [0.73, 0.74, 0.75, 0.76, 0.75, 0.76],
    "ftt_auc": [0.77, 0.78, 0.79, 0.80, 0.81, 0.81],
    "fusion_auc": [0.80, 0.81, 0.82, 0.82, 0.83, 0.83],
}

MOCK_INTERVENTIONS = [
    {"name": "Reduce HbA1c to 6.5", "delta_risk": -0.12, "effort": "Medium", "outcome": "Metformin adjustment"},
    {"name": "Lower LDL to 100", "delta_risk": -0.08, "effort": "Easy", "outcome": "Statin increase"},
    {"name": "Weight loss to BMI 25", "delta_risk": -0.10, "effort": "Hard", "outcome": "Diet + exercise program"},
    {"name": "Reduce heart rate to 70", "delta_risk": -0.05, "effort": "Medium", "outcome": "Beta-blocker titration"},
]

MOCK_COT_REASONING = [
    {"step": 1, "finding": "HbA1c elevated at 7.8%", "implication": "Poor glycemic control over 3-month window"},
    {"step": 2, "finding": "Age 67 + HbA1c 7.8%", "implication": "Increased risk of microvascular complications"},
    {"step": 3, "finding": "Creatinine 1.2 mg/dL borderline", "implication": "Possible early diabetic nephropathy"},
    {"step": 4, "finding": "WBC 11.2 — mild inflammation", "implication": "Chronic low-grade inflammation consistent with T2DM"},
    {"step": 5, "finding": "Combined risk factors", "implication": "30-day readmission risk elevated at 52%"},
    {"step": 6, "recommendation": "Order comprehensive metabolic panel, adjust diabetes medications, monitor creatinine weekly"},
]

MOCK_ICD_GRAPH = {
    "nodes": [
        {"id": "E11.9", "label": "T2DM\n(no complications)", "type": "primary"},
        {"id": "E11.5", "label": "T2DM\n+ CKD", "type": "complication"},
        {"id": "E11.4", "label": "T2DM\n+ neuropathy", "type": "complication"},
        {"id": "E11.3", "label": "T2DM\n+ retinopathy", "type": "complication"},
        {"id": "I10", "label": "Hypertension", "type": "comorbidity"},
        {"id": "E78.5", "label": "Hyperlipidemia", "type": "comorbidity"},
        {"id": "N18.3", "label": "CKD Stage 3", "type": "downstream"},
    ],
    "edges": [
        {"source": "E11.9", "target": "E11.5", "label": "causes"},
        {"source": "E11.9", "target": "E11.4", "label": "causes"},
        {"source": "E11.9", "target": "E11.3", "label": "causes"},
        {"source": "E11.9", "target": "I10", "label": "associated"},
        {"source": "E11.9", "target": "E78.5", "label": "associated"},
        {"source": "E11.5", "target": "N18.3", "label": "progresses"},
    ],
}


# =============================================================================
# Helper components
# =============================================================================

def risk_color(risk: float) -> str:
    if risk < 0.30:
        return "#22c55e"
    elif risk < 0.60:
        return "#f59e0b"
    else:
        return "#ef4444"


def risk_label(risk: float) -> str:
    if risk < 0.30:
        return "LOW"
    elif risk < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


def confidence_label(risk: float) -> tuple[str, str]:
    if risk < 0.25 or risk > 0.75:
        return "High", "#22c55e"
    elif risk < 0.40 or risk > 0.65:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#ef4444"


def render_risk_gauge(risk: float, height: int = 280):
    """Render an animated risk gauge using Plotly."""
    color = risk_color(risk)
    label = risk_label(risk)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "transparent"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e293b",
            "bordercolor": "transparent",
            "steps": [
                {"range": [0, 30], "color": "#1e293b"},
                {"range": [30, 60], "color": "#1e293b"},
                {"range": [60, 100], "color": "#1e293b"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": risk * 100,
            },
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="transparent",
        font={"color": "#cbd5e1"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_chart(shap_values: list[dict]):
    """Render SHAP-style feature importance."""
    df = shap_values.copy()
    df = sorted(df, key=lambda x: x["value"], reverse=True)

    colors = ["#ef4444" if d["direction"] == "risk" else "#22c55e" for d in df]
    values = [v["value"] for v in df]
    labels = [v["feature"] for v in df]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"+{v:.2f}" for v in values],
        textposition="outside",
        textfont={"color": "#cbd5e1"},
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="transparent",
        plot_bgcolor="transparent",
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, max(values) * 1.3],
        ),
        yaxis=dict(
            showgrid=False,
            tickfont={"color": "#cbd5e1", "size": 13},
            zeroline=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison():
    """Render model comparison charts."""
    models = list(MOCK_MODEL_RESULTS.keys())
    metrics = ["auroc", "auprc", "f1", "precision", "recall"]
    metric_labels = ["AUC-ROC", "AUC-PR", "F1 Score", "Precision", "Recall"]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "bar"}]],
                        subplot_titles=["Performance Metrics", "AUC Comparison Over Time"])

    for i, model_name in enumerate(models):
        color = ["#06b6d4", "#f59e0b", "#22c55e"][i]
        marker = ["circle", "square", "diamond"][["XGBoost Baseline", "FT-Transformer", "BioMistral Fusion"].index(model_name)]

        metric_vals = [MOCK_MODEL_RESULTS[model_name][m] for m in metrics]
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_labels,
            y=metric_vals,
            marker_color=color,
            text=[f"{v:.2f}" for v in metric_vals],
            textposition="outside",
            legendgroup=model_name,
        ), row=1, col=1)

    # Temporal AUC chart
    fig.add_trace(go.Scatter(
        x=MOCK_TEMPORAL["months"],
        y=MOCK_TEMPORAL["xgb_auc"],
        name="XGBoost Baseline",
        line={"color": "#06b6d4", "width": 2, "dash": "dot"},
        marker_symbol="circle",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=MOCK_TEMPORAL["months"],
        y=MOCK_TEMPORAL["ftt_auc"],
        name="FT-Transformer",
        line={"color": "#f59e0b", "width": 2},
        marker_symbol="square",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=MOCK_TEMPORAL["months"],
        y=MOCK_TEMPORAL["fusion_auc"],
        name="BioMistral Fusion",
        line={"color": "#22c55e", "width": 3},
        marker_symbol="diamond",
    ), row=1, col=2)

    fig.update_layout(
        height=340,
        barmode="group",
        paper_bgcolor="transparent",
        plot_bgcolor="#1e293b",
        font={"color": "#cbd5e1"},
        legend=dict(bgcolor="#1e293b", font={"color": "#cbd5e1"}),
        xaxis=dict(gridcolor="#334155", tickfont={"color": "#cbd5e1"}),
        yaxis=dict(gridcolor="#334155", tickfont={"color": "#cbd5e1"}, range=[0.65, 0.90]),
        xaxis2=dict(gridcolor="#334155", tickfont={"color": "#cbd5e1"}),
        yaxis2=dict(gridcolor="#334155", tickfont={"color": "#cbd5e1"}, range=[0.65, 0.90]),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_calibration():
    """Render calibration curves."""
    fig = go.Figure()

    confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    perfect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # XGBoost calibration (slightly overconfident)
    xgb_cal = [0.12, 0.24, 0.35, 0.47, 0.56, 0.68, 0.77, 0.87, 0.94]
    fig.add_trace(go.Scatter(
        x=confs, y=xgb_cal, name="XGBoost Baseline",
        line={"color": "#06b6d4", "width": 2}, mode="lines+markers",
        marker_symbol="circle",
    ))

    # FT-Transformer
    ftt_cal = [0.11, 0.21, 0.31, 0.43, 0.52, 0.62, 0.73, 0.84, 0.92]
    fig.add_trace(go.Scatter(
        x=confs, y=ftt_cal, name="FT-Transformer",
        line={"color": "#f59e0b", "width": 2}, mode="lines+markers",
        marker_symbol="square",
    ))

    # Fusion (best calibrated)
    fus_cal = [0.10, 0.20, 0.30, 0.41, 0.51, 0.61, 0.72, 0.82, 0.91]
    fig.add_trace(go.Scatter(
        x=confs, y=fus_cal, name="BioMistral Fusion",
        line={"color": "#22c55e", "width": 3}, mode="lines+markers",
        marker_symbol="diamond",
    ))

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=perfect, y=perfect, name="Perfect Calibration",
        line={"color": "#64748b", "width": 2, "dash": "dash"},
        mode="lines",
    ))

    fig.update_layout(
        height=300,
        paper_bgcolor="transparent",
        plot_bgcolor="transparent",
        font={"color": "#cbd5e1"},
        legend=dict(bgcolor="#1e293b", font={"color": "#cbd5e1"}),
        xaxis=dict(title="Confidence", gridcolor="#334155", tickfont={"color": "#cbd5e1"}),
        yaxis=dict(title="Fraction Positive", gridcolor="#334155", tickfont={"color": "#cbd5e1"}),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_icd_graph():
    """Render ICD knowledge graph using networkx."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        for node in MOCK_ICD_GRAPH["nodes"]:
            G.add_node(node["id"], label=node["label"], type=node["type"])

        for edge in MOCK_ICD_GRAPH["edges"]:
            G.add_edge(edge["source"], edge["target"], label=edge["label"])

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#1e293b")

        pos = nx.spring_layout(G, k=2, iterations=50)

        color_map = {
            "primary": "#06b6d4",
            "complication": "#ef4444",
            "comorbidity": "#f59e0b",
            "downstream": "#a855f7",
        }
        node_colors = [color_map.get(G.nodes[n]["type"], "#64748b") for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000,
                              alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()},
                                font_size=8, font_color="white", ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#64748b", arrows=True,
                              arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax)

        ax.axis("off")
        st.pyplot(fig)

    except ImportError:
        # Fallback: text-based representation
        st.markdown("""
        <div style="background:#1e293b; padding:1.5rem; border-radius:8px;">
            <pre style="color:#cbd5e1; font-size:0.85rem; line-height:1.6;">
            <span style="color:#06b6d4;">T2DM (E11.9)</span> ──┬── <span style="color:#ef4444;">CKD (E11.5)</span> ──→ CKD Stage 3
                    ├── <span style="color:#ef4444;">Neuropathy (E11.4)</span>
                    ├── <span style="color:#ef4444;">Retinopathy (E11.3)</span>
                    ├── <span style="color:#f59e0b;">Hypertension (I10)</span>
                    └── <span style="color:#f59e0b;">Hyperlipidemia (E78.5)</span>
            </pre>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# Pages
# =============================================================================

def page_risk_predictor():
    st.header("🏥 Patient Risk Predictor")

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("Patient Input")

        name = st.text_input("Patient Name", value=MOCK_PATIENT["name"])
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", value=MOCK_PATIENT["age"], min_value=1, max_value=120)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)

        st.subheader("Clinical Features")
        c3, c4, c5 = st.columns(3)
        with c3:
            hba1c = st.number_input("HbA1c (%)", value=MOCK_PATIENT["hba1c"], min_value=4.0, max_value=14.0, step=0.1)
        with c4:
            bmi = st.number_input("BMI", value=MOCK_PATIENT["bmi"], min_value=15.0, max_value=50.0, step=0.1)
        with c5:
            heart_rate = st.number_input("Heart Rate (bpm)", value=MOCK_PATIENT["heart_rate"], min_value=40, max_value=200)
        c6, c7, c8 = st.columns(3)
        with c6:
            systolic_bp = st.number_input("Systolic BP", value=MOCK_PATIENT["systolic_bp"], min_value=80, max_value=250)
        with c7:
            creatinine = st.number_input("Creatinine (mg/dL)", value=MOCK_PATIENT["creatinine"], min_value=0.5, max_value=6.0, step=0.1)
        with c8:
            wbc = st.number_input("WBC (K/uL)", value=MOCK_PATIENT["wbc"], min_value=3.0, max_value=30.0, step=0.1)

        # Compute mock risk based on inputs
        computed_risk = min(0.95, max(0.05,
            0.08 + (hba1c - 5.5) * 0.06 +
            (age - 40) * 0.004 +
            (bmi - 22) * 0.008 +
            (heart_rate - 60) * 0.003 +
            (creatinine - 0.8) * 0.08 +
            (wbc - 7) * 0.015
        ))

        st.markdown("---")

    with col2:
        st.subheader("Risk Assessment")

        # Risk gauge
        render_risk_gauge(computed_risk, height=250)

        # Risk level and confidence
        rcolor = risk_color(computed_risk)
        rlabel = risk_label(computed_risk)
        conf, conf_color = confidence_label(computed_risk)

        st.markdown(f"""
        <div style="text-align:center; margin-top:0.5rem;">
            <span style="font-size:1.2rem; color:{rcolor}; font-weight:bold;">{rlabel} RISK</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="font-size:1rem; color:{conf_color};">Confidence: {conf}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Feature Attribution")

        # Dynamic SHAP values based on inputs
        dynamic_shap = [
            {"feature": "HbA1c", "value": (hba1c - 5.5) * 0.06 / computed_risk * 0.18, "direction": "risk" if hba1c > 6.5 else "protective"},
            {"feature": "Age", "value": (age - 40) * 0.004 / computed_risk * 0.14, "direction": "risk"},
            {"feature": "Creatinine", "value": (creatinine - 0.8) * 0.08 / computed_risk * 0.11, "direction": "risk" if creatinine > 1.0 else "protective"},
            {"feature": "WBC", "value": (wbc - 7) * 0.015 / computed_risk * 0.09, "direction": "risk" if wbc > 9 else "protective"},
            {"feature": "Heart Rate", "value": (heart_rate - 60) * 0.003 / computed_risk * 0.07, "direction": "risk" if heart_rate > 75 else "protective"},
        ]
        for d in dynamic_shap:
            d["value"] = max(0.01, d["value"])

        render_shap_chart(dynamic_shap)

        st.markdown("---")
        st.subheader("Similar Patient Outcomes")

        import pandas as pd
        df_sim = pd.DataFrame(MOCK_SIMILAR_PATIENTS)
        st.dataframe(df_sim, use_container_width=True, hide_index=True)


def page_model_comparison():
    st.header("📊 Model Comparison")

    st.markdown("""
    <div style="background:#1e293b; padding:1rem; border-radius:8px; margin-bottom:1.5rem;">
        <span style="color:#06b6d4; font-weight:bold;">BioMistral Fusion</span>
        outperforms existing approaches by combining temporal EHR modeling with clinical note understanding.
    </div>
    """, unsafe_allow_html=True)

    render_model_comparison()

    st.markdown("---")
    st.subheader("Calibration Analysis")

    render_calibration()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best AUC-ROC", "0.83", "+0.07 vs baseline")
    with col2:
        st.metric("Best AUC-PR", "0.75", "+0.13 vs baseline")
    with col3:
        st.metric("Best F1 Score", "0.71", "+0.13 vs baseline")


def page_whatif():
    st.header("🔬 What-If Analysis")

    st.markdown("""
    Explore how <b>interventions</b> would change this patient's risk profile.
    Adjust clinical parameters to see counterfactual risk reduction.
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Adjust Parameters")
        hba1c_slider = st.slider("HbA1c (%)", 5.0, 12.0, MOCK_PATIENT["hba1c"], 0.1)
        bmi_slider = st.slider("BMI", 18.0, 45.0, MOCK_PATIENT["bmi"], 0.5)
        hr_slider = st.slider("Heart Rate (bpm)", 50.0, 150.0, float(MOCK_PATIENT["heart_rate"]), 1.0)
        creat_slider = st.slider("Creatinine (mg/dL)", 0.5, 4.0, MOCK_PATIENT["creatinine"], 0.1)

    with col2:
        current_risk = MOCK_RISK
        modified_risk = min(0.95, max(0.05,
            0.08 + (hba1c_slider - 5.5) * 0.06 +
            (MOCK_PATIENT["age"] - 40) * 0.004 +
            (bmi_slider - 22) * 0.008 +
            (hr_slider - 60) * 0.003 +
            (creat_slider - 0.8) * 0.08 +
            (MOCK_PATIENT["wbc"] - 7) * 0.015
        ))
        delta = current_risk - modified_risk

        st.subheader("Counterfactual Risk")
        render_risk_gauge(modified_risk, height=220)

        if delta > 0.01:
            st.markdown(f"""
            <div style="text-align:center; margin-top:0.5rem; padding:1rem; background:#1e293b; border-radius:8px;">
                <span style="color:#22c55e; font-size:1.3rem; font-weight:bold;">
                    Risk reduction: {delta*100:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Intervention Impact")

    interventions = sorted(MOCK_INTERVENTIONS, key=lambda x: x["delta_risk"], reverse=True)

    cols = st.columns(4)
    for i, interv in enumerate(interventions):
        with cols[i]:
            delta_pct = interv["delta_risk"] * 100
            st.markdown(f"""
            <div style="background:#1e293b; padding:1rem; border-radius:8px; text-align:center;">
                <div style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0.5rem;">{interv["name"]}</div>
                <div style="font-size:1.4rem; color:#22c55e; font-weight:bold;">-{delta_pct:.0f}%</div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:0.5rem;">{interv["effort"]} effort</div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.3rem;">{interv["outcome"]}</div>
            </div>
            """, unsafe_allow_html=True)


def page_clinical_reasoning():
    st.header("🧠 Clinical Reasoning")

    st.markdown("""
    <div style="background:#1e293b; padding:1rem; border-radius:8px; margin-bottom:1.5rem;">
        Chain-of-Thought reasoning explaining <b>why</b> the model made this prediction.
        This builds clinician trust and enables <b>human-in-the-loop</b> oversight.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Reasoning Chain")

        for step in MOCK_COT_REASONING:
            is_rec = step["step"] == 6
            bg = "#065f46" if is_rec else "#1e293b"
            border = "#22c55e" if is_rec else "#334155"
            icon = "→" if is_rec else str(step["step"])

            st.markdown(f"""
            <div style="background:{bg}; border-left:3px solid {border}; padding:0.8rem; margin-bottom:0.5rem; border-radius:4px;">
                <div style="display:flex; align-items:center; margin-bottom:0.3rem;">
                    <span style="background:#06b6d4; color:white; width:24px; height:24px; border-radius:50%; text-align:center; line-height:24px; font-size:0.8rem; margin-right:0.8rem;">{icon}</span>
                    <span style="color:#cbd5e1; font-weight:bold;">{step["finding"]}</span>
                </div>
                <div style="color:#94a3b8; font-size:0.85rem; padding-left:2.2rem;">{step["implication"]}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("Disease Knowledge Graph")
        render_icd_graph()

    st.markdown("---")
    st.subheader("Similar Patient Outcomes")

    import pandas as pd
    df_sim = pd.DataFrame(MOCK_SIMILAR_PATIENTS)

    for _, row in df_sim.iterrows():
        outcome_color = "#22c55e" if "Stable" in row["outcome"] else "#ef4444"
        st.markdown(f"""
        <div style="background:#1e293b; padding:0.8rem; border-radius:6px; margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="color:#06b6d4; font-weight:bold;">{row["id"]}</span>
                <span style="color:#64748b; margin-left:1rem;">Age {row["age"]}</span>
            </div>
            <div>
                <span style="color:#f59e0b; margin-right:1rem;">Risk: {row["risk"]*100:.0f}%</span>
                <span style="color:{outcome_color};">{row["outcome"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# Main
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0;">
            <div style="font-size:1.5rem; font-weight:bold; color:#06b6d4; margin-bottom:0.5rem;">🏥 EMR Disease Prediction</div>
            <div style="color:#64748b; font-size:0.85rem;">Predictive AI for ICU patient outcomes</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Model Performance**")

        metrics_summary = [
            ("AUC-ROC", "0.83", "+0.07"),
            ("AUC-PR", "0.75", "+0.13"),
            ("F1 Score", "0.71", "+0.13"),
        ]
        for metric, val, delta in metrics_summary:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid #334155;">
                <span style="color:#94a3b8;">{metric}</span>
                <span style="color:#06b6d4; font-weight:bold;">{val}</span>
                <span style="color:#22c55e; font-size:0.85rem;">{delta}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Dataset**")
        st.markdown("""
        <div style="color:#64748b; font-size:0.8rem;">
            • eICU-CRD: 200K+ ICU stays<br>
            • MIMIC-IV (pending access)<br>
            • Temporal: 48h windows
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Version Info**")
        st.markdown(f"""
        <div style="color:#64748b; font-size:0.75rem;">
            Model v2.1.0<br>
            Last trained: {datetime.now().strftime("%b %d, %Y")}<br>
            Training samples: 142,857
        </div>
        """, unsafe_allow_html=True)

    # Main content
    pages = {
        "Patient Risk Predictor": page_risk_predictor,
        "Model Comparison": page_model_comparison,
        "What-If Analysis": page_whatif,
        "Clinical Reasoning": page_clinical_reasoning,
    }

    st.radio(
        "Navigation",
        list(pages.keys()),
        index=0,
        format_func=lambda x: f"  {x}",
        label_visibility="collapsed",
        horizontal=True,
    )

    selected_page = st.session_state.get("page", "Patient Risk Predictor")

    if selected_page in pages:
        pages[selected_page]()

    # Footer
    st.markdown(f"""
    <div class="footer">
        EMR Disease Prediction Demo v1.0.0 · Built with Streamlit · {datetime.now().strftime("%b %d, %Y %H:%M UTC")}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
