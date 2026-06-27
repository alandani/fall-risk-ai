"""
app.py — Fall Risk AI · Streamlit Dashboard
--------------------------------------------
Run with:  streamlit run app.py

Requires:
  - LM Studio running at http://localhost:1234 with Gemma 4 loaded
  - models/ folder populated (run fall_risk_pipeline.ipynb first)
  - pip install streamlit openai shap joblib xgboost scikit-learn matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from llm_advisor import FallRiskAdvisor, _risk_label, RISK_HIGH, RISK_MEDIUM

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fall Risk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Risk badge */
.badge {
    display: inline-block;
    padding: 0.45em 1.1em;
    border-radius: 8px;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.badge-high     { background:#fee2e2; color:#991b1b; border:1.5px solid #f87171; }
.badge-moderate { background:#fef3c7; color:#92400e; border:1.5px solid #fbbf24; }
.badge-low      { background:#dcfce7; color:#166534; border:1.5px solid #4ade80; }

/* Score gauge */
.gauge-label { font-size: 2.4rem; font-weight: 800; margin: 0; }

/* LLM output boxes — use Streamlit's own CSS variables so they always
   match the active theme (dark or light) automatically.              */
.llm-box {
    background: var(--secondary-background-color);
    color: var(--text-color);
    border-left: 4px solid #6366f1;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
    line-height: 1.6;
}
.rec-box {
    background: var(--secondary-background-color);
    color: var(--text-color);
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

_detected_theme = st.query_params.get("_theme", "light")

# Runs in a real iframe so scripts execute — redirects only when theme changes.
components.html("""
<script>
(function() {
    try {
        var root = window.parent.document.documentElement;
        var bg = getComputedStyle(root).getPropertyValue('--background-color').trim();
        var isDark = false;
        if (bg.startsWith('#')) {
            var hex = bg.slice(1);
            var r = parseInt(hex.substr(0,2),16);
            var g = parseInt(hex.substr(2,2),16);
            var b = parseInt(hex.substr(4,2),16);
            isDark = (0.299*r + 0.587*g + 0.114*b) < 128;
        } else {
            var m = bg.match(/\\d+/g);
            if (m && m.length >= 3)
                isDark = (0.299*+m[0] + 0.587*+m[1] + 0.114*+m[2]) < 128;
        }
        var url = new URL(window.parent.location.href);
        var expected = isDark ? 'dark' : 'light';
        if (url.searchParams.get('_theme') !== expected) {
            url.searchParams.set('_theme', expected);
            window.parent.location.replace(url.toString());
        }
    } catch(e) {}
})();
</script>
""", height=0)

dark_chart = _detected_theme == "dark"


# ── Load advisor (cached — loads model once per session) ─────────────────────
@st.cache_resource(show_spinner="Loading ML model and LLM advisor…")
def load_advisor(model_name: str) -> FallRiskAdvisor:
    return FallRiskAdvisor(model_name=model_name)


# ── Helpers ──────────────────────────────────────────────────────────────────
def risk_badge_html(label: str) -> str:
    cls = {"HIGH": "badge-high", "MODERATE": "badge-moderate", "LOW": "badge-low"}.get(label, "")
    icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(label, "")
    return f'<span class="badge {cls}">{icon} {label} RISK</span>'


def score_color(score: float) -> str:
    if score >= RISK_HIGH:   return "#ef4444"
    if score >= RISK_MEDIUM: return "#f59e0b"
    return "#22c55e"


def shap_waterfall_fig(advisor: FallRiskAdvisor, patient: dict, dark: bool = True) -> plt.Figure:
    """Generate a SHAP waterfall figure for a single patient."""
    df = pd.DataFrame([patient])[advisor.all_features]
    X  = advisor.preprocessor.transform(df)
    # Named DataFrame so SHAP attaches real feature names
    sv = advisor.explainer(pd.DataFrame(X, columns=advisor.feature_names))

    if dark:
        BG, TEXT, SPINE = "#1e293b", "#f1f5f9", "#475569"
        mpl_style = "dark_background"
    else:
        BG, TEXT, SPINE = "#ffffff", "#0f172a", "#cbd5e1"
        mpl_style = "default"

    with plt.style.context(mpl_style):
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor(BG)
        for ax in fig.axes:
            ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE)
            for txt in ax.texts:
                txt.set_color(TEXT)
        plt.tight_layout()

    return fig


def format_recommendations(rec_text: str) -> str:
    """Convert newline-separated bullets to HTML list."""
    lines = [l.strip() for l in rec_text.strip().splitlines() if l.strip()]
    items = "".join(
        f"<li>{l.lstrip('- •').strip()}</li>" for l in lines
    )
    return f"<ul style='margin:0; padding-left:1.2rem'>{items}</ul>"


# ── Sidebar — Patient Input Form ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
    st.title("Patient Input")
    st.caption("Enter patient data to assess fall risk.")

    # LM Studio model name + appearance
    with st.expander("⚙️ Settings", expanded=False):
        model_name = st.text_input(
            "Model ID", value="google/gemma-4-e4b",
            help="Paste the exact model ID from LM Studio (run curl http://localhost:1234/v1/models)"
        )

    st.divider()

    # ── Demographics ─────────────────────────────────────────────────────────
    st.subheader("Demographics")
    age = st.slider("Age", 18, 95, 65)
    sex = st.radio("Sex", ["F", "M"], horizontal=True)
    bmi = st.number_input("BMI", 15.0, 55.0, 26.0, step=0.1, format="%.1f")

    # ── Vitals & Labs ─────────────────────────────────────────────────────────
    st.subheader("Vitals & Labs")
    c1, c2 = st.columns(2)
    systolic_bp  = c1.number_input("Systolic BP",  80,  200, 130)
    diastolic_bp = c2.number_input("Diastolic BP", 50,  120, 82)
    heart_rate   = c1.number_input("Heart Rate",   40,  130, 76)
    sodium       = c2.number_input("Sodium (mEq/L)", 120.0, 148.0, 138.0, format="%.1f")
    hemoglobin   = c1.number_input("Hemoglobin (g/dL)", 6.0, 18.0, 13.5, format="%.1f")
    bun          = c2.number_input("BUN (mg/dL)", 5.0, 80.0, 18.0, format="%.1f")

    # ── Diagnoses ─────────────────────────────────────────────────────────────
    st.subheader("Diagnoses")
    has_parkinsons   = int(st.checkbox("Parkinson's disease"))
    has_osteoporosis = int(st.checkbox("Osteoporosis"))
    has_diabetes     = int(st.checkbox("Diabetes"))
    has_dementia     = int(st.checkbox("Dementia"))
    has_depression   = int(st.checkbox("Depression"))
    has_hypertension = int(st.checkbox("Hypertension"))

    # ── Medications ───────────────────────────────────────────────────────────
    st.subheader("Medications")
    on_sedatives         = int(st.checkbox("Sedatives / Benzodiazepines"))
    on_diuretics         = int(st.checkbox("Diuretics"))
    on_antihypertensives = int(st.checkbox("Antihypertensives"))
    on_anticoagulants    = int(st.checkbox("Anticoagulants"))

    # ── Functional ────────────────────────────────────────────────────────────
    st.subheader("Functional")
    prior_fall            = int(st.checkbox("Prior fall history"))
    uses_assistive_device = int(st.checkbox("Uses assistive device"))

    st.divider()
    run_btn = st.button("🔍 Run Risk Analysis", type="primary", use_container_width=True)


# ── Assemble patient dict ────────────────────────────────────────────────────
patient = {
    "age": age, "sex": sex, "bmi": bmi,
    "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
    "heart_rate": heart_rate, "sodium": sodium,
    "hemoglobin": hemoglobin, "bun": bun,
    "has_parkinsons": has_parkinsons, "has_osteoporosis": has_osteoporosis,
    "has_diabetes": has_diabetes, "has_dementia": has_dementia,
    "has_depression": has_depression, "has_hypertension": has_hypertension,
    "on_sedatives": on_sedatives, "on_diuretics": on_diuretics,
    "on_antihypertensives": on_antihypertensives, "on_anticoagulants": on_anticoagulants,
    "prior_fall": prior_fall, "uses_assistive_device": uses_assistive_device,
}


# ── Main area ────────────────────────────────────────────────────────────────
st.title("🏥 Fall Risk AI")
st.caption("XGBoost · SHAP Explainability · LLM Clinical Advisor (Gemma 4 via LM Studio)")
st.divider()

if not run_btn and "result" not in st.session_state:
    # Landing state
    st.info("👈 Fill in the patient form and click **Run Risk Analysis** to get started.", icon="ℹ️")
    st.stop()

# Load advisor
try:
    advisor = load_advisor(model_name)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Run analysis on button click
if run_btn:
    with st.spinner("Scoring patient and generating clinical advice…"):
        try:
            result = advisor.advise(patient)
            st.session_state["result"]  = result
            st.session_state["patient"] = patient
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

result  = st.session_state["result"]
patient = st.session_state["patient"]

score = result["risk_score"]
label = result["risk_label"]

# ── Risk score header ────────────────────────────────────────────────────────
col_badge, col_score, col_meta = st.columns([2, 1, 3])

with col_badge:
    st.markdown(risk_badge_html(label), unsafe_allow_html=True)
    st.markdown(
        f'<p class="gauge-label" style="color:{score_color(score)}">'
        f'{score:.1%}</p>',
        unsafe_allow_html=True
    )
    st.caption("Predicted fall probability")

with col_score:
    st.metric("Age", patient["age"])
    st.metric("Sex", patient["sex"])

with col_meta:
    dx_active  = [k.replace("has_","").replace("_"," ").title() for k,v in patient.items() if k.startswith("has_") and v]
    med_active = [k.replace("on_","").replace("_"," ").title()  for k,v in patient.items() if k.startswith("on_") and v]
    st.markdown(f"**Diagnoses:** {', '.join(dx_active) if dx_active else 'None'}")
    st.markdown(f"**Medications:** {', '.join(med_active) if med_active else 'None'}")
    st.markdown(f"**Prior fall:** {'Yes ⚠️' if patient['prior_fall'] else 'No'} &nbsp;|&nbsp; "
                f"**Assistive device:** {'Yes' if patient['uses_assistive_device'] else 'No'}")
    st.markdown(f"**Na** {patient['sodium']} &nbsp;|&nbsp; "
                f"**Hgb** {patient['hemoglobin']} &nbsp;|&nbsp; "
                f"**BUN** {patient['bun']}")

st.divider()

# ── Tabs: SHAP | LLM Output ──────────────────────────────────────────────────
tab_shap, tab_llm = st.tabs(["📊 SHAP Explainability", "🤖 LLM Clinical Advisor"])

with tab_shap:
    st.subheader("Feature Impact (SHAP Waterfall)")
    st.caption("Red bars push the risk score up; blue bars push it down. "
               "Starting point is the model's average prediction.")

    with st.spinner("Computing SHAP values…"):
        fig = shap_waterfall_fig(advisor, patient, dark=dark_chart)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Top factors table
    st.subheader("Top Contributing Factors")
    feature_labels = {
        "age": "Age", "bmi": "BMI", "prior_fall": "Prior fall",
        "has_parkinsons": "Parkinson's", "has_dementia": "Dementia",
        "has_osteoporosis": "Osteoporosis", "has_diabetes": "Diabetes",
        "has_depression": "Depression", "has_hypertension": "Hypertension",
        "on_sedatives": "Sedatives", "on_diuretics": "Diuretics",
        "on_antihypertensives": "Antihypertensives", "on_anticoagulants": "Anticoagulants",
        "uses_assistive_device": "Assistive device", "sodium": "Sodium",
        "hemoglobin": "Hemoglobin", "bun": "BUN",
        "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP",
        "heart_rate": "Heart Rate", "sex_M": "Sex (M)", "sex_F": "Sex (F)",
    }
    shap_df = pd.DataFrame(result["top_shap"], columns=["Feature", "SHAP Value"])
    shap_df["Feature"]   = shap_df["Feature"].map(lambda x: feature_labels.get(x, x))
    shap_df["Direction"] = shap_df["SHAP Value"].apply(lambda v: "▲ Increases risk" if v > 0 else "▼ Decreases risk")
    shap_df["Impact"]    = shap_df["SHAP Value"].abs().round(4)
    shap_df = shap_df[["Feature", "Direction", "Impact"]].sort_values("Impact", ascending=False)
    st.dataframe(shap_df, hide_index=True, use_container_width=True)

with tab_llm:
    explanation     = result.get("explanation", "")
    recommendations = result.get("recommendations", "")

    if not explanation and not recommendations:
        # Fallback: show raw response
        st.warning("LLM response could not be parsed into sections. Showing raw output.")
        st.text(result.get("full_response", "No response received."))
    else:
        st.subheader("Risk Explanation")
        st.markdown(f'<div class="llm-box">{explanation}</div>', unsafe_allow_html=True)

        st.subheader("Clinical Recommendations")
        rec_html = format_recommendations(recommendations)
        st.markdown(f'<div class="rec-box">{rec_html}</div>', unsafe_allow_html=True)

    with st.expander("🔍 Raw LLM response", expanded=False):
        st.text(result.get("full_response", ""))

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Clinical disclaimer:** This tool is for research and educational purposes only. "
    "AI-generated risk scores and recommendations must not replace clinical judgment. "
    "Always consult a qualified healthcare professional."
)
