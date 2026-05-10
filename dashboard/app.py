from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.models.predict_model import predict_customer_churn


# ============================================================
# Path Configuration
# ============================================================

PROCESSED_DATA_PATH = "data/processed/processed_churn.csv"
FINAL_METRICS_PATH = "reports/final_evaluation/final_metrics.json"
FEATURE_IMPORTANCE_PATH = "reports/final_evaluation/final_feature_importance.csv"

CONFUSION_MATRIX_PATH = "reports/final_evaluation/final_confusion_matrix.png"
ROC_CURVE_PATH = "reports/final_evaluation/final_roc_curve.png"
PR_CURVE_PATH = "reports/final_evaluation/final_precision_recall_curve.png"

SHAP_GLOBAL_PATH = "reports/shap/shap_global_feature_importance.png"
SHAP_SUMMARY_PATH = "reports/shap/shap_summary_plot.png"
SHAP_WATERFALL_PATH = "reports/shap/shap_waterfall_high_risk_customer.png"

DRIFT_REPORT_PATH = "reports/monitoring/data_drift_report.html"


# ============================================================
# Streamlit Config
# ============================================================

st.set_page_config(
    page_title="Retail Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# Custom CSS Injection
# ============================================================

def inject_custom_css() -> None:
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">

        <style>
        /* ── Root Variables ────────────────────────────── */
        :root {
            --bg-base:       #0d0f14;
            --bg-surface:    #13161e;
            --bg-card:       #181c27;
            --bg-card-hover: #1e2333;
            --border:        #252a3a;
            --border-accent: #2e3550;

            --text-primary:   #e8ecf7;
            --text-secondary: #8b93b0;
            --text-muted:     #555d7a;

            --accent-cyan:    #00d4ff;
            --accent-violet:  #7c6cfa;
            --accent-emerald: #00e5a0;
            --accent-amber:   #f5a623;
            --accent-red:     #ff4d6d;

            --gradient-hero: linear-gradient(135deg, #0d0f14 0%, #131728 50%, #0d1220 100%);
            --gradient-card: linear-gradient(145deg, #181c27, #13161e);
            --glow-cyan:      0 0 30px rgba(0, 212, 255, 0.12);
            --glow-violet:    0 0 30px rgba(124, 108, 250, 0.15);

            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;

            --font-display: 'Syne', sans-serif;
            --font-body:    'DM Sans', sans-serif;
        }

        /* ── Global Reset ──────────────────────────────── */
        html, body, [class*="css"] {
            font-family: var(--font-body) !important;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }

        .stApp {
            background: var(--gradient-hero) !important;
        }

        /* ── Sidebar ───────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--bg-surface) !important;
            border-right: 1px solid var(--border) !important;
            padding-top: 1rem;
        }

        [data-testid="stSidebar"]::before {
            content: '';
            display: block;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
            margin-bottom: 1.5rem;
            border-radius: 0 0 4px 0;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] .stMarkdown h1 {
            font-family: var(--font-display) !important;
            font-size: 1.1rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            color: var(--text-secondary) !important;
            padding: 0 1rem 1rem !important;
        }

        [data-testid="stSidebarNav"],
        div[data-testid="stSidebar"] .stRadio {
            padding: 0 0.5rem;
        }

        [data-testid="stSidebar"] .stRadio label {
            font-family: var(--font-body) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
            padding: 0.55rem 0.9rem !important;
            border-radius: var(--radius-sm) !important;
            display: block !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            color: var(--text-primary) !important;
            background: rgba(0, 212, 255, 0.07) !important;
        }

        /* ── Page Titles ───────────────────────────────── */
        h1 {
            font-family: var(--font-display) !important;
            font-size: 2rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
            background: linear-gradient(90deg, var(--text-primary) 60%, var(--accent-cyan));
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            margin-bottom: 0.25rem !important;
        }

        h2, h3 {
            font-family: var(--font-display) !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.01em !important;
        }

        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1rem !important; color: var(--text-secondary) !important; }

        /* ── Metric Cards ──────────────────────────────── */
        [data-testid="stMetric"] {
            background: var(--gradient-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 1.2rem 1.4rem !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }

        [data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
            opacity: 0;
            transition: opacity 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-3px) !important;
            border-color: var(--border-accent) !important;
            box-shadow: var(--glow-cyan) !important;
        }

        [data-testid="stMetric"]:hover::before {
            opacity: 1;
        }

        [data-testid="stMetricLabel"] {
            font-family: var(--font-body) !important;
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
        }

        [data-testid="stMetricValue"] {
            font-family: var(--font-display) !important;
            font-size: 1.75rem !important;
            font-weight: 800 !important;
            color: var(--text-primary) !important;
            line-height: 1.1 !important;
        }

        /* ── Divider ───────────────────────────────────── */
        hr {
            border: none !important;
            border-top: 1px solid var(--border) !important;
            margin: 2rem 0 !important;
        }

        /* ── Dataframe / Tables ────────────────────────── */
        [data-testid="stDataFrame"],
        .stDataFrame {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
        }

        /* ── Charts ────────────────────────────────────── */
        [data-testid="stArrowVegaLiteChart"],
        [data-testid="stVegaLiteChart"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem !important;
        }

        /* ── Buttons ───────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
            color: #000 !important;
            font-family: var(--font-display) !important;
            font-weight: 700 !important;
            font-size: 0.875rem !important;
            letter-spacing: 0.04em !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.65rem 2rem !important;
            transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
        }

        .stButton > button:hover {
            opacity: 0.88 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 24px rgba(0, 212, 255, 0.25) !important;
        }

        /* ── Form Submit Button ────────────────────────── */
        [data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
            color: #000 !important;
            font-family: var(--font-display) !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.65rem 2rem !important;
            width: 100% !important;
            letter-spacing: 0.04em !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stFormSubmitButton"] > button:hover {
            opacity: 0.85 !important;
            box-shadow: 0 6px 24px rgba(0, 212, 255, 0.3) !important;
        }

        /* ── Inputs, Selects, Sliders ──────────────────── */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] > div > div {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text-primary) !important;
            font-family: var(--font-body) !important;
            transition: border-color 0.2s ease !important;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus {
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
        }

        .stSlider [data-testid="stThumbValue"] {
            background: var(--accent-cyan) !important;
            color: #000 !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--accent-cyan) !important;
            border-color: var(--accent-cyan) !important;
        }

        /* ── Alerts ────────────────────────────────────── */
        [data-testid="stAlert"][data-baseweb="notification"] {
            border-radius: var(--radius-md) !important;
            border-left-width: 3px !important;
        }

        .stSuccess {
            background: rgba(0, 229, 160, 0.08) !important;
            border-color: var(--accent-emerald) !important;
        }

        .stWarning {
            background: rgba(245, 166, 35, 0.08) !important;
            border-color: var(--accent-amber) !important;
        }

        .stError {
            background: rgba(255, 77, 109, 0.08) !important;
            border-color: var(--accent-red) !important;
        }

        .stInfo {
            background: rgba(0, 212, 255, 0.06) !important;
            border-color: var(--accent-cyan) !important;
        }

        /* ── Form Container ────────────────────────────── */
        [data-testid="stForm"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.75rem !important;
        }

        /* ── Images ────────────────────────────────────── */
        [data-testid="stImage"] img {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border) !important;
        }

        /* ── Scrollbar ─────────────────────────────────── */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb {
            background: var(--border-accent);
            border-radius: 99px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

        /* ── Column gaps ───────────────────────────────── */
        [data-testid="stHorizontalBlock"] {
            gap: 1rem !important;
        }

        /* ── Subheader pill badges ─────────────────────── */
        .badge {
            display: inline-block;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.25);
            color: var(--accent-cyan);
            font-family: var(--font-body);
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.2rem 0.65rem;
            border-radius: 99px;
            margin-bottom: 0.75rem;
        }

        /* ── Section card wrapper ──────────────────────── */
        .section-card {
            background: var(--gradient-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        /* ── Page header strip ─────────────────────────── */
        .page-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .page-header-icon {
            font-size: 1.6rem;
            line-height: 1;
        }

        /* ── Subheader override ────────────────────────── */
        .stSubheader {
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Helper Functions
# ============================================================

@st.cache_data
def load_processed_data() -> pd.DataFrame:
    data_path = Path(PROCESSED_DATA_PATH)
    if not data_path.exists():
        return pd.DataFrame()
    return pd.read_csv(data_path)


@st.cache_data
def load_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    file_path = Path(FEATURE_IMPORTANCE_PATH)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def format_percentage(value: float) -> str:
    return f"{value:.2%}"


def show_missing_file_warning(path: str) -> None:
    st.warning(f"File not found: `{path}`")


def get_risk_summary(df: pd.DataFrame) -> tuple[int, int]:
    if "risk_segment" not in df.columns:
        return 0, 0
    high_risk_count = df[df["risk_segment"].isin(["High Risk", "Critical Risk"])].shape[0]
    critical_risk_count = df[df["risk_segment"] == "Critical Risk"].shape[0]
    return high_risk_count, critical_risk_count


def badge(text: str) -> None:
    st.markdown(f'<span class="badge">{text}</span>', unsafe_allow_html=True)


def section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a styled page header with icon."""
    sub = f"<p style='color:var(--text-secondary);font-size:0.9rem;margin:0.2rem 0 0;font-family:var(--font-body)'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div style="margin-bottom:2rem;padding-bottom:1.25rem;border-bottom:1px solid var(--border)">
          <div style="display:flex;align-items:center;gap:0.75rem">
            <span style="font-size:1.75rem;line-height:1">{icon}</span>
            <div>
              <h1 style="margin:0">{title}</h1>
              {sub}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Page 1: Executive Summary
# ============================================================

def page_executive_summary(df: pd.DataFrame) -> None:
    section_header("📊", "Executive Summary", "High-level churn metrics and business overview")

    if df.empty:
        show_missing_file_warning(PROCESSED_DATA_PATH)
        return

    total_customers = len(df)
    overall_churn_rate = df["churn_flag"].mean() if "churn_flag" in df.columns else 0
    high_risk_customers, critical_risk_customers = get_risk_summary(df)
    estimated_revenue_at_risk = (
        df["revenue_at_risk"].sum()
        if "revenue_at_risk" in df.columns
        else df.loc[df["churn_flag"] == 1, "total_spent"].sum()
        if {"churn_flag", "total_spent"}.issubset(df.columns)
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Overall Churn Rate", format_percentage(overall_churn_rate))
    col3.metric("High-Risk Customers", f"{high_risk_customers:,}")
    col4.metric("Critical-Risk Customers", f"{critical_risk_customers:,}")
    col5.metric("Revenue at Risk", f"{estimated_revenue_at_risk:,.2f}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        badge("Distribution")
        st.subheader("Churn Distribution")
        churn_counts = df["churn_flag"].value_counts().sort_index()
        churn_counts.index = churn_counts.index.map({0: "Not Churn", 1: "Churn"})
        st.bar_chart(churn_counts, color="#00d4ff")

    with col_b:
        badge("Segments")
        st.subheader("Risk Segment Distribution")
        if "risk_segment" in df.columns:
            risk_counts = df["risk_segment"].value_counts()
            st.bar_chart(risk_counts, color="#7c6cfa")
        else:
            st.info("`risk_segment` column is not available.")

    st.divider()

    badge("Business Summary")
    st.markdown(
        f"""
        <div class="section-card">
        <p style="font-family:var(--font-body);font-size:0.95rem;line-height:1.8;color:var(--text-secondary);margin:0">
            The dataset contains <strong style="color:var(--text-primary)">{total_customers:,} customers</strong>
            with an overall churn rate of
            <strong style="color:var(--accent-cyan)">{overall_churn_rate:.2%}</strong>.<br>
            There are <strong style="color:var(--accent-red)">{high_risk_customers:,} high-risk or critical-risk customers</strong>.
            The estimated revenue at risk is
            <strong style="color:var(--accent-amber)">{estimated_revenue_at_risk:,.2f}</strong>.<br><br>
            From a business perspective, the company should prioritize customers who combine
            high churn risk with high customer value — these represent the most urgent retention opportunity.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Page 2: Customer Segmentation
# ============================================================

def page_customer_segmentation(df: pd.DataFrame) -> None:
    section_header("👥", "Customer Segmentation", "Behavioral and value breakdown by segment")

    if df.empty:
        show_missing_file_warning(PROCESSED_DATA_PATH)
        return

    badge("Churn Analysis")
    st.subheader("Churn Rate by Customer Segment")

    if {"customer_segment", "churn_flag"}.issubset(df.columns):
        churn_by_segment = (
            df.groupby("customer_segment")["churn_flag"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(churn_by_segment, color="#00d4ff")
        st.dataframe(
            churn_by_segment.reset_index().rename(columns={"churn_flag": "churn_rate"}),
            use_container_width=True,
        )
    else:
        st.info("Required columns are not available.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        badge("Spending")
        st.subheader("Spending Behavior by Segment")
        if {"customer_segment", "total_spent"}.issubset(df.columns):
            spending_by_segment = (
                df.groupby("customer_segment")["total_spent"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(spending_by_segment, color="#00e5a0")
        else:
            st.info("Spending data is not available.")

    with col2:
        badge("Loyalty")
        st.subheader("Loyalty Behavior by Risk Segment")
        if {"risk_segment", "loyalty_score"}.issubset(df.columns):
            loyalty_by_risk = (
                df.groupby("risk_segment")["loyalty_score"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(loyalty_by_risk, color="#7c6cfa")
        else:
            st.info("Risk segment or loyalty score is not available.")

    st.divider()

    badge("Cross-Tab")
    st.subheader("Risk Distribution by Customer Segment")
    if {"customer_segment", "risk_segment"}.issubset(df.columns):
        risk_segment_table = pd.crosstab(
            df["customer_segment"],
            df["risk_segment"],
            normalize="index",
        )
        st.dataframe(risk_segment_table, use_container_width=True)
    else:
        st.info("Required columns are not available.")


# ============================================================
# Page 3: Prediction App
# ============================================================

def page_prediction_app() -> None:
    section_header("🔮", "Prediction App", "Generate churn probability and recommended retention action")

    st.markdown(
        "<p style='color:var(--text-secondary);font-size:0.9rem;margin-bottom:1.5rem'>"
        "Fill in the customer profile below and click <strong style='color:var(--text-primary)'>Predict Churn</strong> to get results."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            badge("Identity")
            customer_id = st.text_input("Customer ID", value="CUST_001")
            age_group = st.selectbox(
                "Age Group",
                ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
                index=2,
            )
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
            customer_segment = st.selectbox("Customer Segment", ["New", "Returning", "VIP"])

        with col2:
            badge("Behavior")
            preferred_channel = st.selectbox("Preferred Channel", ["Online", "Mobile App", "In-Store"])
            purchase_frequency = st.number_input("Purchase Frequency", min_value=0.0, value=3.0)
            avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=45.5)
            total_spent = st.number_input("Total Spent", min_value=0.0, value=1250.0)
            recency_days = st.number_input("Recency Days", min_value=0.0, value=75.0)

        with col3:
            badge("Engagement")
            website_visits = st.number_input("Website Visits", min_value=0.0, value=20.0)
            discount_usage_rate = st.slider("Discount Usage Rate", min_value=0.0, max_value=1.0, value=0.65)
            email_open_rate = st.slider("Email Open Rate", min_value=0.0, max_value=1.0, value=0.30)
            cart_abandonment_rate = st.slider("Cart Abandonment Rate", min_value=0.0, max_value=1.0, value=0.72)
            loyalty_score = st.slider("Loyalty Score", min_value=0.0, max_value=100.0, value=35.0)
            engagement_score = st.slider("Engagement Score", min_value=0.0, max_value=100.0, value=40.0)

        submitted = st.form_submit_button("⚡ Predict Churn")

    if submitted:
        input_data = {
            "customer_id": customer_id,
            "age_group": age_group,
            "gender": gender,
            "region": region,
            "customer_segment": customer_segment,
            "preferred_channel": preferred_channel,
            "purchase_frequency": purchase_frequency,
            "avg_order_value": avg_order_value,
            "total_spent": total_spent,
            "recency_days": recency_days,
            "website_visits": website_visits,
            "discount_usage_rate": discount_usage_rate,
            "email_open_rate": email_open_rate,
            "cart_abandonment_rate": cart_abandonment_rate,
            "loyalty_score": loyalty_score,
            "engagement_score": engagement_score,
        }

        try:
            prediction_df = predict_customer_churn(input_data)
            result = prediction_df.iloc[0]

            st.success("✅ Prediction completed successfully.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            col2.metric("Prediction", result["prediction_text"])
            col3.metric("Risk Level", result["risk_level"])

            st.subheader("Recommended Action")
            st.info(f"💡 {result['recommended_action']}")

            badge("Raw Output")
            st.subheader("Prediction Output")
            st.dataframe(prediction_df, use_container_width=True)

        except Exception as error:
            st.error(f"❌ Prediction failed: {error}")


# ============================================================
# Page 4: Model Insights
# ============================================================

def page_model_insights() -> None:
    section_header("🧠", "Model Insights", "Performance metrics, feature importance, and SHAP explainability")

    metrics = load_json(FINAL_METRICS_PATH)

    badge("Evaluation")
    st.subheader("Final Model Metrics")

    if metrics:
        metric_table = pd.DataFrame(
            [
                {"Metric": key, "Value": value}
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            ]
        )
        st.dataframe(metric_table, use_container_width=True)
    else:
        show_missing_file_warning(FINAL_METRICS_PATH)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        badge("Evaluation")
        st.subheader("Confusion Matrix")
        if Path(CONFUSION_MATRIX_PATH).exists():
            st.image(CONFUSION_MATRIX_PATH, use_container_width=True)
        else:
            show_missing_file_warning(CONFUSION_MATRIX_PATH)

    with col2:
        badge("Importance")
        st.subheader("Feature Importance")
        feature_importance = load_feature_importance()
        if not feature_importance.empty:
            top_features = feature_importance.head(20).set_index("feature")
            st.bar_chart(top_features["importance"], color="#7c6cfa")
            st.dataframe(feature_importance.head(20), use_container_width=True)
        else:
            show_missing_file_warning(FEATURE_IMPORTANCE_PATH)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        badge("Curves")
        st.subheader("ROC Curve")
        if Path(ROC_CURVE_PATH).exists():
            st.image(ROC_CURVE_PATH, use_container_width=True)
        else:
            show_missing_file_warning(ROC_CURVE_PATH)

    with col4:
        badge("Curves")
        st.subheader("Precision-Recall Curve")
        if Path(PR_CURVE_PATH).exists():
            st.image(PR_CURVE_PATH, use_container_width=True)
        else:
            show_missing_file_warning(PR_CURVE_PATH)

    st.divider()

    badge("Explainability")
    st.subheader("SHAP Analysis")

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("### Global SHAP Importance")
        if Path(SHAP_GLOBAL_PATH).exists():
            st.image(SHAP_GLOBAL_PATH, use_container_width=True)
        else:
            show_missing_file_warning(SHAP_GLOBAL_PATH)

    with shap_col2:
        st.markdown("### SHAP Summary Plot")
        if Path(SHAP_SUMMARY_PATH).exists():
            st.image(SHAP_SUMMARY_PATH, use_container_width=True)
        else:
            show_missing_file_warning(SHAP_SUMMARY_PATH)

    st.markdown("### Local Explanation: High-Risk Customer")
    if Path(SHAP_WATERFALL_PATH).exists():
        st.image(SHAP_WATERFALL_PATH, use_container_width=True)
    else:
        show_missing_file_warning(SHAP_WATERFALL_PATH)

    st.markdown(
        """
        <div class="section-card" style="margin-top:1rem">
        <p style="font-family:var(--font-body);font-size:0.9rem;line-height:1.8;color:var(--text-secondary);margin:0">
            <strong style="color:var(--accent-cyan)">SHAP</strong> explains which features push the model toward churn or non-churn.
            <strong style="color:var(--accent-red)">Positive SHAP values</strong> increase the churn prediction,
            while <strong style="color:var(--accent-emerald)">negative SHAP values</strong> reduce it.
            This is critical for business users: the model not only predicts churn, but also surfaces
            the behavioral drivers behind each prediction.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Page 5: Monitoring
# ============================================================

def page_monitoring() -> None:
    section_header("📈", "Monitoring", "Data drift, prediction drift, and model health tracking")

    if Path(DRIFT_REPORT_PATH).exists():
        badge("Drift Report")
        st.subheader("Data Drift Report")
        with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as file:
            drift_html = file.read()
        components.html(drift_html, height=900, scrolling=True)

    else:
        st.warning(
            "⚠️ Monitoring report is not available yet. "
            "Generate `reports/monitoring/data_drift_report.html` in the monitoring step."
        )

        st.markdown(
            """
            <div class="section-card">
            <p style="font-family:var(--font-body);font-size:0.875rem;color:var(--text-secondary);margin:0 0 0.75rem">
                Expected monitoring outputs:
            </p>
            <ul style="font-family:var(--font-body);font-size:0.875rem;color:var(--text-secondary);margin:0;padding-left:1.25rem;line-height:2">
                <li>Data drift report</li>
                <li>Prediction drift report</li>
                <li>Feature drift report</li>
                <li>Monthly model performance comparison</li>
                <li>Alert if new data distribution differs from training data</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0 0.5rem 1.5rem">
              <p style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;
                         color:#e8ecf7;letter-spacing:0.02em;margin:0 0 0.2rem">
                Churn Dashboard
              </p>
              <p style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
                         color:#555d7a;margin:0;letter-spacing:0.04em;text-transform:uppercase">
                Retail Analytics · v1.0
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            [
                "📊  Executive Summary",
                "👥  Customer Segmentation",
                "🔮  Prediction App",
                "🧠  Model Insights",
                "📈  Monitoring",
            ],
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div style="position:fixed;bottom:1.5rem;left:0;width:var(--sidebar-width,280px);
                         padding:0 1.25rem;box-sizing:border-box">
              <p style="font-family:'DM Sans',sans-serif;font-size:0.72rem;
                         color:#555d7a;margin:0;text-align:center">
                Built with Streamlit · Powered by XGBoost
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ============================================================
# Main App
# ============================================================

def main() -> None:
    inject_custom_css()

    page = render_sidebar()
    df = load_processed_data()

    if "Executive Summary" in page:
        page_executive_summary(df)
    elif "Customer Segmentation" in page:
        page_customer_segmentation(df)
    elif "Prediction App" in page:
        page_prediction_app()
    elif "Model Insights" in page:
        page_model_insights()
    elif "Monitoring" in page:
        page_monitoring()


if __name__ == "__main__":
    main()