# =====================================================
# PREMIUM XAI PREGNANCY RISK DASHBOARD
# FULL UPDATED APP.PY
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import lime
import lime.lime_tabular
import joblib

from recommendation_engine import generate_recommendations

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="XAI Pregnancy Risk Dashboard",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* =====================================================
BACKGROUND
===================================================== */

.stApp {

    background:
        radial-gradient(circle at top left, #7c3aed22, transparent 25%),
        radial-gradient(circle at top right, #2563eb22, transparent 25%),
        radial-gradient(circle at bottom left, #06b6d422, transparent 25%),
        linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #111827
        );

    color: white;
}

.block-container {

    padding-top: 1rem;

    max-width: 96%;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            rgba(15,23,42,0.98),
            rgba(30,41,59,0.98)
        );

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* =====================================================
HERO CARD
===================================================== */

.hero-card {

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.85),
            rgba(37,99,235,0.85)
        );

    border-radius: 35px;

    padding: 40px;

    margin-bottom: 25px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0px 10px 40px rgba(59,130,246,0.25);
}

.hero-title {

    font-size: 4rem;

    font-weight: 800;

    color: white;
}

.hero-sub {

    font-size: 1.2rem;

    color: #e2e8f0;
}

/* =====================================================
METRIC CARDS
===================================================== */

div[data-testid="metric-container"] {

    background:
        rgba(15,23,42,0.72);

    border-radius: 25px;

    padding: 22px;

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    box-shadow:
        0px 8px 25px rgba(0,0,0,0.3);
}

/* =====================================================
GLASS CARDS
===================================================== */

.glass-card {

    background:
        rgba(15,23,42,0.72);

    border-radius: 30px;

    padding: 25px;

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    margin-bottom: 20px;
}

/* =====================================================
RISK CARDS
===================================================== */

.low-card {

    background:
        linear-gradient(
            135deg,
            rgba(34,197,94,0.2),
            rgba(21,128,61,0.25)
        );

    border-left: 8px solid #22c55e;

    padding: 25px;

    border-radius: 25px;

    margin-bottom: 20px;
}

.moderate-card {

    background:
        linear-gradient(
            135deg,
            rgba(245,158,11,0.2),
            rgba(217,119,6,0.25)
        );

    border-left: 8px solid #f59e0b;

    padding: 25px;

    border-radius: 25px;

    margin-bottom: 20px;
}

.critical-card {

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.2),
            rgba(185,28,28,0.25)
        );

    border-left: 8px solid #ef4444;

    padding: 25px;

    border-radius: 25px;

    margin-bottom: 20px;
}

/* =====================================================
TABS
===================================================== */

button[data-baseweb="tab"] {

    background:
        rgba(15,23,42,0.8) !important;

    border-radius: 18px !important;

    padding: 14px 24px !important;

    border:
        1px solid rgba(255,255,255,0.08) !important;

    margin-right: 8px;

    color: white !important;

    font-weight: 600 !important;
}

button[aria-selected="true"] {

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        ) !important;

    color: white !important;
}

/* =====================================================
SCROLLBAR
===================================================== */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            180deg,
            #7c3aed,
            #2563eb
        );

    border-radius: 20px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD FILES
# =====================================================

model = joblib.load("rf_model.pkl")

scaler = joblib.load("scaler.pkl")

feature_names = joblib.load("feature_names.pkl")

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("dataa.csv")

df.columns = df.columns.str.strip().str.replace(" ", "")

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""

<div class='hero-card'>

<div class='hero-title'>
🩺 XAI Pregnancy Risk Dashboard
</div>

<div class='hero-sub'>

Explainable AI-powered maternal healthcare dashboard
using Random Forest, SHAP, and LIME.

</div>

</div>

""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Patient Parameters")

user_input = {}

for feature in feature_names:

    min_val = float(df[feature].min())

    max_val = float(df[feature].max())

    mean_val = float(df[feature].mean())

    user_input[feature] = st.sidebar.slider(

        feature,

        min_value=min_val,

        max_value=max_val,

        value=mean_val
    )

# =====================================================
# INPUT DATAFRAME
# =====================================================

input_df = pd.DataFrame([user_input])

scaled_input = scaler.transform(input_df)

# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(scaled_input)[0]

prediction_proba = model.predict_proba(scaled_input)[0]

# =====================================================
# THRESHOLD-BASED RISK BANDING
# =====================================================

max_prob = np.max(prediction_proba)

confidence = max_prob * 100

if max_prob < 0.60:

    predicted_risk = "Medium Risk"

    risk_status = "UNCERTAIN"

    risk_description = (
        "The model detected borderline clinical patterns "
        "with moderate prediction certainty."
    )

elif prediction == 1:

    predicted_risk = "High Risk"

    risk_status = "HIGH"

    risk_description = (
        "The model detected strong high-risk maternal indicators."
    )

else:

    predicted_risk = "Low Risk"

    risk_status = "LOW"

    risk_description = (
        "The model detected relatively stable maternal indicators."
    )

risk_score = round(
    prediction_proba[1] * 100,
    2
)

# =====================================================
# METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Risk",
        predicted_risk
    )

with col2:
    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

with col3:
    st.metric(
        "Risk Status",
        risk_status
    )

# =====================================================
# RISK CARD
# =====================================================

if predicted_risk == "High Risk":

    card_class = "critical-card"

elif predicted_risk == "Medium Risk":

    card_class = "moderate-card"

else:

    card_class = "low-card"

st.markdown(f"""

<div class='{card_class}'>

<h2>{predicted_risk}</h2>

<p>{risk_description}</p>

<p><b>Risk Score:</b> {risk_score}%</p>

<p><b>Prediction Confidence:</b> {confidence:.2f}%</p>

</div>

""", unsafe_allow_html=True)

# =====================================================
# TABS
# =====================================================

dashboard_tab, shap_tab, lime_tab, graphs_tab, rec_tab = st.tabs([
    "Dashboard",
    "SHAP Analysis",
    "LIME Analysis",
    "XAI Graphs",
    "Recommendations"
])

# =====================================================
# DASHBOARD TAB
# =====================================================

with dashboard_tab:

    col1, col2 = st.columns(2)

    # =================================================
    # DONUT CHART
    # =================================================

    fig1 = go.Figure(data=[go.Pie(

        labels=["Low Risk", "High Risk"],

        values=prediction_proba,

        hole=.65
    )])

    fig1.update_layout(

        paper_bgcolor="#111827",

        font_color="white",

        title="Risk Probability Distribution"
    )

    # =================================================
    # FEATURE IMPORTANCE
    # =================================================

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
            model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig2 = px.bar(

        importance_df.tail(5),

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        color_continuous_scale="blues"
    )

    fig2.update_layout(

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        font_color="white",

        title="Top Important Features"
    )

    with col1:
        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# =====================================================
# SHAP
# =====================================================

X = df.drop(columns=["RiskLevel"])

X = X.fillna(X.mean())

X_scaled = scaler.transform(X)


# =====================================================
# GLOBAL SHAP
# =====================================================

global_explainer = shap.TreeExplainer(model)

global_shap_values = global_explainer.shap_values(X_scaled)

if isinstance(global_shap_values, list):

    global_shap_data = np.abs(global_shap_values[1])

else:

    global_shap_values = np.array(global_shap_values)

    if len(global_shap_values.shape) == 3:

        global_shap_data = np.abs(
            global_shap_values[:, :, 1]
        )

    else:

        global_shap_data = np.abs(
            global_shap_values
        )

global_importance = np.mean(
    global_shap_data,
    axis=0
)

global_shap_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": global_importance

})

global_shap_df = global_shap_df.sort_values(
    by="Importance",
    ascending=False
)


# =====================================================
# SHAP TAB
# =====================================================
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(scaled_input)

if isinstance(shap_values, list):

    shap_data = np.array(shap_values[1][0])

else:

    shap_values = np.array(shap_values)

    if len(shap_values.shape) == 3:

        shap_data = shap_values[0, :, 1]

    elif len(shap_values.shape) == 2:

        shap_data = shap_values[0]

    else:

        shap_data = shap_values.flatten()

shap_data = np.array(shap_data).flatten()
with shap_tab:

    shap_df = pd.DataFrame({

        "Feature": feature_names,

        "SHAP":
            shap_data
    })

    shap_df["Abs"] = np.abs(
        shap_df["SHAP"]
    )

    shap_df = shap_df.sort_values(
        by="Abs",
        ascending=False
    ).head(10)

    if predicted_risk == "Medium Risk":

        st.warning("""

        SHAP detected mixed feature contributions,
        suggesting moderate prediction uncertainty.

        """)

    elif predicted_risk == "High Risk":

        st.error("""

        SHAP identified strong feature contributions
        toward high maternal risk.

        """)

    else:

        st.success("""

        SHAP contributions favor low maternal risk.

        """)

    fig3 = px.bar(

        shap_df,

        x="SHAP",

        y="Feature",

        orientation="h",

        color="SHAP",

        color_continuous_scale=[
            "#22c55e",
            "#f59e0b",
            "#ef4444"
        ]
    )

    fig3.update_layout(

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        font_color="white",

        title="SHAP Contributions"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =====================================================
# LIME TAB
# =====================================================

with lime_tab:

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(

        training_data=np.array(X_scaled),

        feature_names=feature_names,

        class_names=[
            "Low Risk",
            "High Risk"
        ],

        mode="classification"
    )

    lime_exp = lime_explainer.explain_instance(

        scaled_input[0],

        model.predict_proba,

        num_features=5
    )

    lime_list = lime_exp.as_list()

    lime_df = pd.DataFrame(

        lime_list,

        columns=[
            "Feature",
            "Contribution"
        ]
    )

    if predicted_risk == "Medium Risk":

        st.warning("""

        LIME identified competing local feature behavior,
        indicating prediction uncertainty.

        """)

    elif predicted_risk == "High Risk":

        st.error("""

        LIME identified strong local contributions
        toward high maternal risk.

        """)

    else:

        st.success("""

        LIME local explanations favor low maternal risk.

        """)

    fig4 = px.bar(

        lime_df,

        x="Contribution",

        y="Feature",

        orientation="h",

        color="Contribution",

        color_continuous_scale=[
            "#22c55e",
            "#f59e0b",
            "#ef4444"
        ]
    )

    fig4.update_layout(

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        font_color="white",

        title="LIME Contributions"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# =====================================================
# XAI GRAPHS TAB
# =====================================================

with graphs_tab:

    st.header("Explainable AI Visualizations")

    # ================================================
    # SHAP WATERFALL
    # ================================================

    st.subheader("SHAP Waterfall Plot")

    try:

        import matplotlib.pyplot as plt

        input_df = pd.DataFrame(
              scaled_input,
              columns=feature_names
        )

        explanation = explainer(
             input_df
        )

        shap.plots.waterfall(

            explanation[0,:,1],

            max_display=10,

            show=False
        )

        st.pyplot(
            plt.gcf(),
            clear_figure=True
        )

    except Exception as e:

        st.warning(
            f"Unable to generate SHAP Waterfall Plot: {e}"
        )

    st.markdown("---")

    # ================================================
    # SHAP SUMMARY BAR
    # ================================================

    st.subheader("SHAP Feature Importance")

    try:

        shap_importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance":
                np.abs(shap_data)

        })

        shap_importance = shap_importance.sort_values(
            by="Importance",
            ascending=False
        )

        fig_shap = px.bar(

            shap_importance.head(10),

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance",

            color_continuous_scale="Reds",

            title="Top SHAP Features"
        )

        fig_shap.update_layout(

            paper_bgcolor="#111827",

            plot_bgcolor="#111827",

            font_color="white"
        )

        st.plotly_chart(
            fig_shap,
            width="stretch"
        )

    except Exception as e:

        st.warning(
            f"Unable to generate SHAP Importance Plot: {e}"
        )

    st.markdown("---")

    # ================================================
    # LIME BAR CHART
    # ================================================

    st.subheader("LIME Feature Contributions")

    try:

        lime_sorted = lime_df.copy()

        lime_sorted = lime_sorted.sort_values(
            by="Contribution"
        )

        fig_lime = px.bar(

            lime_sorted,

            x="Contribution",

            y="Feature",

            orientation="h",

            color="Contribution",

            color_continuous_scale="RdYlGn_r",

            title="LIME Local Explanation"
        )

        fig_lime.update_layout(

            paper_bgcolor="#111827",

            plot_bgcolor="#111827",

            font_color="white"
        )

        st.plotly_chart(
            fig_lime,
            width="stretch"
        )

    except Exception as e:

        st.warning(
            f"Unable to generate LIME Bar Chart: {e}"
        )

    st.markdown("---")

    # ================================================
    # LIME PIE CHART
    # ================================================

    st.subheader("LIME Contribution Distribution")

    try:

        lime_pie = lime_df.copy()

        lime_pie["Contribution"] = (
            lime_pie["Contribution"]
            .abs()
        )

        fig_pie = px.pie(

            lime_pie,

            names="Feature",

            values="Contribution",

            title="Feature Contribution Share"
        )

        fig_pie.update_layout(

            paper_bgcolor="#111827",

            font_color="white"
        )

        st.plotly_chart(
            fig_pie,
            width="stretch"
        )

    except Exception as e:

        st.warning(
            f"Unable to generate LIME Pie Chart: {e}"
        )




# =====================================================
# RECOMMENDATIONS
# =====================================================

with rec_tab:

    recommendations = generate_recommendations(
        user_input,
        shap_data,
        lime_df
    )

    filtered_recommendations = [

        rec for rec in recommendations

        if rec["severity"] in [
            "Critical",
            "High",
            "Moderate"
        ]
    ]

    critical_count = len([
        r for r in filtered_recommendations
        if r["severity"] == "Critical"
    ])

    high_count = len([
        r for r in filtered_recommendations
        if r["severity"] == "High"
    ])

    moderate_count = len([
        r for r in filtered_recommendations
        if r["severity"] == "Moderate"
    ])

    st.info(
        f"""
        🚨 Critical: {critical_count}
        | ⚠️ High: {high_count}
        | ℹ️ Moderate: {moderate_count}
        """
    )

    if len(filtered_recommendations) == 0:

        st.success(
            "✅ No clinically significant maternal risk factors detected."
        )

    for rec in filtered_recommendations:

        severity = rec["severity"]

        if severity == "Critical":

            card = "critical-card"

        elif severity == "High":

            card = "moderate-card"

        else:

            card = "low-card"

        st.markdown(f"""

        <div class='{card}'>

        <h3>
        {rec['feature']} ({severity})
        </h3>

        <p>
        <b>Current Value:</b>
        {rec['value']}
        </p>

        <p>
        <b>SHAP Contribution:</b>
        {rec['shap_value']}
        </p>

        <p>
        <b>LIME Contribution:</b>
        {rec['lime_value']}
        </p>

        <p>
        <b>Observation:</b>
        {rec['observation']}
        </p>

        <p>
        <b>Clinical Risk:</b>
        {rec['clinical_risk']}
        </p>

        <p>
        <b>Recommendation:</b>
        {rec['recommendation']}
        </p>

        </div>

        """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""

<center>

<p>
Explainable AI-based Maternal Healthcare Dashboard
</p>


</center>

""", unsafe_allow_html=True)
