# =============================================================================
# FinproDatmin.py — FIXED VERSION
# Streamlit Hybrid Country Recommendation Dashboard
# =============================================================================

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Cost of Living Dashboard",
    page_icon="🌍",
    layout="wide"
)

# =============================================================================
# THEME
# =============================================================================
LIGHT = {
    "app_bg": "#fcfcfc",
    "surface": "#f4f2ef",
    "card_bg": "#ffffff",
    "primary": "#6f816e",
    "accent": "#936b43",
    "text": "#0e1011",
    "muted": "#7a7872",
    "border": "#e0dbd4",
    "chart": "#faf9f7",
    "tbl_hdr": "#ece9e3",
}

DARK = {
    "app_bg": "#141618",
    "surface": "#1e2123",
    "card_bg": "#1e2123",
    "primary": "#7c9179",
    "accent": "#b8895a",
    "text": "#e4e1da",
    "muted": "#888680",
    "border": "#2c3033",
    "chart": "#232729",
    "tbl_hdr": "#272b2d",
}

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("🌍 Cost of Living")

    dark_mode = st.toggle("Dark Mode", value=False)

T = DARK if dark_mode else LIGHT

# =============================================================================
# CSS FIXED
# =============================================================================
css = """
<style>

html, body, .stApp {{
    background-color: {app_bg};
    color: {text};
    font-family: sans-serif;
}}

section[data-testid="stSidebar"] {{
    background-color: {surface};
}}

.metric-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 10px;
}}

.metric-title {{
    color: {muted};
    font-size: 12px;
}}

.metric-value {{
    font-size: 28px;
    font-weight: bold;
    color: {text};
}}

.custom-table {{
    width:100%;
    border-collapse: collapse;
}}

.custom-table th {{
    background:{tbl_hdr};
    color:{text};
    padding:10px;
    text-align:left;
}}

.custom-table td {{
    padding:10px;
    border-bottom:1px solid {border};
}}

.stButton > button {{
    background:{primary};
    color:white;
    border:none;
    border-radius:10px;
}}

</style>
""".format(
    app_bg=T["app_bg"],
    text=T["text"],
    surface=T["surface"],
    card_bg=T["card_bg"],
    border=T["border"],
    muted=T["muted"],
    primary=T["primary"],
    tbl_hdr=T["tbl_hdr"]
)

st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# LOAD DATA
# =============================================================================
@st.cache_data
def load_data():

    file_path = "cost-of-living.csv"

    df = pd.read_csv(file_path)

    x_cols = [c for c in df.columns if c.startswith("x")]

    df[x_cols] = df[x_cols].fillna(df[x_cols].median())

    country_df = df.groupby("country")[x_cols].mean().reset_index()

    country_df["CLI"] = (
        0.35 * country_df["x1"] +
        0.40 * country_df["x28"] +
        0.15 * country_df["x36"] +
        0.10 * country_df["x48"]
    )

    country_df["Recommendation_Score"] = (
        country_df["x54"] / country_df["CLI"]
    )

    feature_cols = [
        "x1",
        "x3",
        "x8",
        "x28",
        "x33",
        "x36",
        "x48",
        "x49",
        "x54",
        "CLI"
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(country_df[feature_cols])

    cosine = cosine_similarity(scaled)

    cosine_df = pd.DataFrame(
        cosine,
        index=country_df["country"],
        columns=country_df["country"]
    )

    return country_df, scaled, cosine_df


country_data, feature_scaled, cosine_df = load_data()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
with st.sidebar:

    countries = sorted(country_data["country"].unique())

    default_index = 0

    if "Germany" in countries:
        default_index = countries.index("Germany")

    user_country = st.selectbox(
        "Reference Country",
        countries,
        index=default_index
    )

    budget = st.slider(
        "Max Budget",
        200,
        5000,
        2000
    )

    salary = st.slider(
        "Min Salary",
        100,
        10000,
        1000
    )

    top_n = st.slider(
        "Top N",
        3,
        20,
        10
    )

    k = st.slider(
        "Clusters",
        2,
        8,
        4
    )

# =============================================================================
# HTML TABLE
# =============================================================================
def html_table(df):

    headers = "".join([f"<th>{c}</th>" for c in df.columns])

    rows = ""

    for _, row in df.iterrows():

        cells = ""

        for val in row:
            cells += f"<td>{val}</td>"

        rows += f"<tr>{cells}</tr>"

    return f"""
    <table class='custom-table'>
        <thead>
            <tr>{headers}</tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

# =============================================================================
# HEADER
# =============================================================================
st.title("🌍 Cost of Living Recommendation Dashboard")

# =============================================================================
# METRICS
# =============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Countries</div>
        <div class='metric-value'>{len(country_data)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Average CLI</div>
        <div class='metric-value'>
            ${country_data["CLI"].mean():.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Average Salary</div>
        <div class='metric-value'>
            ${country_data["x54"].mean():.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAP
# =============================================================================
st.subheader("Global Recommendation Score")

fig_map = px.choropleth(
    country_data,
    locations="country",
    locationmode="country names",
    color="Recommendation_Score",
    hover_name="country",
    color_continuous_scale="Viridis"
)

fig_map.update_layout(
    paper_bgcolor=T["card_bg"],
    plot_bgcolor=T["card_bg"],
    font=dict(color=T["text"])
)

st.plotly_chart(fig_map, use_container_width=True)

# =============================================================================
# TOP COUNTRIES
# =============================================================================
st.subheader("Top Countries")

top_df = (
    country_data[
        ["country", "CLI", "x54", "Recommendation_Score"]
    ]
    .sort_values("Recommendation_Score", ascending=False)
    .head(10)
)

top_df.columns = [
    "Country",
    "CLI",
    "Salary",
    "Score"
]

st.markdown(
    html_table(top_df.round(2)),
    unsafe_allow_html=True
)

# =============================================================================
# CORRELATION
# =============================================================================
st.subheader("Correlation Heatmap")

sel_cols = [
    "x1",
    "x3",
    "x8",
    "x28",
    "x33",
    "x36",
    "x48",
    "x49",
    "x54",
    "CLI"
]

corr = country_data[sel_cols].corr()

fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor(T["card_bg"])

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# =============================================================================
# FEATURE SELECTION
# =============================================================================
st.subheader("Feature Importance")

X = country_data[[c for c in country_data.columns if c.startswith("x")]]

y = country_data["x54"]

mi = mutual_info_regression(X, y)

mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI Score": mi
}).sort_values("MI Score", ascending=False)

fig_mi = px.bar(
    mi_df.head(15),
    x="MI Score",
    y="Feature",
    orientation="h",
    color="MI Score",
    color_continuous_scale="Viridis"
)

fig_mi.update_layout(
    paper_bgcolor=T["card_bg"],
    plot_bgcolor=T["card_bg"],
    font=dict(color=T["text"])
)

st.plotly_chart(fig_mi, use_container_width=True)

# =============================================================================
# KMEANS
# =============================================================================
st.subheader("KMeans Clustering")

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(feature_scaled)

pca = PCA(n_components=2)

coords = pca.fit_transform(feature_scaled)

pca_df = pd.DataFrame({
    "PC1": coords[:, 0],
    "PC2": coords[:, 1],
    "Cluster": labels.astype(str),
    "Country": country_data["country"]
})

fig_cluster = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name="Country"
)

fig_cluster.update_layout(
    paper_bgcolor=T["card_bg"],
    plot_bgcolor=T["card_bg"],
    font=dict(color=T["text"])
)

st.plotly_chart(fig_cluster, use_container_width=True)

# =============================================================================
# RECOMMENDER
# =============================================================================
st.subheader("Country Recommendation")

sim_df = (
    cosine_df[user_country]
    .drop(user_country)
    .sort_values(ascending=False)
    .reset_index()
)

sim_df.columns = [
    "country",
    "similarity"
]

filtered = country_data[
    (country_data["CLI"] <= budget) &
    (country_data["x54"] >= salary)
]

result = sim_df.merge(
    filtered,
    on="country"
).head(top_n)

if result.empty:

    st.warning("No recommendation found.")

else:

    result["Similarity %"] = (
        result["similarity"] * 100
    ).round(1)

    show_df = result[
        [
            "country",
            "Similarity %",
            "CLI",
            "x54",
            "Recommendation_Score"
        ]
    ]

    show_df.columns = [
        "Country",
        "Similarity %",
        "CLI",
        "Salary",
        "Score"
    ]

    st.markdown(
        html_table(show_df.round(2)),
        unsafe_allow_html=True
    )

    fig_rec = px.bar(
        show_df.sort_values("Similarity %"),
        x="Similarity %",
        y="Country",
        orientation="h",
        color="Similarity %",
        color_continuous_scale="Viridis"
    )

    fig_rec.update_layout(
        paper_bgcolor=T["card_bg"],
        plot_bgcolor=T["card_bg"],
        font=dict(color=T["text"])
    )

    st.plotly_chart(fig_rec, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

st.caption(
    "Final Project Data Mining • KMeans • PCA • Cosine Similarity"
)
