=============================================================================
# FinproDatmin.py — Hybrid Country Recommendation Dashboard (Streamlit)
# Final Project Data Mining — Cost of Living Analysis
# Theme: Earthy Minimalist — Sage green + Warm tan, with Dark Mode toggle
# =============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title            = "Cost of Living · Data Mining",
    page_icon             = "🌍",
    layout                = "wide",
    initial_sidebar_state = "auto"   
)

# ===========================================================================
# THEME PALETTES
# ===========================================================================
LIGHT = {
    "app_bg"   : "#fcfcfc",
    "surface"  : "#f4f2ef",
    "surface2" : "#ece9e3",
    "card_bg"  : "#ffffff",
    "primary"  : "#6f816e",
    "primary_d": "#556354",
    "second"   : "#c9b79c",
    "accent"   : "#936b43",
    "text"     : "#0e1011",
    "muted"    : "#7a7872",
    "border"   : "#e0dbd4",
    "pink_lt"  : "#f4f0ea",
    "chart_pl" : "#faf9f7",
    # tabel HTML
    "tbl_hdr"  : "#ece9e3",
    "tbl_row"  : "#ffffff",
    "tbl_alt"  : "#f4f2ef",
    "tbl_txt"  : "#0e1011",
    "tbl_muted": "#7a7872",
}
DARK = {
    "app_bg"   : "#141618",
    "surface"  : "#1e2123",
    "surface2" : "#272b2d",
    "card_bg"  : "#1e2123",
    "primary"  : "#7c9179",
    "primary_d": "#a3b8a0",
    "second"   : "#c9b79c",
    "accent"   : "#b8895a",
    "text"     : "#e4e1da",
    "muted"    : "#888680",
    "border"   : "#2c3033",
    "pink_lt"  : "#1c2420",
    "chart_pl" : "#232729",
    # tabel HTML
    "tbl_hdr"  : "#272b2d",
    "tbl_row"  : "#1e2123",
    "tbl_alt"  : "#232729",
    "tbl_txt"  : "#e4e1da",
    "tbl_muted": "#888680",
}

# ===========================================================================
# SIDEBAR — Dark Mode Toggle HARUS PERTAMA sebelum T dipakai
# ===========================================================================
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;padding:4px 0 14px 0;'>
        <div style='width:42px;height:42px;border-radius:12px;
                    background:linear-gradient(135deg,#6f816e,#936b43);
                    display:flex;align-items:center;justify-content:center;font-size:20px;
                    box-shadow:0 3px 12px rgba(111,129,110,0.35);flex-shrink:0;'>&#127757;</div>
        <div>
            <div style='font-size:14px;font-weight:700;line-height:1.2;'>Cost of Living</div>
            <div style='font-size:11px;opacity:0.5;font-weight:500;margin-top:2px;'>
                Data Mining &middot; Final Project</div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0 10px 0;'>", unsafe_allow_html=True)
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    st.markdown("<hr style='margin:10px 0 8px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:10px;font-weight:700;letter-spacing:0.12em;"
        "opacity:0.5;text-transform:uppercase;'>Filters</span>",
        unsafe_allow_html=True
    )

# Resolve theme SEGERA setelah toggle dibaca
T = DARK if dark_mode else LIGHT

# ===========================================================================
# CSS INJECTION — semua warna sudah pakai T yang benar
# ===========================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp {{
    font-family: 'Sora', sans-serif !important;
    background-color: {T['app_bg']} !important;
    color: {T['text']} !important;
}}

#MainMenu {{ display: none !important; }}
footer {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}

header[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {T['surface']} !important;
    border-right: 1px solid {T['border']} !important;
}}

.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1450px !important;
}}

.js-plotly-plot, .plot-container {{
    border-radius: 18px !important;
    overflow: hidden !important;
}}
/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {T['surface']} !important;
    border-right: 1px solid {T['border']} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {T['text']} !important;
    font-family: 'Sora', sans-serif !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: {T['border']} !important;
    margin: 6px 0 !important;
}}

/* Sidebar widget cards */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stMultiSelect {{
    background-color: {T['card_bg']} !important;
    padding: 8px 10px !important;
    border-radius: 10px !important;
    border: 1px solid {T['border']} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    margin-bottom: 6px !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background-color: {T['card_bg']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
}}
section[data-testid="stSidebar"] .stNumberInput input {{
    background-color: {T['card_bg']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
    font-weight: 600 !important;
}}
section[data-testid="stSidebar"] .stNumberInput input:focus {{
    border-color: {T['primary']} !important;
    box-shadow: 0 0 0 3px {T['primary']}28 !important;
    outline: none !important;
}}

/* Slider — track abu-abu bersih, thumb sage */
section[data-testid="stSidebar"] .stSlider > div > div > div {{
    background: {T['border']} !important;
}}
section[data-testid="stSidebar"] .stSlider > div > div > div > div {{
    background: {T['primary']} !important;
}}
section[data-testid="stSidebar"] .stSlider [role="slider"] {{
    background-color: {T['primary']} !important;
    border: 2px solid {T['card_bg']} !important;
    box-shadow: 0 0 0 2.5px {T['primary']} !important;
}}

/* FIX: angka tick slider — hapus background putih/abu stabilo */
section[data-testid="stSidebar"] .stSlider span,
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {{
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    color: {T['muted']} !important;
    font-size: 10px !important;
}}

/* Number +/- buttons */
section[data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] {{
    background-color: {T['card_bg']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] .stNumberInput button {{
    background-color: {T['surface2']} !important;
    color: {T['text']} !important;
    border: none !important;
    border-left: 1px solid {T['border']} !important;
}}

/* Run button — gradient sage→amber */
section[data-testid="stSidebar"] .stButton > button {{
    background: linear-gradient(135deg, {T['primary']}, {T['accent']}) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 3px 12px {T['primary']}55 !important;
    transition: all 0.18s ease !important;
    font-family: 'Sora', sans-serif !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}}

/* Dark Mode Toggle */
[data-baseweb="toggle"] {{
    background-color: #bdb7ae !important;
    transition: background-color 0.2s !important;
}}
[data-baseweb="toggle"][aria-checked="true"] {{
    background-color: {T['primary']} !important;
}}
[data-baseweb="toggle"] div {{
    background-color: #ffffff !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {T['surface']} !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1px solid {T['border']} !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: {T['muted']} !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    padding: 8px 22px !important;
    border: none !important;
    background: transparent !important;
    font-family: 'Sora', sans-serif !important;
    transition: all 0.18s ease !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: {T['card_bg']} !important;
    color: {T['text']} !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.09) !important;
    font-weight: 700 !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

/* ── Metric Cards ── */
.metric-card {{
    background-color: {T['card_bg']};
    border: 1px solid {T['border']};
    border-top: 3px solid {T['primary']};
    border-radius: 16px;
    padding: 20px 22px 16px 22px;
    margin-bottom: 8px;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}}
.metric-card:hover {{
    box-shadow: 0 6px 24px rgba(111,129,110,0.15);
    transform: translateY(-2px);
}}
.metric-top  {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
.metric-name {{ font-size:11px; font-weight:700; color:{T['muted']}; text-transform:uppercase; letter-spacing:0.09em; }}
.metric-icon {{ font-size:18px; opacity:0.7; }}
.metric-value {{ font-size:32px; font-weight:700; color:{T['text']}; letter-spacing:-1.5px; line-height:1; font-family:'Sora',sans-serif; }}
.metric-trend {{
    display:inline-flex; align-items:center; gap:5px;
    font-size:12px; font-weight:600; color:{T['primary']}; margin-top:12px;
    background:{T['primary']}1a; padding:3px 10px; border-radius:20px;
}}

/* ── Section titles ── */
.section-title {{
    font-size:11px; font-weight:700; color:{T['muted']};
    margin: 24px 0 12px 0;
    display:flex; align-items:center; gap:10px;
    text-transform:uppercase; letter-spacing:0.10em;
}}
.section-title::after {{ content:''; flex:1; height:1px; background:{T['border']}; }}

/* ── Info box ── */
.info-box {{
    background: {T['surface']};
    border-radius: 12px;
    padding: 11px 16px;
    margin-bottom: 14px;
    border-left: 3px solid {T['primary']};
    color: {T['muted']};
    font-size: 12.5px;
    line-height: 1.65;
}}

/* ── Page header ── */
.page-header {{
    display:flex; align-items:baseline; justify-content:space-between;
    margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid {T['border']};
}}
.page-title {{ font-size:20px; font-weight:700; color:{T['text']}; margin:0; font-family:'Sora',sans-serif; }}
.page-sub   {{ font-size:12px; color:{T['muted']}; }}

/* ── HTML Custom Table — sepenuhnya dikontrol, bukan canvas ── */
.custom-table {{
    width:100%;
    border-collapse: collapse;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid {T['border']};
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    font-family: 'Sora', sans-serif;
    font-size: 13px;
}}
.custom-table thead tr {{
    background-color: {T['tbl_hdr']};
}}
.custom-table thead th {{
    padding: 11px 16px;
    text-align: left;
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {T['tbl_muted']};
    border-bottom: 2px solid {T['primary']}55;
}}
.custom-table tbody tr:nth-child(odd) {{
    background-color: {T['tbl_row']};
}}
.custom-table tbody tr:nth-child(even) {{
    background-color: {T['tbl_alt']};
}}
.custom-table tbody tr:hover {{
    background-color: {T['primary']}14;
}}
.custom-table tbody td {{
    padding: 9px 16px;
    color: {T['tbl_txt']};
    border-bottom: 1px solid {T['border']};
    font-size: 13px;
}}
.custom-table tbody tr:last-child td {{
    border-bottom: none;
}}
/* Rank badge */
.rank-badge {{
    display:inline-flex; align-items:center; justify-content:center;
    width:24px; height:24px; border-radius:50%;
    background:{T['primary']}22; color:{T['primary']};
    font-size:11px; font-weight:700;
}}
.rank-badge.gold   {{ background:#ffd70022; color:#b8860b; }}
.rank-badge.silver {{ background:#c0c0c022; color:#707070; }}
.rank-badge.bronze {{ background:#cd7f3222; color:#8b4513; }}

/* ── Method pills ── */
.method-pill {{
    display:inline-block;
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 7px; padding:3px 10px;
    font-size:10.5px; color:{T['text']} !important;
    margin:2px 2px 2px 0;
    font-family:'JetBrains Mono', monospace;
}}

/* ── Alerts ── */
.stAlert {{ border-radius:12px !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:{T['border']}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:{T['muted']}; }}

/* st.dataframe fallback — tetap bersih kalau masih dipakai */
[data-testid="stDataFrame"] > div {{
    border-radius: 14px !important;
    border: 1px solid {T['border']} !important;
    overflow: hidden !important;
    background-color: {T['card_bg']} !important;
}}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# UTILITY: render HTML table yang SELALU terlihat (bukan canvas)
# ===========================================================================
def html_table(df: pd.DataFrame, rank_col: bool = False) -> str:
    """
    Render DataFrame sebagai HTML table yang fully-styled dan selalu terlihat.
    Tidak bergantung pada canvas Streamlit — teks pasti muncul.
    """
    badge_colors = ["gold", "silver", "bronze"]

    headers = "".join(
        f"<th>{'#' if rank_col else ''}{col}</th>"
        if col == df.columns[0] and rank_col
        else f"<th>{col}</th>"
        for col in df.columns
    )

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        cells = ""
        for j, val in enumerate(row):
            if j == 0 and rank_col:
                rank = i + 1
                badge_cls = badge_colors[i] if i < 3 else ""
                cells += f"<td><span class='rank-badge {badge_cls}'>{rank}</span>&nbsp;&nbsp;{val}</td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    return f"""
    <div style='overflow-x:auto; margin-bottom:8px;'>
    <table class='custom-table'>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """


# ===========================================================================
# LOAD & PREPROCESSING
# ===========================================================================
@st.cache_data
def load_and_preprocess():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path   = os.path.join(current_dir, "cost-of-living.csv")
    df     = pd.read_csv(file_path)
    x_cols = [c for c in df.columns if c.startswith("x")]
    df[x_cols] = df[x_cols].fillna(df[x_cols].median())

    country_df = df.groupby("country")[x_cols].mean().reset_index()
    country_df["CLI"] = (
        0.35 * country_df["x1"]  +
        0.40 * country_df["x28"] +
        0.15 * country_df["x36"] +
        0.10 * country_df["x48"]
    )
    country_df["Recommendation_Score"] = np.where(
        country_df["CLI"] > 0, country_df["x54"] / country_df["CLI"], np.nan
    )
    country_df = country_df.dropna(subset=["CLI", "Recommendation_Score", "x54"])

    feature_cols   = ["x1","x3","x8","x28","x33","x36","x48","x49","x54","CLI"]
    feature_matrix = country_df[feature_cols].values
    scaler         = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_matrix)

    cosine_sim = cosine_similarity(feature_scaled)
    cosine_df  = pd.DataFrame(cosine_sim,
                               index   = country_df["country"].values,
                               columns = country_df["country"].values)
    return country_df, feature_scaled, feature_cols, cosine_df

country_data, feature_scaled, feature_cols, cosine_sim_matrix = load_and_preprocess()


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================
@st.cache_data
def run_kmeans(k, _fs):
    km = KMeans(n_clusters=k, random_state=42, n_init=25)
    return km, km.fit_predict(_fs)

@st.cache_data
def compute_elbow(_fs, max_k=8):
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(_fs); inertias.append(km.inertia_)
    return list(range(2, max_k + 1)), inertias

@st.cache_data
def compute_silhouette(_fs, max_k=8):
    scores = []
    for k in range(2, max_k + 1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(_fs)
        scores.append(silhouette_score(_fs, lbl))
    return list(range(2, max_k + 1)), scores

@st.cache_data
def compute_feature_analysis(_country_data):
    x_cols = [c for c in _country_data.columns if c.startswith("x")]
    X = _country_data[x_cols]
    y = _country_data["x54"]
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({
        "Feature" : x_cols,
        "MI_Score": mi_scores,
        "Variance": X.var().values
    }).sort_values("MI_Score", ascending=False).reset_index(drop=True)
    selector = VarianceThreshold(threshold=1.0)
    selector.fit(X)
    mi_df["Lolos_VarThreshold"] = selector.get_support()
    return mi_df, mi_df.head(10)["Feature"].tolist()

mi_df, top10_features = compute_feature_analysis(country_data)

def get_cluster_label(cid):
    return {
        0: "High Cost · High Income", 1: "Affordable Emerging",
        2: "Mid-Tier Balanced",       3: "Budget Frontier",
        4: "Developing Low-Cost",     5: "Transitional Economy",
        6: "Resource-Rich",           7: "Small Island Economy"
    }.get(cid, f"Cluster {cid+1}")

def hybrid_recommend(ref, budget, salary, n):
    if ref not in cosine_sim_matrix.index:
        return pd.DataFrame()
    sim_df = cosine_sim_matrix[ref].drop(ref).sort_values(ascending=False).reset_index()
    sim_df.columns = ["country", "similarity"]
    filt = country_data[
        (country_data["CLI"] <= budget) & (country_data["x54"] >= salary)
    ][["country", "CLI", "x54", "Recommendation_Score"]]
    res = sim_df.merge(filt, on="country").head(n)
    res["Similarity (%)"] = (res["similarity"] * 100).round(1)
    res["CLI ($)"]        = res["CLI"].round(0).astype(int)
    res["Avg Salary ($)"] = res["x54"].round(0).astype(int)
    res["Rec. Score"]     = res["Recommendation_Score"].round(2)
    return res[["country", "Similarity (%)", "CLI ($)", "Avg Salary ($)", "Rec. Score"]]

def pl(fig, h=400):
    """
    Unified Plotly layout.
    FIX: axis label + tick warna PENUH (bukan muted) agar kontras
         light mode → hitam (#0e1011), dark mode → putih (#e4e1da)
    """
    ax_color = T["text"]   # hitam pekat (light) atau putih (dark)

    fig.update_layout(
        height        = h,
        paper_bgcolor = T["card_bg"],
        plot_bgcolor  = T["chart_pl"],
        font          = dict(color=T["text"], family="Sora, sans-serif", size=12),
        margin        = dict(t=30, b=40, l=10, r=10),
        xaxis = dict(
            gridcolor   = T["border"],
            linecolor   = T["border"],
            showline    = True,
            linewidth   = 1,
            zeroline    = False,
            tickfont    = dict(size=11, color=ax_color, family="Sora, sans-serif"),
            title_font  = dict(size=12, color=ax_color, family="Sora, sans-serif"),
        ),
        yaxis = dict(
            gridcolor   = T["border"],
            linecolor   = T["border"],
            showline    = True,
            linewidth   = 1,
            zeroline    = False,
            tickfont    = dict(size=11, color=ax_color, family="Sora, sans-serif"),
            title_font  = dict(size=12, color=ax_color, family="Sora, sans-serif"),
        ),
        legend = dict(
            bgcolor     = T["card_bg"],
            bordercolor = T["border"],
            borderwidth = 1,
            font        = dict(size=11, color=T["text"])
        )
    )
    # update_xaxes/yaxes override semua subplot sekaligus
    fig.update_xaxes(
        tickfont   = dict(color=ax_color, size=11, family="Sora, sans-serif"),
        title_font = dict(color=ax_color, size=12, family="Sora, sans-serif"),
    )
    fig.update_yaxes(
        tickfont   = dict(color=ax_color, size=11, family="Sora, sans-serif"),
        title_font = dict(color=ax_color, size=12, family="Sora, sans-serif"),
    )
    return fig

CSCALE = [[0, T["pink_lt"]], [0.5, T["primary"]], [1, T["accent"]]]
CLUSTER_COLORS = [T["primary"], T["accent"], T["second"], "#7a9bb5",
                  "#7A4F5E", "#8A9B7A", "#b08ea0", "#6B7A5E"]
BOX_COLORS     = [T["primary"], T["accent"], T["second"], "#7a9bb5", T["muted"]]


# ===========================================================================
# SIDEBAR — Filters lanjutan
# ===========================================================================
with st.sidebar:
    user_country = st.selectbox(
        "Negara Referensi",
        options = sorted(country_data["country"].unique()),
        index   = list(sorted(country_data["country"].unique())).index("Germany")
                  if "Germany" in country_data["country"].values else 0
    )
    max_budget = st.slider("Max Monthly Budget (USD)", 200, 5000, 2000, 100)
    min_salary = st.slider("Min Average Salary (USD)", 100, 8000, 1000, 100)
    top_n      = st.number_input("Top-N Negara", min_value=3, max_value=30, value=10)
    k_clusters = st.slider("Jumlah Cluster (K)", 2, 8, 4)

    st.markdown("<hr style='margin:10px 0 8px 0;'>", unsafe_allow_html=True)
    st.button("Run Analysis", use_container_width=True)
    st.markdown("<hr style='margin:10px 0 8px 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<span style='font-size:10px;font-weight:700;letter-spacing:0.12em;"
        "opacity:0.5;text-transform:uppercase;'>Metode</span>",
        unsafe_allow_html=True
    )
    st.markdown("""<div style='margin-top:6px;'>
    <span class='method-pill'>K-Means</span>
    <span class='method-pill'>PCA</span>
    <span class='method-pill'>Cosine Sim</span>
    <span class='method-pill'>Rule-Based</span>
    <span class='method-pill'>Mutual Info</span>
    </div>""", unsafe_allow_html=True)


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊 Overview  ",
    "  🔍 Analysis  ",
    "  🤖 Clustering  ",
    "  🌍 Recommender  "
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab1:
    st.markdown("""
    <div class='page-header'>
        <span class='page-title'>📊 Global Overview</span>
        <span class='page-sub'>Cost of Living &middot; Country-Level Analysis</span>
    </div>""", unsafe_allow_html=True)

    top_c     = country_data.nlargest(1, "Recommendation_Score")["country"].values[0]
    top_score = country_data.nlargest(1, "Recommendation_Score")["Recommendation_Score"].values[0]

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Total Negara",       str(len(country_data)),                  "Aktif dalam dataset",  "🌐"),
        ("Avg CLI",            f"${country_data['CLI'].mean():,.0f}",   "Rata-rata global",      "📊"),
        ("Avg Monthly Salary", f"${country_data['x54'].mean():,.0f}",  "Across all countries",  "💼"),
        ("Top Score Country",  top_c,                                   f"Score {top_score:.2f}","🏆"),
    ]
    for col, (name, val, sub, icon) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-top'>
                    <span class='metric-name'>{name}</span>
                    <span class='metric-icon'>{icon}</span>
                </div>
                <div class='metric-value'>{val}</div>
                <div class='metric-trend'>&#8599; {sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Peta Distribusi Recommendation Score</div>",
                unsafe_allow_html=True)

    fig_map = px.choropleth(
        country_data, locations="country", locationmode="country names",
        color="Recommendation_Score", hover_name="country",
        hover_data={"CLI":":.0f","x54":":.0f","Recommendation_Score":":.2f"},
        color_continuous_scale=CSCALE,
        labels={"Recommendation_Score":"Rec. Score","x54":"Salary","CLI":"CLI"}
    )
    fig_map.update_layout(
        height=440, paper_bgcolor=T["card_bg"],
        geo=dict(bgcolor=T["chart_pl"], showframe=False, showcoastlines=True,
                 coastlinecolor=T["border"], landcolor=T["surface"],
                 oceancolor="#d8eaf2" if not dark_mode else "#1a2830", showocean=True),
        coloraxis_colorbar=dict(title="Score", tickfont=dict(color=T["text"], size=11)),
        margin=dict(t=10, b=10, l=0, r=0),
        font=dict(color=T["text"], family="Sora, sans-serif")
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Top 7 sebagai HTML table (bukan st.dataframe) ──
    st.markdown("<div class='section-title'>Top 7 Negara — Recommendation Score Tertinggi</div>",
                unsafe_allow_html=True)
    display_df         = country_data[["country","CLI","x54","Recommendation_Score"]].copy()
    display_df.columns = ["Country", "CLI ($)", "Avg Salary ($)", "Rec. Score"]
    display_df         = (display_df
                          .round(2)
                          .sort_values("Rec. Score", ascending=False)
                          .head(7)
                          .reset_index(drop=True))
    st.markdown(html_table(display_df, rank_col=True), unsafe_allow_html=True)

    # ── Keterangan variabel ──
    st.markdown(f"""
    <div style='display:flex; gap:12px; margin-top:12px; flex-wrap:wrap;'>
        <div style='background:{T["surface"]};border:1px solid {T["border"]};border-radius:12px;
                    padding:12px 16px;flex:1;min-width:160px;'>
            <div style='font-size:10px;font-weight:700;color:{T["primary"]};
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;'>
                CLI ($) — Cost of Living Index
            </div>
            <div style='font-size:12.5px;color:{T["text"]};line-height:1.55;'>
                Indeks gabungan biaya hidup bulanan: makan 35%, sewa 40%,
                utilitas 15%, bensin 10%. Makin kecil = makin terjangkau.
            </div>
        </div>
        <div style='background:{T["surface"]};border:1px solid {T["border"]};border-radius:12px;
                    padding:12px 16px;flex:1;min-width:160px;'>
            <div style='font-size:10px;font-weight:700;color:{T["primary"]};
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;'>
                Avg Salary ($) — Gaji Bulanan Bersih
            </div>
            <div style='font-size:12.5px;color:{T["text"]};line-height:1.55;'>
                Rata-rata gaji bulanan bersih (variabel x54 dataset).
                Makin tinggi = potensi pendapatan lebih besar.
            </div>
        </div>
        <div style='background:{T["surface"]};border:1px solid {T["border"]};border-radius:12px;
                    padding:12px 16px;flex:1;min-width:160px;'>
            <div style='font-size:10px;font-weight:700;color:{T["primary"]};
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;'>
                Rec. Score — Recommendation Score
            </div>
            <div style='font-size:12.5px;color:{T["text"]};line-height:1.55;'>
                = Avg Salary &divide; CLI. Mengukur seberapa <i>worth it</i> suatu negara.
                Makin tinggi = gaji besar relatif terhadap biaya hidup.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 2: ANALYSIS
# ============================================================
with tab2:
    st.markdown("""
    <div class='page-header'>
        <span class='page-title'>🔍 Statistical Analysis</span>
        <span class='page-sub'>Feature Correlation &middot; Distributions &middot; Feature Selection</span>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("<div class='section-title'>Correlation Heatmap: Selected Features</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
        10 fitur terpilih + CLI. Pemilihan berdasarkan relevansi terhadap x54 (Salary).
        </div>""", unsafe_allow_html=True)

        sel_cols   = ["x1","x3","x8","x28","x33","x36","x48","x49","x54","CLI"]
        sel_labels = ["Meal","Beer","McMeal","Rent_1BR","Rent_3BR","Utilities","Gasoline","Transport","Salary","CLI"]
        corr_df    = country_data[sel_cols].copy()
        corr_df.columns = sel_labels

        fig_corr, ax = plt.subplots(figsize=(8, 6))
        fig_corr.patch.set_facecolor(T["card_bg"])
        ax.set_facecolor(T["chart_pl"])
        cmap = sns.diverging_palette(140, 30, s=60, l=50, as_cmap=True)
        sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax,
                    linewidths=0.4, linecolor=T["border"],
                    annot_kws={"size": 8.5, "color": T["text"]},
                    cbar_kws={"shrink": 0.8})
        # FIX: paksa semua tick label heatmap warna penuh
        ax.tick_params(axis="both", colors=T["text"], labelsize=9, which="both")
        for lbl in ax.get_xticklabels():
            lbl.set_color(T["text"]); lbl.set_alpha(1.0)
        for lbl in ax.get_yticklabels():
            lbl.set_color(T["text"]); lbl.set_alpha(1.0)
        plt.xticks(rotation=45, ha="right", color=T["text"], fontsize=9)
        plt.yticks(color=T["text"], fontsize=9)
        plt.title("Correlation Matrix: Selected Features",
                  color=T["text"], pad=12, fontweight="bold", fontsize=11)
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors=T["text"], labelsize=8)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=T["text"], alpha=1.0)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)

    with col_right:
        st.markdown("<div class='section-title'>Distribusi CLI</div>", unsafe_allow_html=True)
        fig_cli = px.histogram(country_data, x="CLI", nbins=30,
                               color_discrete_sequence=[T["primary"]])
        pl(fig_cli, 230)
        fig_cli.update_layout(xaxis_title="Cost of Living Index (USD)",
                              yaxis_title="Jumlah Negara",
                              showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cli, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='section-title'>Distribusi Salary</div>", unsafe_allow_html=True)
        fig_sal = px.histogram(country_data, x="x54", nbins=30,
                               color_discrete_sequence=[T["accent"]])
        pl(fig_sal, 230)
        fig_sal.update_layout(xaxis_title="Monthly Salary (USD)",
                              yaxis_title="Jumlah Negara",
                              showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_sal, use_container_width=True, config={"displayModeBar": False})

    # ── Feature Selection ──
    st.markdown("<div class='section-title'>Feature Selection — Mutual Information &amp; Variance</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    <b>Mutual Information (MI)</b> mengukur informasi setiap fitur terhadap x54 (Salary).
    <b>Variance Threshold</b> menyaring fitur hampir konstan (threshold = 1.0).
    </div>""", unsafe_allow_html=True)

    col_mi, col_var = st.columns([3, 2])

    with col_mi:
        st.markdown("<div class='section-title'>MI Score: Semua Fitur vs x54</div>",
                    unsafe_allow_html=True)
        sel_feat = ["x1","x3","x8","x28","x33","x36","x48","x49","x54","CLI"]
        mi_plot  = mi_df.copy()
        mi_plot["Warna"] = mi_plot["Feature"].isin(sel_feat).map(
            {True: T["primary"], False: T["border"]}
        )
        mi_sorted = mi_plot.sort_values("MI_Score", ascending=True).tail(30)

        fig_mi = go.Figure()
        fig_mi.add_trace(go.Bar(
            x=mi_sorted["MI_Score"], y=mi_sorted["Feature"], orientation="h",
            marker_color=mi_sorted["Warna"], marker_line_width=0,
            text=mi_sorted["MI_Score"].round(3), textposition="outside",
            textfont=dict(size=9, color=T["text"]),
            hovertemplate="<b>%{y}</b><br>MI Score: %{x:.4f}<extra></extra>"
        ))
        pl(fig_mi, 520)
        fig_mi.update_layout(xaxis_title="Mutual Information Score", yaxis_title="Feature",
                              showlegend=False, margin=dict(t=10, b=30, l=10, r=60))
        st.plotly_chart(fig_mi, use_container_width=True, config={"displayModeBar": False})

    with col_var:
        st.markdown("<div class='section-title'>Variance Threshold Analysis</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
        Fitur dengan varians sangat rendah tidak informatif. Threshold = 1.0.
        </div>""", unsafe_allow_html=True)

        fig_var = px.scatter(
            mi_df, x="Variance", y="MI_Score", text="Feature",
            color="Lolos_VarThreshold",
            color_discrete_map={True: T["primary"], False: T["border"]},
            labels={"Variance":"Variance","MI_Score":"MI Score",
                    "Lolos_VarThreshold":"Lolos Threshold"}
        )
        fig_var.update_traces(
            textposition="top center",
            textfont=dict(size=7, color=T["text"]),
            marker=dict(size=8, opacity=0.85, line=dict(color=T["card_bg"], width=1))
        )
        fig_var.add_vline(x=1.0, line_dash="dash", line_color=T["accent"],
                          annotation_text="Threshold=1.0",
                          annotation_font_color=T["accent"], annotation_font_size=10)
        pl(fig_var, 280)
        fig_var.update_layout(showlegend=True,
                              legend=dict(font=dict(size=10, color=T["text"]), title=""),
                              margin=dict(t=10, b=10),
                              xaxis_title="Variance", yaxis_title="MI Score")
        st.plotly_chart(fig_var, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='section-title'>Ringkasan Feature Selection</div>",
                    unsafe_allow_html=True)
        summary_data = pd.DataFrame({
            "Metode"    : ["Total fitur awal","Lolos Var Threshold","Top 10 MI","Final model"],
            "Jumlah"    : [len(mi_df), int(mi_df["Lolos_VarThreshold"].sum()), 10, 10],
            "Keterangan": ["x1-x55","Variance > 1.0","MI tertinggi","K-Means & Cosine Sim"]
        })
        st.markdown(html_table(summary_data), unsafe_allow_html=True)

    # ── Normalisasi Box Plots ──
    st.markdown("<div class='section-title'>Normalisasi: Sebelum vs Sesudah StandardScaler</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    StandardScaler mengubah semua fitur ke mean=0, std=1 — setiap fitur berkontribusi setara.
    </div>""", unsafe_allow_html=True)

    fd   = ["x1","x28","x36","x48","x54"]
    fl   = ["Meal","Rent","Utilities","Gasoline","Salary"]
    rmlt = country_data[fd].copy(); rmlt.columns = fl
    rmlt = rmlt.melt(var_name="Feature", value_name="Value")
    sarr = StandardScaler().fit_transform(country_data[fd])
    smlt = pd.DataFrame(sarr, columns=fl).melt(var_name="Feature", value_name="Value")

    cb, ca = st.columns(2)
    with cb:
        st.markdown("<div class='section-title'>Sebelum Normalisasi (Raw)</div>",
                    unsafe_allow_html=True)
        fb = px.box(rmlt, x="Feature", y="Value", color="Feature",
                    color_discrete_sequence=BOX_COLORS)
        pl(fb, 300)
        fb.update_layout(showlegend=False, xaxis_title="Feature",
                          yaxis_title="Value (USD)", margin=dict(t=10, b=10))
        st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})

    with ca:
        st.markdown("<div class='section-title'>Sesudah Normalisasi (Z-Score)</div>",
                    unsafe_allow_html=True)
        fa = px.box(smlt, x="Feature", y="Value", color="Feature",
                    color_discrete_sequence=BOX_COLORS)
        pl(fa, 300)
        fa.update_layout(showlegend=False, xaxis_title="Feature",
                          yaxis_title="Z-Score", margin=dict(t=10, b=10))
        st.plotly_chart(fa, use_container_width=True, config={"displayModeBar": False})

    # ── Scatter + Top 15 ──
    st.markdown("<div class='section-title'>Scatter: Monthly Salary vs Cost of Living Index</div>",
                unsafe_allow_html=True)
    fig_sc = px.scatter(
        country_data, x="CLI", y="x54",
        size="Recommendation_Score", color="Recommendation_Score", hover_name="country",
        color_continuous_scale=CSCALE,
        labels={"CLI":"Cost of Living Index (USD)","x54":"Monthly Salary (USD)",
                "Recommendation_Score":"Recommendation Score"}
    )
    pl(fig_sc, 400)
    fig_sc.update_layout(xaxis_title="Cost of Living Index (USD)",
                          yaxis_title="Monthly Salary (USD)")
    st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-title'>Top 15 Negara by Recommendation Score</div>",
                unsafe_allow_html=True)
    top15   = country_data.nlargest(15, "Recommendation_Score").sort_values("Recommendation_Score")
    fig_top = px.bar(top15, x="Recommendation_Score", y="country", orientation="h",
                     color="Recommendation_Score", color_continuous_scale=CSCALE,
                     labels={"Recommendation_Score":"Recommendation Score","country":"Country"})
    pl(fig_top, 450)
    fig_top.update_layout(showlegend=False,
                           xaxis_title="Recommendation Score", yaxis_title="Country")
    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})


# ============================================================
# TAB 3: CLUSTERING
# ============================================================
with tab3:
    st.markdown("""
    <div class='page-header'>
        <span class='page-title'>🤖 K-Means Clustering</span>
        <span class='page-sub'>Elbow &middot; Silhouette &middot; PCA Visualization</span>
    </div>""", unsafe_allow_html=True)

    ce, cs = st.columns(2)

    with ce:
        st.markdown("<div class='section-title'>Elbow Method</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>WSS makin kecil = cluster makin padat.
        Pilih K di titik "siku" kurva.</div>""", unsafe_allow_html=True)

        ks, inertias = compute_elbow(feature_scaled)
        fe = go.Figure()
        fe.add_trace(go.Scatter(
            x=ks, y=inertias, mode="lines+markers",
            line=dict(color=T["primary"], width=2.5),
            marker=dict(size=9, color=T["card_bg"], line=dict(color=T["primary"], width=2.5))
        ))
        fe.add_vline(x=k_clusters, line_dash="dash", line_color=T["accent"],
                     annotation_text=f"K = {k_clusters}",
                     annotation_font_color=T["accent"], annotation_font_size=12)
        pl(fe, 340)
        fe.update_layout(xaxis_title="Jumlah Cluster (K)",
                          yaxis_title="Inertia — Within-Cluster Sum of Squares")
        st.plotly_chart(fe, use_container_width=True, config={"displayModeBar": False})

    with cs:
        st.markdown("<div class='section-title'>Silhouette Score</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>Score 0-1. Makin tinggi = pemisahan makin baik.</div>""",
                    unsafe_allow_html=True)
        ks_s, sil_s = compute_silhouette(feature_scaled)
        fs = go.Figure()
        fs.add_trace(go.Bar(
            x=ks_s, y=sil_s,
            marker_color=[T["primary"] if k == k_clusters else T["border"] for k in ks_s],
            marker_line_width=0
        ))
        pl(fs, 340)
        fs.update_layout(xaxis_title="Jumlah Cluster (K)",
                          yaxis_title="Silhouette Score (0 - 1)")
        st.plotly_chart(fs, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-title'>PCA Cluster Plot — Reduksi Dimensi 10D ke 2D</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>PCA merangkum 10 fitur ke 2 komponen utama.
    Tiap titik = 1 negara, warna = cluster.</div>""", unsafe_allow_html=True)

    km_model, cluster_labels = run_kmeans(k_clusters, feature_scaled)
    pca        = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(feature_scaled)
    var_exp    = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1"    : pca_coords[:, 0], "PC2": pca_coords[:, 1],
        "country": country_data["country"].values,
        "Cluster": [get_cluster_label(c) for c in cluster_labels],
        "CLI"    : country_data["CLI"].values.round(0),
        "Salary" : country_data["x54"].values.round(0)
    })
    fp = px.scatter(
        pca_df, x="PC1", y="PC2", color="Cluster",
        hover_name="country", hover_data={"CLI": True, "Salary": True},
        labels={"PC1": f"PC1 — {var_exp[0]*100:.1f}% var",
                "PC2": f"PC2 — {var_exp[1]*100:.1f}% var"},
        color_discrete_sequence=CLUSTER_COLORS
    )
    fp.update_traces(marker=dict(size=10, opacity=0.82, line=dict(color=T["card_bg"], width=1)))
    pl(fp, 500)
    fp.update_layout(
        xaxis_title=f"PC1 — {var_exp[0]*100:.1f}% Variance Explained",
        yaxis_title=f"PC2 — {var_exp[1]*100:.1f}% Variance Explained"
    )
    st.plotly_chart(fp, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-title'>Cluster Profiling</div>", unsafe_allow_html=True)
    clustered_df            = country_data.copy()
    clustered_df["Cluster"] = [get_cluster_label(c) for c in cluster_labels]
    profile = clustered_df.groupby("Cluster").agg(
        N_Negara   = ("country",             "count"),
        Avg_CLI    = ("CLI",                 lambda x: round(x.mean(), 0)),
        Avg_Salary = ("x54",                 lambda x: round(x.mean(), 0)),
        Avg_Score  = ("Recommendation_Score", lambda x: round(x.mean(), 2))
    ).reset_index()
    profile.columns = ["Cluster","N Negara","Avg CLI ($)","Avg Salary ($)","Avg Score"]
    st.markdown(html_table(profile), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Anggota per Cluster</div>", unsafe_allow_html=True)
    members         = clustered_df[["country","Cluster","CLI","x54","Recommendation_Score"]].copy()
    members.columns = ["Country","Cluster","CLI ($)","Salary ($)","Score"]
    members         = members.round(1).sort_values("Cluster").reset_index(drop=True)
    st.markdown(html_table(members), unsafe_allow_html=True)


# ============================================================
# TAB 4: RECOMMENDER
# ============================================================
with tab4:
    st.markdown("""
    <div class='page-header'>
        <span class='page-title'>🌍 Country Recommender</span>
        <span class='page-sub'>Content-Based Cosine Similarity + Rule-Based Filter</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
    <b>Hybrid Method:</b> Content-Based mencari negara dengan profil biaya hidup paling mirip
    menggunakan Cosine Similarity pada normalized feature space. Rule-Based menyaring hasil
    berdasarkan Max Budget dan Min Salary dari sidebar.
    </div>""", unsafe_allow_html=True)

    cr, crec = st.columns([1, 2])

    with cr:
        st.markdown("<div class='section-title'>Profil Negara Referensi</div>",
                    unsafe_allow_html=True)
        ref_row = country_data[country_data["country"] == user_country]
        if not ref_row.empty:
            rvars  = ["x1","x28","x36","x48","x54"]
            rlbls  = ["Meal","Rent","Utilities","Gasoline","Salary"]
            vals   = ref_row[rvars].values.flatten()
            vals_n = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=np.append(vals_n, vals_n[0]), theta=rlbls + [rlbls[0]],
                fill="toself", name=user_country,
                fillcolor="rgba(111,129,110,0.18)" if not dark_mode else "rgba(124,145,121,0.22)",
                line=dict(color=T["primary"], width=2.5)
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1],
                                    gridcolor=T["border"],
                                    tickfont=dict(color=T["text"], size=10)),
                    angularaxis=dict(tickfont=dict(color=T["text"], size=11)),
                    bgcolor=T["chart_pl"]
                ),
                height=300, paper_bgcolor=T["card_bg"], showlegend=False,
                font=dict(color=T["text"], family="Sora, sans-serif"),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

            ref_data = pd.DataFrame({
                "Metrik": ["CLI ($)", "Avg Salary ($)", "Rec. Score"],
                "Nilai" : [f"${ref_row['CLI'].values[0]:,.0f}",
                           f"${ref_row['x54'].values[0]:,.0f}",
                           f"{ref_row['Recommendation_Score'].values[0]:.2f}"]
            })
            st.markdown(html_table(ref_data), unsafe_allow_html=True)

    with crec:
        st.markdown("<div class='section-title'>Top Rekomendasi Negara</div>",
                    unsafe_allow_html=True)
        recs = hybrid_recommend(user_country, max_budget, min_salary, int(top_n))

        if recs.empty:
            st.warning("Tidak ada negara yang memenuhi kriteria. Coba longgarkan budget atau salary minimum.")
        else:
            st.markdown(html_table(recs.reset_index(drop=True), rank_col=True),
                        unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Cosine Similarity Score</div>",
                        unsafe_allow_html=True)
            fig_sim = px.bar(
                recs.sort_values("Similarity (%)"),
                x="Similarity (%)", y="country", orientation="h",
                color="Similarity (%)", color_continuous_scale=CSCALE,
                labels={"country":"Country","Similarity (%)":"Cosine Similarity (%)"}
            )
            pl(fig_sim, 300)
            fig_sim.update_layout(showlegend=False,
                                   xaxis_title="Cosine Similarity (%)",
                                   yaxis_title="Country")
            st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})

    if not recs.empty:
        st.markdown("<div class='section-title'>Perbandingan: Referensi vs Top 5 (Z-Score Heatmap)</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
        Z-Score agar perbandingan antar variabel dengan skala berbeda tetap adil.
        </div>""", unsafe_allow_html=True)

        all_c     = [user_country] + recs.head(5)["country"].tolist()
        comp_vars = ["x1","x28","x33","x36","x48","x54","CLI","Recommendation_Score"]
        comp_lbls = ["Meal","Rent","Groceries","Utilities","Gasoline","Salary","CLI","Score"]
        comp_df   = (country_data[country_data["country"].isin(all_c)]
                     [["country"] + comp_vars].set_index("country"))
        comp_sc   = (comp_df - comp_df.mean()) / (comp_df.std() + 1e-9)
        comp_sc.columns = comp_lbls

        fh = px.imshow(
            comp_sc,
            color_continuous_scale=[[0,T["accent"]],[0.5,T["card_bg"]],[1,T["primary"]]],
            aspect="auto", text_auto=".2f"
        )
        fh.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(color=T["text"], family="Sora, sans-serif", size=12),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Feature", yaxis_title="Country",
            xaxis=dict(tickfont=dict(size=11, color=T["text"]),
                       title_font=dict(size=12, color=T["text"])),
            yaxis=dict(tickfont=dict(size=11, color=T["text"]),
                       title_font=dict(size=12, color=T["text"])),
            coloraxis_colorbar=dict(tickfont=dict(color=T["text"], size=10))
        )
        fh.update_traces(textfont=dict(color=T["text"], size=11))
        st.plotly_chart(fh, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:{T['muted']};font-size:12px;
             padding:10px 0 6px 0;font-family:Sora,sans-serif;'>
    Final Project Data Mining &nbsp;&middot;&nbsp; Hybrid Country Recommendation System<br>
    <span style='color:{T['border']};'>
    K-Means &nbsp;&middot;&nbsp; PCA &nbsp;&middot;&nbsp; Cosine Similarity
    &nbsp;&middot;&nbsp; Mutual Information &nbsp;&middot;&nbsp; Rule-Based Filtering
    </span>
</div>
""", unsafe_allow_html=True)
