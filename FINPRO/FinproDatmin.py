# FinproDatmin.py — Hybrid Country Recommendation Dashboard (Streamlit)
# Final Project Data Mining — Cost of Living Analysis

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(
    page_title            = "Cost of Living · Data Mining",
    page_icon             = "🌍",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# THEME PALETTES
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
    "tbl_hdr"  : "#ece9e3",
    "tbl_row"  : "#ffffff",
    "tbl_alt"  : "#f4f2ef",
    "tbl_txt"  : "#0e1011",
    "tbl_muted": "#7a7872",
    # FIX #1: Radio button adaptive colors
    "radio_bg"       : "#f4f2ef",
    "radio_bg_sel"   : "#6f816e",
    "radio_txt"      : "#0e1011",
    "radio_txt_sel"  : "#ffffff",
    "radio_border"   : "#e0dbd4",
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
    "tbl_hdr"  : "#272b2d",
    "tbl_row"  : "#1e2123",
    "tbl_alt"  : "#232729",
    "tbl_txt"  : "#e4e1da",
    "tbl_muted": "#888680",
    # FIX #1: Radio button adaptive colors
    "radio_bg"       : "#272b2d",
    "radio_bg_sel"   : "#7c9179",
    "radio_txt"      : "#e4e1da",
    "radio_txt_sel"  : "#ffffff",
    "radio_border"   : "#2c3033",
}

# SIDEBAR
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
    dark_mode = st.toggle("Dark Mode", value=False)
    st.markdown("<hr style='margin:10px 0 8px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:10px;font-weight:700;letter-spacing:0.12em;"
        "opacity:0.5;text-transform:uppercase;'>Filters</span>",
        unsafe_allow_html=True
    )

T = DARK if dark_mode else LIGHT

# CSS INJECTION
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp {{
    font-family: 'Sora', sans-serif !important;
    background-color: {T['app_bg']} !important;
    color: {T['text']} !important;
}}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{
    display: none !important;
    visibility: hidden !important;
}}
.stDeployButton {{ display: none !important; }}

.block-container {{
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
    background-color: {T['app_bg']} !important;
}}

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

.section-title {{
    font-size:11px; font-weight:700; color:{T['muted']};
    margin: 24px 0 12px 0;
    display:flex; align-items:center; gap:10px;
    text-transform:uppercase; letter-spacing:0.10em;
}}
.section-title::after {{ content:''; flex:1; height:1px; background:{T['border']}; }}

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

.page-header {{
    display:flex; align-items:baseline; justify-content:space-between;
    margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid {T['border']};
}}
.page-title {{ font-size:20px; font-weight:700; color:{T['text']}; margin:0; font-family:'Sora',sans-serif; }}
.page-sub   {{ font-size:12px; color:{T['muted']}; }}

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
.rank-badge {{
    display:inline-flex; align-items:center; justify-content:center;
    width:24px; height:24px; border-radius:50%;
    background:{T['primary']}22; color:{T['primary']};
    font-size:11px; font-weight:700;
}}
.rank-badge.gold   {{ background:#ffd70022; color:#b8860b; }}
.rank-badge.silver {{ background:#c0c0c022; color:#707070; }}
.rank-badge.bronze {{ background:#cd7f3222; color:#8b4513; }}

.method-pill {{
    display:inline-block;
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 7px; padding:3px 10px;
    font-size:10.5px; color:{T['text']} !important;
    margin:2px 2px 2px 0;
    font-family:'JetBrains Mono', monospace;
}}

.stAlert {{ border-radius:12px !important; }}

::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:{T['border']}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:{T['muted']}; }}

[data-testid="stDataFrame"] > div {{
    border-radius: 14px !important;
    border: 1px solid {T['border']} !important;
    overflow: hidden !important;
    background-color: {T['card_bg']} !important;
}}

/* =====================================================
   FIX #1: Adaptive radio button cluster filter
   Targets all stRadio contexts (horizontal & vertical)
   ===================================================== */
div[data-testid="stRadio"] > label {{
    color: {T['text']} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
    display: block !important;
}}

/* Radio option labels */
div[data-testid="stRadio"] div[role="radiogroup"] label {{
    background-color: {T['radio_bg']} !important;
    color: {T['radio_txt']} !important;
    border: 1px solid {T['radio_border']} !important;
    border-radius: 8px !important;
    padding: 5px 14px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    font-family: 'Sora', sans-serif !important;
    margin-right: 4px !important;
    margin-bottom: 4px !important;
}}

div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
    background-color: {T['primary']}22 !important;
    border-color: {T['primary']} !important;
    color: {T['primary']} !important;
}}

/* Selected radio option */
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
    background-color: {T['radio_bg_sel']} !important;
    color: {T['radio_txt_sel']} !important;
    border-color: {T['radio_bg_sel']} !important;
}}

/* Hide native radio circle */
div[data-testid="stRadio"] div[role="radiogroup"] label input[type="radio"] {{
    display: none !important;
}}

/* Force text inside label spans to inherit */
div[data-testid="stRadio"] div[role="radiogroup"] label p,
div[data-testid="stRadio"] div[role="radiogroup"] label span,
div[data-testid="stRadio"] div[role="radiogroup"] label div {{
    color: inherit !important;
    font-family: 'Sora', sans-serif !important;
}}

/* Pipeline flow boxes */
.pipeline-step {{
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    flex: 1;
}}
.pipeline-step-title {{
    font-size: 11px;
    font-weight: 700;
    color: {T['primary']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
}}
.pipeline-step-desc {{
    font-size: 11.5px;
    color: {T['muted']};
    line-height: 1.5;
}}
.pipeline-arrow {{
    display: flex;
    align-items: center;
    color: {T['border']};
    font-size: 20px;
    padding: 0 4px;
}}

/* Quadrant labels on scatter */
.quadrant-label {{
    font-size: 10px;
    font-weight: 700;
    opacity: 0.55;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
</style>
""", unsafe_allow_html=True)


# UTILITY: render HTML table
def html_table(df: pd.DataFrame, rank_col: bool = False) -> str:
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


# PREPROCESSING

NUMERIC_COLS = [f"x{i}" for i in range(1, 56)]
CLI_COLS     = ["x1", "x48", "x36", "x33"]
CAP_COLS     = ["x48", "x54"]
FEATURE_COLS = ["x3", "x8", "x28", "x49", "Recommendation_Score"]
CLUSTER_COLS = ["x1", "x3", "x8", "x28", "x33", "x36", "x48", "x49", "x54", "CLI"]


def _validate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "city"    in df.columns: df["city"]    = df["city"].astype(str).str.strip()
    if "country" in df.columns: df["country"] = df["country"].astype(str).str.strip()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.sort_values("data_quality", ascending=False)
    df = df.drop_duplicates(subset=["city", "country"], keep="first")
    return df.reset_index(drop=True)


def _remove_illogical_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    return df


def _smart_impute(df: pd.DataFrame) -> pd.DataFrame:
    numeric_in_df  = [c for c in NUMERIC_COLS if c in df.columns]
    country_median = df.groupby("country")[numeric_in_df].transform("median")
    df[numeric_in_df] = df[numeric_in_df].fillna(country_median)
    global_median      = df[numeric_in_df].median()
    df[numeric_in_df] = df[numeric_in_df].fillna(global_median)
    return df


def _iqr_capping(df: pd.DataFrame) -> pd.DataFrame:
    for col in CAP_COLS:
        if col not in df.columns: continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(upper=Q3 + 1.5*(Q3-Q1))
    return df


def clean_city_level(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = _validate_dtypes(df)
    df = _handle_duplicates(df)
    df = _remove_illogical_values(df)
    df = _smart_impute(df)
    df = _iqr_capping(df)
    return df


def aggregate_and_engineer(df_clean: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in NUMERIC_COLS if c in df_clean.columns]
    dq1 = df_clean[df_clean["data_quality"] == 1]
    dq0 = df_clean[df_clean["data_quality"] == 0]
    countries_dq1      = set(dq1["country"].unique())
    countries_dq0_only = set(dq0["country"].unique()) - countries_dq1
    agg_dq1      = dq1.groupby("country")[numeric_cols].mean()
    agg_fallback = (dq0[dq0["country"].isin(countries_dq0_only)]
                    .groupby("country")[numeric_cols].mean())
    country_df = pd.concat([agg_dq1, agg_fallback]).reset_index()
    city_counts = (df_clean.groupby("country")["city"].nunique().rename("n_cities"))
    country_df  = country_df.merge(city_counts.reset_index(), on="country", how="left")
    country_df["CLI"] = (
        0.35 * country_df["x1"]  +
        0.40 * country_df["x48"] +
        0.15 * country_df["x36"] +
        0.10 * country_df["x33"]
    )
    country_df["Recommendation_Score"] = np.where(
        country_df["CLI"] > 0, country_df["x54"] / country_df["CLI"], np.nan
    )
    country_df["CLI"] = country_df["CLI"].fillna(country_df["CLI"].median())
    country_df["Recommendation_Score"] = (
        country_df["Recommendation_Score"]
        .fillna(country_df["Recommendation_Score"].median())
    )
    return country_df.reset_index(drop=True)


def select_features(country_df: pd.DataFrame):
    feature_cols = ["x3", "x8", "x28", "x49", "Recommendation_Score"]
    missing = [c for c in feature_cols if c not in country_df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")
    df_features   = country_df[["country"] + feature_cols].copy()
    df_features[feature_cols] = df_features[feature_cols].fillna(
        df_features[feature_cols].median()
    )
    scaler_minmax = MinMaxScaler()
    feat_minmax   = scaler_minmax.fit_transform(df_features[feature_cols])
    df_feat_minmax = pd.DataFrame(feat_minmax, columns=feature_cols,
                                  index=df_features["country"])
    scaler_std = StandardScaler()
    feat_std   = scaler_std.fit_transform(df_features[feature_cols])
    df_feat_std = pd.DataFrame(feat_std, columns=feature_cols,
                               index=df_features["country"])
    return {
        "df_features"   : df_features,
        "df_feat_minmax": df_feat_minmax,
        "df_feat_std"   : df_feat_std,
        "scaler_minmax" : scaler_minmax,
        "scaler_std"    : scaler_std,
        "feature_cols"  : feature_cols,
    }


@st.cache_data
def run_preprocessing_pipeline(filepath: str) -> dict:
    df_raw        = pd.read_csv(filepath)
    df_clean      = clean_city_level(df_raw)
    country_df    = aggregate_and_engineer(df_clean)
    feat_results  = select_features(country_df)
    df_feat        = feat_results["df_features"]
    df_feat_scaled = feat_results["df_feat_minmax"]
    scaler_mm      = feat_results["scaler_minmax"]
    stats = {
        "n_raw"          : len(df_raw),
        "n_after_clean"  : len(df_clean),
        "n_exact_dup"    : len(df_raw) - len(df_raw.drop_duplicates()),
        "n_country_raw"  : int(df_raw["country"].nunique()),
        "n_country_final": len(country_df),
        "city_counts"    : df_clean.groupby("country")["city"].nunique(),
    }
    return {
        "df_raw"            : df_raw,
        "df_clean"          : df_clean,
        "country_df"        : country_df,
        "df_features"       : df_feat,
        "df_features_scaled": df_feat_scaled,
        "scaler_mm"         : scaler_mm,
        "stats"             : stats,
    }


@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path   = os.path.join(current_dir, "cost-of-living.csv")
    results     = run_preprocessing_pipeline(file_path)
    country_df  = results["country_df"]
    cluster_cols = [
        "x1", "x3", "x8", "x28",
        "x33", "x36", "x48",
        "x49", "x54", "CLI"
    ]
    cluster_matrix = country_df[cluster_cols].fillna(
        country_df[cluster_cols].median()
    )
    scaler_std    = StandardScaler()
    feature_scaled = scaler_std.fit_transform(cluster_matrix)
    df_feat_scaled = results["df_features_scaled"]
    cosine_sim     = cosine_similarity(df_feat_scaled.values)
    cosine_df      = pd.DataFrame(cosine_sim,
                                  index=df_feat_scaled.index,
                                  columns=df_feat_scaled.index)
    return (country_df, feature_scaled, cluster_cols, cosine_df, results)


country_data, feature_scaled, feature_cols, cosine_sim_matrix, pipeline_results = load_data()


# HELPER FUNCTIONS
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
    x_cols     = [c for c in _country_data.columns if c.startswith("x")]
    X          = _country_data[x_cols]
    y          = _country_data["x54"]
    mi_scores  = mutual_info_regression(X, y, random_state=42)
    mi_df      = pd.DataFrame({
        "Feature" : x_cols,
        "MI_Score": mi_scores,
        "Variance": X.var().values
    }).sort_values("MI_Score", ascending=False).reset_index(drop=True)
    selector   = VarianceThreshold(threshold=1.0)
    selector.fit(X)
    mi_df["Lolos_VarThreshold"] = selector.get_support()
    return mi_df, mi_df.head(10)["Feature"].tolist()


mi_df, top10_features = compute_feature_analysis(country_data)


def get_cluster_label(cluster_id: int, clustered_df: pd.DataFrame) -> str:
    if "Cluster_ID" not in clustered_df.columns:
        return f"Cluster {cluster_id + 1}"
    profile = clustered_df.groupby("Cluster_ID")[["CLI", "x54"]].median()
    if cluster_id not in profile.index:
        return f"Cluster {cluster_id + 1}"
    global_cli    = clustered_df["CLI"].median()
    global_salary = clustered_df["x54"].median()
    row           = profile.loc[cluster_id]
    if   row["CLI"] >  global_cli and row["x54"] >  global_salary: label = "High Cost · High Income"
    elif row["CLI"] >  global_cli and row["x54"] <= global_salary:  label = "Expensive · Low Income"
    elif row["CLI"] <= global_cli and row["x54"] >  global_salary:  label = "Affordable · High Income"
    else:                                                            label = "Affordable Emerging"
    # FIX #11: Prefix C{id+1} always present → label selalu unik
    return f"C{cluster_id + 1}: {label}"


def hybrid_recommend(ref, budget, salary, n):
    if ref not in cosine_sim_matrix.index:
        return pd.DataFrame()
    sim_df = (cosine_sim_matrix[ref].drop(ref)
              .sort_values(ascending=False).reset_index())
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
    ax_color = T["text"]
    fig.update_layout(
        height        = h,
        paper_bgcolor = T["card_bg"],
        plot_bgcolor  = T["chart_pl"],
        font          = dict(color=T["text"], family="Sora, sans-serif", size=12),
        margin        = dict(t=30, b=40, l=10, r=10),
        xaxis = dict(
            gridcolor  = T["border"], linecolor=T["border"],
            showline   = True, linewidth=1, zeroline=False,
            tickfont   = dict(size=11, color=ax_color, family="Sora, sans-serif"),
            title_font = dict(size=12, color=ax_color, family="Sora, sans-serif"),
        ),
        yaxis = dict(
            gridcolor  = T["border"], linecolor=T["border"],
            showline   = True, linewidth=1, zeroline=False,
            tickfont   = dict(size=11, color=ax_color, family="Sora, sans-serif"),
            title_font = dict(size=12, color=ax_color, family="Sora, sans-serif"),
        ),
        legend = dict(
            bgcolor=T["card_bg"], bordercolor=T["border"], borderwidth=1,
            font=dict(size=11, color=T["text"])
        )
    )
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
BOX_COLORS = [T["primary"], T["accent"], T["second"], "#7a9bb5", T["muted"]]


# SIDEBAR (filters)
with st.sidebar:
    user_country = st.selectbox(
        "Negara Referensi",
        options=sorted(country_data["country"].unique()),
        index=list(sorted(country_data["country"].unique())).index("Germany")
              if "Germany" in country_data["country"].values else 0
    )
    max_budget = st.slider("Max Monthly Budget (USD)", 200, 5000, 2000, 100)
    min_salary = st.slider("Min Average Salary (USD)", 100, 8000, 1000, 100)
    top_n      = st.number_input("Top-N Negara", min_value=3, max_value=30, value=10)

    ks_s, sil_s = compute_silhouette(feature_scaled)
    # FIX #10: Default K=4 (domain-knowledge), bukan otomatis K=2
    k_clusters = st.slider("Jumlah Cluster (K)", 2, 8, 4)

    st.markdown("<hr style='margin:10px 0 8px 0;'>", unsafe_allow_html=True)

    # FIX #9: Hapus "Run Analysis" button yang tidak berfungsi
    # Diganti dengan reset filter yang meaningful
    if st.button("↺ Reset Filter", use_container_width=True):
        st.rerun()

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


# TABS
tab1, tab_prep, tab4 = st.tabs([
    "  📊 Overview  ",
    "  🔎 Data Exploration  ",
    "  🌍 Recommender  "
])


# ═══════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════
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
        ("Total Negara",       str(len(country_data)),               "Aktif dalam dataset", "🌐"),
        ("Avg CLI",            f"${country_data['CLI'].mean():,.0f}", "Rata-rata global",    "📊"),
        ("Avg Monthly Salary", f"${country_data['x54'].mean():,.0f}","Across all countries","💼"),
        ("Top Score Country",  top_c,                                 f"Score {top_score:.2f}","🏆"),
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

    st.markdown("<div class='section-title'>Top 7 Negara — Recommendation Score Tertinggi</div>",
                unsafe_allow_html=True)
    display_df         = country_data[["country","CLI","x54","Recommendation_Score"]].copy()
    display_df.columns = ["Country", "CLI ($)", "Avg Salary ($)", "Rec. Score"]
    display_df         = (display_df.round(2)
                          .sort_values("Rec. Score", ascending=False)
                          .head(7).reset_index(drop=True))
    st.markdown(html_table(display_df, rank_col=True), unsafe_allow_html=True)

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
                Avg Salary &divide; CLI. Mengukur seberapa <i>worth it</i> suatu negara.
                Makin tinggi = gaji besar relatif terhadap biaya hidup.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# PREPROCESSING
with tab_prep:
    st.markdown("""
    <div class='page-header'>
        <span class='page-title'>🔬 Data Exploration</span>
        <span class='page-sub'>Preprocessing · Statistical Analysis · Clustering</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("🛠️ Preprocessing Pipeline", expanded=True):

        pp_stats    = pipeline_results["stats"]
        df_raw_pp   = pipeline_results["df_raw"]
        df_clean_pp = pipeline_results["df_clean"]
    
        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        prep_cards = [
            ("Raw Rows",      f"{pp_stats['n_raw']:,}",         "Total data mentah",     "📂"),
            ("After Cleaning",f"{pp_stats['n_after_clean']:,}", "Setelah cleaning",       "🧹"),
            ("Negara Raw",    f"{pp_stats['n_country_raw']}",   "Sebelum preprocessing", "🌐"),
            ("Negara Final",  f"{pp_stats['n_country_final']}", "Setelah preprocessing", "✅"),
        ]
        for col, (name, val, sub, icon) in zip([c1, c2, c3, c4], prep_cards):
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
    
        # Pipeline Flow Diagram
        st.markdown("<div class='section-title'>Alur Pipeline</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex; align-items:stretch; gap:6px; margin-bottom:20px; flex-wrap:wrap;'>
            <div class='pipeline-step'>
                <div class='pipeline-step-title'>📂 Raw Data</div>
                <div class='pipeline-step-desc'>{pp_stats['n_raw']:,} baris<br>CSV kota-level</div>
            </div>
            <div class='pipeline-arrow'>→</div>
            <div class='pipeline-step' style='border-top:3px solid {T["primary"]};'>
                <div class='pipeline-step-title'>🧹 Clean</div>
                <div class='pipeline-step-desc'>Dedup · Impute<br>IQR Capping</div>
            </div>
            <div class='pipeline-arrow'>→</div>
            <div class='pipeline-step' style='border-top:3px solid {T["accent"]};'>
                <div class='pipeline-step-title'>🏙️ Aggregate</div>
                <div class='pipeline-step-desc'>Kota → Negara<br>Mean per country</div>
            </div>
            <div class='pipeline-arrow'>→</div>
            <div class='pipeline-step' style='border-top:3px solid {T["second"]};'>
                <div class='pipeline-step-title'>⚙️ Engineer</div>
                <div class='pipeline-step-desc'>CLI · Rec. Score<br>Feature derivation</div>
            </div>
            <div class='pipeline-arrow'>→</div>
            <div class='pipeline-step' style='border-top:3px solid {T["primary"]};'>
                <div class='pipeline-step-title'>📐 Scale</div>
                <div class='pipeline-step-desc'>MinMax → Cosine<br>StdScaler → KMeans</div>
            </div>
            <div class='pipeline-arrow'>→</div>
            <div class='pipeline-step' style='border-top:3px solid {T["accent"]};'>
                <div class='pipeline-step-title'>🤖 Model</div>
                <div class='pipeline-step-desc'>{pp_stats['n_country_final']} negara<br>Ready for analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        # Ringkasan Pipeline Stats
        st.markdown("<div class='section-title'>Ringkasan Pipeline</div>", unsafe_allow_html=True)
        n_cleaned = pp_stats['n_raw'] - pp_stats['n_after_clean']
        pipeline_summary = pd.DataFrame({
            "Tahap"      : ["Raw data", "Exact duplicates removed", "Logical duplicates (city+country)",
                            "Rows setelah cleaning", "Negara raw", "Negara final"],
            "Nilai"      : [f"{pp_stats['n_raw']:,}", f"{pp_stats['n_exact_dup']:,}",
                            f"{n_cleaned - pp_stats['n_exact_dup']:,}",
                            f"{pp_stats['n_after_clean']:,}",
                            f"{pp_stats['n_country_raw']}", f"{pp_stats['n_country_final']}"],
            "Keterangan" : ["Input", "Duplikat identik dihapus", "Kota sama per negara, simpan data_quality tertinggi",
                            "Setelah semua cleaning steps", "Sebelum agregasi", "Setelah agregasi negara"],
        })
        st.markdown(html_table(pipeline_summary), unsafe_allow_html=True)
    
        # FIX #4: Detail teknis tersembunyi dalam expander
        with st.expander("🔬 Detail Teknis — untuk Data Scientist"):
            st.markdown("<div class='section-title'>Missing Value Comparison</div>", unsafe_allow_html=True)
            sample_cols = ["x1","x3","x8","x28","x33","x36","x48","x49","x54"]
            null_raw    = df_raw_pp[sample_cols].isnull().sum()
            null_clean  = df_clean_pp[sample_cols].isnull().sum()
            null_df = pd.DataFrame({
                "Kolom"    : sample_cols,
                "NaN Raw"  : null_raw.values,
                "NaN Clean": null_clean.values,
                "Berkurang": (null_raw - null_clean).values
            })
            st.markdown(html_table(null_df), unsafe_allow_html=True)
    
            st.markdown("<div class='section-title'>IQR Capping — x54 Salary</div>", unsafe_allow_html=True)
            raw_x54   = pd.to_numeric(df_raw_pp["x54"], errors="coerce").dropna()
            clean_x54 = df_clean_pp["x54"].dropna()
            fig_iqr = go.Figure()
            fig_iqr.add_trace(go.Box(y=raw_x54,   name="Raw",     marker_color=T["accent"], boxmean=True))
            fig_iqr.add_trace(go.Box(y=clean_x54, name="Cleaned", marker_color=T["primary"], boxmean=True))
            fig_iqr.update_layout(margin=dict(t=10, b=10),
                                   yaxis=dict(title=dict(text="Monthly Salary (USD)",
                                                         font=dict(color=T["muted"]))))
            pl(fig_iqr, 300)
            st.plotly_chart(fig_iqr, use_container_width=True, config={"displayModeBar": False})
    
            st.markdown("<div class='section-title'>CLI Component Weights</div>", unsafe_allow_html=True)
            cli_weights = pd.DataFrame({
                "Komponen": ["x1 — Meal","x48 — Rent","x36 — Utilities","x33 — Gasoline"],
                "Bobot"   : ["35%","40%","15%","10%"]
            })
            st.markdown(html_table(cli_weights), unsafe_allow_html=True)
    
            st.markdown("<div class='section-title'>Feature Selection — Variabel Model</div>",
                        unsafe_allow_html=True)
            feat_table = pd.DataFrame({
                "Kode" : ["x3","x8","x28","x49","Recommendation_Score"],
                "Fitur": ["McMeal","Water (1.5L)","Transport (Monthly)",
                          "Apartment Outside Centre","Purchasing Power Index"],
                "Scaler": ["MinMax→Cosine","MinMax→Cosine","MinMax→Cosine",
                           "MinMax→Cosine","MinMax→Cosine"]
            })
            st.markdown(html_table(feat_table), unsafe_allow_html=True)
    
            mi_tab, var_tab, dist_tab, norm_tab = st.tabs([
                "MI Score", "Variance Analysis", "Feature Distribution", "Normalisasi"
            ])
    
            with mi_tab:
                cosine_feat_set = {"x3","x8","x28","x49"}
                mi_plot = mi_df.copy()
                mi_plot["Warna"] = mi_plot["Feature"].apply(
                    lambda f: T["accent"]  if f in cosine_feat_set
                    else (T["primary"] if f in {"x1","x33","x36","x48","x54","CLI"} else T["border"])
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
                fig_mi.update_layout(
                    xaxis=dict(title=dict(text="Mutual Information Score", font=dict(color=T["muted"]))),
                    yaxis=dict(title=dict(text="Feature",                  font=dict(color=T["muted"]))),
                    showlegend=False, margin=dict(t=10, b=30, l=10, r=60)
                )
                st.plotly_chart(fig_mi, use_container_width=True, config={"displayModeBar": False})
    
            with var_tab:
                _, col_var, _ = st.columns([1, 2, 1])
                with col_var:
                    fig_var = px.scatter(
                        mi_df, x="Variance", y="MI_Score", text="Feature",
                        color="Lolos_VarThreshold",
                        color_discrete_map={True: T["primary"], False: T["border"]},
                        labels={"Variance":"Variance","MI_Score":"MI Score","Lolos_VarThreshold":"Lolos Threshold"}
                    )
                    fig_var.update_traces(
                        textposition="top center",
                        textfont=dict(size=7, color=T["text"]),
                        marker=dict(size=8, opacity=0.85, line=dict(color=T["card_bg"], width=1))
                    )
                    fig_var.add_vline(x=1.0, line_dash="dash", line_color=T["accent"],
                                      annotation_text="Threshold=1.0",
                                      annotation_font_color=T["accent"], annotation_font_size=10)
                    pl(fig_var, 400)
                    fig_var.update_layout(
                        showlegend=True,
                        legend=dict(font=dict(size=10, color=T["text"]), title=""),
                        margin=dict(t=10, b=10),
                        xaxis=dict(title=dict(text="Variance", font=dict(color=T["muted"]))),
                        yaxis=dict(title=dict(text="MI Score",  font=dict(color=T["muted"]))),
                    )
                    st.plotly_chart(fig_var, use_container_width=True, config={"displayModeBar": False})
    
            with dist_tab:
                _, col_dist, _ = st.columns([1, 2, 1])
                with col_dist:
                    df_fs = pipeline_results["df_features_scaled"].copy()
                    feat_labels = {
                        "x3":"McMeal","x8":"Water","x28":"Transport",
                        "x49":"Apartment","Recommendation_Score":"Rec.Score"
                    }
                    df_fs_plot = df_fs.rename(columns=feat_labels).melt(var_name="Feature", value_name="Scaled Value")
                    fig_feat = px.box(df_fs_plot, x="Feature", y="Scaled Value",
                                      color="Feature", color_discrete_sequence=BOX_COLORS)
                    pl(fig_feat, 380)
                    fig_feat.update_layout(
                        showlegend=False, margin=dict(t=10, b=10),
                        xaxis=dict(title=dict(text="Feature",            font=dict(color=T["muted"]))),
                        yaxis=dict(title=dict(text="Scaled Value [0–1]", font=dict(color=T["muted"]))),
                    )
                    st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})
    
            with norm_tab:
                fd   = ["x1","x28","x36","x48","x54"]
                fl   = ["Meal","Rent","Utilities","Gasoline","Salary"]
                rmlt = country_data[fd].copy()
                rmlt.columns = fl
                rmlt = rmlt.melt(var_name="Feature", value_name="Value")
                sarr = StandardScaler().fit_transform(country_data[fd])
                smlt = pd.DataFrame(sarr, columns=fl).melt(var_name="Feature", value_name="Value")
                nb, na = st.columns(2)
                with nb:
                    st.markdown("<div class='section-title'>Raw</div>", unsafe_allow_html=True)
                    fb = px.box(rmlt, x="Feature", y="Value", color="Feature",
                                color_discrete_sequence=BOX_COLORS)
                    pl(fb, 300); fb.update_layout(showlegend=False, margin=dict(t=10, b=10))
                    st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})
                with na:
                    st.markdown("<div class='section-title'>Z-Score</div>", unsafe_allow_html=True)
                    fa = px.box(smlt, x="Feature", y="Value", color="Feature",
                                color_discrete_sequence=BOX_COLORS)
                    pl(fa, 300); fa.update_layout(showlegend=False, margin=dict(t=10, b=10))
                    st.plotly_chart(fa, use_container_width=True, config={"displayModeBar": False})

# ANALYSIS
    with st.expander("🔍 Statistical Analysis", expanded=False):
        col_left, col_right = st.columns([2, 1])
    
        with col_left:
            st.markdown("<div class='section-title'>Correlation Heatmap: Selected Features</div>",
                        unsafe_allow_html=True)
            sel_cols   = ["x1","x3","x8","x28","x33","x36","x48","x49","x54","CLI"]
            sel_labels = ["Meal","McMeal","Water","Transport","Gasoline",
                          "Utilities","Apt-Centre","Apt-Outside","Salary","CLI"]
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
            ax.tick_params(axis="both", colors=T["text"], labelsize=9)
            for lbl in ax.get_xticklabels(): lbl.set_color(T["text"])
            for lbl in ax.get_yticklabels(): lbl.set_color(T["text"])
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
            fig_cli.update_layout(
                xaxis=dict(title=dict(text="Cost of Living Index (USD)", font=dict(color=T["muted"]))),
                yaxis=dict(title=dict(text="Jumlah Negara",              font=dict(color=T["muted"]))),
                showlegend=False, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_cli, use_container_width=True, config={"displayModeBar": False})
    
            st.markdown("<div class='section-title'>Distribusi Salary</div>", unsafe_allow_html=True)
            fig_sal = px.histogram(country_data, x="x54", nbins=30,
                                   color_discrete_sequence=[T["accent"]])
            pl(fig_sal, 230)
            fig_sal.update_layout(
                showlegend=False, margin=dict(t=10, b=10),
                xaxis=dict(title=dict(text="Monthly Salary (USD)", font=dict(color=T["muted"]))),
                yaxis=dict(title=dict(text="Jumlah Negara",        font=dict(color=T["muted"]))),
            )
            st.plotly_chart(fig_sal, use_container_width=True, config={"displayModeBar": False})
    
        # FIX #6: Scatter dengan quadrant lines + zona label
        st.markdown("<div class='section-title'>Scatter: Monthly Salary vs Cost of Living Index — Analisis Kuadran</div>",
                    unsafe_allow_html=True)
    
        med_cli    = country_data["CLI"].median()
        med_salary = country_data["x54"].median()
    
        fig_sc = px.scatter(
            country_data, x="CLI", y="x54",
            size="Recommendation_Score", color="Recommendation_Score",
            hover_name="country",
            color_continuous_scale=CSCALE,
            labels={"CLI":"Cost of Living Index (USD)","x54":"Monthly Salary (USD)",
                    "Recommendation_Score":"Recommendation Score"}
        )
        # Quadrant lines
        fig_sc.add_hline(
            y=med_salary, line_dash="dash", line_color=T["muted"], line_width=1.2,
            annotation_text=f"Median Salary ${med_salary:,.0f}",
            annotation_font_color=T["muted"], annotation_font_size=10,
            annotation_position="top right"
        )
        fig_sc.add_vline(
            x=med_cli, line_dash="dash", line_color=T["muted"], line_width=1.2,
            annotation_text=f"Median CLI ${med_cli:,.0f}",
            annotation_font_color=T["muted"], annotation_font_size=10,
            annotation_position="top right"
        )
        # Quadrant annotations
        x_range = country_data["CLI"].max() - country_data["CLI"].min()
        y_range = country_data["x54"].max() - country_data["x54"].min()
        # Plotly tidak support hex 8-digit — gunakan rgba() untuk alpha
        ann_bg = "rgba(255,255,255,0.80)" if not dark_mode else "rgba(30,33,35,0.80)"
        quadrant_labels = [
            (med_cli * 0.35,               med_salary + y_range * 0.12, "🎯 Sweet Spot", T["primary"]),
            (med_cli + x_range * 0.22,     med_salary + y_range * 0.12, "💰 Premium",    T["accent"]),
            (med_cli * 0.35,               med_salary - y_range * 0.12, "💸 Budget",     T["second"]),
            (med_cli + x_range * 0.22,     med_salary - y_range * 0.12, "⚠️ Trap",      "#c0392b"),
        ]
        for qx, qy, qlabel, qcolor in quadrant_labels:
            fig_sc.add_annotation(
                x=qx, y=qy, text=qlabel, showarrow=False,
                font=dict(size=11, color=qcolor, family="Sora, sans-serif"),
                bgcolor=ann_bg, bordercolor=qcolor,
                borderwidth=1, borderpad=5, opacity=0.9
            )
        pl(fig_sc, 430)
        fig_sc.update_layout(
            xaxis=dict(title=dict(text="Cost of Living Index (USD)", font=dict(color=T["muted"]))),
            yaxis=dict(title=dict(text="Monthly Salary (USD)",       font=dict(color=T["muted"]))),
            coloraxis_colorbar=dict(
                title=dict(font=dict(color=T["muted"])),
                tickfont=dict(color=T["muted"])
            )
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
    
        st.markdown(f"""
        <div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;'>
            <div style='flex:1;min-width:140px;background:{T["primary"]}15;border:1px solid {T["primary"]}44;
                        border-radius:10px;padding:10px 14px;'>
                <b style='color:{T["primary"]};'>🎯 Sweet Spot</b>
                <span style='font-size:12px;color:{T["muted"]};'> — CLI rendah, gaji tinggi. Negara ideal untuk relokasi.</span>
            </div>
            <div style='flex:1;min-width:140px;background:{T["accent"]}15;border:1px solid {T["accent"]}44;
                        border-radius:10px;padding:10px 14px;'>
                <b style='color:{T["accent"]};'>💰 Premium</b>
                <span style='font-size:12px;color:{T["muted"]};'> — CLI tinggi, gaji tinggi. Worth it jika karir mendukung.</span>
            </div>
            <div style='flex:1;min-width:140px;background:{T["second"]}22;border:1px solid {T["second"]}55;
                        border-radius:10px;padding:10px 14px;'>
                <b style='color:{T["accent"]};'>💸 Budget</b>
                <span style='font-size:12px;color:{T["muted"]};'> — CLI rendah, gaji rendah. Terjangkau tapi terbatas.</span>
            </div>
            <div style='flex:1;min-width:140px;background:#c0392b15;border:1px solid #c0392b44;
                        border-radius:10px;padding:10px 14px;'>
                <b style='color:#c0392b;'>⚠️ Trap</b>
                <span style='font-size:12px;color:{T["muted"]};'> — CLI tinggi, gaji rendah. Hindari untuk relokasi jangka panjang.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# TAB 4: CLUSTERING
    with st.expander("🤖 K-Means Clustering", expanded=False):
    

            # Info K default
            st.markdown(f"""
            <div class='info-box' style='border-left:3px solid {T["accent"]};'>
                <b>Panduan Pemilihan K:</b> Default K=4 direkomendasikan berdasarkan kombinasi
                domain knowledge (4 kuadran biaya hidup) dan validasi elbow method.
                Geser slider di sidebar untuk mengeksplorasi konfigurasi lain.
            </div>
            """, unsafe_allow_html=True)
        
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
                    fe.update_layout(
                        xaxis=dict(title=dict(text="Jumlah Cluster (K)", font=dict(color=T["muted"]))),
                        yaxis=dict(title=dict(text="Inertia — Within-Cluster Sum of Squares", font=dict(color=T["muted"]))),
                        font=dict(color=T["muted"])
                    )
                    st.plotly_chart(fe, use_container_width=True, config={"displayModeBar": False})
        
            with cs:
                st.markdown("<div class='section-title'>Silhouette Score</div>", unsafe_allow_html=True)
                fs = go.Figure()
                fs.add_trace(go.Bar(
                    x=ks_s, y=sil_s,
                    marker_color=[T["primary"] if k == k_clusters else T["border"] for k in ks_s],
                    marker_line_width=0
                ))
                pl(fs, 340)
                fs.update_layout(
                    xaxis=dict(title=dict(text="Jumlah Cluster (K)", font=dict(color=T["muted"]))),
                    yaxis=dict(title=dict(text="Silhouette Score (0 - 1)", font=dict(color=T["muted"])))
                )
                st.plotly_chart(fs, use_container_width=True, config={"displayModeBar": False})
        
            st.markdown("<div class='section-title'>PCA Cluster Plot — Reduksi Dimensi 10D ke 2D</div>",
                        unsafe_allow_html=True)
        
            km_model, cluster_labels = run_kmeans(k_clusters, feature_scaled)
            pca        = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(feature_scaled)
            var_exp    = pca.explained_variance_ratio_
        
            clustered_temp = country_data.copy()
            clustered_temp["Cluster_ID"] = cluster_labels
        
            pca_df = pd.DataFrame({
                "PC1"    : pca_coords[:, 0],
                "PC2"    : pca_coords[:, 1],
                "country": country_data["country"].values,
                "Cluster": [get_cluster_label(c, clustered_temp) for c in cluster_labels],
                "CLI"    : country_data["CLI"].values.round(0),
                "Salary" : country_data["x54"].values.round(0)
            })
        
            unique_clusters = pca_df["Cluster"].unique()
            color_map = {
                label: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                for i, label in enumerate(sorted(unique_clusters))
            }
        
            fp = px.scatter(
                pca_df, x="PC1", y="PC2", color="Cluster",
                hover_name="country",
                hover_data={"CLI":True,"Salary":True,"PC1":False,"PC2":False},
                color_discrete_map=color_map,
            )
            fp.update_traces(marker=dict(size=10, opacity=0.82, line=dict(color=T["card_bg"], width=1)))
            pl(fp, 500)
            fp.update_layout(
                xaxis=dict(title=dict(text=f"PC1 — {var_exp[0]*100:.1f}% Variance Explained", font=dict(color=T["muted"]))),
                yaxis=dict(title=dict(text=f"PC2 — {var_exp[1]*100:.1f}% Variance Explained", font=dict(color=T["muted"]))),
            )
            st.plotly_chart(fp, use_container_width=True, config={"displayModeBar": False})
        
            sil_now  = silhouette_score(feature_scaled, cluster_labels)
            best_k   = ks_s[sil_s.index(max(sil_s))]
            sil_best = max(sil_s)
            sil_msg  = (
                f"— cluster overlap. K={best_k} menghasilkan silhouette tertinggi ({sil_best:.3f}), "
                "tapi K=4 lebih interpretatif secara domain."
                if sil_now < 0.35
                else "— pemisahan cluster baik ✓"
            )
            st.markdown(f"""
            <div class='info-box' style='border-left:3px solid {T["accent"] if sil_now < 0.35 else T["primary"]};'>
                <b>Kualitas Cluster K={k_clusters}:</b> Silhouette Score = <b>{sil_now:.3f}</b> {sil_msg}
            </div>
            """, unsafe_allow_html=True)
        
            # Cluster Profiling
            st.markdown("<div class='section-title'>Cluster Profiling</div>", unsafe_allow_html=True)
            clustered_df = country_data.copy()
            clustered_df["Cluster_ID"] = cluster_labels
            clustered_df["Cluster"] = [get_cluster_label(c, clustered_df) for c in cluster_labels]
        
            profile = clustered_df.groupby("Cluster").agg(
                N_Negara   = ("country","count"),
                Avg_CLI    = ("CLI",    lambda x: round(x.mean(), 0)),
                Avg_Salary = ("x54",   lambda x: round(x.mean(), 0)),
                Avg_Score  = ("Recommendation_Score", lambda x: round(x.mean(), 2))
            ).reset_index()
            profile.columns = ["Cluster", "N Negara", "Avg CLI ($)", "Avg Salary ($)", "Avg Score"]
            st.markdown(html_table(profile), unsafe_allow_html=True)
        
            # Anggota per Cluster — FIX #1: Radio adaptive via CSS
            st.markdown("<div class='section-title'>Anggota per Cluster</div>", unsafe_allow_html=True)
            members = clustered_df[["country","Cluster","CLI","x54","Recommendation_Score"]].copy()
            members.columns = ["Country","Cluster","CLI ($)","Salary ($)","Score"]
            members = members.round(1).sort_values("Cluster").reset_index(drop=True)
        
            cluster_options  = ["All"] + sorted(members["Cluster"].unique().tolist())
            selected_cluster = st.radio(
                "Filter Cluster:",
                options=cluster_options,
                horizontal=True,
                key="cluster_filter"
            )
        
            filtered_members = (
                members[members["Cluster"] == selected_cluster].reset_index(drop=True)
                if selected_cluster != "All"
                else members
            )
            st.markdown(html_table(filtered_members), unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 5: RECOMMENDER
# FIX #7: Tambah cosine similarity bar chart
# FIX #8: Selaraskan radar chart dengan fitur model + label eksplisit
# FIX #12: Info-box dengan bahasa manusiawi
# ═══════════════════════════════════════════
with cr:
    st.markdown(f"""
    <div class='section-title'>Profil Negara Referensi</div>
    <div style='font-size:11px;color:{T["muted"]};margin-bottom:10px;'>
        Skala 0–1 = posisi relatif terhadap semua negara (MinMax global)
    </div>
    """, unsafe_allow_html=True)

    radar_vars = ["x3", "x8", "x28", "x49", "Recommendation_Score"]
    radar_lbls = ["McMeal", "Water", "Transport", "Apt-Outside", "Rec.Score"]

    # Ambil nilai mentah dari country_data lalu normalisasi GLOBAL
    ref_row = country_data[country_data["country"] == user_country]
    if not ref_row.empty:
        raw_vals = ref_row[radar_vars].values.flatten().astype(float)
        all_vals = country_data[radar_vars].fillna(0).values
        min_v = np.nanmin(all_vals, axis=0)
        max_v = np.nanmax(all_vals, axis=0)
        vals_n = (raw_vals - min_v) / (max_v - min_v + 1e-9)
        vals_n = np.clip(vals_n, 0, 1)
    else:
        vals_n = np.zeros(len(radar_vars))

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=np.append(vals_n, vals_n[0]),
        theta=radar_lbls + [radar_lbls[0]],
        fill="toself",
        name=user_country,
        fillcolor="rgba(111,129,110,0.18)" if not dark_mode else "rgba(124,145,121,0.22)",
        line=dict(color=T["primary"], width=2.5)
    ))
    fig_r.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=T["border"],
                tickfont=dict(color=T["text"], size=10),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "25%", "50%", "75%", "100%"],
            ),
            angularaxis=dict(tickfont=dict(color=T["text"], size=11)),
            bgcolor=T["chart_pl"]
        ),
        height=300,
        paper_bgcolor=T["card_bg"],
        showlegend=False,
        font=dict(color=T["text"], family="Sora, sans-serif"),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

    if not ref_row.empty:
        ref_data = pd.DataFrame({
            "Metrik": ["CLI ($)", "Avg Salary ($)", "Rec. Score"],
            "Nilai": [
                f"${ref_row['CLI'].values[0]:,.0f}",
                f"${ref_row['x54'].values[0]:,.0f}",
                f"{ref_row['Recommendation_Score'].values[0]:.2f}"
            ]
        })
        st.markdown(html_table(ref_data), unsafe_allow_html=True)

        
        # Tabel ringkas negara referensi
        ref_row = country_data[country_data["country"] == user_country]
        if not ref_row.empty:
            ref_data = pd.DataFrame({
                "Metrik": ["CLI ($)", "Avg Salary ($)", "Rec. Score"],
                "Nilai": [
                    f"${ref_row['CLI'].values[0]:,.0f}",
                    f"${ref_row['x54'].values[0]:,.0f}",
                    f"{ref_row['Recommendation_Score'].values[0]:.2f}"
                ]
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

    # FIX #7: Similarity bar chart — gap antar kandidat jadi jelas
    if not recs.empty:
        st.markdown("<div class='section-title'>Similarity Score — Perbandingan Kandidat</div>",
                    unsafe_allow_html=True)
        sim_sorted = recs.sort_values("Similarity (%)", ascending=True)
        fig_sim = px.bar(
            sim_sorted, x="Similarity (%)", y="country", orientation="h",
            color="Similarity (%)", color_continuous_scale=CSCALE,
            labels={"Similarity (%)":"Cosine Similarity (%)","country":"Negara"}
        )
        fig_sim.update_traces(
            text=sim_sorted["Similarity (%)"].astype(str) + "%",
            textposition="outside",
            textfont=dict(size=11, color=T["text"])
        )
        pl(fig_sim, max(280, len(recs) * 36))
        fig_sim.update_layout(
            showlegend=False,
            xaxis=dict(title=dict(text="Cosine Similarity (%)", font=dict(color=T["muted"])),
                       range=[0, 110]),
            yaxis=dict(title=dict(text="Negara", font=dict(color=T["muted"]))),
            coloraxis_colorbar=dict(title=dict(font=dict(color=T["muted"])),
                                    tickfont=dict(color=T["muted"]))
        )
        st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})

# FOOTER
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
