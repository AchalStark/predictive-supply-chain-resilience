# ============================================================
# Navi-Shield: Predictive Supply Chain Resilience
# GRV Pipeline + Streamlit Prototype
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Navi-Shield | Supply Chain Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    .metric-card {
        background: linear-gradient(135deg, #1e2235 0%, #252a3d 100%);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .recommended-card {
        background: linear-gradient(135deg, #0d2b1e 0%, #1a4a2e 100%);
        border: 2px solid #00c875;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .risk-high { color: #ff4b4b; font-weight: 700; }
    .risk-medium { color: #ffb347; font-weight: 700; }
    .risk-low { color: #00c875; font-weight: 700; }
    h1 { color: #e8eaf6 !important; }
    .stMetric label { color: #8892b0 !important; font-size: 13px !important; }
    .stMetric [data-testid="metric-container"] > div:first-child {
        color: #ccd6f6 !important;
    }
    .section-header {
        color: #64ffda;
        font-size: 18px;
        font-weight: 600;
        padding: 8px 0;
        border-bottom: 1px solid #2d3250;
        margin-bottom: 16px;
    }
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .pill-green { background: #0d2b1e; color: #00c875; border: 1px solid #00c875; }
    .pill-red { background: #2b0d0d; color: #ff4b4b; border: 1px solid #ff4b4b; }
    .pill-yellow { background: #2b220d; color: #ffb347; border: 1px solid #ffb347; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ROUTE MAPPING
# Each route is associated with country_codes
# from the sanctions/embargo sheets and
# geographic corridor bounding boxes for
# conflict/piracy.
# ─────────────────────────────────────────────
ROUTE_INFO = {
    1: {
        "name": "Suez Canal",
        "emoji": "⚓",
        "desc": "Mediterranean → Red Sea → Indian Ocean",
        "country_codes": [651, 522, 663],   # Egypt, Djibouti, Somalia
        "lat_range": (10, 32), "lon_range": (32, 50)
    },
    2: {
        "name": "Cape of Good Hope",
        "emoji": "🌊",
        "desc": "Atlantic → Southern Africa → Indian Ocean",
        "country_codes": [560, 540],        # South Africa, Mozambique
        "lat_range": (-35, 0), "lon_range": (15, 50)
    },
    3: {
        "name": "INSTC Rail",
        "emoji": "🚂",
        "desc": "India → Iran → Russia overland corridor",
        "country_codes": [7, 86, 91, 98],   # Russia, Iran, India, China
        "lat_range": (20, 60), "lon_range": (50, 80)
    },
    4: {
        "name": "Air Freight",
        "emoji": "✈️",
        "desc": "Global air cargo network",
        "country_codes": [7, 98, 91, 86],
        "lat_range": (0, 90), "lon_range": (0, 180)
    },
}

ROUTE_NAMES = {k: v["name"] for k, v in ROUTE_INFO.items()}
COLOR_MAP = {
    "Suez Canal":            "#4fc3f7",
    "Cape of Good Hope":     "#81c784",
    "INSTC Rail":            "#ffb74d",
    "Air Freight":           "#ce93d8",
}


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_conflict(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    return pd.read_csv("sheet_1_conflict_data.csv")

@st.cache_data(show_spinner=False)
def load_piracy(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    return pd.read_csv("sheet_2_piracy_data.csv")

@st.cache_data(show_spinner=False)
def load_sanctions(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    return pd.read_csv("sheet_3_sanctions_data-3.csv")

@st.cache_data(show_spinner=False)
def load_embargo(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    return pd.read_csv("sheet_4_embargo_data-4.csv")


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_conflict_features(conflict_df):
    """
    Aggregate per-corridor conflict events into (route_id, snapshot_quarter)
    raw features: event_count and total_fatalities.
    Because conflict_df already has in_corridor=1 for relevant events,
    we use latitude/longitude bounding boxes to assign route_id.
    """
    df = conflict_df[conflict_df["in_corridor"] == 1].copy()

    def assign_route(row):
        lat, lon = row["latitude"], row["longitude"]
        for rid, info in ROUTE_INFO.items():
            if (info["lat_range"][0] <= lat <= info["lat_range"][1] and
                    info["lon_range"][0] <= lon <= info["lon_range"][1]):
                return rid
        return 1  # default to Suez if unmatched

    df["route_id"] = df.apply(assign_route, axis=1)
    agg = df.groupby(["route_id", "snapshot_quarter"]).agg(
        event_count=("conflict_id", "count"),
        total_fatalities=("fatalities", "sum")
    ).reset_index()
    agg["raw_conflict"] = agg["event_count"] + 0.1 * agg["total_fatalities"]
    return agg[["route_id", "snapshot_quarter", "raw_conflict"]]


def build_piracy_features(piracy_df):
    """
    Aggregate per-corridor piracy incidents into (route_id, snapshot_quarter).
    Weighted metric: incident count + normalized severity.
    """
    df = piracy_df[piracy_df["in_corridor"] == 1].copy()

    def assign_route(row):
        lat, lon = row["latitude"], row["longitude"]
        for rid, info in ROUTE_INFO.items():
            if (info["lat_range"][0] <= lat <= info["lat_range"][1] and
                    info["lon_range"][0] <= lon <= info["lon_range"][1]):
                return rid
        return 1

    df["route_id"] = df.apply(assign_route, axis=1)
    agg = df.groupby(["route_id", "snapshot_quarter"]).agg(
        incident_count=("incident_id", "count"),
        total_severity=("severity_economic_usd", "sum")
    ).reset_index()
    sev_max = agg["total_severity"].max() if agg["total_severity"].max() > 0 else 1
    agg["raw_piracy"] = agg["incident_count"] + (agg["total_severity"] / sev_max) * 5
    return agg[["route_id", "snapshot_quarter", "raw_piracy"]]


def build_sanction_features(sanctions_df):
    """
    For each route, sum severity_score of active sanctions for countries
    that belong to that corridor.
    """
    df = sanctions_df[sanctions_df["active_flag"] == 1].copy()
    rows = []
    for _, row in df.iterrows():
        for rid, info in ROUTE_INFO.items():
            if row["country_code"] in info["country_codes"]:
                rows.append({
                    "route_id": rid,
                    "snapshot_quarter": row["snapshot_quarter"],
                    "severity_score": row["severity_score"]
                })
    if not rows:
        return pd.DataFrame(columns=["route_id", "snapshot_quarter", "raw_sanction"])
    agg = pd.DataFrame(rows).groupby(["route_id", "snapshot_quarter"]).agg(
        raw_sanction=("severity_score", "sum")
    ).reset_index()
    return agg


def build_embargo_features(embargo_df):
    """
    For each route, sum restriction_value of embargoes that apply
    to countries in that corridor. Negative values (blocked trade)
    are taken as absolute value.
    """
    rows = []
    for _, row in embargo_df.iterrows():
        for rid, info in ROUTE_INFO.items():
            if row["country_code"] in info["country_codes"]:
                rows.append({
                    "route_id": rid,
                    "snapshot_quarter": row["snapshot_quarter"],
                    "restriction_value": abs(row["restriction_value"])
                })
    if not rows:
        return pd.DataFrame(columns=["route_id", "snapshot_quarter", "raw_embargo"])
    agg = pd.DataFrame(rows).groupby(["route_id", "snapshot_quarter"]).agg(
        raw_embargo=("restriction_value", "sum")
    ).reset_index()
    return agg


# ─────────────────────────────────────────────
# GRV PIPELINE (MASTER ENGINE)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_grv_pipeline(conflict_df, piracy_df, sanctions_df, embargo_df):
    """
    Full pipeline: load → feature engineering → normalization → GRV → label.
    Returns Sheet 5 (GRV Master).
    """
    # Step 1: Build raw features
    conflict_feat  = build_conflict_features(conflict_df)
    piracy_feat    = build_piracy_features(piracy_df)
    sanction_feat  = build_sanction_features(sanctions_df)
    embargo_feat   = build_embargo_features(embargo_df)

    # Step 2: Full cross-join of routes × all quarters
    all_quarters = sorted(set(
        list(conflict_feat["snapshot_quarter"].unique()) +
        list(piracy_feat["snapshot_quarter"].unique())
    ))
    route_ids = [1, 2, 3, 4]
    base = pd.DataFrame(
        [(r, q) for r in route_ids for q in all_quarters],
        columns=["route_id", "snapshot_quarter"]
    )

    # Step 3: Merge features
    df = base.copy()
    df = df.merge(conflict_feat,  on=["route_id", "snapshot_quarter"], how="left")
    df = df.merge(piracy_feat,    on=["route_id", "snapshot_quarter"], how="left")
    df = df.merge(sanction_feat,  on=["route_id", "snapshot_quarter"], how="left")
    df = df.merge(embargo_feat,   on=["route_id", "snapshot_quarter"], how="left")
    df = df.fillna(0)

    # Step 4: MinMaxScaler normalization (0–1 per feature)
    scaler = MinMaxScaler()
    raw_cols    = ["raw_conflict", "raw_piracy", "raw_sanction", "raw_embargo"]
    score_cols  = ["conflict_score", "piracy_score", "sanction_score", "embargo_score"]
    if df[raw_cols].sum().sum() > 0:
        df[score_cols] = scaler.fit_transform(df[raw_cols])
    else:
        df[score_cols] = 0.0

    # Step 5: GRV weighted formula
    df["GRV"] = (
        0.35 * df["conflict_score"] +
        0.25 * df["piracy_score"]   +
        0.25 * df["sanction_score"] +
        0.15 * df["embargo_score"]
    )

    # Step 6: Recommendation label (lowest GRV per quarter = 1)
    df["recommended"] = 0
    idx = df.groupby("snapshot_quarter")["GRV"].idxmin()
    df.loc[idx, "recommended"] = 1

    # Step 7: Add route names
    df["route_name"] = df["route_id"].map(ROUTE_NAMES)

    return df


# ─────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────

def train_rf_classifier(grv_df):
    """
    RandomForestClassifier to predict 'recommended' binary label
    from the 4 component scores.
    """
    features = ["conflict_score", "piracy_score", "sanction_score", "embargo_score", "route_id"]
    df = grv_df[features + ["recommended"]].dropna()
    if df["recommended"].nunique() < 2 or len(df) < 20:
        return None, None, None

    X, y = df[features], df["recommended"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    return clf, report, importances


def train_gbr_forecaster(grv_df):
    """
    GradientBoostingRegressor to forecast next-quarter GRV per route,
    using lag-1, lag-2, and rolling-4Q mean of GRV.
    """
    results = {}
    for rid in [1, 2, 3, 4]:
        rdf = grv_df[grv_df["route_id"] == rid].sort_values("snapshot_quarter").copy()
        rdf = rdf.reset_index(drop=True)
        rdf["lag1"] = rdf["GRV"].shift(1)
        rdf["lag2"] = rdf["GRV"].shift(2)
        rdf["roll4"] = rdf["GRV"].rolling(4).mean()
        rdf = rdf.dropna()
        if len(rdf) < 15:
            continue
        X = rdf[["lag1", "lag2", "roll4", "conflict_score", "piracy_score",
                  "sanction_score", "embargo_score"]]
        y = rdf["GRV"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))

        # Forecast next quarter
        last = rdf.iloc[-1]
        next_X = pd.DataFrame([{
            "lag1":            last["GRV"],
            "lag2":            last["lag1"],
            "roll4":           rdf["GRV"].tail(4).mean(),
            "conflict_score":  last["conflict_score"],
            "piracy_score":    last["piracy_score"],
            "sanction_score":  last["sanction_score"],
            "embargo_score":   last["embargo_score"],
        }])
        forecast = float(model.predict(next_X)[0])
        results[rid] = {"model": model, "rmse": rmse, "forecast": max(0, min(1, forecast))}
    return results


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Navi-Shield")
    st.markdown("*Predictive Supply Chain Resilience*")
    st.divider()

    st.markdown("### 📂 Data Sources")
    up1 = st.file_uploader("Sheet 1 — Conflict",  type="csv", key="s1")
    up2 = st.file_uploader("Sheet 2 — Piracy",    type="csv", key="s2")
    up3 = st.file_uploader("Sheet 3 — Sanctions", type="csv", key="s3")
    up4 = st.file_uploader("Sheet 4 — Embargo",   type="csv", key="s4")

    st.divider()
    st.markdown("### ⚖️ GRV Weight Override")
    w_conflict  = st.slider("Conflict weight",  0.0, 1.0, 0.35, 0.05)
    w_piracy    = st.slider("Piracy weight",    0.0, 1.0, 0.25, 0.05)
    w_sanction  = st.slider("Sanction weight",  0.0, 1.0, 0.25, 0.05)
    w_embargo   = st.slider("Embargo weight",   0.0, 1.0, 0.15, 0.05)
    total_w = w_conflict + w_piracy + w_sanction + w_embargo
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_w:.2f}. Should equal 1.0.")

    st.divider()
    st.markdown("### 🔬 Analysis Mode")
    mode = st.radio("", ["📊 GRV Dashboard", "📈 Time Series", "🤖 ML Prediction", "📥 Export Sheet 5"])


# ─────────────────────────────────────────────
# LOAD DATA + RUN PIPELINE
# ─────────────────────────────────────────────

with st.spinner("⚙️ Running GRV Pipeline..."):
    try:
        conflict_df  = load_conflict(up1)
        piracy_df    = load_piracy(up2)
        sanctions_df = load_sanctions(up3)
        embargo_df   = load_embargo(up4)
        grv_df       = run_grv_pipeline(conflict_df, piracy_df, sanctions_df, embargo_df)

        # Apply custom weights if changed
        if abs(total_w - 1.0) <= 0.01 and (w_conflict != 0.35 or w_piracy != 0.25):
            grv_df["GRV"] = (
                w_conflict  * grv_df["conflict_score"] +
                w_piracy    * grv_df["piracy_score"]   +
                w_sanction  * grv_df["sanction_score"] +
                w_embargo   * grv_df["embargo_score"]
            )
            grv_df["recommended"] = 0
            grv_df.loc[grv_df.groupby("snapshot_quarter")["GRV"].idxmin(), "recommended"] = 1

        pipeline_ok = True
    except FileNotFoundError as e:
        st.error(f"CSV file not found: {e}. Upload files in the sidebar or place CSVs in the same folder as app.py.")
        st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🛡️")
with col_title:
    st.markdown("## Navi-Shield — Global Risk Variable Dashboard")
    st.caption("Predictive Supply Chain Resilience: Navigating Disruptions via Custom Risk Variables")

st.divider()

all_quarters = sorted(grv_df["snapshot_quarter"].unique())
latest_q     = all_quarters[-1]


# ─────────────────────────────────────────────
# MODE: GRV DASHBOARD
# ─────────────────────────────────────────────
if "GRV Dashboard" in mode:

    selected_q = st.selectbox(
        "📅 Select Snapshot Quarter (YYYYQQ)",
        options=all_quarters,
        index=len(all_quarters) - 1,
        format_func=lambda x: f"Q{str(x)[-1]} {str(x)[:4]}"
    )

    q_data = grv_df[grv_df["snapshot_quarter"] == selected_q].sort_values("GRV")

    # Top row: recommended route
    rec_row  = q_data[q_data["recommended"] == 1].iloc[0]
    rec_info = ROUTE_INFO[int(rec_row["route_id"])]
    worst    = q_data.iloc[-1]

    st.markdown(f"""
    <div class="recommended-card">
        <div style="font-size:36px">{rec_info['emoji']}</div>
        <div style="color:#00c875; font-size:22px; font-weight:700; margin:8px 0">
            ✅ RECOMMENDED ROUTE
        </div>
        <div style="color:#e8eaf6; font-size:28px; font-weight:800">{rec_row['route_name']}</div>
        <div style="color:#8892b0; margin-top:4px">{rec_info['desc']}</div>
        <div style="color:#64ffda; font-size:20px; margin-top:12px">
            GRV: <strong>{rec_row['GRV']:.4f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    kpi_cols = st.columns(4)
    for i, (_, row) in enumerate(q_data.iterrows()):
        with kpi_cols[i]:
            risk_class = "risk-low" if row["GRV"] < 0.2 else ("risk-medium" if row["GRV"] < 0.5 else "risk-high")
            badge = "🥇" if row["recommended"] == 1 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:24px">{ROUTE_INFO[int(row['route_id'])]['emoji']}</div>
                <div style="color:#ccd6f6; font-weight:600; font-size:14px; margin:4px 0">
                    {row['route_name']} {badge}
                </div>
                <div class="{risk_class}" style="font-size:26px">{row['GRV']:.4f}</div>
                <div style="color:#8892b0; font-size:11px">GRV Score</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # GRV bar chart
    with c1:
        st.markdown('<div class="section-header">📊 GRV by Route</div>', unsafe_allow_html=True)
        colors = [COLOR_MAP.get(n, "#888") for n in q_data["route_name"]]
        fig_bar = go.Figure(go.Bar(
            x=q_data["route_name"],
            y=q_data["GRV"],
            marker_color=colors,
            text=[f"{v:.4f}" for v in q_data["GRV"]],
            textposition="outside"
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#ccd6f6", yaxis_range=[0, 1],
            margin=dict(t=20, b=10), height=320,
            yaxis=dict(gridcolor="#2d3250"),
            xaxis=dict(gridcolor="#2d3250")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar / component breakdown
    with c2:
        st.markdown('<div class="section-header">🕸️ Component Score Breakdown</div>', unsafe_allow_html=True)
        categories = ["Conflict", "Piracy", "Sanction", "Embargo"]
        fig_radar = go.Figure()
        for _, row in q_data.iterrows():
            vals = [row["conflict_score"], row["piracy_score"],
                    row["sanction_score"], row["embargo_score"]]
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["route_name"],
                line_color=COLOR_MAP.get(row["route_name"], "#888"),
                opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(bgcolor="#1a1d27",
                       radialaxis=dict(visible=True, range=[0, 1], color="#8892b0"),
                       angularaxis=dict(color="#ccd6f6")),
            paper_bgcolor="#0f1117", font_color="#ccd6f6",
            legend=dict(bgcolor="#0f1117"), height=320,
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Component score table
    st.markdown('<div class="section-header">📋 Detailed Score Table</div>', unsafe_allow_html=True)
    display_df = q_data[["route_name", "conflict_score", "piracy_score",
                           "sanction_score", "embargo_score", "GRV", "recommended"]].copy()
    display_df.columns = ["Route", "Conflict", "Piracy", "Sanction",
                           "Embargo", "GRV", "Recommended"]
    display_df["Recommended"] = display_df["Recommended"].map({1: "✅ YES", 0: "❌ NO"})
    for col in ["Conflict", "Piracy", "Sanction", "Embargo", "GRV"]:
        display_df[col] = display_df[col].map("{:.4f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# MODE: TIME SERIES
# ─────────────────────────────────────────────
elif "Time Series" in mode:
    st.markdown("### 📈 GRV Historical Trends")

    max_q = st.slider("Number of most-recent quarters to display", 4, min(200, len(all_quarters)), 40)
    recent = sorted(all_quarters)[-max_q:]
    ts_df  = grv_df[grv_df["snapshot_quarter"].isin(recent)].copy()
    ts_df["Q_label"] = ts_df["snapshot_quarter"].astype(str).apply(
        lambda x: f"Q{x[-1]} {x[:4]}"
    )

    fig_ts = go.Figure()
    for rid in [1, 2, 3, 4]:
        rdf = ts_df[ts_df["route_id"] == rid].sort_values("snapshot_quarter")
        fig_ts.add_trace(go.Scatter(
            x=rdf["Q_label"], y=rdf["GRV"],
            mode="lines+markers",
            name=ROUTE_NAMES[rid],
            line=dict(color=COLOR_MAP[ROUTE_NAMES[rid]], width=2),
            marker=dict(size=5)
        ))
    fig_ts.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#ccd6f6", height=400,
        yaxis=dict(gridcolor="#2d3250", range=[0, 1], title="GRV"),
        xaxis=dict(gridcolor="#2d3250", title="Quarter"),
        legend=dict(bgcolor="#1a1d27"),
        margin=dict(t=20)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Stacked component area per route
    st.markdown("### 🧩 Component Breakdown Over Time")
    sel_route = st.selectbox("Select route", options=[1, 2, 3, 4],
                              format_func=lambda x: ROUTE_NAMES[x])
    rdf2 = ts_df[ts_df["route_id"] == sel_route].sort_values("snapshot_quarter")
    score_labels = {
        "conflict_score": "Conflict",
        "piracy_score": "Piracy",
        "sanction_score": "Sanction",
        "embargo_score": "Embargo"
    }
    comp_colors = ["#ff4b4b", "#4fc3f7", "#ffb347", "#ce93d8"]
    fig_area = go.Figure()
    for (col, label), color in zip(score_labels.items(), comp_colors):
        fig_area.add_trace(go.Scatter(
            x=rdf2["Q_label"], y=rdf2[col],
            name=label, fill="tozeroy",
            line=dict(color=color, width=1.5),
            opacity=0.6
        ))
    fig_area.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#ccd6f6", height=350,
        yaxis=dict(gridcolor="#2d3250", range=[0, 1]),
        xaxis=dict(gridcolor="#2d3250"),
        legend=dict(bgcolor="#1a1d27"),
        margin=dict(t=20)
    )
    st.plotly_chart(fig_area, use_container_width=True)


# ─────────────────────────────────────────────
# MODE: ML PREDICTION
# ─────────────────────────────────────────────
elif "ML Prediction" in mode:
    st.markdown("### 🤖 Machine Learning — Prediction & Forecasting")

    tab1, tab2 = st.tabs(["🌳 Route Recommender (RF Classifier)", "📉 GRV Forecaster (GBR)"])

    with tab1:
        st.markdown("**RandomForestClassifier** trained on component scores to predict the recommended route.")
        with st.spinner("Training Random Forest..."):
            clf, report, importances = train_rf_classifier(grv_df)

        if clf is None:
            st.warning("Insufficient labelled data (need ≥20 rows with both classes). More quarters needed.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Model Performance")
                m = report.get("1", report.get("1.0", {}))
                colA, colB, colC = st.columns(3)
                colA.metric("Precision", f"{m.get('precision',0):.2f}")
                colB.metric("Recall",    f"{m.get('recall',0):.2f}")
                colC.metric("F1-Score",  f"{m.get('f1-score',0):.2f}")
                acc = report.get("accuracy", 0)
                st.metric("Accuracy", f"{acc:.2%}")

            with c2:
                st.markdown("#### Feature Importances")
                fig_imp = go.Figure(go.Bar(
                    x=importances.values,
                    y=importances.index,
                    orientation="h",
                    marker_color="#64ffda"
                ))
                fig_imp.update_layout(
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="#ccd6f6", height=280, margin=dict(t=10, b=10),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            # Live prediction
            st.markdown("#### 🧪 Live Prediction — Enter Custom Scores")
            pc1, pc2, pc3, pc4, pc5 = st.columns(5)
            in_c = pc1.number_input("Conflict", 0.0, 1.0, 0.3, 0.01)
            in_p = pc2.number_input("Piracy",   0.0, 1.0, 0.2, 0.01)
            in_s = pc3.number_input("Sanction", 0.0, 1.0, 0.4, 0.01)
            in_e = pc4.number_input("Embargo",  0.0, 1.0, 0.1, 0.01)
            in_r = pc5.selectbox("Route", [1, 2, 3, 4], format_func=lambda x: ROUTE_NAMES[x])
            pred_prob = clf.predict_proba([[in_c, in_p, in_s, in_e, in_r]])[0]
            pred_cls  = clf.predict([[in_c, in_p, in_s, in_e, in_r]])[0]
            grv_custom = 0.35*in_c + 0.25*in_p + 0.25*in_s + 0.15*in_e
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Predicted GRV", f"{grv_custom:.4f}")
            res_col2.metric("Recommendation Probability", f"{pred_prob[1]:.2%}")
            res_col3.metric("Model Decision", "✅ Recommended" if pred_cls == 1 else "❌ Not Recommended")

    with tab2:
        st.markdown("**GradientBoostingRegressor** per route, using lag features to forecast next quarter GRV.")
        with st.spinner("Training GBR forecasters (one per route)..."):
            gbr_results = train_gbr_forecaster(grv_df)

        if not gbr_results:
            st.warning("Not enough time-series data per route for forecasting. Need ≥15 quarters per route.")
        else:
            st.markdown("#### Next-Quarter GRV Forecast")
            fc_cols = st.columns(len(gbr_results))
            for i, (rid, res) in enumerate(gbr_results.items()):
                with fc_cols[i]:
                    risk = "🟢" if res["forecast"] < 0.2 else ("🟡" if res["forecast"] < 0.5 else "🔴")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:22px">{ROUTE_INFO[rid]['emoji']}</div>
                        <div style="color:#ccd6f6; font-size:13px; margin:4px 0">{ROUTE_NAMES[rid]}</div>
                        <div style="color:#64ffda; font-size:24px; font-weight:800">
                            {res['forecast']:.4f}
                        </div>
                        <div style="color:#8892b0; font-size:11px">Forecasted GRV {risk}</div>
                        <div style="color:#8892b0; font-size:11px; margin-top:4px">
                            RMSE: {res['rmse']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Forecast vs actuals chart
            st.markdown("#### 📊 Forecast vs Actual (last 20 quarters)")
            fc_route = st.selectbox("Select route for forecast chart",
                                     options=list(gbr_results.keys()),
                                     format_func=lambda x: ROUTE_NAMES[x], key="fc_rt")
            rdf_fc = grv_df[grv_df["route_id"] == fc_route].sort_values("snapshot_quarter").tail(20)
            rdf_fc = rdf_fc.copy()
            rdf_fc["lag1"]  = rdf_fc["GRV"].shift(1)
            rdf_fc["lag2"]  = rdf_fc["GRV"].shift(2)
            rdf_fc["roll4"] = rdf_fc["GRV"].rolling(4).mean()
            rdf_fc = rdf_fc.dropna()
            model  = gbr_results[fc_route]["model"]
            feat_cols = ["lag1", "lag2", "roll4", "conflict_score", "piracy_score",
                         "sanction_score", "embargo_score"]
            rdf_fc["predicted"] = model.predict(rdf_fc[feat_cols])
            rdf_fc["Q_label"] = rdf_fc["snapshot_quarter"].astype(str).apply(
                lambda x: f"Q{x[-1]} {x[:4]}"
            )
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=rdf_fc["Q_label"], y=rdf_fc["GRV"],
                                         mode="lines+markers", name="Actual GRV",
                                         line=dict(color="#4fc3f7", width=2)))
            fig_fc.add_trace(go.Scatter(x=rdf_fc["Q_label"], y=rdf_fc["predicted"],
                                         mode="lines+markers", name="Predicted GRV",
                                         line=dict(color="#ffb347", width=2, dash="dot")))
            fig_fc.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font_color="#ccd6f6", height=360,
                yaxis=dict(gridcolor="#2d3250", range=[0, 1]),
                xaxis=dict(gridcolor="#2d3250"),
                legend=dict(bgcolor="#1a1d27"),
                margin=dict(t=20)
            )
            st.plotly_chart(fig_fc, use_container_width=True)


# ─────────────────────────────────────────────
# MODE: EXPORT SHEET 5
# ─────────────────────────────────────────────
elif "Export" in mode:
    st.markdown("### 📥 Sheet 5 — GRV Master (Pipeline Output)")
    st.caption("This is the auto-generated output of the pipeline. Not manually authored.")

    export_cols = ["route_id", "route_name", "snapshot_quarter",
                   "conflict_score", "piracy_score", "sanction_score",
                   "embargo_score", "GRV", "recommended"]
    export_df = grv_df[export_cols].sort_values(["snapshot_quarter", "route_id"])

    st.dataframe(export_df, use_container_width=True, hide_index=True)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download sheet_5_grv_master.csv",
        data=csv_bytes,
        file_name="sheet_5_grv_master.csv",
        mime="text/csv"
    )

    # Quick stats
    st.markdown("#### Pipeline Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Rows",      len(export_df))
    s2.metric("Quarters Covered", export_df["snapshot_quarter"].nunique())
    s3.metric("Max GRV",          f"{export_df['GRV'].max():.4f}")
    s4.metric("Min GRV",          f"{export_df['GRV'].min():.4f}")

    # Recommendations distribution
    rec_dist = export_df.groupby("route_name")["recommended"].sum().reset_index()
    rec_dist.columns = ["Route", "Times Recommended"]
    fig_pie = px.pie(rec_dist, names="Route", values="Times Recommended",
                      color="Route", color_discrete_map=COLOR_MAP)
    fig_pie.update_layout(
        paper_bgcolor="#0f1117", font_color="#ccd6f6",
        legend=dict(bgcolor="#0f1117"), height=360
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("🛡️ Navi-Shield | Predictive Supply Chain Resilience | GRV Formula: 0.35×C + 0.25×P + 0.25×S + 0.15×E")
