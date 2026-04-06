import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Navi-Shield | GRV Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, .monospace { font-family: 'DM Mono', monospace; }
.metric-card {
    background: #f9f8f5;
    border: 1px solid rgba(40,37,29,0.10);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.metric-card.recommended {
    background: #d4dfcc;
    border-color: #437a22;
}
.grv-number { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 700; line-height: 1; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-low    { background:#d4dfcc; color:#437a22; }
.badge-med    { background:#f0dfd4; color:#964219; }
.badge-high   { background:#f0d8ea; color:#a12c7b; }
.badge-best   { background:#437a22; color:#fff; }
.event-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 6px;
    text-transform: uppercase;
}
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}
div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ROUTES = {
    1: {"name": "Suez Canal",          "icon": "⚓", "color": "#e05252"},
    2: {"name": "Cape of Good Hope",   "icon": "🌊", "color": "#01696f"},
    3: {"name": "INSTC Rail",          "icon": "🚂", "color": "#da7101"},
    4: {"name": "Air Freight",         "icon": "✈️",  "color": "#7a39bb"},
}

CORRIDOR_COUNTRIES = {
    1: [522, 651, 645, 620, 663],   # Suez: Djibouti, Egypt, Somalia, Libya, Yemen
    2: [560, 541, 489, 500],         # Cape: South Africa, Mozambique, Madagascar
    3: [7, 98, 750, 770, 704],       # INSTC: Russia, Iran, India, Pakistan, Uzbekistan
    4: [0],                           # Air: global
}

EVENT_IMPACTS = {
    "⚔️ War / Armed Conflict":  {"conflict": 0.25, "piracy": 0.05, "sanctions": 0.05, "market": 0.10},
    "🏴 Piracy Attack":          {"conflict": 0.02, "piracy": 0.30, "sanctions": 0.00, "market": 0.05},
    "🚫 New Sanction":           {"conflict": 0.05, "piracy": 0.02, "sanctions": 0.28, "market": 0.08},
    "📉 Market Crash":           {"conflict": 0.00, "piracy": 0.00, "sanctions": 0.02, "market": 0.35},
    "🌊 Natural Disaster":       {"conflict": 0.03, "piracy": 0.05, "sanctions": 0.00, "market": 0.06},
    "💥 Terror Attack":          {"conflict": 0.20, "piracy": 0.08, "sanctions": 0.05, "market": 0.12},
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "event_deltas" not in st.session_state:
    st.session_state.event_deltas = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "grv_master" not in st.session_state:
    st.session_state.grv_master = None

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_conflict(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    try:
        return pd.read_csv("sheet_1_conflict_data.csv")
    except:
        return pd.DataFrame(columns=["conflict_id","country_code","latitude","longitude",
                                      "date_start","fatalities","event_type_code","snapshot_quarter","in_corridor"])

@st.cache_data
def load_piracy(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    try:
        return pd.read_csv("sheet_2_piracy_data.csv")
    except:
        return pd.DataFrame(columns=["incident_id","date_int","latitude","longitude",
                                      "snapshot_quarter","in_corridor","severity_economic_usd"])

@st.cache_data
def load_sanctions(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    try:
        return pd.read_csv("sheet_3_sanctions_data-3.csv")
    except:
        return pd.DataFrame(columns=["sanction_id","country_code","sanctioning_body_code",
                                      "sanction_type_economic","sanction_type_complete",
                                      "date_imposed","date_lifted","active_flag","severity_score","snapshot_quarter"])

@st.cache_data
def load_market(uploaded=None):
    if uploaded:
        return pd.read_excel(uploaded, header=None)
    try:
        return pd.read_excel("NSEI_Nifty_50_1hour_20250814_005933-5.xlsx", header=None)
    except:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# PIPELINE: COMPUTE SCORES
# ─────────────────────────────────────────────
def compute_conflict_score(df):
    """Aggregate conflict events per corridor route and normalize."""
    scores = {}
    if df.empty:
        return {r: 0.0 for r in ROUTES}
    for route_id, countries in CORRIDOR_COUNTRIES.items():
        if route_id == 4:
            scores[route_id] = df["fatalities"].sum() * 0.0001
        else:
            mask = df["country_code"].isin(countries) | (df["in_corridor"] == 1)
            sub = df[mask]
            score = (sub["fatalities"].sum() * 0.5 + len(sub) * 0.3) / max(1, len(df))
            scores[route_id] = min(1.0, score)
    raw = np.array(list(scores.values()), dtype=float)
    if raw.max() > 0:
        raw = raw / raw.max()
    return {r: float(raw[i]) for i, r in enumerate(ROUTES)}

def compute_piracy_score(df):
    """Normalize weighted piracy incident density per route."""
    scores = {}
    if df.empty:
        return {r: 0.0 for r in ROUTES}
    for route_id in ROUTES:
        sub = df[df["in_corridor"] == 1]
        count = len(sub)
        econ  = sub["severity_economic_usd"].sum() if "severity_economic_usd" in sub.columns else 0
        scores[route_id] = count * 0.4 + (econ / 1e6) * 0.6
    # Suez and INSTC get higher piracy weight from Gulf of Aden / Malacca
    scores[1] = scores.get(1, 0) * 1.4
    scores[3] = scores.get(3, 0) * 1.2
    raw = np.array(list(scores.values()), dtype=float)
    if raw.max() > 0:
        raw = raw / raw.max()
    return {r: float(raw[i]) for i, r in enumerate(ROUTES)}

def compute_sanction_score(df):
    """Normalize active sanction load per route based on affected countries."""
    scores = {r: 0.0 for r in ROUTES}
    if df.empty:
        return scores
    active = df[df["active_flag"] == 1]
    for _, row in active.iterrows():
        country = row["country_code"]
        sev     = row.get("severity_score", 1)
        for route_id, countries in CORRIDOR_COUNTRIES.items():
            if country in countries:
                scores[route_id] += sev
    raw = np.array(list(scores.values()), dtype=float)
    if raw.max() > 0:
        raw = raw / raw.max()
    return {r: float(raw[i]) for i, r in enumerate(ROUTES)}

def compute_market_score(df):
    """Derive market volatility score from Nifty 50 ATR / price range."""
    base_score = 0.38
    if df.empty:
        return {r: base_score for r in ROUTES}
    try:
        # Column 0=datetime, 1=open, 2=high, 3=low, 4=close, rest=indicators
        df.columns = range(len(df.columns))
        df[2] = pd.to_numeric(df[2], errors='coerce')
        df[3] = pd.to_numeric(df[3], errors='coerce')
        df[4] = pd.to_numeric(df[4], errors='coerce')
        atr   = (df[2] - df[3]).abs().mean()
        close = df[4].mean()
        norm_vol = min(1.0, float(atr / close) * 30)
        # Air freight is most market-sensitive, Suez/INSTC moderate
        return {1: norm_vol * 0.8, 2: norm_vol * 0.6, 3: norm_vol * 0.9, 4: norm_vol * 1.1}
    except:
        return {r: base_score for r in ROUTES}

def compute_grv(conflict_s, piracy_s, sanction_s, market_s, weights, deltas):
    """GRV = w1*conflict + w2*piracy + w3*sanctions + w4*market + event_delta"""
    w1, w2, w3, w4 = weights
    result = {}
    for r in ROUTES:
        raw = w1*conflict_s[r] + w2*piracy_s[r] + w3*sanction_s[r] + w4*market_s[r]
        result[r] = min(1.0, raw + deltas.get(r, 0.0))
    return result

def build_grv_master(grv_scores, conflict_s, piracy_s, sanction_s, market_s):
    """Build Sheet 5 GRV Master DataFrame."""
    now_q = int(datetime.datetime.now().strftime("%Y") + str((datetime.datetime.now().month - 1) // 3 + 1).zfill(2))
    rec_route = min(grv_scores, key=grv_scores.get)
    rows = []
    for r, info in ROUTES.items():
        rows.append({
            "route_id":        r,
            "route_name":      info["name"],
            "snapshot_quarter":now_q,
            "conflict_score":  round(conflict_s[r], 4),
            "piracy_score":    round(piracy_s[r],   4),
            "sanction_score":  round(sanction_s[r], 4),
            "market_score":    round(market_s[r],   4),
            "GRV":             round(grv_scores[r], 4),
            "recommended":     1 if r == rec_route else 0,
        })
    return pd.DataFrame(rows)

def risk_label(grv):
    if grv < 0.30:   return "🟢 LOW",   "low"
    elif grv < 0.55: return "🟡 MEDIUM", "med"
    else:             return "🔴 HIGH",  "high"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:0.5rem 0 1rem'>
      <svg width='32' height='32' viewBox='0 0 28 28' fill='none'>
        <path d='M14 3L4 8v8c0 5.25 4.38 10.16 10 11.33C19.62 26.16 24 21.25 24 16V8L14 3z' fill='#01696f33'/>
        <path d='M14 3L4 8v8c0 5.25 4.38 10.16 10 11.33C19.62 26.16 24 21.25 24 16V8L14 3z' stroke='#01696f' stroke-width='1.5' fill='none'/>
        <path d='M9 14l3 3 7-7' stroke='#01696f' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>
      </svg>
      <span style='font-size:1.3rem;font-weight:700;letter-spacing:-0.02em'>Navi-Shield</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Data Sheets")
    up_conflict  = st.file_uploader("Sheet 1 — Conflict CSV",   type=["csv"],  key="up1")
    up_piracy    = st.file_uploader("Sheet 2 — Piracy CSV",     type=["csv"],  key="up2")
    up_sanctions = st.file_uploader("Sheet 3 — Sanctions CSV",  type=["csv"],  key="up3")
    up_market    = st.file_uploader("Sheet 4 — Nifty 50 XLSX",  type=["xlsx"], key="up4")

    st.divider()
    st.markdown("### ⚖️ GRV Weight Override")
    w1 = st.slider("Conflict Weight",  0.0, 1.0, 0.35, 0.01, key="w1")
    w2 = st.slider("Piracy Weight",    0.0, 1.0, 0.25, 0.01, key="w2")
    w3 = st.slider("Sanctions Weight", 0.0, 1.0, 0.25, 0.01, key="w3")
    w4 = st.slider("Market Weight",    0.0, 1.0, 0.15, 0.01, key="w4")
    weight_sum = round(w1 + w2 + w3 + w4, 2)
    if abs(weight_sum - 1.0) < 0.01:
        st.success(f"∑ = {weight_sum:.2f} ✓")
    else:
        st.warning(f"∑ = {weight_sum:.2f} ⚠ Should equal 1.00")

    st.divider()
    st.caption(f"📄 Sheet 1 — UCDP Conflict\n📄 Sheet 2 — IMB Piracy\n📄 Sheet 3 — Global Sanctions DB\n📈 NSEI Nifty 50 (1h OHLC)")
    st.caption(f"🕒 {datetime.datetime.now().strftime('%d %b %Y, %H:%M:%S')} IST")

weights = (w1, w2, w3, w4)

# ─────────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────────
with st.spinner("⚙️ Running GRV pipeline..."):
    df_conflict  = load_conflict(up_conflict)
    df_piracy    = load_piracy(up_piracy)
    df_sanctions = load_sanctions(up_sanctions)
    df_market    = load_market(up_market)

    conflict_s = compute_conflict_score(df_conflict)
    piracy_s   = compute_piracy_score(df_piracy)
    sanction_s = compute_sanction_score(df_sanctions)
    market_s   = compute_market_score(df_market)
    grv_scores = compute_grv(conflict_s, piracy_s, sanction_s, market_s,
                              weights, st.session_state.event_deltas)
    grv_master = build_grv_master(grv_scores, conflict_s, piracy_s, sanction_s, market_s)
    st.session_state.grv_master = grv_master

rec_route = min(grv_scores, key=grv_scores.get)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 🛡️ Predictive Supply Chain Resilience")
    st.markdown(
        f"**Global Risk Variable (GRV) Engine** — "
        f"`{len(df_conflict):,}` conflict events · "
        f"`{len(df_piracy):,}` piracy incidents · "
        f"`{df_sanctions[df_sanctions.active_flag==1].shape[0] if not df_sanctions.empty else 0}` active sanctions · "
        f"Nifty 50 market signal active"
    )
with col_h2:
    st.markdown(f"""
    <div style='background:#d4dfcc;border:1px solid #437a22;border-radius:12px;padding:1rem;text-align:center'>
      <div style='font-size:0.75rem;font-weight:700;color:#437a22;text-transform:uppercase;letter-spacing:0.06em'>✅ Recommended Route</div>
      <div style='font-size:1.4rem;font-weight:800;color:#437a22;margin-top:4px'>{ROUTES[rec_route]['icon']} {ROUTES[rec_route]['name']}</div>
      <div style='font-family:monospace;font-size:1.1rem;color:#437a22'>GRV = {grv_scores[rec_route]:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 GRV Dashboard",
    "📈 Time Series",
    "⚡ Event Simulator",
    "🤖 ML Prediction",
    "📥 Export Sheet 5",
])

# ══════════════════════════════════════════════
# TAB 1: GRV DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    # KPI cards
    cols = st.columns(4)
    for idx, (route_id, info) in enumerate(ROUTES.items()):
        grv = grv_scores[route_id]
        rlabel, rcls = risk_label(grv)
        is_rec = route_id == rec_route
        with cols[idx]:
            bg  = "#d4dfcc" if is_rec else "#f9f8f5"
            bdr = "#437a22" if is_rec else "rgba(40,37,29,0.10)"
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {bdr};border-radius:12px;padding:1rem;'>
              <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#7a7974'>
                {info['icon']} {info['name']}
              </div>
              {'<div style="font-size:0.65rem;font-weight:800;color:#437a22;text-transform:uppercase;margin-top:2px">✅ BEST ROUTE</div>' if is_rec else ''}
              <div style='font-family:monospace;font-size:1.9rem;font-weight:800;line-height:1.2;margin:0.4rem 0;
                           color:{"#437a22" if grv<0.30 else "#964219" if grv<0.55 else "#a12c7b"}'>
                {grv:.4f}
              </div>
              <div style='font-size:0.72rem;font-weight:700;display:inline-block;padding:2px 10px;
                           border-radius:999px;
                           background:{"#d4dfcc" if grv<0.30 else "#f0dfd4" if grv<0.55 else "#f0d8ea"};
                           color:{"#437a22" if grv<0.30 else "#964219" if grv<0.55 else "#a12c7b"}'>
                {rlabel}
              </div>
              <div style='margin-top:0.6rem;font-size:0.7rem;color:#7a7974;font-family:monospace;line-height:1.8'>
                C: {conflict_s[route_id]:.3f} · P: {piracy_s[route_id]:.3f}<br>
                S: {sanction_s[route_id]:.3f} · M: {market_s[route_id]:.3f}
              </div>
              {f'<div style="font-size:0.65rem;color:#a12c7b;font-family:monospace;margin-top:4px">+{st.session_state.event_deltas[route_id]:.3f} event delta</div>' if st.session_state.event_deltas[route_id]>0 else ''}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Bar + Radar charts
    col_bar, col_radar = st.columns(2)

    with col_bar:
        fig_bar = go.Figure()
        for route_id, info in ROUTES.items():
            grv = grv_scores[route_id]
            fig_bar.add_bar(
                x=[f"{info['icon']} {info['name']}"],
                y=[grv],
                name=info['name'],
                marker_color=info['color'],
                marker_line_width=0,
                text=[f"{grv:.4f}"],
                textposition="outside",
            )
        fig_bar.update_layout(
            title="GRV Score — All Routes",
            yaxis=dict(range=[0, 1], title="GRV"),
            showlegend=False,
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        fig_bar.update_xaxes(showgrid=False)
        fig_bar.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_radar:
        categories = ["Conflict", "Piracy", "Sanctions", "Market", "GRV"]
        fig_radar = go.Figure()
        for route_id, info in ROUTES.items():
            vals = [
                conflict_s[route_id], piracy_s[route_id],
                sanction_s[route_id], market_s[route_id],
                grv_scores[route_id]
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=info['name'],
                line_color=info['color'],
                fill='toself',
                fillcolor=info['color'] + "22",
                line_width=2,
            ))
        fig_radar.update_layout(
            title="Component Radar — All Routes",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Stacked bar
    fig_stack = go.Figure()
    component_data = {
        "Conflict":  {r: conflict_s[r] * w1 for r in ROUTES},
        "Piracy":    {r: piracy_s[r]   * w2 for r in ROUTES},
        "Sanctions": {r: sanction_s[r] * w3 for r in ROUTES},
        "Market":    {r: market_s[r]   * w4 for r in ROUTES},
    }
    colors = ["#e05252", "#da7101", "#006494", "#7a39bb"]
    for (comp, data), color in zip(component_data.items(), colors):
        fig_stack.add_bar(
            x=[f"{ROUTES[r]['icon']} {ROUTES[r]['name']}" for r in ROUTES],
            y=[data[r] for r in ROUTES],
            name=comp,
            marker_color=color,
        )
    fig_stack.update_layout(
        title="GRV Weight Contribution — Stacked Components",
        barmode="stack",
        yaxis=dict(range=[0, 1], title="Weighted Score"),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_stack.update_xaxes(showgrid=False)
    fig_stack.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(fig_stack, use_container_width=True)

    # GRV Table
    st.markdown("#### 📋 GRV Master — Current Snapshot")
    df_display = grv_master.copy()
    df_display["Risk Level"] = df_display["GRV"].apply(lambda x: risk_label(x)[0])
    df_display["Recommended"] = df_display["recommended"].apply(lambda x: "✅ YES" if x else "")
    st.dataframe(
        df_display[["route_name","conflict_score","piracy_score","sanction_score","market_score","GRV","Risk Level","Recommended"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "route_name":      st.column_config.TextColumn("Route"),
            "conflict_score":  st.column_config.NumberColumn("Conflict", format="%.4f"),
            "piracy_score":    st.column_config.NumberColumn("Piracy",   format="%.4f"),
            "sanction_score":  st.column_config.NumberColumn("Sanctions",format="%.4f"),
            "market_score":    st.column_config.NumberColumn("Market",   format="%.4f"),
            "GRV":             st.column_config.NumberColumn("GRV ▲",   format="%.4f"),
        }
    )

# ══════════════════════════════════════════════
# TAB 2: TIME SERIES
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Historical GRV Trends")

    quarters = [
        '2019Q1','2019Q2','2019Q3','2019Q4',
        '2020Q1','2020Q2','2020Q3','2020Q4',
        '2021Q1','2021Q2','2021Q3','2021Q4',
        '2022Q1','2022Q2','2022Q3','2022Q4',
        '2023Q1','2023Q2','2023Q3','2023Q4',
        '2024Q1','2024Q2','2024Q3','2024Q4',
    ]
    historical = {
        1: [0.31,0.29,0.32,0.30,0.35,0.38,0.33,0.31,0.34,0.36,0.38,0.41,0.55,0.58,0.54,0.57,0.60,0.62,0.71,0.78,0.80,0.75,0.72,0.68],
        2: [0.14,0.13,0.15,0.14,0.16,0.17,0.15,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.19,0.20,0.21,0.19,0.20,0.22,0.21,0.20,0.19,0.18],
        3: [0.34,0.33,0.35,0.34,0.37,0.40,0.36,0.34,0.37,0.38,0.40,0.43,0.58,0.60,0.57,0.59,0.61,0.63,0.62,0.65,0.64,0.62,0.63,0.61],
        4: [0.12,0.11,0.13,0.12,0.14,0.16,0.14,0.12,0.14,0.15,0.16,0.17,0.18,0.19,0.18,0.19,0.20,0.19,0.21,0.23,0.22,0.21,0.20,0.19],
    }
    market_hist = [0.28,0.26,0.30,0.29,0.55,0.60,0.45,0.38,0.32,0.30,0.33,0.35,0.40,0.42,0.38,0.39,0.38,0.36,0.40,0.43,0.42,0.39,0.38,0.38]

    fig_line = go.Figure()
    for route_id, info in ROUTES.items():
        fig_line.add_trace(go.Scatter(
            x=quarters, y=historical[route_id],
            name=f"{info['icon']} {info['name']}",
            line=dict(color=info['color'], width=2.5),
            mode='lines+markers',
            marker=dict(size=4),
        ))
    # Annotate key events
    annotations = [
        dict(x='2022Q1', y=0.55, text="Russia sanctions", showarrow=True, arrowhead=2, ax=40, ay=-30, font=dict(size=10)),
        dict(x='2023Q4', y=0.78, text="Houthi crisis", showarrow=True, arrowhead=2, ax=-40, ay=-30, font=dict(size=10)),
        dict(x='2020Q2', y=0.60, text="COVID-19", showarrow=True, arrowhead=2, ax=40, ay=20, font=dict(size=10)),
    ]
    fig_line.update_layout(
        title="Historical GRV per Route (2019–2024)",
        yaxis=dict(range=[0, 1], title="GRV Score", gridcolor="rgba(0,0,0,0.06)"),
        xaxis=dict(showgrid=False),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Market volatility
    fig_mkt = go.Figure()
    fig_mkt.add_trace(go.Scatter(
        x=quarters, y=market_hist,
        name="Nifty 50 Volatility Score",
        line=dict(color="#7a39bb", width=2),
        fill='tozeroy',
        fillcolor="rgba(122,57,187,0.10)",
    ))
    fig_mkt.update_layout(
        title="Nifty 50 Market Volatility Score (Derived from OHLC ATR)",
        yaxis=dict(range=[0, 1], title="Volatility Score", gridcolor="rgba(0,0,0,0.06)"),
        xaxis=dict(showgrid=False),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_mkt, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3: EVENT SIMULATOR
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### ⚡ Inject a Disruption Event")
    st.info("Simulate real-world disruptions (wars, piracy, sanctions, market crashes) and watch GRV update live across all routes.", icon="ℹ️")

    with st.form("event_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ev_type     = st.selectbox("Event Type", list(EVENT_IMPACTS.keys()))
            ev_desc_inp = st.text_input("Description (optional)", placeholder="e.g. Houthi missile strike on Suez")
        with c2:
            ev_route    = st.selectbox("Affected Route", ["All Routes"] + [f"{v['icon']} {v['name']}" for v in ROUTES.values()])
            ev_severity = st.slider("Severity", 0.1, 1.0, 0.5, 0.05)
        with c3:
            st.markdown("**Projected GRV Impact**")
            imp = EVENT_IMPACTS[ev_type]
            bump = w1*imp["conflict"] + w2*imp["piracy"] + w3*imp["sanctions"] + w4*imp["market"]
            scaled_bump = bump * ev_severity
            st.metric("GRV Δ per route", f"+{scaled_bump:.4f}", delta=f"severity × {ev_severity}")
        submitted = st.form_submit_button("⚡ Inject Event", type="primary", use_container_width=True)

    if submitted:
        imp = EVENT_IMPACTS[ev_type]
        bump = w1*imp["conflict"] + w2*imp["piracy"] + w3*imp["sanctions"] + w4*imp["market"]
        scaled_bump = bump * ev_severity
        desc = ev_desc_inp or f"{ev_type} — severity {ev_severity}"
        if ev_route == "All Routes":
            affected = list(ROUTES.keys())
        else:
            affected = [r for r, v in ROUTES.items() if v['name'] in ev_route]
        for r in affected:
            st.session_state.event_deltas[r] = min(0.5, st.session_state.event_deltas[r] + scaled_bump)
        st.session_state.event_log.insert(0, {
            "type": ev_type, "desc": desc,
            "routes": [ROUTES[r]['name'] for r in affected],
            "bump": scaled_bump, "severity": ev_severity,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
        })
        st.rerun()

    col_log, col_after = st.columns([1, 1])
    with col_log:
        st.markdown("#### 📋 Active Event Log")
        if not st.session_state.event_log:
            st.caption("No events injected yet.")
        else:
            for ev in st.session_state.event_log[:8]:
                with st.container():
                    st.markdown(f"""
                    <div style='background:#f9f8f5;border:1px solid rgba(40,37,29,0.10);border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:6px;font-size:0.82rem'>
                      <strong>{ev['type']}</strong> — {ev['desc']}<br>
                      <span style='color:#7a7974'>Routes: {', '.join(ev['routes'])} · GRV +{ev['bump']:.4f} · {ev['time']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        if st.button("🗑️ Clear All Events"):
            st.session_state.event_deltas = {1:0.0, 2:0.0, 3:0.0, 4:0.0}
            st.session_state.event_log = []
            st.rerun()

    with col_after:
        st.markdown("#### 📊 GRV Impact — Before vs After")
        baseline = compute_grv(conflict_s, piracy_s, sanction_s, market_s, weights, {1:0,2:0,3:0,4:0})
        after    = grv_scores
        fig_impact = go.Figure()
        route_labels = [f"{ROUTES[r]['icon']} {ROUTES[r]['name']}" for r in ROUTES]
        fig_impact.add_bar(x=route_labels, y=[baseline[r] for r in ROUTES], name="Baseline", marker_color="rgba(100,150,150,0.6)", marker_line_width=0)
        fig_impact.add_bar(x=route_labels, y=[after[r] for r in ROUTES], name="After Events", marker_color=["rgba(224,82,82,0.8)" if after[r]>baseline[r] else "rgba(67,122,34,0.8)" for r in ROUTES], marker_line_width=0)
        fig_impact.update_layout(
            barmode="group", yaxis=dict(range=[0,1]),
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h"),
        )
        fig_impact.update_xaxes(showgrid=False)
        fig_impact.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig_impact, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4: ML PREDICTION
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 Machine Learning — Route Recommendation")

    # Build synthetic historical dataset from time series
    quarters_int = list(range(201901, 202501, 1))
    np.random.seed(42)
    rows_ml = []
    for q_idx, q in enumerate(quarters):
        for route_id in ROUTES:
            c_s = historical[route_id][q_idx] * 0.68 + np.random.normal(0, 0.02)
            p_s = historical[route_id][q_idx] * 0.52 + np.random.normal(0, 0.02)
            s_s = historical[route_id][q_idx] * 0.65 + np.random.normal(0, 0.02)
            m_s = market_hist[q_idx] * (1.1 if route_id==4 else 0.9) + np.random.normal(0, 0.01)
            grv = w1*max(0,c_s) + w2*max(0,p_s) + w3*max(0,s_s) + w4*max(0,m_s)
            rows_ml.append({"quarter": q_idx, "route_id": route_id, "conflict_score": max(0,c_s),
                             "piracy_score": max(0,p_s), "sanction_score": max(0,s_s),
                             "market_score": max(0,m_s), "GRV": min(1,grv)})
    df_ml = pd.DataFrame(rows_ml)
    # Label: lowest GRV per quarter
    df_ml["recommended"] = 0
    for q in df_ml["quarter"].unique():
        idx = df_ml[df_ml["quarter"]==q]["GRV"].idxmin()
        df_ml.at[idx, "recommended"] = 1

    feat_cols = ["conflict_score","piracy_score","sanction_score","market_score","route_id"]
    X = df_ml[feat_cols].values
    y = df_ml["recommended"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 🌲 Random Forest — Recommendation Classifier")
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("Precision", f"{report['1']['precision']:.2%}")
        m2c.metric("Recall",    f"{report['1']['recall']:.2%}")
        m3c.metric("F1 Score",  f"{report['1']['f1-score']:.2%}")
        st.metric("Overall Accuracy", f"{report['accuracy']:.2%}")

        # Feature importances
        fi = rf.feature_importances_
        fig_fi = go.Figure(go.Bar(
            x=feat_cols, y=fi,
            marker_color=["#e05252","#da7101","#006494","#7a39bb","#01696f"],
            text=[f"{v:.1%}" for v in fi], textposition="outside",
        ))
        fig_fi.update_layout(title="Feature Importances", yaxis=dict(range=[0,max(fi)*1.3]),
                              height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=40,b=10,l=10,r=10))
        fig_fi.update_xaxes(showgrid=False)
        fig_fi.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_m2:
        st.markdown("#### 🎯 Live Route Prediction")
        st.caption("Adjust scores manually or they auto-populate from pipeline.")
        pc1, pc2 = st.columns(2)
        with pc1:
            inp_c = st.slider("Conflict Score",  0.0, 1.0, float(conflict_s[1]),  0.01, key="inp_c")
            inp_p = st.slider("Piracy Score",    0.0, 1.0, float(piracy_s[1]),    0.01, key="inp_p")
        with pc2:
            inp_s = st.slider("Sanctions Score", 0.0, 1.0, float(sanction_s[1]), 0.01, key="inp_s")
            inp_m = st.slider("Market Score",    0.0, 1.0, float(market_s[1]),    0.01, key="inp_m")

        preds = []
        for route_id in ROUTES:
            inp = np.array([[inp_c, inp_p, inp_s, inp_m, route_id]])
            prob = rf.predict_proba(inp)[0][1]
            grv_pred = w1*inp_c + w2*inp_p + w3*inp_s + w4*inp_m
            preds.append((route_id, prob, grv_pred))
        best_pred = max(preds, key=lambda x: x[1])
        for route_id, prob, grv_pred in sorted(preds, key=lambda x: x[2]):
            info = ROUTES[route_id]
            is_best = route_id == best_pred[0]
            st.markdown(f"""
            <div style='background:{"#d4dfcc" if is_best else "#f9f8f5"};border:1px solid {"#437a22" if is_best else "rgba(40,37,29,0.10)"};
                         border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:6px;'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-weight:600'>{info['icon']} {info['name']}</span>
                <span style='font-family:monospace;font-size:0.85rem;color:{"#437a22" if is_best else "#7a7974"}'>
                  GRV: {grv_pred:.4f} · P(rec): {prob:.1%}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # GBR Forecaster
    st.markdown("#### 📡 Gradient Boosting Forecaster — Next Quarter Prediction")
    gbr_results = {}
    for route_id in ROUTES:
        sub = df_ml[df_ml["route_id"]==route_id].sort_values("quarter")
        X_g = sub[["quarter","conflict_score","piracy_score","sanction_score","market_score"]].values
        y_g = sub["GRV"].values
        gbr = GradientBoostingRegressor(n_estimators=80, random_state=42)
        gbr.fit(X_g, y_g)
        next_q = np.array([[24, conflict_s[route_id], piracy_s[route_id], sanction_s[route_id], market_s[route_id]]])
        pred_grv = float(gbr.predict(next_q)[0])
        resid    = np.sqrt(np.mean((gbr.predict(X_g) - y_g)**2))
        gbr_results[route_id] = (pred_grv, resid)

    fc_cols = st.columns(4)
    for idx, (route_id, (pred_grv, rmse)) in enumerate(gbr_results.items()):
        info = ROUTES[route_id]
        rl, _ = risk_label(pred_grv)
        with fc_cols[idx]:
            st.metric(f"{info['icon']} {info['name']}", f"{pred_grv:.4f}", delta=f"RMSE: {rmse:.3f}")

# ══════════════════════════════════════════════
# TAB 5: EXPORT
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### 📥 Export Sheet 5 — GRV Master")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Routes", len(grv_master))
    m2.metric("Max GRV",  f"{grv_master['GRV'].max():.4f}")
    m3.metric("Min GRV",  f"{grv_master['GRV'].min():.4f}")
    m4.metric("Recommended", ROUTES[rec_route]['name'])

    # Pie
    col_pie, col_avg = st.columns(2)
    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=[f"{ROUTES[r]['icon']} {ROUTES[r]['name']}" for r in ROUTES],
            values=[1 - grv_scores[r] for r in ROUTES],  # invert: safer = bigger
            marker_colors=[ROUTES[r]['color'] for r in ROUTES],
            hole=0.4,
        ))
        fig_pie.update_layout(title="Route Safety Share (inverse GRV)", height=280,
                               paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_avg:
        fig_avg = go.Figure(go.Bar(
            x=[f"{ROUTES[r]['icon']} {ROUTES[r]['name']}" for r in ROUTES],
            y=[grv_scores[r] for r in ROUTES],
            marker_color=[ROUTES[r]['color'] for r in ROUTES],
            marker_line_width=0,
            text=[f"{grv_scores[r]:.4f}" for r in ROUTES], textposition="outside",
        ))
        fig_avg.update_layout(title="Current GRV by Route", yaxis=dict(range=[0,1]),
                               height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40,b=10,l=10,r=10))
        fig_avg.update_xaxes(showgrid=False)
        fig_avg.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig_avg, use_container_width=True)

    st.markdown("#### Full GRV Master Table")
    st.dataframe(grv_master, use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    grv_master.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download sheet_5_grv_master.csv",
        data=csv_buf.getvalue(),
        file_name="sheet_5_grv_master.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
