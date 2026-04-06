"""
Predictive Supply Chain Resilience
A Streamlit application for analyzing geopolitical risk on global trade routes
for electrical equipment & petroleum logistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

# Local imports
from data_processor import (
    load_conflict_data, load_piracy_data, load_sanctions_data,
    compute_route_scores, compute_grv, build_quarterly_features
)
from route_engine import ROUTES, COMMODITIES, get_all_route_keys, get_all_commodity_keys
from ml_model import run_all_models
from styles import (
    get_custom_css, render_metric_card, render_route_card,
    render_weight_bar
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictive Supply Chain Resilience",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_data():
    """Load and cache all datasets."""
    conflict = load_conflict_data()
    piracy = load_piracy_data()
    sanctions = load_sanctions_data()
    return conflict, piracy, sanctions


@st.cache_data(show_spinner=False)
def get_route_scores(_conflict, _piracy, _sanctions):
    """Compute and cache route scores."""
    return compute_route_scores(_conflict, _piracy, _sanctions, sample_conflict=60000)


@st.cache_data(show_spinner=False)
def get_ml_results(_conflict, _piracy, _sanctions):
    """Train ML models and cache results."""
    features = build_quarterly_features(_conflict, _piracy, _sanctions, sample_conflict=60000)
    return run_all_models(features), features


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0;">
        <div style="font-size:2.5rem;">🛡️</div>
        <div style="font-size:1.1rem;font-weight:800;background:linear-gradient(135deg,#58a6ff,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            Predictive Supply Chain Resilience
        </div>
        <div style="font-size:0.65rem;color:#8b949e;margin-top:4px;letter-spacing:0.1em;">
            GEOPOLITICAL RISK INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "NAVIGATION",
        ["🎯 Dashboard", "📊 Route Analysis", "🤖 ML Model Insights", "📈 Data Explorer", "📐 Methodology"],
        label_visibility="visible",
    )
    
    st.markdown("---")
    
    # Commodity Selection
    st.markdown('<div style="font-size:0.75rem;color:#8b949e;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">SELECT COMMODITY</div>', unsafe_allow_html=True)
    
    commodity_options = {k: f"{v['icon']} {v['name']}" for k, v in COMMODITIES.items()}
    selected_commodity = st.selectbox(
        "Commodity",
        options=list(commodity_options.keys()),
        format_func=lambda x: commodity_options[x],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    # Weight Source
    st.markdown('<div style="font-size:0.75rem;color:#8b949e;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">WEIGHT SOURCE</div>', unsafe_allow_html=True)
    
    weight_source = st.selectbox(
        "Weight Source",
        ["ML Ensemble (Recommended)", "Gradient Boosting", "Random Forest",
         "Ridge Regression", "Variance Analysis", "Stock Market Validation", "Custom"],
        label_visibility="collapsed",
    )
    
    if weight_source == "Custom":
        st.markdown('<div style="font-size:0.7rem;color:#8b949e;margin:8px 0 4px;">Adjust weights (must sum to 1.0):</div>', unsafe_allow_html=True)
        cw = st.slider("Conflict", 0.0, 1.0, 0.36, 0.01, key="cw")
        pw = st.slider("Piracy", 0.0, 1.0, 0.54, 0.01, key="pw")
        sw = st.slider("Sanctions", 0.0, 1.0, 0.10, 0.01, key="sw")
        total = cw + pw + sw
        if abs(total - 1.0) > 0.02:
            st.warning(f"Sum = {total:.2f} (should be 1.0)")
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;font-size:0.65rem;color:#484f58;padding:8px;">
        <div>Built for LSCM Project</div>
        <div style="margin-top:4px;">Data: 1995–2025</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading geopolitical datasets..."):
    conflict_df, piracy_df, sanctions_df = load_all_data()

with st.spinner("📊 Computing route risk scores..."):
    route_scores = get_route_scores(conflict_df, piracy_df, sanctions_df)

with st.spinner("🤖 Training ML models to predict weights..."):
    ml_results, features_df = get_ml_results(conflict_df, piracy_df, sanctions_df)


# ─────────────────────────────────────────────────────────────
# GET SELECTED WEIGHTS
# ─────────────────────────────────────────────────────────────

weight_map = {
    "ML Ensemble (Recommended)": "ensemble",
    "Gradient Boosting": "gradient_boosting",
    "Random Forest": "random_forest",
    "Ridge Regression": "ridge_regression",
    "Variance Analysis": "variance_analysis",
    "Stock Market Validation": "stock_market_validation",
}

if weight_source == "Custom":
    total = cw + pw + sw
    weights = {"conflict": cw/total, "piracy": pw/total, "sanctions": sw/total}
else:
    model_key = weight_map[weight_source]
    weights = ml_results[model_key]["weights"]

# Compute GRV
grv_df = compute_grv(route_scores, weights, selected_commodity)


# ═══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════

if page == "🎯 Dashboard":
    
    # Hero header
    st.markdown("""
    <div class="hero-header animate-in">
        <h1>Predictive Supply Chain Resilience</h1>
        <div class="subtitle">Geopolitical risk intelligence across global trade routes</div>
        <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top metrics
    commodity_name = COMMODITIES[selected_commodity]["name"]
    commodity_icon = COMMODITIES[selected_commodity]["icon"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(render_metric_card(
            commodity_icon, "Selected Commodity", commodity_name, "grv"
        ), unsafe_allow_html=True)
    
    with col2:
        avg_grv = grv_df["grv"].mean()
        st.markdown(render_metric_card(
            "📊", "Avg GRV Score", f"{avg_grv:.1f}", "grv"
        ), unsafe_allow_html=True)
    
    with col3:
        best_route = grv_df.iloc[0]["short_name"]
        st.markdown(render_metric_card(
            "✅", "Safest Route", best_route, "piracy"
        ), unsafe_allow_html=True)
    
    with col4:
        worst_route = grv_df.iloc[-1]["short_name"]
        st.markdown(render_metric_card(
            "⚠️", "Highest Risk", worst_route, "conflict"
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Weight display
    st.markdown(f'<div class="section-header">⚖️ Active Weights — {weight_source}</div>', unsafe_allow_html=True)
    
    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        st.markdown(render_metric_card("⚔️", "Conflict Weight", f"{weights['conflict']:.3f}", "conflict"), unsafe_allow_html=True)
    with wcol2:
        st.markdown(render_metric_card("🏴‍☠️", "Piracy Weight", f"{weights['piracy']:.3f}", "piracy"), unsafe_allow_html=True)
    with wcol3:
        st.markdown(render_metric_card("🚫", "Sanctions Weight", f"{weights['sanctions']:.3f}", "sanctions"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ───── MAP ─────
    st.markdown('<div class="section-header">🗺️ Global Route Risk Map</div>', unsafe_allow_html=True)
    
    fig_map = go.Figure()
    
    best_route_key = grv_df.iloc[0]["route_key"] if len(grv_df) > 0 else None
    worst_route_key = grv_df.iloc[-1]["route_key"] if len(grv_df) > 0 else None
    
    for _, row in grv_df.iterrows():
        route = ROUTES[row["route_key"]]
        waypoints = route["waypoints"]
        lats = [wp[0] for wp in waypoints]
        lons = [wp[1] for wp in waypoints]
        
        rk = row["route_key"]
        if rk == best_route_key:
            color = "#3fb950"  # Green for safest
            line_w = 4
        elif rk == worst_route_key:
            color = "#f85149"  # Red for riskiest
            line_w = 4
        else:
            color = "rgba(255,255,255,0.6)"  # White for middle routes
            line_w = 2
        
        grv_val = row["grv"]
        risk_level = str(row["risk_level"]) if str(row["risk_level"]) != "nan" else "Low"
        
        fig_map.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines+markers",
            line=dict(width=line_w, color=color),
            marker=dict(size=5, color=color, line=dict(width=1, color="white")),
            name=f"{route['icon']} {route['short_name']} (GRV: {grv_val:.1f})",
            hovertemplate=(
                f"<b>{route['name']}</b><br>"
                f"GRV: {grv_val:.1f}<br>"
                f"Risk: {risk_level}<br>"
                f"Distance: {route.get('distance_nm', 'N/A')} nm<br>"
                f"<extra></extra>"
            ),
        ))
    
    fig_map.update_geos(
        bgcolor="rgba(0,0,0,0)",
        landcolor="rgba(22, 27, 34, 0.8)",
        oceancolor="rgba(10, 14, 23, 0.9)",
        lakecolor="rgba(10, 14, 23, 0.9)",
        coastlinecolor="rgba(48, 54, 61, 0.6)",
        countrycolor="rgba(48, 54, 61, 0.4)",
        showframe=False,
        projection_type="natural earth",
        showland=True,
        showocean=True,
        showlakes=True,
        showcountries=True,
        resolution=50,
        lataxis_range=[-50, 70],
        lonaxis_range=[-130, 160],
    )
    
    fig_map.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="rgba(48,54,61,0.5)",
            borderwidth=1,
            font=dict(color="#e6edf3", size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        font=dict(color="#e6edf3"),
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # ───── GRV ROUTE TABLE ─────
    st.markdown(f'<div class="section-header">📋 GRV Route Summary — {commodity_name} (sorted by risk)</div>', unsafe_allow_html=True)
    
    table_df = grv_df[["short_name", "distance_nm", "commodity_multiplier", "grv", "risk_level", "rank"]].copy()
    table_df.columns = ["Route", "Distance (nm)", "Sensitivity", "GRV (1-10)", "Risk Level", "Rank"]
    table_df["Risk Level"] = table_df["Risk Level"].astype(str).replace("nan", "Low")
    
    st.dataframe(
        table_df.style
        .format({"Sensitivity": "{:.2f}", "GRV (1-10)": "{:.1f}"})
        .background_gradient(subset=["GRV (1-10)"], cmap="RdYlGn_r", vmin=1, vmax=10),
        use_container_width=True,
        height=min(len(table_df) * 40 + 60, 500),
    )
    
    # ───── ROUTE RANKING CARDS ─────
    st.markdown(f'<div class="section-header">🏆 Route Rankings — {commodity_name}</div>', unsafe_allow_html=True)
    
    # Render each card using st.html() to avoid Streamlit HTML sanitization issues
    for idx, (_, row) in enumerate(grv_df.iterrows()):
        route = ROUTES[row["route_key"]]
        card_html = render_route_card(
            route_name=route["name"],
            route_desc=route["description"],
            grv_score=row["grv"],
            risk_level=str(row["risk_level"]),
            rank=row["rank"],
            is_recommended=(idx == 0),
            icon=route["icon"],
            chokepoints=route["chokepoints"],
        )
        # Wrap in a full HTML doc with inline styles so it renders in st.html()'s iframe
        st.html(f"""
        <style>
            body {{ margin: 0; padding: 0; background: transparent; font-family: 'Inter', -apple-system, sans-serif; }}
            .route-card {{
                background: linear-gradient(135deg, rgba(22, 27, 34, 0.95), rgba(13, 17, 23, 0.98));
                border: 1px solid rgba(48, 54, 61, 0.5);
                border-radius: 14px;
                padding: 20px;
                margin-bottom: 4px;
                position: relative;
                color: #e6edf3;
            }}
            .route-card:hover {{ border-color: rgba(88,166,255,0.4); }}
            .recommended-route {{
                background: linear-gradient(135deg, rgba(35, 134, 54, 0.15), rgba(22, 27, 34, 0.95));
                border: 2px solid rgba(35, 134, 54, 0.5) !important;
            }}
            .recommended-badge {{
                position: absolute; top: -10px; right: 16px;
                background: linear-gradient(135deg, #238636, #2ea043);
                color: white; padding: 3px 12px; border-radius: 10px;
                font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.05em; box-shadow: 0 4px 12px rgba(35,134,54,0.3);
            }}
            .route-name {{ font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 6px; }}
            .route-desc {{ font-size: 0.8rem; color: #8b949e; margin-bottom: 10px; }}
            .grv-score {{ font-size: 1.6rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
            .grv-low {{ color: #3fb950; }}
            .grv-moderate {{ color: #d29922; }}
            .grv-high {{ color: #db6d28; }}
            .grv-critical {{ color: #f85149; }}
            .risk-badge {{
                display: inline-block; padding: 2px 10px; border-radius: 8px;
                font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
            }}
            .risk-low {{ background: rgba(63,185,80,0.2); color: #3fb950; border: 1px solid rgba(63,185,80,0.4); }}
            .risk-moderate {{ background: rgba(210,153,34,0.2); color: #d29922; border: 1px solid rgba(210,153,34,0.4); }}
            .risk-high {{ background: rgba(219,109,40,0.2); color: #db6d28; border: 1px solid rgba(219,109,40,0.4); }}
            .risk-critical {{ background: rgba(248,81,73,0.2); color: #f85149; border: 1px solid rgba(248,81,73,0.4); }}
            .grv-bar-bg {{
                height: 6px; background: rgba(48,54,61,0.5); border-radius: 3px; overflow: hidden;
            }}
            .grv-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.5s ease; }}
            .grv-bar-low {{ background: linear-gradient(90deg, #238636, #3fb950); }}
            .grv-bar-moderate {{ background: linear-gradient(90deg, #9e6a03, #d29922); }}
            .grv-bar-high {{ background: linear-gradient(90deg, #bd561d, #db6d28); }}
            .grv-bar-critical {{ background: linear-gradient(90deg, #da3633, #f85149); }}
        </style>
        {card_html}
        """)
    
    # ───── COMPARATIVE BAR CHART ─────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Comparative Risk Breakdown</div>', unsafe_allow_html=True)
    
    fig_bar = go.Figure()
    
    sorted_df = grv_df.sort_values("grv", ascending=True)
    
    fig_bar.add_trace(go.Bar(
        y=sorted_df["short_name"],
        x=sorted_df["conflict_score"] * weights["conflict"] * 10,
        name="⚔️ Conflict",
        orientation="h",
        marker_color="#FF6B6B",
        hovertemplate="Conflict: %{x:.1f}<extra></extra>",
    ))
    
    fig_bar.add_trace(go.Bar(
        y=sorted_df["short_name"],
        x=sorted_df["piracy_score"] * weights["piracy"] * 10,
        name="🏴‍☠️ Piracy",
        orientation="h",
        marker_color="#4ECDC4",
        hovertemplate="Piracy: %{x:.1f}<extra></extra>",
    ))
    
    fig_bar.add_trace(go.Bar(
        y=sorted_df["short_name"],
        x=sorted_df["sanctions_score"] * weights["sanctions"] * 10,
        name="🚫 Sanctions",
        orientation="h",
        marker_color="#F7DC6F",
        hovertemplate="Sanctions: %{x:.1f}<extra></extra>",
    ))
    
    fig_bar.update_layout(
        barmode="stack",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3", family="Inter"),
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="rgba(48,54,61,0.5)",
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            x=0.5, y=1.1, xanchor="center",
        ),
        xaxis=dict(
            title="Weighted Risk Score",
            gridcolor="rgba(48,54,61,0.3)",
            zerolinecolor="rgba(48,54,61,0.3)",
        ),
        yaxis=dict(gridcolor="rgba(48,54,61,0.3)"),
        margin=dict(l=120, r=20, t=30, b=40),
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: ROUTE ANALYSIS
# ═══════════════════════════════════════════════════════════════

elif page == "📊 Route Analysis":
    
    st.markdown("""
    <div class="hero-header animate-in">
        <h1>Route Analysis</h1>
        <div class="subtitle">Detailed GRV analysis per commodity across all trade routes</div>
        <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Side-by-side commodity comparison
    tab_single, tab_compare = st.tabs(["📦 Single Commodity", "⚖️ Compare Commodities"])
    
    with tab_single:
        commodity_name = COMMODITIES[selected_commodity]["name"]
        commodity_icon = COMMODITIES[selected_commodity]["icon"]
        
        st.markdown(f'<div class="section-header">{commodity_icon} {commodity_name} — GRV by Route</div>', unsafe_allow_html=True)
        
        # Detailed table
        display_df = grv_df[[
            "short_name", "route_name", "conflict_score", "piracy_score",
            "sanctions_score", "commodity_multiplier", "grv", "risk_level", "rank",
            "distance_nm", "conflict_events", "piracy_events", "sanctions_count"
        ]].copy()
        
        display_df.columns = [
            "Route", "Full Name", "Conflict Score", "Piracy Score",
            "Sanctions Score", "Commodity Sensitivity", "GRV (1-10)", "Risk Level", "Rank",
            "Distance (nm)", "Conflict Events", "Piracy Events", "Active Sanctions"
        ]
        display_df["Risk Level"] = display_df["Risk Level"].astype(str).replace("nan", "Low")
        
        st.dataframe(
            display_df.style
            .format({
                "Conflict Score": "{:.3f}",
                "Piracy Score": "{:.3f}",
                "Sanctions Score": "{:.3f}",
                "Commodity Sensitivity": "{:.2f}",
                "GRV (1-10)": "{:.1f}",
            })
            .background_gradient(subset=["GRV (1-10)"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=450,
        )
        
        # Recommendation box
        best = grv_df.iloc[0]
        worst = grv_df.iloc[-1]
        best_route = ROUTES[best["route_key"]]
        worst_route = ROUTES[worst["route_key"]]
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown(f"""
            <div class="glass-card" style="border-color:rgba(35,134,54,0.5);">
                <div style="font-size:0.8rem;color:#3fb950;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">✅ RECOMMENDED ROUTE</div>
                <div style="font-size:1.3rem;font-weight:800;color:#e6edf3;">{best_route['icon']} {best_route['name']}</div>
                <div style="font-size:0.85rem;color:#8b949e;margin:8px 0;">{best_route['description']}</div>
                <div style="display:flex;gap:16px;margin-top:12px;">
                    <div><span style="color:#3fb950;font-weight:700;font-size:1.5rem;font-family:'JetBrains Mono';">{best['grv']:.1f}</span><span style="color:#8b949e;font-size:0.7rem;"> GRV</span></div>
                    <div><span style="color:#8b949e;font-size:0.85rem;">{best_route.get('distance_nm', 'N/A')} nm</span></div>
                </div>
                <div style="margin-top:8px;font-size:0.75rem;color:#8b949e;">
                    Chokepoints: {', '.join(best_route['chokepoints'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown(f"""
            <div class="glass-card" style="border-color:rgba(248,81,73,0.5);">
                <div style="font-size:0.8rem;color:#f85149;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">⚠️ HIGHEST RISK ROUTE</div>
                <div style="font-size:1.3rem;font-weight:800;color:#e6edf3;">{worst_route['icon']} {worst_route['name']}</div>
                <div style="font-size:0.85rem;color:#8b949e;margin:8px 0;">{worst_route['description']}</div>
                <div style="display:flex;gap:16px;margin-top:12px;">
                    <div><span style="color:#f85149;font-weight:700;font-size:1.5rem;font-family:'JetBrains Mono';">{worst['grv']:.1f}</span><span style="color:#8b949e;font-size:0.7rem;"> GRV</span></div>
                    <div><span style="color:#8b949e;font-size:0.85rem;">{worst_route.get('distance_nm', 'N/A')} nm</span></div>
                </div>
                <div style="margin-top:8px;font-size:0.75rem;color:#8b949e;">
                    Chokepoints: {', '.join(worst_route['chokepoints'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Radar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🕸️ Risk Profile Radar</div>', unsafe_allow_html=True)
        
        fig_radar = go.Figure()
        
        for _, row in grv_df.head(5).iterrows():
            route = ROUTES[row["route_key"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=[row["conflict_score"], row["piracy_score"], row["sanctions_score"],
                   row["commodity_multiplier"] / 3, row["grv"] / 10],
                theta=["Conflict", "Piracy", "Sanctions", "Sensitivity", "GRV"],
                fill="toself",
                name=route["short_name"],
                line=dict(color=route["color"]),
                opacity=0.7,
            ))
        
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="rgba(48,54,61,0.3)",
                    linecolor="rgba(48,54,61,0.3)",
                    tickfont=dict(color="#8b949e"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(48,54,61,0.3)",
                    linecolor="rgba(48,54,61,0.3)",
                    tickfont=dict(color="#e6edf3", size=12),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            legend=dict(
                bgcolor="rgba(13,17,23,0.8)",
                bordercolor="rgba(48,54,61,0.5)",
                borderwidth=1,
            ),
            height=450,
            margin=dict(l=60, r=60, t=30, b=30),
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab_compare:
        st.markdown('<div class="section-header">⚖️ Commodity Comparison</div>', unsafe_allow_html=True)
        
        # Let user pick 2 commodities to compare
        all_commodity_keys = get_all_commodity_keys()
        all_commodity_labels = {k: f"{COMMODITIES[k]['icon']} {COMMODITIES[k]['name']}" for k in all_commodity_keys}
        
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            commodity_a = st.selectbox("Commodity A", all_commodity_keys,
                                       format_func=lambda x: all_commodity_labels[x],
                                       index=0, key="comp_a")
        with comp_col2:
            default_b = 1 if len(all_commodity_keys) > 1 else 0
            commodity_b = st.selectbox("Commodity B", all_commodity_keys,
                                       format_func=lambda x: all_commodity_labels[x],
                                       index=default_b, key="comp_b")
        
        # Compute GRV for both selected commodities
        grv_a = compute_grv(route_scores, weights, commodity_a)
        grv_b = compute_grv(route_scores, weights, commodity_b)
        
        name_a = COMMODITIES[commodity_a]["name"]
        name_b = COMMODITIES[commodity_b]["name"]
        icon_a = COMMODITIES[commodity_a]["icon"]
        icon_b = COMMODITIES[commodity_b]["icon"]
        
        # Merge for comparison
        comparison = grv_a[["short_name", "route_key", "grv", "rank"]].merge(
            grv_b[["route_key", "grv", "rank"]],
            on="route_key",
            suffixes=("_a", "_b"),
        )
        comparison.columns = ["Route", "route_key", f"GRV ({name_a})", f"Rank ({name_a})", f"GRV ({name_b})", f"Rank ({name_b})"]
        comparison["Δ GRV"] = (comparison[f"GRV ({name_a})"] - comparison[f"GRV ({name_b})"]).round(1)
        
        st.dataframe(
            comparison[["Route", f"GRV ({name_a})", f"Rank ({name_a})", f"GRV ({name_b})", f"Rank ({name_b})", "Δ GRV"]]
            .style.format({f"GRV ({name_a})": "{:.1f}", f"GRV ({name_b})": "{:.1f}", "Δ GRV": "{:+.1f}"}),
            use_container_width=True,
            height=450,
        )
        
        # Grouped bar chart
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            x=comparison["Route"],
            y=comparison[f"GRV ({name_a})"],
            name=f"{icon_a} {name_a}",
            marker_color="#a855f7",
            marker_line=dict(width=1, color="#c084fc"),
        ))
        
        fig_comp.add_trace(go.Bar(
            x=comparison["Route"],
            y=comparison[f"GRV ({name_b})"],
            name=f"{icon_b} {name_b}",
            marker_color="#06b6d4",
            marker_line=dict(width=1, color="#22d3ee"),
        ))
        
        fig_comp.update_layout(
            barmode="group",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3", family="Inter"),
            legend=dict(
                bgcolor="rgba(13,17,23,0.8)",
                bordercolor="rgba(48,54,61,0.5)", borderwidth=1,
                orientation="h", x=0.5, y=1.12, xanchor="center",
            ),
            xaxis=dict(gridcolor="rgba(48,54,61,0.3)", tickangle=-30),
            yaxis=dict(title="GRV Score (1-10)", gridcolor="rgba(48,54,61,0.3)", range=[0, 11]),
            margin=dict(l=40, r=20, t=40, b=80),
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Recommendation per commodity
        st.markdown("<br>", unsafe_allow_html=True)
        cr1, cr2 = st.columns(2)
        
        with cr1:
            best_a = grv_a.iloc[0]
            best_a_route = ROUTES[best_a["route_key"]]
            st.markdown(f"""
            <div class="glass-card" style="border-color:rgba(168,85,247,0.5);">
                <div style="font-size:0.8rem;color:#a855f7;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">{icon_a} BEST ROUTE FOR {name_a.upper()}</div>
                <div style="font-size:1.2rem;font-weight:800;color:#e6edf3;">{best_a_route['icon']} {best_a_route['name']}</div>
                <div style="color:#3fb950;font-weight:700;font-size:1.8rem;font-family:'JetBrains Mono';margin-top:8px;">GRV: {best_a['grv']:.1f}/10</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cr2:
            best_b = grv_b.iloc[0]
            best_b_route = ROUTES[best_b["route_key"]]
            st.markdown(f"""
            <div class="glass-card" style="border-color:rgba(6,182,212,0.5);">
                <div style="font-size:0.8rem;color:#06b6d4;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">{icon_b} BEST ROUTE FOR {name_b.upper()}</div>
                <div style="font-size:1.2rem;font-weight:800;color:#e6edf3;">{best_b_route['icon']} {best_b_route['name']}</div>
                <div style="color:#3fb950;font-weight:700;font-size:1.8rem;font-family:'JetBrains Mono';margin-top:8px;">GRV: {best_b['grv']:.1f}/10</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: ML MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════

elif page == "🤖 ML Model Insights":
    
    st.markdown("""
    <div class="hero-header animate-in">
        <h1>ML Weight Prediction</h1>
        <div class="subtitle">Machine learning models trained on conflict, piracy & sanctions data to predict optimal GRV weights</div>
        <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Weight comparison bars ──
    st.markdown('<div class="section-header">📊 Predicted Weights by Model</div>', unsafe_allow_html=True)
    
    model_order = ["gradient_boosting", "random_forest", "ridge_regression", "variance_analysis", "ensemble", "stock_market_validation"]
    
    for model_key in model_order:
        if model_key in ml_results:
            result = ml_results[model_key]
            w = result["weights"]
            m = result["metrics"]
            model_name = m.get("model", model_key.replace("_", " ").title())
            
            extra_info = ""
            if "r2_mean" in m:
                extra_info = f"  |  R² = {m['r2_mean']:.3f} ± {m.get('r2_std', 0):.3f}"
            
            st.markdown(render_weight_bar(
                w["conflict"], w["piracy"], w["sanctions"],
                model_name=f"{model_name}{extra_info}"
            ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Detailed model metrics ──
    st.markdown('<div class="section-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    metrics_data = []
    for model_key in ["gradient_boosting", "random_forest", "ridge_regression", "variance_analysis"]:
        if model_key in ml_results:
            result = ml_results[model_key]
            w = result["weights"]
            m = result["metrics"]
            metrics_data.append({
                "Model": m.get("model", model_key),
                "Conflict Weight": w["conflict"],
                "Piracy Weight": w["piracy"],
                "Sanctions Weight": w["sanctions"],
                "R² Score": m.get("r2_mean", "N/A"),
                "R² Std": m.get("r2_std", "N/A"),
                "PCA Var Explained": m.get("explained_variance_pca", "N/A"),
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(
        metrics_df.style.format({
            "Conflict Weight": "{:.4f}",
            "Piracy Weight": "{:.4f}",
            "Sanctions Weight": "{:.4f}",
        }),
        use_container_width=True,
    )
    
    # ── Ensemble breakdown chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Ensemble Weight Visualization</div>', unsafe_allow_html=True)
    
    ensemble_w = ml_results["ensemble"]["weights"]
    
    fig_weights = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["Weight Distribution", "Model Comparison"],
    )
    
    # Pie chart
    fig_weights.add_trace(
        go.Pie(
            labels=["⚔️ Conflict", "🏴‍☠️ Piracy", "🚫 Sanctions"],
            values=[ensemble_w["conflict"], ensemble_w["piracy"], ensemble_w["sanctions"]],
            marker=dict(colors=["#FF6B6B", "#4ECDC4", "#F7DC6F"]),
            textinfo="label+percent",
            textfont=dict(size=13, color="#1c1c1c"),
            hole=0.4,
        ),
        row=1, col=1,
    )
    
    # Bar chart comparison
    model_names = []
    conflict_vals = []
    piracy_vals = []
    sanctions_vals = []
    
    for model_key in ["gradient_boosting", "random_forest", "ridge_regression", "variance_analysis", "ensemble", "stock_market_validation"]:
        if model_key in ml_results:
            w = ml_results[model_key]["weights"]
            label = model_key.replace("_", " ").title()
            if len(label) > 15:
                label = label[:14] + "…"
            model_names.append(label)
            conflict_vals.append(w["conflict"])
            piracy_vals.append(w["piracy"])
            sanctions_vals.append(w["sanctions"])
    
    fig_weights.add_trace(
        go.Bar(name="Conflict", x=model_names, y=conflict_vals, marker_color="#FF6B6B", showlegend=False),
        row=1, col=2,
    )
    fig_weights.add_trace(
        go.Bar(name="Piracy", x=model_names, y=piracy_vals, marker_color="#4ECDC4", showlegend=False),
        row=1, col=2,
    )
    fig_weights.add_trace(
        go.Bar(name="Sanctions", x=model_names, y=sanctions_vals, marker_color="#F7DC6F", showlegend=False),
        row=1, col=2,
    )
    
    fig_weights.update_layout(
        barmode="group",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3", family="Inter"),
        margin=dict(l=40, r=20, t=40, b=80),
    )
    fig_weights.update_xaxes(gridcolor="rgba(48,54,61,0.3)", tickangle=-25, row=1, col=2)
    fig_weights.update_yaxes(gridcolor="rgba(48,54,61,0.3)", row=1, col=2)
    fig_weights.update_annotations(font=dict(color="#e6edf3", size=14))
    
    st.plotly_chart(fig_weights, use_container_width=True)
    
    # ── Stock market comparison ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📉 Validation: Indian Stock Market Analysis</div>', unsafe_allow_html=True)
    
    smv = ml_results["stock_market_validation"]
    sm_metrics = smv["metrics"]
    sm_weights = smv["weights"]
    ens_weights = ml_results["ensemble"]["weights"]
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="info-box">
            <strong>Methodology:</strong> Weights derived from BSE/NSE daily return suppression analysis.<br>
            • <strong>Quiet day avg return:</strong> +0.067%<br>
            • <strong>Conflict day avg return:</strong> +0.040% (n={sm_metrics['sample_conflict_days']:,})<br>
            • <strong>Piracy day avg return:</strong> +0.019% (n={sm_metrics['sample_piracy_days']:,})<br>
            • <strong>Conflict suppression:</strong> {sm_metrics['conflict_suppression']}%<br>
            • <strong>Piracy suppression:</strong> {sm_metrics['piracy_suppression']}%
        </div>
        <div style="display:flex;gap:20px;margin-top:16px;">
            <div style="flex:1;">
                <div style="font-size:0.8rem;color:#8b949e;font-weight:600;margin-bottom:8px;">STOCK MARKET WEIGHTS</div>
                <div style="font-size:1.1rem;color:#e6edf3;">
                    ⚔️ {sm_weights['conflict']:.2f} &nbsp; 🏴‍☠️ {sm_weights['piracy']:.2f} &nbsp; 🚫 {sm_weights['sanctions']:.2f}
                </div>
            </div>
            <div style="flex:1;">
                <div style="font-size:0.8rem;color:#8b949e;font-weight:600;margin-bottom:8px;">ML ENSEMBLE WEIGHTS</div>
                <div style="font-size:1.1rem;color:#e6edf3;">
                    ⚔️ {ens_weights['conflict']:.2f} &nbsp; 🏴‍☠️ {ens_weights['piracy']:.2f} &nbsp; 🚫 {ens_weights['sanctions']:.2f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════

elif page == "📈 Data Explorer":
    
    st.markdown("""
    <div class="hero-header animate-in">
        <h1>Data Explorer</h1>
        <div class="subtitle">Browse and analyze raw conflict, piracy & sanctions datasets</div>
        <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card("⚔️", "Conflict Events", f"{len(conflict_df):,}", "conflict"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("🏴‍☠️", "Piracy Incidents", f"{len(piracy_df):,}", "piracy"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("🚫", "Sanctions Records", f"{len(sanctions_df):,}", "sanctions"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_conflict, tab_piracy, tab_sanctions, tab_temporal = st.tabs([
        "⚔️ Conflict Data", "🏴‍☠️ Piracy Data", "🚫 Sanctions Data", "📅 Temporal Trends"
    ])
    
    with tab_conflict:
        st.markdown('<div class="section-header">⚔️ Global Conflict Events (1995–2025)</div>', unsafe_allow_html=True)
        
        # Year filter
        years = sorted(conflict_df["year"].dropna().unique())
        year_range = st.slider("Year Range", int(min(years)), int(max(years)), (2015, 2025), key="conflict_year")
        
        filtered = conflict_df[(conflict_df["year"] >= year_range[0]) & (conflict_df["year"] <= year_range[1])]
        
        # Heatmap of conflict events
        fig_conflict_map = go.Figure(go.Densitymapbox(
            lat=filtered["latitude"].head(20000),
            lon=filtered["longitude"].head(20000),
            z=filtered["fatalities"].head(20000).clip(upper=100),
            radius=8,
            colorscale="hot",
            opacity=0.7,
        ))
        
        fig_conflict_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=20, lon=50),
                zoom=2,
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        
        st.plotly_chart(fig_conflict_map, use_container_width=True)
        
        st.dataframe(filtered.head(500), use_container_width=True, height=300)
    
    with tab_piracy:
        st.markdown('<div class="section-header">🏴‍☠️ Global Piracy Incidents (1995–2025)</div>', unsafe_allow_html=True)
        
        fig_piracy_map = go.Figure(go.Scattermapbox(
            lat=piracy_df["latitude"],
            lon=piracy_df["longitude"],
            mode="markers",
            marker=dict(
                size=5,
                color=piracy_df["severity_casualties"],
                colorscale="turbo",
                opacity=0.6,
                colorbar=dict(title="Casualties", tickfont=dict(color="#e6edf3")),
            ),
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Casualties: %{marker.color}<extra></extra>",
        ))
        
        fig_piracy_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=10, lon=70),
                zoom=2,
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        
        st.plotly_chart(fig_piracy_map, use_container_width=True)
        
        st.dataframe(piracy_df.head(500), use_container_width=True, height=300)
    
    with tab_sanctions:
        st.markdown('<div class="section-header">🚫 Active Sanctions (1995–2025)</div>', unsafe_allow_html=True)
        
        # Sanctions by severity
        fig_sanctions = px.histogram(
            sanctions_df, x="severity_score", nbins=5,
            color_discrete_sequence=["#F7DC6F"],
            title="Distribution of Sanction Severity Scores",
        )
        fig_sanctions.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="rgba(48,54,61,0.3)"),
            yaxis=dict(gridcolor="rgba(48,54,61,0.3)"),
        )
        st.plotly_chart(fig_sanctions, use_container_width=True)
        
        st.dataframe(sanctions_df, use_container_width=True, height=300)
    
    with tab_temporal:
        st.markdown('<div class="section-header">📅 Temporal Risk Trends</div>', unsafe_allow_html=True)
        
        # Conflict events per year
        conflict_yearly = conflict_df.groupby("year").agg(
            events=("conflict_id", "count"),
            fatalities=("fatalities", "sum"),
        ).reset_index()
        
        piracy_yearly = piracy_df.groupby("year").agg(
            events=("incident_id", "count"),
            casualties=("severity_casualties", "sum"),
        ).reset_index()
        
        fig_temporal = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Conflict Events & Fatalities by Year", "Piracy Incidents & Casualties by Year"],
            shared_xaxes=True,
            vertical_spacing=0.12,
        )
        
        fig_temporal.add_trace(
            go.Bar(x=conflict_yearly["year"], y=conflict_yearly["events"], name="Conflict Events", marker_color="#FF6B6B", opacity=0.7),
            row=1, col=1,
        )
        fig_temporal.add_trace(
            go.Scatter(x=conflict_yearly["year"], y=conflict_yearly["fatalities"], name="Fatalities", line=dict(color="#ff4444", width=2), yaxis="y2"),
            row=1, col=1,
        )
        
        fig_temporal.add_trace(
            go.Bar(x=piracy_yearly["year"], y=piracy_yearly["events"], name="Piracy Incidents", marker_color="#4ECDC4", opacity=0.7),
            row=2, col=1,
        )
        fig_temporal.add_trace(
            go.Scatter(x=piracy_yearly["year"], y=piracy_yearly["casualties"], name="Casualties", line=dict(color="#00b894", width=2)),
            row=2, col=1,
        )
        
        fig_temporal.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3", family="Inter"),
            legend=dict(
                bgcolor="rgba(13,17,23,0.8)",
                bordercolor="rgba(48,54,61,0.5)", borderwidth=1,
            ),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        fig_temporal.update_xaxes(gridcolor="rgba(48,54,61,0.3)")
        fig_temporal.update_yaxes(gridcolor="rgba(48,54,61,0.3)")
        fig_temporal.update_annotations(font=dict(color="#e6edf3"))
        
        st.plotly_chart(fig_temporal, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: METHODOLOGY
# ═══════════════════════════════════════════════════════════════

elif page == "📐 Methodology":
    
    st.markdown("""
    <div class="hero-header animate-in">
        <h1>Methodology</h1>
        <div class="subtitle">How GRV is computed and how ML models predict the component weights</div>
        <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">📐 GRV Formula</div>
        <div class="formula-box">
            GRV = w<sub>conflict</sub> × Conflict<sub>score</sub> + w<sub>piracy</sub> × Piracy<sub>score</sub> + w<sub>sanctions</sub> × Sanctions<sub>score</sub>
        </div>
        <div class="info-box" style="margin-top:16px;">
            Where each <strong>sub-score</strong> is normalized to [0, 1] and <strong>weights sum to 1.0</strong>.
            The final GRV is scaled to <strong>1–10</strong> and adjusted by a commodity-specific sensitivity multiplier.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">⚔️ Conflict Score Computation</div>
        <div class="info-box">
            <strong>Data:</strong> ~327K conflict events (1995–2025) with lat/lon, fatalities, event type<br><br>
            <strong>Per Route:</strong> Events are matched to route corridors using geographic bounding boxes.<br><br>
            <strong>Score = 0.4 × Events<sub>norm</sub> + 0.4 × Fatalities<sub>norm</sub> + 0.2 × Intensity<sub>norm</sub></strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">🏴‍☠️ Piracy Score Computation</div>
        <div class="info-box">
            <strong>Data:</strong> ~8.5K piracy incidents (1995–2025) with lat/lon, casualties, economic loss<br><br>
            <strong>Per Route:</strong> Incidents matched via bounding boxes around maritime corridors.<br><br>
            <strong>Score = 0.4 × Events<sub>norm</sub> + 0.3 × Casualties<sub>norm</sub> + 0.3 × Severity<sub>norm</sub></strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">🚫 Sanctions Score Computation</div>
        <div class="info-box">
            <strong>Data:</strong> 201 active sanctions with country codes, severity scores, types<br><br>
            <strong>Per Route:</strong> Sanctions are matched via country codes along the route.<br><br>
            <strong>Score = 0.5 × Count<sub>norm</sub> + 0.5 × Severity<sub>norm</sub></strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">🤖 ML Weight Prediction Pipeline</div>
        <div class="info-box">
            <strong>Step 1:</strong> Build quarterly feature matrix per route (conflict, piracy, sanctions aggregates)<br><br>
            <strong>Step 2:</strong> Create composite risk target using PCA (1st principal component)<br><br>
            <strong>Step 3:</strong> Train 4 models — Gradient Boosting, Random Forest, Ridge Regression, Variance Analysis<br><br>
            <strong>Step 4:</strong> Extract feature importances / coefficients as component weights<br><br>
            <strong>Step 5:</strong> Compute ensemble weights (R²-weighted average of all models)<br><br>
            <strong>Validation:</strong> Compare ML-predicted weights against Indian stock market suppression analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">📦 Commodity Sensitivity Multipliers</div>
        <div class="info-box">
            Each commodity has route-specific <strong>sensitivity multipliers</strong> reflecting supply chain criticality.
            Only routes where the commodity is actually shipped are displayed. 12 commodity categories are supported:<br><br>
            <strong>Electrical Equipment:</strong><br>
            <strong>🔋 EV Batteries:</strong> Malacca (China→India) and Transpacific (China→US)<br>
            <strong>💽 Semiconductors:</strong> Transpacific (Taiwan/Korea→US) and Malacca<br>
            <strong>☀️ Solar Panels:</strong> Malacca (80%+ of India's imports from China)<br>
            <strong>⚡ Power Transformers:</strong> Suez (Germany/EU→India heavy equipment)<br>
            <strong>⚙️ Electric Motors:</strong> Suez (EU) and Malacca (China/Japan)<br>
            <strong>🔌 Cables & Wiring:</strong> Cape (African copper ore) and India Coastal<br>
            <strong>💡 LED Lighting:</strong> Malacca (China→India LED imports)<br><br>
            <strong>Petroleum Products:</strong><br>
            <strong>🛢️ Crude Oil:</strong> Hormuz (Middle East→India/Asia, dominant)<br>
            <strong>🏗️ Sweet Crude:</strong> Cape & Suez (Nigeria/Norway→India)<br>
            <strong>🔥 LPG:</strong> Hormuz (Gulf→India) and India Coastal<br>
            <strong>⛽ Petrol:</strong> Hormuz, Malacca, India Coastal<br>
            <strong>🚛 Diesel:</strong> Hormuz, Malacca, Cape, India Coastal
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show commodity sensitivity table
    st.markdown('<div class="section-header">📋 Sensitivity Matrix (— = not shipped on this route)</div>', unsafe_allow_html=True)
    
    sensitivity_data = []
    for route_key in get_all_route_keys():
        route = ROUTES[route_key]
        row = {"Route": route["short_name"]}
        for commodity_key in get_all_commodity_keys():
            commodity = COMMODITIES[commodity_key]
            applicable = commodity.get("applicable_routes", [])
            if route_key in applicable:
                row[commodity["icon"] + " " + commodity["name"]] = commodity["route_sensitivity"].get(route_key, 1.0)
            else:
                row[commodity["icon"] + " " + commodity["name"]] = None
        sensitivity_data.append(row)
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    # Format: show numbers for applicable routes, — for non-applicable
    value_cols = [c for c in sensitivity_df.columns if c != "Route"]
    
    def format_sensitivity(val):
        if pd.isna(val):
            return "—"
        return f"{val:.1f}"
    
    styled = sensitivity_df.style
    for col in value_cols:
        styled = styled.format({col: format_sensitivity})
    
    # Apply gradient only to non-null values
    styled = styled.background_gradient(
        cmap="RdYlGn_r", subset=value_cols, vmin=0.5, vmax=3.0
    )
    
    st.dataframe(styled, use_container_width=True, height=500)
