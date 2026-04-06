"""
Styles — Premium dark theme CSS for the Streamlit GRV Calculator app.
Glassmorphism, gradients, animations, and modern typography.
"""


def get_custom_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* ═══════════════════════════════════════════════════
       GLOBAL STYLES & DARK THEME
       ═══════════════════════════════════════════════════ */
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 40%, #0f1923 70%, #0a0e17 100%);
        color: #e6edf3;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .stApp > header {
        background: transparent !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ═══════════════════════════════════════════════════
       SIDEBAR STYLING
       ═══════════════════════════════════════════════════ */
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border-right: 1px solid rgba(48, 54, 61, 0.6);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #8b949e !important;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
    }
    
    /* ═══════════════════════════════════════════════════
       GLASSMORPHISM CARDS
       ═══════════════════════════════════════════════════ */
    
    .glass-card {
        background: rgba(13, 17, 23, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .glass-card:hover {
        border-color: rgba(88, 166, 255, 0.3);
        box-shadow: 0 8px 32px rgba(88, 166, 255, 0.1);
        transform: translateY(-2px);
    }
    
    /* ═══════════════════════════════════════════════════
       METRIC CARDS
       ═══════════════════════════════════════════════════ */
    
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    
    .metric-card.conflict::before { background: linear-gradient(90deg, #FF6B6B, #FF8E8E); }
    .metric-card.piracy::before { background: linear-gradient(90deg, #4ECDC4, #6BE5DC); }
    .metric-card.sanctions::before { background: linear-gradient(90deg, #F7DC6F, #F9E89B); }
    .metric-card.grv::before { background: linear-gradient(90deg, #a855f7, #d946ef); }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 8px 0;
        background: linear-gradient(135deg, #58a6ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 4px;
    }
    
    /* ═══════════════════════════════════════════════════
       GRV GAUGE / RISK LEVEL BADGES
       ═══════════════════════════════════════════════════ */
    
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .risk-low { background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid rgba(35, 134, 54, 0.4); }
    .risk-moderate { background: rgba(187, 128, 9, 0.2); color: #d29922; border: 1px solid rgba(187, 128, 9, 0.4); }
    .risk-high { background: rgba(218, 109, 40, 0.2); color: #db6d28; border: 1px solid rgba(218, 109, 40, 0.4); }
    .risk-critical { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }
    
    /* ═══════════════════════════════════════════════════
       ROUTE CARDS
       ═══════════════════════════════════════════════════ */
    
    .route-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.7), rgba(13, 17, 23, 0.85));
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .route-card:hover {
        border-color: rgba(88, 166, 255, 0.4);
        transform: translateX(4px);
    }
    
    .route-card .route-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 6px;
    }
    
    .route-card .route-desc {
        font-size: 0.8rem;
        color: #8b949e;
        margin-bottom: 10px;
    }
    
    .route-card .grv-score {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .grv-low { color: #3fb950; }
    .grv-moderate { color: #d29922; }
    .grv-high { color: #db6d28; }
    .grv-critical { color: #f85149; }
    
    /* ═══════════════════════════════════════════════════
       RECOMMENDED ROUTE HIGHLIGHT
       ═══════════════════════════════════════════════════ */
    
    .recommended-route {
        background: linear-gradient(135deg, rgba(35, 134, 54, 0.15), rgba(22, 27, 34, 0.8));
        border: 2px solid rgba(35, 134, 54, 0.5) !important;
        position: relative;
    }
    
    .recommended-badge {
        position: absolute;
        top: -10px;
        right: 16px;
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        padding: 3px 12px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3);
    }
    
    /* ═══════════════════════════════════════════════════
       WEIGHT MODEL CARDS
       ═══════════════════════════════════════════════════ */
    
    .weight-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.75), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 10px;
    }
    
    .weight-card .model-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .weight-bar-container {
        background: rgba(48, 54, 61, 0.3);
        border-radius: 6px;
        height: 28px;
        display: flex;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .weight-bar-conflict {
        background: linear-gradient(90deg, #FF6B6B, #FF8E8E);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #1c1c1c;
        transition: width 1s ease;
    }
    
    .weight-bar-piracy {
        background: linear-gradient(90deg, #4ECDC4, #6BE5DC);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #1c1c1c;
        transition: width 1s ease;
    }
    
    .weight-bar-sanctions {
        background: linear-gradient(90deg, #F7DC6F, #F9E89B);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        color: #1c1c1c;
        transition: width 1s ease;
    }
    
    /* ═══════════════════════════════════════════════════
       HERO HEADER
       ═══════════════════════════════════════════════════ */
    
    .hero-header {
        text-align: center;
        padding: 30px 0 20px 0;
        margin-bottom: 30px;
        position: relative;
    }
    
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #58a6ff 0%, #a855f7 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    
    .hero-header .subtitle {
        font-size: 1rem;
        color: #8b949e;
        font-weight: 400;
        letter-spacing: 0.02em;
    }
    
    .hero-header .divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #58a6ff, #a855f7);
        margin: 16px auto;
        border-radius: 2px;
    }
    
    /* ═══════════════════════════════════════════════════
       SECTION HEADERS
       ═══════════════════════════════════════════════════ */
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 30px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(48, 54, 61, 0.5);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* ═══════════════════════════════════════════════════
       TABLE STYLING
       ═══════════════════════════════════════════════════ */
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame table {
        background: rgba(13, 17, 23, 0.8) !important;
    }
    
    /* ═══════════════════════════════════════════════════
       TAB STYLING
       ═══════════════════════════════════════════════════ */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13, 17, 23, 0.5);
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #8b949e;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(88, 166, 255, 0.15) !important;
        color: #58a6ff !important;
    }
    
    /* ═══════════════════════════════════════════════════
       ANIMATIONS
       ═══════════════════════════════════════════════════ */
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease forwards;
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* ═══════════════════════════════════════════════════
       COMMODITY SELECTOR CARDS
       ═══════════════════════════════════════════════════ */
    
    .commodity-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8), rgba(13, 17, 23, 0.9));
        border: 2px solid rgba(48, 54, 61, 0.5);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .commodity-card.selected {
        border-color: #58a6ff;
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(13, 17, 23, 0.9));
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.15);
    }
    
    .commodity-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .commodity-icon {
        font-size: 2.5rem;
        margin-bottom: 8px;
    }
    
    .commodity-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e6edf3;
    }
    
    /* ═══════════════════════════════════════════════════
       PROGRESS BAR / GRV BAR
       ═══════════════════════════════════════════════════ */
    
    .grv-bar-bg {
        background: rgba(48, 54, 61, 0.3);
        border-radius: 8px;
        height: 12px;
        width: 100%;
        overflow: hidden;
    }
    
    .grv-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .grv-bar-low { background: linear-gradient(90deg, #238636, #3fb950); }
    .grv-bar-moderate { background: linear-gradient(90deg, #bb8009, #d29922); }
    .grv-bar-high { background: linear-gradient(90deg, #da6d28, #db6d28); }
    .grv-bar-critical { background: linear-gradient(90deg, #da3633, #f85149); }
    
    /* ═══════════════════════════════════════════════════
       METHODOLOGY / INFO BOXES
       ═══════════════════════════════════════════════════ */
    
    .info-box {
        background: rgba(56, 139, 253, 0.08);
        border: 1px solid rgba(56, 139, 253, 0.25);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #8b949e;
        line-height: 1.6;
    }
    
    .info-box strong {
        color: #58a6ff;
    }
    
    .formula-box {
        background: rgba(13, 17, 23, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 10px;
        padding: 16px 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: #a855f7;
        text-align: center;
        margin: 12px 0;
        letter-spacing: 0.02em;
    }
    
    /* ═══════════════════════════════════════════════════
       SCROLLBAR
       ═══════════════════════════════════════════════════ */
    
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(13, 17, 23, 0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(48, 54, 61, 0.8); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(88, 166, 255, 0.3); }
    
    </style>
    """


def render_metric_card(icon, label, value, card_class=""):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card {card_class}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def render_route_card(route_name, route_desc, grv_score, risk_level, rank, is_recommended=False, icon="🚢", chokepoints=None):
    """Render a styled route card with GRV score."""
    # Ensure risk_level is valid
    risk_level = str(risk_level) if risk_level and str(risk_level) != "nan" else "Low"
    risk_class = risk_level.lower() if risk_level else "low"
    rec_class = "recommended-route" if is_recommended else ""
    rec_badge = '<div class="recommended-badge">✓ RECOMMENDED</div>' if is_recommended else ""
    
    # Ensure grv_score is numeric
    try:
        grv_val = float(grv_score)
    except (TypeError, ValueError):
        grv_val = 1.0
    
    chokepoint_html = ""
    if chokepoints:
        chips = " ".join([f'<span style="background:rgba(88,166,255,0.15);color:#58a6ff;padding:2px 8px;border-radius:6px;font-size:0.7rem;margin-right:4px;">{cp}</span>' for cp in chokepoints])
        chokepoint_html = f'<div style="margin-top:8px;">{chips}</div>'
    
    return f"""
    <div class="route-card {rec_class}">
        {rec_badge}
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
            <div style="flex:1;min-width:0;">
                <div class="route-name">{icon} {route_name}</div>
                <div class="route-desc">{route_desc}</div>
                {chokepoint_html}
            </div>
            <div style="text-align:right;min-width:100px;flex-shrink:0;">
                <div class="grv-score grv-{risk_class}">{grv_val:.1f}/10</div>
                <div class="risk-badge risk-{risk_class}">{risk_level}</div>
                <div style="font-size:0.7rem;color:#8b949e;margin-top:4px;">Rank #{rank}</div>
            </div>
        </div>
        <div class="grv-bar-bg" style="margin-top:12px;">
            <div class="grv-bar-fill grv-bar-{risk_class}" style="width:{min(grv_val * 10, 100):.0f}%;"></div>
        </div>
    </div>
    """


def render_weight_bar(conflict_w, piracy_w, sanctions_w, model_name=""):
    """Render a horizontal stacked weight bar."""
    c_pct = conflict_w * 100
    p_pct = piracy_w * 100
    s_pct = sanctions_w * 100
    
    return f"""
    <div class="weight-card">
        <div class="model-name">{model_name}</div>
        <div class="weight-bar-container">
            <div class="weight-bar-conflict" style="width:{c_pct}%">⚔️ {c_pct:.1f}%</div>
            <div class="weight-bar-piracy" style="width:{p_pct}%">🏴‍☠️ {p_pct:.1f}%</div>
            <div class="weight-bar-sanctions" style="width:{s_pct}%">🚫 {s_pct:.1f}%</div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#8b949e;margin-top:4px;">
            <span>Conflict: {conflict_w:.3f}</span>
            <span>Piracy: {piracy_w:.3f}</span>
            <span>Sanctions: {sanctions_w:.3f}</span>
        </div>
    </div>
    """
