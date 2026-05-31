"""
app.py — CA Intelligence Suite v2.0
Streamlit UI Orchestrator

This file is a PURE UI LAYER. All business logic lives in:
    config.py             — Central constants
    modules/ml_model.py   — ML inference + hybrid fallback
    modules/rule_engine.py— Backward chaining engine
    modules/financial_engine.py — Computations & health
    modules/compliance_engine.py— Insights & scoring
    modules/anomaly_detection.py— Anomaly detection
    modules/visualization.py   — All Plotly charts
    modules/report_generator.py — 14-section PDF
    utils/                — Helpers, logger

To train the model:  python run_pipeline.py
To launch the app:   streamlit run app.py
"""

import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# 
# Page config — MUST be first Streamlit call
# 
st.set_page_config(
    page_title="CA Intelligence Suite — AI Accounting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 
# Module imports (after set_page_config)
# 
from config import APP_TITLE, APP_VERSION
from utils.helpers import inr_format
from modules.ml_model import load_artifacts, predict_categories
from modules.financial_engine import (
    compute_financials, classify_financial_health,
    estimate_cash_flow, compute_comparative_analysis,
)
from modules.compliance_engine import (
    generate_compliance_insights, compute_compliance_score,
    validate_balance_sheet, validate_tax_provision,
)
from modules.data_validation import validate_data_quality
from modules.visualization import (
    chart_income_vs_expense, chart_expense_distribution_pie,
    chart_category_bar, chart_category_pie, chart_monthly_trend,
    chart_subcategory_expense, chart_payment_mode, chart_gst_breakdown,
    chart_compliance_score_gauge,
    # Phase 8 — Advanced Interactive Charts
    chart_sankey_flow, chart_network_graph, chart_sunburst_hierarchy,
)
from modules.report_generator import generate_pdf_report
from utils.helpers import data_quality_score

# Convenience alias
inr = inr_format

# 
# Premium Dark CSS (UI responsibility only — no business logic here)
# 
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap');

/* ═══════════════════════════════════════════════════════════
   PREMIUM DESIGN TOKENS — Navy + Gold Financial Theme
═══════════════════════════════════════════════════════════ */
:root {
  --bg-primary:      #060d1a;
  --bg-secondary:    #0b1628;
  --bg-card:         #0f1f38;
  --bg-card2:        #132540;
  --bg-card3:        #172d4c;
  --accent-gold:     #d4a843;
  --accent-gold-lt:  #f0c96e;
  --accent-gold-dk:  #a07830;
  --accent-emerald:  #00c896;
  --accent-rose:     #f43f5e;
  --accent-sky:      #38bdf8;
  --accent-violet:   #a78bfa;
  --accent-amber:    #fbbf24;
  --accent-teal:     #2dd4bf;
  --text-primary:    #eef2ff;
  --text-secondary:  #a8b8d8;
  --text-muted:      #5a7099;
  --border:          #1a3055;
  --border-light:    #243f66;
  --glow-gold:       0 0 28px rgba(212,168,67,0.30);
  --glow-emerald:    0 0 28px rgba(0,200,150,0.28);
  --glow-rose:       0 0 28px rgba(244,63,94,0.28);
  --glow-sky:        0 0 28px rgba(56,189,248,0.28);
  --glow-violet:     0 0 28px rgba(167,139,250,0.28);
  --glow-amber:      0 0 28px rgba(251,191,36,0.28);
  --radius-sm:  8px;
  --radius-md:  12px;
  --radius-lg:  18px;
  --radius-xl:  24px;
}

/* ═══ KEYFRAME ANIMATIONS ═══════════════════════════════════ */
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGold {
  0%, 100% { box-shadow: 0 0 0 0 rgba(212,168,67,0); }
  50%       { box-shadow: 0 0 0 8px rgba(212,168,67,0.12); }
}
@keyframes spinDot {
  to { transform: rotate(360deg); }
}
@keyframes glowPulse {
  0%, 100% { opacity: 0.6; }
  50%       { opacity: 1.0; }
}

/* ═══ BASE LAYOUT ════════════════════════════════════════════ */
html, body {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(212,168,67,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(56,189,248,0.04) 0%, transparent 50%) !important;
    background-attachment: fixed !important;
}

[data-testid="stMain"] {
    background: transparent !important;
}

/* ═══ SCROLLBAR ══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent-gold-dk), var(--border-light));
    border-radius: 10px;
}

/* ═══ SIDEBAR ════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1525 0%, #070e1c 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 180px;
    background: linear-gradient(180deg, rgba(212,168,67,0.05) 0%, transparent 100%);
    pointer-events: none;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 2px rgba(212,168,67,0.15) !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    text-transform: none !important;
    font-size: 0.85rem !important;
    letter-spacing: 0 !important;
    color: var(--text-primary) !important;
}

/* ═══ NAVIGATION TABS — FIXED ════════════════════════════════ */
/* Target Streamlit tab bar container */
.stTabs > div:first-child,
[data-testid="stTabs"] > div:first-child,
div[class*="stTabs"] > div:first-child {
    background: transparent !important;
}

/* Tab list wrapper */
.stTabs [data-baseweb="tab-list"],
[role="tablist"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 6px 8px !important;
    gap: 4px !important;
    margin-bottom: 28px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    display: flex !important;
    overflow-x: auto !important;
    scrollbar-width: none !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }

/* Individual Tab */
.stTabs [data-baseweb="tab"],
[role="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 9px 18px !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    white-space: nowrap !important;
    position: relative !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
    outline: none !important;
}

/* Tab Hover */
.stTabs [data-baseweb="tab"]:hover,
[role="tab"]:hover {
    background: rgba(212,168,67,0.08) !important;
    color: var(--accent-gold-lt) !important;
    transform: translateY(-1px) !important;
}

/* Active / Selected Tab */
.stTabs [data-baseweb="tab"][aria-selected="true"],
[role="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-gold-dk)) !important;
    color: #0a0f1a !important;
    font-weight: 700 !important;
    box-shadow: var(--glow-gold), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transform: translateY(-1px) !important;
}

/* Remove Streamlit default tab highlight bar (the underline indicator) */
.stTabs [data-baseweb="tab-highlight"],
[data-baseweb="tab-highlight"] {
    background: transparent !important;
    height: 0 !important;
    display: none !important;
}

/* Tab panel padding */
.stTabs [data-baseweb="tab-panel"] { padding-top: 4px !important; }

/* ═══ HERO HEADER ════════════════════════════════════════════ */
.hero-header {
    background:
        linear-gradient(135deg, #08122a 0%, #0d1e3d 45%, #101b38 100%);
    border: 1px solid var(--border-light);
    border-top: 1px solid rgba(212,168,67,0.25);
    border-radius: var(--radius-xl);
    padding: 36px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.5s ease both;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(212,168,67,0.08);
}
.hero-header::before {
    content: '';
    position: absolute; top: -80px; right: -60px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(212,168,67,0.10) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-header::after {
    content: '';
    position: absolute; bottom: -60px; left: 10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 60%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-logo-wrap {
    display: flex; align-items: center; gap: 16px; margin-bottom: 14px;
}
.hero-logo {
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-gold-dk));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; box-shadow: var(--glow-gold);
    animation: pulseGold 3s ease infinite;
}
.hero-title {
    font-family: 'Outfit', 'Inter', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #f0c96e 0%, #d4a843 40%, #ffffff 70%, #38bdf8 100%);
    background-size: 400px 100%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; line-height: 1.15;
    animation: shimmer 4s linear infinite;
}
.hero-sub {
    font-size: 0.97rem; color: var(--text-secondary);
    margin-top: 8px; line-height: 1.6; max-width: 680px;
}
.hero-badges { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(212,168,67,0.10);
    border: 1px solid rgba(212,168,67,0.30);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.73rem; color: var(--accent-gold-lt); font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.hero-badge.sky {
    background: rgba(56,189,248,0.08);
    border-color: rgba(56,189,248,0.25);
    color: #7dd3fc;
}
.hero-badge.emerald {
    background: rgba(0,200,150,0.08);
    border-color: rgba(0,200,150,0.25);
    color: #5eead4;
}
.hero-stats {
    display: flex; gap: 28px; margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.hero-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem; font-weight: 700;
    color: var(--accent-gold-lt);
}
.hero-stat-lbl {
    font-size: 0.7rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px;
}

/* ═══ KPI CARDS ══════════════════════════════════════════════ */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin: 24px 0;
}
@media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 22px 24px;
    position: relative; overflow: hidden;
    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1),
                box-shadow 0.25s cubic-bezier(0.4,0,0.2,1);
    animation: fadeSlideUp 0.4s ease both;
    cursor: default;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.kpi-card:hover {
    transform: translateY(-5px);
}
.kpi-card.gold   { border-color: rgba(212,168,67,0.25); }
.kpi-card.gold::before   { background: linear-gradient(90deg, var(--accent-gold), transparent); }
.kpi-card.gold:hover     { box-shadow: var(--glow-gold); }

.kpi-card.emerald { border-color: rgba(0,200,150,0.25); }
.kpi-card.emerald::before { background: linear-gradient(90deg, var(--accent-emerald), transparent); }
.kpi-card.emerald:hover   { box-shadow: var(--glow-emerald); }

.kpi-card.rose  { border-color: rgba(244,63,94,0.25); }
.kpi-card.rose::before  { background: linear-gradient(90deg, var(--accent-rose), transparent); }
.kpi-card.rose:hover    { box-shadow: var(--glow-rose); }

.kpi-card.sky   { border-color: rgba(56,189,248,0.25); }
.kpi-card.sky::before   { background: linear-gradient(90deg, var(--accent-sky), transparent); }
.kpi-card.sky:hover     { box-shadow: var(--glow-sky); }

.kpi-card.violet { border-color: rgba(167,139,250,0.25); }
.kpi-card.violet::before { background: linear-gradient(90deg, var(--accent-violet), transparent); }
.kpi-card.violet:hover   { box-shadow: var(--glow-violet); }

.kpi-card.amber { border-color: rgba(251,191,36,0.25); }
.kpi-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), transparent); }
.kpi-card.amber:hover   { box-shadow: var(--glow-amber); }

.kpi-label {
    font-size: 0.7rem; color: var(--text-muted);
    font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 12px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem; font-weight: 700;
    color: var(--text-primary); line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-sub   { font-size: 0.72rem; color: var(--text-muted); margin-top: 8px; line-height: 1.4; }
.kpi-icon  {
    position: absolute; right: 20px; top: 18px;
    font-size: 2.2rem; opacity: 0.12;
    transition: opacity 0.2s;
}
.kpi-card:hover .kpi-icon { opacity: 0.22; }
.kpi-glow-dot {
    position: absolute; bottom: -20px; right: -20px;
    width: 80px; height: 80px; border-radius: 50%;
    opacity: 0.06; blur: 20px;
    pointer-events: none;
}

/* ═══ SECTION HEADERS ════════════════════════════════════════ */
.section-header {
    display: flex; align-items: center; gap: 12px;
    font-family: 'Outfit', 'Inter', sans-serif;
    font-size: 1.15rem; font-weight: 700;
    color: var(--text-primary);
    padding: 12px 16px 12px 18px;
    margin: 30px 0 18px 0;
    background: linear-gradient(90deg, rgba(212,168,67,0.06) 0%, transparent 70%);
    border-left: 3px solid var(--accent-gold);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    animation: fadeSlideUp 0.35s ease both;
}
.section-header-icon {
    font-size: 1.2rem;
    background: rgba(212,168,67,0.12);
    width: 34px; height: 34px;
    border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ═══ INSIGHT CARDS ══════════════════════════════════════════ */
.insight-card {
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin: 10px 0;
    border-left: 3px solid;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeSlideUp 0.35s ease both;
}
.insight-card:hover {
    transform: translateX(4px);
}
.insight-info {
    background: linear-gradient(90deg, rgba(56,189,248,0.07) 0%, rgba(56,189,248,0.03) 100%);
    border-color: var(--accent-sky);
    box-shadow: -3px 0 12px rgba(56,189,248,0.1);
}
.insight-warning {
    background: linear-gradient(90deg, rgba(251,191,36,0.09) 0%, rgba(251,191,36,0.03) 100%);
    border-color: var(--accent-amber);
    box-shadow: -3px 0 12px rgba(251,191,36,0.12);
}
.insight-critical {
    background: linear-gradient(90deg, rgba(244,63,94,0.09) 0%, rgba(244,63,94,0.03) 100%);
    border-color: var(--accent-rose);
    box-shadow: -3px 0 12px rgba(244,63,94,0.12);
}
.insight-icon  { font-size: 1.4rem; }
.insight-title {
    font-size: 0.93rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: 0.01em;
}
.insight-msg   {
    font-size: 0.84rem; color: var(--text-secondary);
    margin-top: 5px; line-height: 1.6;
}

/* ═══ HEALTH BADGE ═══════════════════════════════════════════ */
.health-strong {
    background: linear-gradient(135deg, rgba(0,200,150,0.15), rgba(0,200,150,0.08));
    border: 1px solid rgba(0,200,150,0.4); color: #34d399;
    box-shadow: 0 0 16px rgba(0,200,150,0.15);
}
.health-moderate {
    background: linear-gradient(135deg, rgba(251,191,36,0.15), rgba(251,191,36,0.08));
    border: 1px solid rgba(251,191,36,0.4); color: #fcd34d;
    box-shadow: 0 0 16px rgba(251,191,36,0.15);
}
.health-weak {
    background: linear-gradient(135deg, rgba(244,63,94,0.15), rgba(244,63,94,0.08));
    border: 1px solid rgba(244,63,94,0.4); color: #fb7185;
    box-shadow: 0 0 16px rgba(244,63,94,0.15);
}
.health-badge {
    display: inline-flex; align-items: center; gap: 8px;
    border-radius: 30px; padding: 8px 20px;
    font-size: 0.92rem; font-weight: 700;
    letter-spacing: 0.05em; margin: 6px 0;
    font-family: 'Outfit', sans-serif;
}

/* ═══ VALIDATION BOXES ═══════════════════════════════════════ */
.val-ok {
    background: linear-gradient(90deg, rgba(0,200,150,0.08), rgba(0,200,150,0.03));
    border: 1px solid rgba(0,200,150,0.30);
    border-left: 3px solid var(--accent-emerald);
    border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0;
}
.val-warn {
    background: linear-gradient(90deg, rgba(251,191,36,0.08), rgba(251,191,36,0.03));
    border: 1px solid rgba(251,191,36,0.28);
    border-left: 3px solid var(--accent-amber);
    border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0;
}
.val-crit {
    background: linear-gradient(90deg, rgba(244,63,94,0.09), rgba(244,63,94,0.03));
    border: 1px solid rgba(244,63,94,0.30);
    border-left: 3px solid var(--accent-rose);
    border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0;
}
.val-info {
    background: linear-gradient(90deg, rgba(56,189,248,0.07), rgba(56,189,248,0.02));
    border: 1px solid rgba(56,189,248,0.25);
    border-left: 3px solid var(--accent-sky);
    border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0;
}
.val-label {
    font-size: 0.85rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 4px;
}
.val-text  {
    font-size: 0.82rem; color: var(--text-secondary); line-height: 1.55;
}

/* ═══ CHART INTERPRETATION BOX ═══════════════════════════════ */
.chart-interp {
    background: rgba(212,168,67,0.05);
    border-left: 3px solid var(--accent-gold-dk);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 14px 18px; margin: 10px 0 24px 0;
    font-size: 0.84rem; color: var(--text-secondary); line-height: 1.65;
}
.chart-interp b { color: var(--text-primary); }

/* ═══ BUTTONS ════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-gold-dk)) !important;
    color: #0a0f1a !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 26px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 2px 12px rgba(212,168,67,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--glow-gold) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Download button variant */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #1a3055, #0f2040) !important;
    color: var(--accent-gold-lt) !important;
    border: 1px solid rgba(212,168,67,0.30) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    padding: 10px 26px !important;
    transition: all 0.22s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent-gold) !important;
    box-shadow: var(--glow-gold) !important;
    transform: translateY(-1px) !important;
}

/* ═══ FILE UPLOADER ══════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    padding: 20px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-gold) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ═══ METRICS ════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ═══ DATAFRAMES / TABLES ═══════════════════════════════════ */
[data-testid="stDataFrame"], iframe {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ═══ EXPANDERS ══════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: var(--bg-card2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ═══ PROGRESS BAR ═══════════════════════════════════════════ */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-gold-dk), var(--accent-gold)) !important;
    border-radius: 10px !important;
}

/* ═══ ALERTS / INFO ══════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-width: 1px !important;
}

/* ═══ DIVIDER ════════════════════════════════════════════════ */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
    opacity: 0.6;
}

/* ═══ SIDEBAR SECTION LABEL ══════════════════════════════════ */
.sidebar-section {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--accent-gold) !important;
    padding: 10px 0 6px 0;
    display: flex; align-items: center; gap: 8px;
}
.sidebar-section::before {
    content: '';
    display: inline-block;
    width: 16px; height: 2px;
    background: var(--accent-gold);
    border-radius: 2px;
}

/* ═══ EMPTY STATE ════════════════════════════════════════════ */
.empty-state {
    text-align: center; padding: 80px 20px;
    animation: fadeSlideUp 0.5s ease both;
}
.empty-icon {
    font-size: 5rem; margin-bottom: 20px;
    display: block; animation: glowPulse 3s ease infinite;
}
.empty-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 10px;
}
.empty-sub {
    font-size: 1rem; color: var(--text-secondary);
    max-width: 480px; margin: 0 auto; line-height: 1.6;
}
.empty-hint {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(212,168,67,0.08);
    border: 1px solid rgba(212,168,67,0.2);
    border-radius: 20px; padding: 8px 20px;
    margin-top: 24px;
    font-size: 0.85rem; color: var(--accent-gold-lt);
    font-weight: 600;
}

/* ═══ HIDE STREAMLIT CHROME ══════════════════════════════════ */
#MainMenu { visibility: hidden !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# LIGHT MODE CSS — Clean Ivory + Gold Professional Theme
# ─────────────────────────────────────────────────────────────────────────────
LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
  --bg-primary:      #f2f5fc;
  --bg-secondary:    #e8edf7;
  --bg-card:         #ffffff;
  --bg-card2:        #f7faff;
  --bg-card3:        #eef3fd;
  --accent-gold:     #b8860b;
  --accent-gold-lt:  #d4a843;
  --accent-gold-dk:  #8b6914;
  --accent-emerald:  #059669;
  --accent-rose:     #e11d48;
  --accent-sky:      #0284c7;
  --accent-violet:   #7c3aed;
  --accent-amber:    #d97706;
  --accent-teal:     #0d9488;
  --text-primary:    #0f1c35;
  --text-secondary:  #2d3f60;
  --text-muted:      #6b7fa3;
  --border:          #cdd8ee;
  --border-light:    #b8cae0;
  --glow-gold:       0 0 22px rgba(184,134,11,0.22);
  --glow-emerald:    0 0 22px rgba(5,150,105,0.20);
  --glow-rose:       0 0 22px rgba(225,29,72,0.20);
  --glow-sky:        0 0 22px rgba(2,132,199,0.20);
  --glow-violet:     0 0 22px rgba(124,58,237,0.20);
  --glow-amber:      0 0 22px rgba(217,119,6,0.20);
  --radius-sm:  8px;
  --radius-md:  12px;
  --radius-lg:  18px;
  --radius-xl:  24px;
}

@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGold {
  0%, 100% { box-shadow: 0 0 0 0 rgba(184,134,11,0); }
  50%       { box-shadow: 0 0 0 8px rgba(184,134,11,0.10); }
}
@keyframes glowPulse {
  0%, 100% { opacity: 0.6; }
  50%       { opacity: 1.0; }
}

html, body {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Outfit', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(184,134,11,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(2,132,199,0.03) 0%, transparent 50%) !important;
    background-attachment: fixed !important;
}
[data-testid="stMain"] { background: transparent !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent-gold-dk), var(--border-light));
    border-radius: 10px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #edf2fd 0%, #e4eaf7 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.08) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 2px rgba(184,134,11,0.12) !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    text-transform: none !important; font-size: 0.85rem !important;
    letter-spacing: 0 !important; color: var(--text-primary) !important;
}

/* NAVIGATION TABS */
.stTabs > div:first-child,
[data-testid="stTabs"] > div:first-child,
div[class*="stTabs"] > div:first-child { background: transparent !important; }

.stTabs [data-baseweb="tab-list"], [role="tablist"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 6px 8px !important; gap: 4px !important;
    margin-bottom: 28px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.8) !important;
    display: flex !important; overflow-x: auto !important; scrollbar-width: none !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }

.stTabs [data-baseweb="tab"], [role="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 9px 18px !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    white-space: nowrap !important; cursor: pointer !important; outline: none !important;
}
.stTabs [data-baseweb="tab"]:hover, [role="tab"]:hover {
    background: rgba(184,134,11,0.08) !important;
    color: var(--accent-gold) !important;
    transform: translateY(-1px) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"],
[role="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-gold-dk)) !important;
    color: #ffffff !important; font-weight: 700 !important;
    box-shadow: var(--glow-gold), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-1px) !important;
}
.stTabs [data-baseweb="tab-highlight"], [data-baseweb="tab-highlight"] {
    background: transparent !important; height: 0 !important; display: none !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 4px !important; }

/* HERO */
.hero-header {
    background: linear-gradient(135deg, #ffffff 0%, #f5f8ff 50%, #eef3fd 100%);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent-gold-lt);
    border-radius: var(--radius-xl);
    padding: 36px 44px; margin-bottom: 32px;
    position: relative; overflow: hidden;
    animation: fadeSlideUp 0.5s ease both;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.9);
}
.hero-header::before {
    content: '';
    position: absolute; top: -80px; right: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(184,134,11,0.07) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.hero-header::after {
    content: '';
    position: absolute; bottom: -60px; left: 10%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(2,132,199,0.04) 0%, transparent 60%);
    border-radius: 50%; pointer-events: none;
}
.hero-logo-wrap { display: flex; align-items: center; gap: 16px; margin-bottom: 14px; }
.hero-logo {
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-gold-dk));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; box-shadow: var(--glow-gold);
    animation: pulseGold 3s ease infinite;
}
.hero-title {
    font-family: 'Outfit', 'Inter', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #8b6914 0%, #b8860b 40%, #0f1c35 70%, #0284c7 100%);
    background-size: 400px 100%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; line-height: 1.15;
    animation: shimmer 4s linear infinite;
}
.hero-sub { font-size: 0.97rem; color: var(--text-secondary); margin-top: 8px; line-height: 1.6; max-width: 680px; }
.hero-badges { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(184,134,11,0.08); border: 1px solid rgba(184,134,11,0.28);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.73rem; color: var(--accent-gold-dk); font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.hero-badge.sky { background: rgba(2,132,199,0.07); border-color: rgba(2,132,199,0.22); color: #0369a1; }
.hero-badge.emerald { background: rgba(5,150,105,0.07); border-color: rgba(5,150,105,0.22); color: #047857; }
.hero-stats { display: flex; gap: 28px; margin-top: 22px; padding-top: 18px; border-top: 1px solid rgba(0,0,0,0.06); }
.hero-stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: var(--accent-gold); }
.hero-stat-lbl { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px; }

/* KPI CARDS */
.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0; }
@media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 22px 24px;
    position: relative; overflow: hidden;
    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1), box-shadow 0.25s cubic-bezier(0.4,0,0.2,1);
    animation: fadeSlideUp 0.4s ease both; cursor: default;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.kpi-card:hover { transform: translateY(-5px); }
.kpi-card.gold   { border-color: rgba(184,134,11,0.25); }
.kpi-card.gold::before   { background: linear-gradient(90deg, var(--accent-gold), transparent); }
.kpi-card.gold:hover     { box-shadow: var(--glow-gold), 0 8px 24px rgba(0,0,0,0.08); }
.kpi-card.emerald { border-color: rgba(5,150,105,0.25); }
.kpi-card.emerald::before { background: linear-gradient(90deg, var(--accent-emerald), transparent); }
.kpi-card.emerald:hover   { box-shadow: var(--glow-emerald), 0 8px 24px rgba(0,0,0,0.08); }
.kpi-card.rose  { border-color: rgba(225,29,72,0.25); }
.kpi-card.rose::before  { background: linear-gradient(90deg, var(--accent-rose), transparent); }
.kpi-card.rose:hover    { box-shadow: var(--glow-rose), 0 8px 24px rgba(0,0,0,0.08); }
.kpi-card.sky   { border-color: rgba(2,132,199,0.25); }
.kpi-card.sky::before   { background: linear-gradient(90deg, var(--accent-sky), transparent); }
.kpi-card.sky:hover     { box-shadow: var(--glow-sky), 0 8px 24px rgba(0,0,0,0.08); }
.kpi-card.violet { border-color: rgba(124,58,237,0.25); }
.kpi-card.violet::before { background: linear-gradient(90deg, var(--accent-violet), transparent); }
.kpi-card.violet:hover   { box-shadow: var(--glow-violet), 0 8px 24px rgba(0,0,0,0.08); }
.kpi-card.amber { border-color: rgba(217,119,6,0.25); }
.kpi-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), transparent); }
.kpi-card.amber:hover   { box-shadow: var(--glow-amber), 0 8px 24px rgba(0,0,0,0.08); }
.kpi-label { font-size: 0.7rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; font-weight: 700; color: var(--text-primary); line-height: 1; letter-spacing: -0.02em; }
.kpi-sub   { font-size: 0.72rem; color: var(--text-muted); margin-top: 8px; line-height: 1.4; }
.kpi-icon  { position: absolute; right: 20px; top: 18px; font-size: 2.2rem; opacity: 0.10; transition: opacity 0.2s; }
.kpi-card:hover .kpi-icon { opacity: 0.18; }

/* SECTION HEADERS */
.section-header {
    display: flex; align-items: center; gap: 12px;
    font-family: 'Outfit', 'Inter', sans-serif;
    font-size: 1.15rem; font-weight: 700; color: var(--text-primary);
    padding: 12px 16px 12px 18px; margin: 30px 0 18px 0;
    background: linear-gradient(90deg, rgba(184,134,11,0.07) 0%, transparent 70%);
    border-left: 3px solid var(--accent-gold);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    animation: fadeSlideUp 0.35s ease both;
}
.section-header-icon {
    font-size: 1.2rem; background: rgba(184,134,11,0.10);
    width: 34px; height: 34px; border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
}

/* INSIGHT CARDS */
.insight-card {
    border-radius: var(--radius-md); padding: 16px 20px; margin: 10px 0;
    border-left: 3px solid; backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeSlideUp 0.35s ease both;
}
.insight-card:hover { transform: translateX(4px); }
.insight-info     { background: linear-gradient(90deg, rgba(2,132,199,0.07) 0%, rgba(2,132,199,0.02) 100%); border-color: var(--accent-sky); box-shadow: -3px 0 10px rgba(2,132,199,0.08); }
.insight-warning  { background: linear-gradient(90deg, rgba(217,119,6,0.08) 0%, rgba(217,119,6,0.02) 100%); border-color: var(--accent-amber); box-shadow: -3px 0 10px rgba(217,119,6,0.08); }
.insight-critical { background: linear-gradient(90deg, rgba(225,29,72,0.07) 0%, rgba(225,29,72,0.02) 100%); border-color: var(--accent-rose); box-shadow: -3px 0 10px rgba(225,29,72,0.08); }
.insight-icon  { font-size: 1.4rem; }
.insight-title { font-size: 0.93rem; font-weight: 700; color: var(--text-primary); letter-spacing: 0.01em; }
.insight-msg   { font-size: 0.84rem; color: var(--text-secondary); margin-top: 5px; line-height: 1.6; }

/* HEALTH BADGE */
.health-strong   { background: linear-gradient(135deg, rgba(5,150,105,0.12), rgba(5,150,105,0.06)); border: 1px solid rgba(5,150,105,0.35); color: #047857; box-shadow: 0 0 14px rgba(5,150,105,0.10); }
.health-moderate { background: linear-gradient(135deg, rgba(217,119,6,0.12), rgba(217,119,6,0.06));  border: 1px solid rgba(217,119,6,0.35);  color: #b45309; box-shadow: 0 0 14px rgba(217,119,6,0.10); }
.health-weak     { background: linear-gradient(135deg, rgba(225,29,72,0.10), rgba(225,29,72,0.05));  border: 1px solid rgba(225,29,72,0.30);  color: #be123c; box-shadow: 0 0 14px rgba(225,29,72,0.10); }
.health-badge    { display: inline-flex; align-items: center; gap: 8px; border-radius: 30px; padding: 8px 20px; font-size: 0.92rem; font-weight: 700; letter-spacing: 0.05em; margin: 6px 0; font-family: 'Outfit', sans-serif; }

/* VALIDATION BOXES */
.val-ok   { background: linear-gradient(90deg, rgba(5,150,105,0.07), rgba(5,150,105,0.02));  border: 1px solid rgba(5,150,105,0.25);  border-left: 3px solid var(--accent-emerald); border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0; }
.val-warn { background: linear-gradient(90deg, rgba(217,119,6,0.07), rgba(217,119,6,0.02));  border: 1px solid rgba(217,119,6,0.22);  border-left: 3px solid var(--accent-amber);   border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0; }
.val-crit { background: linear-gradient(90deg, rgba(225,29,72,0.07), rgba(225,29,72,0.02));  border: 1px solid rgba(225,29,72,0.25);  border-left: 3px solid var(--accent-rose);    border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0; }
.val-info { background: linear-gradient(90deg, rgba(2,132,199,0.06), rgba(2,132,199,0.02));  border: 1px solid rgba(2,132,199,0.20);  border-left: 3px solid var(--accent-sky);     border-radius: var(--radius-md); padding: 14px 18px; margin: 8px 0; }
.val-label { font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
.val-text  { font-size: 0.82rem; color: var(--text-secondary); line-height: 1.55; }

/* CHART INTERP */
.chart-interp { background: rgba(184,134,11,0.05); border-left: 3px solid var(--accent-gold-dk); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: 14px 18px; margin: 10px 0 24px 0; font-size: 0.84rem; color: var(--text-secondary); line-height: 1.65; }
.chart-interp b { color: var(--text-primary); }

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-gold-dk)) !important;
    color: #ffffff !important; border: none !important;
    border-radius: var(--radius-sm) !important; font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important; padding: 10px 26px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 2px 12px rgba(184,134,11,0.20) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: var(--glow-gold) !important; filter: brightness(1.06) !important; }
.stButton > button:active { transform: translateY(0) !important; }
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #eef3fd, #e0eaf8) !important;
    color: var(--accent-gold) !important;
    border: 1px solid rgba(184,134,11,0.28) !important;
    border-radius: var(--radius-sm) !important; font-weight: 600 !important; padding: 10px 26px !important; transition: all 0.22s !important;
}
[data-testid="stDownloadButton"] > button:hover { border-color: var(--accent-gold) !important; box-shadow: var(--glow-gold) !important; transform: translateY(-1px) !important; }

/* FILE UPLOADER */
[data-testid="stFileUploader"] { background: #ffffff !important; border: 2px dashed var(--border-light) !important; border-radius: var(--radius-md) !important; padding: 20px !important; transition: border-color 0.2s !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent-gold) !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* METRICS */
[data-testid="metric-container"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; padding: 16px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: var(--text-primary) !important; font-size: 1.4rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; font-weight: 600 !important; }

/* DATAFRAMES */
[data-testid="stDataFrame"], iframe { border-radius: var(--radius-md) !important; overflow: hidden !important; border: 1px solid var(--border) !important; }

/* PROGRESS BAR */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent-gold-dk), var(--accent-gold)) !important; border-radius: 10px !important; }

/* ALERTS */
[data-testid="stAlert"] { border-radius: var(--radius-md) !important; border-width: 1px !important; }

/* DIVIDER */
.divider { border: none; border-top: 1px solid var(--border); margin: 24px 0; opacity: 0.7; }

/* SIDEBAR SECTION LABEL */
.sidebar-section { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent-gold) !important; padding: 10px 0 6px 0; display: flex; align-items: center; gap: 8px; }
.sidebar-section::before { content: ''; display: inline-block; width: 16px; height: 2px; background: var(--accent-gold); border-radius: 2px; }

/* EMPTY STATE */
.empty-state { text-align: center; padding: 80px 20px; animation: fadeSlideUp 0.5s ease both; }
.empty-icon  { font-size: 5rem; margin-bottom: 20px; display: block; animation: glowPulse 3s ease infinite; }
.empty-title { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--text-primary); margin-bottom: 10px; }
.empty-sub   { font-size: 1rem; color: var(--text-secondary); max-width: 480px; margin: 0 auto; line-height: 1.6; }
.empty-hint  { display: inline-flex; align-items: center; gap: 8px; background: rgba(184,134,11,0.07); border: 1px solid rgba(184,134,11,0.20); border-radius: 20px; padding: 8px 20px; margin-top: 24px; font-size: 0.85rem; color: var(--accent-gold); font-weight: 600; }

/* STREAMLIT EXPANDER */
.streamlit-expanderHeader { background: var(--bg-card2) !important; border-radius: var(--radius-sm) !important; color: var(--text-primary) !important; font-weight: 600 !important; }

/* HIDE CHROME */
#MainMenu { visibility: hidden !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
</style>
"""


UX_FIX_CSS = """
<style>
/* Shared UI/UX fixes applied to both light and dark themes. */

/* ── Sidebar COLLAPSE button (arrow inside the open sidebar) ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
}

/* ── Sidebar COLLAPSED CONTROL (floating ☰ button shown when sidebar is closed) ── */
/* This is the button that re-opens the sidebar — must ALWAYS be visible */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="header"] {
    visibility: visible !important;
    opacity: 1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 999999 !important;
    pointer-events: all !important;
    width: 42px !important;
    height: 42px !important;
    border-radius: 12px !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18) !important;
    cursor: pointer !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
button[kind="header"] svg {
    color: var(--text-primary) !important;
    fill: currentColor !important;
    pointer-events: none !important;
}

/* Ensure NO parent hides the collapsed control button */
section[data-testid="stSidebarCollapsedControl"],
div[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: all !important;
}

/* ── Make the white Streamlit header bar transparent so children (toggle buttons) are visible ── */
[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    pointer-events: none !important;
}


/* ── Fix sidebar inner content containers ── */
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background: transparent !important;
    padding: 0.5rem 0.8rem !important;
    overflow-y: auto !important;
}

/* Ensure sidebar text and widgets are visible */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span {
    visibility: visible !important;
    opacity: 1 !important;
}

/* ── Model status cards ── */
.model-loaded {
    background: linear-gradient(135deg, rgba(0,200,150,0.10), rgba(0,200,150,0.04));
    border: 1px solid rgba(0,200,150,0.30);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    text-align: center;
    margin: 8px 0;
}
.model-loaded-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,200,150,0.12);
    border-radius: 20px; padding: 3px 10px; margin-bottom: 8px;
}
.model-acc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 800;
    color: var(--accent-emerald); line-height: 1;
}
.model-acc-lbl {
    font-size: 0.68rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;
}
.model-error {
    background: linear-gradient(135deg, rgba(244,63,94,0.10), rgba(244,63,94,0.04));
    border: 1px solid rgba(244,63,94,0.30);
    border-radius: var(--radius-md);
    padding: 14px 16px; text-align: center; margin: 8px 0;
}
.model-error-lbl {
    font-size: 0.85rem; font-weight: 700;
    color: var(--accent-rose); margin-bottom: 4px;
}
.model-error-sub {
    font-size: 0.75rem; color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar separators ── */
.sidebar-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-light), transparent);
    margin: 12px 0; opacity: 0.6;
}
.sidebar-sep-wide {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-light), transparent);
    margin: 20px 0; opacity: 0.5;
}

/* ── Sidebar footer ── */
.sidebar-footer {
    font-size: 0.72rem; color: var(--text-muted);
    text-align: center; line-height: 1.7; padding: 10px 8px;
}
.sidebar-footer-brand {
    color: var(--accent-gold); font-weight: 700; letter-spacing: 0.04em;
}

[data-testid="stSidebar"] {
    min-width: 300px !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
}

.stButton > button,
[data-testid="stDownloadButton"] > button {
    min-height: 42px !important;
    width: auto !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    margin: 4px 0 14px 0 !important;
}

.stTabs [data-baseweb="tab-list"],
[role="tablist"] {
    position: sticky !important;
    top: 0.75rem !important;
    z-index: 50 !important;
    backdrop-filter: blur(16px) !important;
}

.stTabs [data-baseweb="tab"],
[role="tab"] {
    min-height: 40px !important;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 96px !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] button {
    color: var(--text-primary) !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] li {
    color: inherit !important;
}

@media (max-width: 900px) {
    .hero-header {
        padding: 28px 24px !important;
        border-radius: var(--radius-lg) !important;
    }

    .hero-title {
        font-size: 2rem !important;
    }

    .hero-stats {
        flex-wrap: wrap !important;
        gap: 16px !important;
    }

    .kpi-grid {
        grid-template-columns: 1fr !important;
    }

    .stTabs [data-baseweb="tab-list"],
    [role="tablist"] {
        margin-left: -0.25rem !important;
        margin-right: -0.25rem !important;
        border-radius: var(--radius-md) !important;
    }
}
</style>
"""


def get_active_css() -> str:
    """Return the CSS block matching the current theme stored in session state."""
    base_css = LIGHT_CSS if st.session_state.get("theme", "dark") == "light" else CUSTOM_CSS
    return base_css + UX_FIX_CSS


# 
# Cached artifact loader (st.cache_resource — loaded once per session)
# 
@st.cache_resource(show_spinner=False)
def _cached_load_artifacts():
    return load_artifacts()


# 
# Pure UI components (no business logic)
# 
def render_hero():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-logo-wrap">
            <div class="hero-logo">🏦</div>
            <div>
                <div class="hero-title">CA Intelligence Suite</div>
            </div>
        </div>
        <div class="hero-sub">
            AI-Powered Chartered Accountant Assistant &nbsp;·&nbsp;
            ML Transaction Classifier &nbsp;·&nbsp;
            Backward Chaining Rule Engine
        </div>
        <div class="hero-badges">
            <span class="hero-badge">🏛 Schedule III</span>
            <span class="hero-badge">⚖ Companies Act 2013</span>
            <span class="hero-badge sky">🤖 Neural Network ML</span>
            <span class="hero-badge emerald">✅ Audit-Quality Reports</span>
            <span class="hero-badge">🔖 v2.0</span>
        </div>
        <div class="hero-stats">
            <div>
                <div class="hero-stat-val">14</div>
                <div class="hero-stat-lbl">Report Sections</div>
            </div>
            <div>
                <div class="hero-stat-val">6</div>
                <div class="hero-stat-lbl">Analysis Modules</div>
            </div>
            <div>
                <div class="hero-stat-val">100+</div>
                <div class="hero-stat-lbl">Compliance Rules</div>
            </div>
            <div>
                <div class="hero-stat-val">GST+TDS</div>
                <div class="hero-stat-lbl">Tax Coverage</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(summary) -> None:
    sd         = summary.to_dict()
    npl        = sd["Net Profit / Loss"]
    pm         = sd["Profit Margin %"]
    npl_color  = "emerald" if npl >= 0 else "rose"
    npl_symbol = "▲" if npl >= 0 else "▼"
    npl_label  = "Net Profit" if npl >= 0 else "Net Loss"
    npl_icon   = "📈" if npl >= 0 else "📉"
    health, _  = classify_financial_health(pm)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card gold">
        <div class="kpi-icon">💰</div>
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value">{inr(sd['Total Income'])}</div>
        <div class="kpi-sub">📊 All revenue streams consolidated</div>
      </div>
      <div class="kpi-card rose">
        <div class="kpi-icon">💸</div>
        <div class="kpi-label">Total Expense</div>
        <div class="kpi-value">{inr(sd['Total Expense'])}</div>
        <div class="kpi-sub">📋 All expenditure heads</div>
      </div>
      <div class="kpi-card {npl_color}">
        <div class="kpi-icon">{npl_icon}</div>
        <div class="kpi-label">{npl_label}</div>
        <div class="kpi-value">{npl_symbol} {inr(abs(npl))}</div>
        <div class="kpi-sub">Margin: {pm:.1f}% &nbsp;·&nbsp; {health}</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-icon">🧾</div>
        <div class="kpi-label">Net GST Payable</div>
        <div class="kpi-value">{inr(sd['Net GST Payable'])}</div>
        <div class="kpi-sub">⚡ After input tax credit</div>
      </div>
      <div class="kpi-card sky">
        <div class="kpi-icon">🏗</div>
        <div class="kpi-label">Total Assets</div>
        <div class="kpi-value">{inr(sd['Total Assets'])}</div>
        <div class="kpi-sub">🔷 Capital employed</div>
      </div>
      <div class="kpi-card violet">
        <div class="kpi-icon">⚖</div>
        <div class="kpi-label">Total Liabilities</div>
        <div class="kpi-value">{inr(sd['Total Liabilities'])}</div>
        <div class="kpi-sub">🔹 Outstanding obligations</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_section(title: str, icon: str = "") -> None:
    icon_html = f'<span class="section-header-icon">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="section-header">{icon_html}{title}</div>',
        unsafe_allow_html=True
    )


def render_insight_card(ins: dict) -> None:
    level_class = {
        "info":     "insight-info",
        "warning":  "insight-warning",
        "critical": "insight-critical",
    }.get(ins["level"], "insight-info")
    level_tag = {
        "critical": '<span style="font-size:0.65rem;background:rgba(244,63,94,0.25);color:#fb7185;border-radius:4px;padding:2px 7px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">CRITICAL</span>',
        "warning":  '<span style="font-size:0.65rem;background:rgba(251,191,36,0.20);color:#fcd34d;border-radius:4px;padding:2px 7px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">WARNING</span>',
        "info":     '<span style="font-size:0.65rem;background:rgba(56,189,248,0.20);color:#7dd3fc;border-radius:4px;padding:2px 7px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">INFO</span>',
    }.get(ins["level"], "")
    st.markdown(f"""
    <div class="insight-card {level_class}">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <span class="insight-icon">{ins['icon']}</span>
        <span class="insight-title">{ins['title']}</span>
        {level_tag}
      </div>
      <div class="insight-msg">{ins['message']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_health_badge(health: str) -> None:
    css  = {"Strong": "health-strong", "Moderate": "health-moderate"}.get(health, "health-weak")
    icon = {"Strong": "💚", "Moderate": "🟡", "Weak": "🔴", "Loss-Making": "🚨"}.get(health, "⚪")
    sub_text = {
        "Strong":      "Profit margin ≥ 15% — Excellent financial health",
        "Moderate":    "Profit margin 5–15% — Acceptable, monitor closely",
        "Weak":        "Profit margin 0–5% — Requires immediate attention",
        "Loss-Making": "Negative margin — Critical intervention needed",
    }.get(health, "")
    st.markdown(
        f'<span class="health-badge {css}">{icon}&nbsp; Financial Health: <strong>{health}</strong></span>'
        f'<div style="font-size:0.78rem;color:#5a7099;margin:4px 0 10px 4px;">{sub_text}</div>',
        unsafe_allow_html=True
    )


def render_validation_box(severity: str, label: str, text: str) -> None:
    css  = {
        "ok":       "val-ok",
        "warn":     "val-warn",
        "warning":  "val-warn",
        "critical": "val-crit",
        "info":     "val-info",
    }.get(severity, "val-info")
    icon = {
        "ok":       "✅",
        "warn":     "⚠️",
        "warning":  "⚠️",
        "critical": "🚨",
        "info":     "ℹ️",
    }.get(severity, "ℹ️")
    st.markdown(
        f'<div class="{css}">'
        f'<div class="val-label">{icon} {label}</div>'
        f'<div class="val-text">{text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def color_category(val: str) -> str:
    colors = {"Expense": "#1e40af", "Income": "#3b82f6", "Asset": "#93c5fd", "Liability": "#60a5fa"}
    return f"color: {colors.get(val, '#e6edf3')}; font-weight: 700"


def render_data_quality_panel(df: pd.DataFrame) -> None:
    """Display the data quality validation panel with scores and issues."""
    render_section("Data Quality Analysis", "")
    score  = data_quality_score(df)
    issues = validate_data_quality(df)

    score_color = "#10b981" if score >= 80 else ("#f59e0b" if score >= 60 else "#ef4444")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:20px;margin-bottom:16px;">
        <div style="background:var(--bg-card2);border:2px solid {score_color};border-radius:12px;
                    padding:16px 24px;text-align:center;min-width:120px;">
            <div style="font-size:2rem;font-weight:800;color:{score_color};">{score}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);">Quality Score / 100</div>
        </div>
        <div style="color:var(--text-muted);font-size:0.9rem;">
            {' Data quality is high — minimal issues detected.' if score >= 80 else
             ' Moderate issues detected — review before analysis.' if score >= 60 else
             ' Poor data quality — high risk of inaccurate financial results.'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    for issue in issues:
        sev = issue["severity"]
        render_validation_box(
            sev if sev in ("ok", "warn", "critical", "info") else "info",
            issue["type"],
            f"{issue['description']}<br><i>Impact: {issue['impact']}</i>",
        )


# 
# Main Application
# 
def main():
    # ── Theme initialisation ─────────────────────────────────────────────
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    st.markdown(get_active_css(), unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    #  Sidebar — Premium Redesign
    # ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # ── Theme toggle button ───────────────────────────────────────────
        is_dark  = st.session_state["theme"] == "dark"
        tog_icon = "☀️" if is_dark else "🌙"
        tog_lbl  = "Light Mode" if is_dark else "Dark Mode"
        sep_grad = "linear-gradient(90deg,transparent,#1a3055,transparent)" if is_dark else "linear-gradient(90deg,transparent,#cdd8ee,transparent)"

        st.markdown('<div style="padding:10px 8px 2px;">', unsafe_allow_html=True)
        if st.button(f"{tog_icon} {tog_lbl}", key="theme_toggle_btn",
                     help="Switch between Dark and Light mode"):
            st.session_state["theme"] = "light" if is_dark else "dark"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Brand header
        st.markdown("""
        <div style="text-align:center; padding:12px 8px 16px;">
            <div style="
                width:64px; height:64px; border-radius:18px;
                background:linear-gradient(135deg,#d4a843,#a07830);
                display:inline-flex; align-items:center; justify-content:center;
                font-size:2rem; box-shadow: var(--glow-gold);
                margin-bottom:14px;
            ">🏦</div>
            <div style="
                font-family:'Outfit',sans-serif;
                font-size:1.15rem; font-weight:800;
                color: var(--accent-gold);
                letter-spacing:0.01em;
            ">CA Intelligence</div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:4px; letter-spacing:0.04em;">
                AI Accounting Assistant · v2.0
            </div>
        </div>
        <div class="sidebar-sep" style="margin:4px 0 20px;"></div>
        """, unsafe_allow_html=True)

        # Upload section
        st.markdown('<div class="sidebar-section">📂 Upload Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

        # Company details
        st.markdown('<div class="sidebar-section">🏢 Company Details</div>', unsafe_allow_html=True)
        if "company_name" not in st.session_state:
            st.session_state["company_name"] = "ABC Private Limited"
        if "fy_year" not in st.session_state:
            st.session_state["fy_year"] = "2024-25"

        company_name = st.text_input("Company Name",   key="company_name")
        fy_year      = st.text_input("Financial Year", key="fy_year")

        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

        # Options
        st.markdown('<div class="sidebar-section">⚙️ Options</div>', unsafe_allow_html=True)
        use_predictions = st.checkbox("Use AI Predictions for analysis", value=True)

        st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

        # Model status
    
        st.markdown('<div class="sidebar-section">🤖 Model Status</div>', unsafe_allow_html=True)
        artifacts = _cached_load_artifacts()
        if artifacts and "error" not in artifacts:
            metrics = artifacts.get("metrics", {})
            acc     = metrics.get("test_accuracy", 0)
            st.markdown(f"""
            <div class="model-loaded">
                <div class="model-loaded-badge">
                    <span style="width:6px;height:6px;background:var(--accent-emerald);border-radius:50%;display:inline-block;animation:glowPulse 2s ease infinite;"></span>
                    <span style="color:var(--accent-emerald);font-size:0.68rem;font-weight:700;letter-spacing:0.08em;">LOADED</span>
                </div>
                <div class="model-acc">{acc*100:.1f}%</div>
                <div class="model-acc-lbl">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-error">
                <div style="font-size:1.8rem;margin-bottom:6px;">⚠️</div>
                <div class="model-error-lbl">Model Not Found</div>
                <div class="model-error-sub">Run train_model.py first</div>
            </div>
            """, unsafe_allow_html=True)
            use_predictions = False

        # Footer
        st.markdown("""
        <div class="sidebar-sep-wide"></div>
        <div class="sidebar-footer">
            🧠 Neural Networks + Backward Chaining<br>
            📜 Schedule III · Companies Act 2013<br>
            <span class="sidebar-footer-brand">CA Intelligence Suite v2.0</span>
        </div>
        """, unsafe_allow_html=True)

    #  Hero 
    render_hero()

    #  Load data 
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.success(f" Loaded **{uploaded_file.name}** — {len(df):,} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f" Failed to load file: {e}")

    if df is None:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">🏦</span>
            <div class="empty-title">No Company Ledger Uploaded</div>
            <div class="empty-sub">
                Upload your financial dataset (CSV or Excel) using the sidebar panel
                to begin your AI-powered CA Analysis.
            </div>
            <div class="empty-hint">
                📂 Drag &amp; drop or click to upload · CSV / XLSX / XLS supported
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    #  AI Prediction 
    cat_col = "Category"
    if use_predictions and artifacts and "error" not in artifacts:
        with st.spinner("🧠 Running AI classification with hybrid fallback..."):
            df = predict_categories(df, artifacts)
        cat_col = "Predicted_Category"

    #  Tabs 
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋  Data Preview",
        "🤖  AI Predictions",
        "📊  Dashboard",
        "📈  Visualizations",
        "🔍  CA Insights",
        "📄  Report",
    ])

    # 
    # TAB 1: Data Preview + Data Quality
    # 
    with tab1:
        render_section("Raw Data Preview", "")
        cols_to_show = [c for c in df.columns if c not in ["Description_clean", "Amount_scaled", "Payment_Mode_enc", "Category_enc"]]
        display_df   = df[cols_to_show].head(500)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",    f"{len(df):,}")
        c2.metric("Columns",       len(cols_to_show))
        c3.metric("Date Range",    f"{df['Date'].min()} → {df['Date'].max()}" if "Date" in df.columns else "N/A")
        c4.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        if "Category" in display_df.columns:
            st.dataframe(display_df.style.map(color_category, subset=["Category"]), use_container_width=True, height=380)
        else:
            st.dataframe(display_df, use_container_width=True, height=380)

        st.download_button(" Download Processed CSV", data=df.to_csv(index=False).encode(),
                           file_name="processed_ledger.csv", mime="text/csv")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_data_quality_panel(df)

    # 
    # TAB 2: AI Predictions
    # 
    with tab2:
        render_section("AI Transaction Classification", "🤖")

        if "Predicted_Category" not in df.columns:
            if artifacts and "error" not in artifacts:
                if st.button(" Run AI Classification Now"):
                    with st.spinner("Classifying transactions..."):
                        df = predict_categories(df, artifacts)
                    st.success(" Classification complete!")
                    st.rerun()
            else:
                st.warning(" Model not loaded. Run `python train_model.py` first.")
        else:
            pred_counts = df["Predicted_Category"].value_counts()
            st.markdown("#### Prediction Summary")
            pcols = st.columns(4)
            for i, (cat, count) in enumerate(pred_counts.items()):
                pcols[i % 4].metric(cat, f"{count:,}", f"{count/len(df)*100:.1f}%")

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            pred_cols = [c for c in ["Date", "Description", "Amount", "Payment_Mode", "Predicted_Category", "Confidence_%", "Is_Anomaly_Predicted"] if c in df.columns]
            st.markdown(f"**Showing {min(500, len(df)):,} of {len(df):,} rows**")
            st.dataframe(df[pred_cols].head(500).style.map(color_category, subset=["Predicted_Category"]),
                         use_container_width=True, height=420)

            if "Confidence_%" in df.columns:
                low_conf = df[df["Confidence_%"] < 60]
                if not low_conf.empty:
                    st.warning(f" {len(low_conf)} transactions have confidence < 60% — manually review these.")

            if "Is_Anomaly_Predicted" in df.columns:
                anomalies = df[df["Is_Anomaly_Predicted"] == True]
                if not anomalies.empty:
                    st.error(f" {len(anomalies)} transactions flagged as anomalous by Isolation Forest.")
                    st.dataframe(anomalies[pred_cols].head(100), use_container_width=True)

        if artifacts and "metrics" in (artifacts or {}):
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            render_section("Model Performance", "")
            metrics   = artifacts["metrics"]
            ca, cb, cc = st.columns(3)
            ca.metric("Test Accuracy",  f"{metrics.get('test_accuracy', 0)*100:.2f}%")
            cb.metric("Epochs Run",     metrics.get("epochs_run", "N/A"))
            cc.metric("Input Features", metrics.get("input_dim", "N/A"))

            import os
            from config import ARTIFACTS_DIR
            cm_path   = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
            hist_path = os.path.join(ARTIFACTS_DIR, "training_history.png")
            if os.path.exists(cm_path) and os.path.exists(hist_path):
                ic1, ic2 = st.columns(2)
                with ic1: st.image(cm_path,   caption="Confusion Matrix",   use_column_width=True)
                with ic2: st.image(hist_path, caption="Training History",   use_column_width=True)

    # 
    # TAB 3: Dashboard
    # 
    with tab3:
        render_section("Financial Dashboard", "")

        with st.spinner("Running Backward Chaining Rule Engine..."):
            try:
                summary = compute_financials(df, cat_col=cat_col)
                sd      = summary.to_dict()
                pm      = sd["Profit Margin %"]
                health, _ = classify_financial_health(pm)

                render_health_badge(health)
                st.markdown("<br>", unsafe_allow_html=True)
                render_kpi_cards(summary)
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                # Phase 2 Validations
                st.markdown("####  Financial Validations")
                vcol1, vcol2, vcol3 = st.columns(3)

                with vcol1:
                    pm_sev = "ok" if pm >= 15 else ("warn" if pm >= 5 else "critical")
                    render_validation_box(pm_sev, f"Profit Margin: {pm:.2f}%",
                        f"Classification: {health}. " +
                        ("Above 15% — Strong." if pm >= 15 else ("5–15% — Moderate." if pm >= 5 else "Below 5% — Weak.")))

                with vcol2:
                    bv = validate_balance_sheet(summary)
                    render_validation_box("ok" if bv["balanced"] else "critical",
                        "Balance Sheet: " + (" Balanced" if bv["balanced"] else " Mismatch"), bv["note"])

                with vcol3:
                    tv = validate_tax_provision(summary)
                    render_validation_box(tv["status"] if tv["status"] in ("ok", "warning", "critical") else "info",
                        "Tax Provision", tv["finding"])

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                col_left, col_right = st.columns(2)
                with col_left:
                    render_section("Income Breakdown", "")
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Total Revenue | {inr(sd['Total Income'])} |
                    | Revenue from Ops | {inr(summary.revenue_from_operations)} |
                    | Other Income | {inr(summary.other_income)} |
                    | Output GST | {inr(sd['GST Collected (Output)'])} |
                    | Transactions | {len(summary.income_txns):,} |
                    """)

                with col_right:
                    render_section("Expense Breakdown", "")
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Total Expenses | {inr(sd['Total Expense'])} |
                    | Cost of Materials | {inr(summary.cost_of_materials)} |
                    | Employee Benefits | {inr(summary.employee_benefit_expense)} |
                    | Finance Costs | {inr(summary.finance_costs)} |
                    | Depreciation | {inr(summary.depreciation_amortisation)} |
                    | Other Expenses | {inr(summary.other_expenses)} |
                    | Input GST Credit | {inr(sd['GST Paid (Input Credit)'])} |
                    """)

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                cf = estimate_cash_flow(summary, df)
                render_section("Cash Flow Insight", "")
                cf1, cf2, cf3, cf4 = st.columns(4)
                cf1.metric("Operating Cash Flow", inr(cf["operating_cf"]))
                cf2.metric("Cash on Hand",         inr(cf["cash_on_hand"]))
                cf3.metric("Net Cash Flow",         inr(cf["net_cf"]))
                cf4.metric("Liquidity",             cf["liquidity"])
                render_validation_box(
                    {"Strong": "ok", "Moderate": "warn", "Weak": "critical"}.get(cf["liquidity"], "info"),
                    f"Liquidity Assessment: {cf['liquidity']}", cf["interpretation"],
                )

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                comparative = compute_comparative_analysis(df, cat_col=cat_col)
                render_section("Comparative Analysis (YoY)", "")
                if comparative and comparative.get("available"):
                    cc1, cc2, cc3 = st.columns(3)
                    rg, eg, pg = comparative["revenue_growth_pct"], comparative["expense_growth_pct"], comparative["profit_growth_pct"]
                    cc1.metric("Revenue Growth",  inr(comparative["current_revenue"]),
                               f"{'' if rg and rg > 0 else ''} {abs(rg):.1f}% vs {comparative['prev_fy']}" if rg is not None else "N/A",
                               delta_color="normal" if rg and rg > 0 else "inverse")
                    cc2.metric("Expense Growth",  inr(comparative["current_expense"]),
                               f"{'' if eg and eg > 0 else ''} {abs(eg):.1f}% vs {comparative['prev_fy']}" if eg is not None else "N/A",
                               delta_color="inverse" if eg and eg > 0 else "normal")
                    cc3.metric("Profit Trend",    inr(comparative["current_profit"]),
                               f"{'' if pg and pg > 0 else ''} {abs(pg):.1f}% — {comparative['trend']}" if pg is not None else comparative["trend"],
                               delta_color="normal" if pg and pg > 0 else "inverse")
                else:
                    st.info(f"ℹ {comparative['note'] if comparative else 'Date data unavailable for trend analysis.'}")

            except Exception as e:
                st.error(f"Rule engine error: {e}")
                import traceback; st.code(traceback.format_exc())

    # 
    # TAB 4: Visualizations
    # 
    with tab4:
        render_section("Financial Visualizations", "")
        try:
            if "summary" not in dir():
                summary = compute_financials(df, cat_col=cat_col)

            # ── Chart 1: Sankey Flow ──────────────────────────────────────────
            # Replaces: chart_income_vs_expense + chart_gst_breakdown
            render_section("Money Flow — Sankey Diagram", "")
            st.plotly_chart(
                chart_sankey_flow(df, cat_col),
                use_container_width=True,
            )
            st.markdown(
                '<div class="chart-interp">'
                '<b>What this chart shows:</b> End-to-end money flow from income '
                'sources through every expense category and sub-category, '
                'terminating at GST paid and net retained amounts.<br>'
                '<b>What it means:</b> A disproportionately thick GST outlet relative '
                'to net flow signals high indirect-tax burden — review ITC claims '
                'and GST slab applicability. Thin income-side flows highlight '
                'revenue concentration risk.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Chart 2: Network Graph ────────────────────────────────────────
            # Replaces: chart_category_bar + chart_payment_mode
            render_section("Transaction Network — Categories · Sub-Categories · Payment Modes", "")
            st.plotly_chart(
                chart_network_graph(df, cat_col),
                use_container_width=True,
            )
            st.markdown(
                '<div class="chart-interp">'
                '<b>What this chart shows:</b> A three-ring network where Category '
                'hubs sit at centre (circle nodes), Sub-Categories orbit at mid-ring '
                '(blue dots), and Payment Modes anchor the outer ring (amber diamonds). '
                'Node size = total transaction volume; edge opacity = flow strength.<br>'
                '<b>What it means:</b> Dense edges to a single Category node reveal '
                'spending concentration. Amber (Cash) Payment-Mode nodes with thick '
                'connections to Expense sub-categories trigger Section 40A(3) '
                'disallowance risk — digitise those payment channels immediately.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Chart 3: Sunburst Hierarchy ───────────────────────────────────
            # Replaces: chart_expense_distribution_pie + chart_subcategory_expense
            render_section("Spend Hierarchy — Sunburst (GST Intensity)", "")
            st.plotly_chart(
                chart_sunburst_hierarchy(df, cat_col),
                use_container_width=True,
            )
            st.markdown(
                '<div class="chart-interp">'
                '<b>What this chart shows:</b> Three-level drill-down: innermost '
                'ring = Category, middle ring = Sub-Category, outer ring = Payment '
                'Mode. Sector size = rupee amount; colour (blue → amber → red) = '
                'GST ratio intensity. Click any sector to zoom in.<br>'
                '<b>What it means:</b> Red-tinted outer sectors carry the highest '
                'GST burden as a proportion of spend — prime candidates for ITC '
                'reconciliation. Large grey (uncoloured) sectors have zero GST data '
                'and may require invoice-level verification for completeness.'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Chart 4: Monthly Trend (retained) ────────────────────────────
            render_section("Monthly Income vs Expense Trend", "")
            chart_monthly_trend(df, cat_col)

        except Exception as e:
            st.error(f"Visualization error: {e}")
            import traceback; st.code(traceback.format_exc())

    # 
    # TAB 5: CA Insights
    # 
    with tab5:
        render_section("CA Financial Insights & Compliance", "")
        try:
            if "summary" not in dir():
                summary = compute_financials(df, cat_col=cat_col)

            insights     = generate_compliance_insights(summary, df)
            compliance   = compute_compliance_score(insights)

            # Compliance score gauge
            col_g, col_m = st.columns([1, 2])
            with col_g:
                chart_compliance_score_gauge(compliance["compliance_score"], compliance["grade"])
            with col_m:
                st.markdown("#### Compliance Summary")
                cg1, cg2, cg3, cg4 = st.columns(4)
                cg1.metric(" Critical",  compliance["critical_count"])
                cg2.metric(" Warnings",  compliance["warning_count"])
                cg3.metric(" Info",      compliance["info_count"])
                cg4.metric(" Score",     f"{compliance['compliance_score']}/100")

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            if not insights:
                st.info("No insights generated — upload a larger dataset.")
            else:
                critical  = [i for i in insights if i["level"] == "critical"]
                warnings_ = [i for i in insights if i["level"] == "warning"]
                info_ins  = [i for i in insights if i["level"] == "info"]

                if critical:
                    st.markdown("####  Critical Alerts")
                    for ins in critical: render_insight_card(ins)
                if warnings_:
                    st.markdown("#### Warnings")
                    for ins in warnings_: render_insight_card(ins)
                if info_ins:
                    st.markdown("####  Informational")
                    for ins in info_ins: render_insight_card(ins)

        except Exception as e:
            st.error(f"Insights engine error: {e}")
            import traceback; st.code(traceback.format_exc())

    # 
    # TAB 6: Professional CA Report (PDF)
    # 
    with tab6:
        render_section("Professional CA Report — 14 Sections (PDF)", "")
        try:
            if "summary" not in dir():
                summary = compute_financials(df, cat_col=cat_col)
                sd      = summary.to_dict()
            if "insights" not in dir():
                insights = generate_compliance_insights(summary, df)

            now_str  = pd.Timestamp.now().strftime("%d %B %Y, %I:%M %p")
            pm       = sd["Profit Margin %"]
            health, _ = classify_financial_health(pm)
            npl      = sd["Net Profit / Loss"]
            critical  = [i for i in insights if i["level"] == "critical"]
            warnings_ = [i for i in insights if i["level"] == "warning"]

            st.markdown("""
            <div style="background:rgba(37,99,235,0.08); border:1px solid rgba(37,99,235,0.25);
                        border-radius:14px; padding:24px 28px; margin-bottom:20px;">
                <div style="font-size:1.35rem; font-weight:800; color:#93c5fd; margin-bottom:4px;">
                     Professional CA Financial Report — 14 Sections
                </div>
                <div style="color:#8b949e; font-size:0.85rem; margin-bottom:16px;">
                    Schedule III · Companies Act 2013 · Audit-Quality · Client-Ready
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Profit / Loss",  inr(npl), f"{pm:.1f}% margin", delta_color="normal" if npl >= 0 else "inverse")
            c2.metric("Financial Health",   health)
            c3.metric("Total Transactions", f"{len(df):,}")
            c4.metric("Compliance Alerts",  f"{len(critical)} critical · {len(warnings_)} warnings")

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(37,99,235,0.06); border-radius:10px; padding:16px 20px;
                        border-left:4px solid #3b82f6; margin-bottom:16px;">
                <div style="color:#93c5fd; font-weight:700; margin-bottom:8px;"> Report Sections</div>
                <div style="color:#8b949e; font-size:0.83rem; line-height:2.0;">
                    1. Executive Summary (CA Advisory Tone) &nbsp;|&nbsp;
                    2. Profit &amp; Loss Statement &nbsp;|&nbsp;
                    3. Balance Sheet (Validated, A=E+L) &nbsp;|&nbsp;
                    4. Comparative Analysis (YoY) &nbsp;|&nbsp;
                    5. Cash Flow Insight &nbsp;|&nbsp;
                    6. GST Analysis &nbsp;|&nbsp;
                    7. Financial Ratios (Formula + Interpretation) &nbsp;|&nbsp;
                    8. Expense Analysis (Top 3) &nbsp;|&nbsp;
                    9. Graphical Analysis (Bar + Pie) &nbsp;|&nbsp;
                    10. Compliance Analysis (/🟠/) &nbsp;|&nbsp;
                    11. Anomaly Detection &nbsp;|&nbsp;
                    12. CA Recommendations (Data-Backed) &nbsp;|&nbsp;
                    13. Conclusion &nbsp;|&nbsp;
                    14. Disclaimer
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner(" Generating 14-section CA report..."):
                pdf_bytes = generate_pdf_report(
                    sd, df, insights, summary, now_str,
                    company_name=company_name, fy_year=fy_year,
                )

            st.download_button(
                label=" Download Full CA Report — Schedule III (PDF)",
                data=pdf_bytes,
                file_name=f"CA_Report_{company_name.replace(' ','_')}_{fy_year}.pdf",
                mime="application/pdf",
                key="dl_ca_report",
            )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            render_section("Report Preview", "")
            prev1, prev2, prev3 = st.columns(3)
            with prev1:
                st.markdown("** P&L Summary**")
                st.table(pd.DataFrame([
                    {"Particulars": "Revenue from Ops", "Amount": inr(summary.revenue_from_operations)},
                    {"Particulars": "Other Income",      "Amount": inr(summary.other_income)},
                    {"Particulars": "Total Revenue",     "Amount": inr(sd["Total Income"])},
                    {"Particulars": "Total Expenses",    "Amount": inr(sd["Total Expense"])},
                    {"Particulars": "Profit Before Tax", "Amount": inr(summary.profit_before_tax)},
                    {"Particulars": "Tax Provision",     "Amount": inr(summary.short_term_provisions)},
                    {"Particulars": "Net Profit/(Loss)", "Amount": inr(npl)},
                    {"Particulars": "Profit Margin",     "Amount": f"{pm:.2f}%"},
                ]))
            with prev2:
                st.markdown("** Balance Sheet**")
                st.table(pd.DataFrame([
                    {"Item": "Total Assets",       "Value": inr(sd["Total Assets"])},
                    {"Item": "Total Liabilities",  "Value": inr(sd["Total Liabilities"])},
                    {"Item": "Shareholders Equity","Value": inr(sd["Shareholders Funds"])},
                    {"Item": "Net GST Payable",    "Value": inr(sd["Net GST Payable"])},
                ]))
            with prev3:
                st.markdown("** Key Ratios**")
                exp_r  = summary.total_expense / max(summary.total_income, 1)
                d_e = (summary.short_term_borrowings + summary.long_term_borrowings) / max(summary.shareholders_funds, 1)
                c_r = summary.current_assets / max(summary.current_liabilities, 1)
                st.table(pd.DataFrame([
                    {"Ratio": "Profit Margin",  "Value": f"{pm:.1f}%"},
                    {"Ratio": "Expense Ratio",  "Value": f"{exp_r*100:.1f}%"},
                    {"Ratio": "Debt-to-Equity", "Value": f"{d_e:.2f}x"},
                    {"Ratio": "Current Ratio",  "Value": f"{c_r:.2f}x"},
                    {"Ratio": "Health Status",  "Value": health},
                ]))

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("** Compliance Insights Preview**")
            for ins in insights[:8]:
                render_insight_card(ins)

        except Exception as e:
            st.error(f"Report generation error: {e}")
            import traceback; st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
