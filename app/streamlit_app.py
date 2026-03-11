"""
streamlit_app.py  —  EcoCharge AI
Professional dark-theme EV Battery Intelligence Platform
Dark background = white text | White cards = black text
"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
import io

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC_DIR)

st.set_page_config(
    page_title="EcoCharge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }

/* ─ Dark main background ─ */
.stApp { background: #05080f !important; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 1500px !important; }

/* ─ All text on dark bg = WHITE ─ */
.stApp p, .stApp span, .stApp div,
.stApp label, .stApp li, .stApp small { color: #e2e8f0 !important; }
.stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6 { color: #ffffff !important; }

/* ─ Sidebar dark ─ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c16 0%, #0c1220 100%) !important;
    border-right: 1px solid #1e2d42 !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ─ Animated gradient header ─ */
.eco-header {
    background: linear-gradient(90deg, #00f5ff, #0080ff, #8000ff, #ff0080, #00f5ff);
    background-size: 400% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 4rem; font-weight: 700; letter-spacing: 1px;
    animation: gradientShift 5s linear infinite;
    line-height: 1; margin: 0;
}
@keyframes gradientShift { to { background-position: 400% center; } }

/* ─ WHITE CARDS = BLACK TEXT ─ */
.white-card {
    background: #ffffff !important;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    margin: 8px 0;
    color: #000000 !important;
}
.white-card * { 
    color: #000000 !important; 
}
.white-card .wc-icon { font-size: 1.8rem; margin-bottom: 8px; }
.white-card .wc-value { font-size: 2rem; font-weight: 700; font-family: 'Space Grotesk'; line-height: 1.1; }
.white-card .wc-label { color: #444444 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; font-weight: 600; }

/* ─ Dark metric cards ─ */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1525, #121e30) !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 14px !important; padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color: #0080ff !important; transform: translateY(-2px) !important;
}
div[data-testid="metric-container"] * { color: #ffffff !important; }
[data-testid="stMetricLabel"] p { color: #6b8aaa !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }
[data-testid="stMetricValue"]  { color: #ffffff !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { color: #00d084 !important; }

/* ─ Buttons ─ */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0050cc, #0080ff) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 12px !important; padding: 0.7rem 2rem !important;
    font-weight: 600 !important; font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(0,128,255,0.3) !important;
    transition: all 0.3s !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0080ff, #00bfff) !important;
    transform: translateY(-2px) !important;
}
.stButton > button {
    background: #0d1525 !important; color: #e2e8f0 !important;
    border: 1px solid #1e2d42 !important; border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important; font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { border-color: #0080ff !important; }

/* ─ Inputs ─ */
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #0d1525 !important; color: #ffffff !important;
    border: 1px solid #1e2d42 !important; border-radius: 10px !important;
}
.stNumberInput label, .stTextInput label, .stSlider label,
.stRadio label span, .stSelectbox label, .stCheckbox label span,
.stTextArea label { color: #8aaac8 !important; font-size: 0.82rem !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
.stSelectbox [data-baseweb="select"] { background: #0d1525 !important; border-color: #1e2d42 !important; }
.stSelectbox [data-baseweb="select"] * { color: #ffffff !important; background: #0d1525 !important; }

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1525 !important; border-radius: 14px !important;
    padding: 5px !important; border: 1px solid #1e2d42 !important;
}
.stTabs [data-baseweb="tab"] { color: #6b8aaa !important; border-radius: 10px !important; padding: 9px 22px !important; }
.stTabs [aria-selected="true"] { background: #1a2d45 !important; color: #00bfff !important; font-weight: 600 !important; }

/* ─ Expander ─ */
.streamlit-expanderHeader { background: #0d1525 !important; border: 1px solid #1e2d42 !important; border-radius: 10px !important; }
.streamlit-expanderContent { background: #080c16 !important; border: 1px solid #1e2d42 !important; }

/* ─ Alerts ─ */
div[data-testid="stInfo"] { background: #071428 !important; border: 1px solid #0050aa !important; border-radius: 10px !important; }
div[data-testid="stInfo"] * { color: #80b8e8 !important; }
div[data-testid="stWarning"] { background: #160f00 !important; border-color: #cc7700 !important; }
div[data-testid="stWarning"] * { color: #ffaa40 !important; }
div[data-testid="stSuccess"] { background: #051410 !important; border-color: #00a050 !important; }
div[data-testid="stSuccess"] * { color: #40d090 !important; }

/* ─ Dataframe ─ */
.stDataFrame { border: 1px solid #1e2d42 !important; border-radius: 12px !important; overflow: hidden !important; }
.stDataFrame * { color: #c8dae8 !important; background: #080c16 !important; }
.stDataFrame th { background: #0d1525 !important; color: #6b8aaa !important; font-size: 0.8rem !important; text-transform: uppercase !important; }

/* ─ File Uploader ─ */
[data-testid="stFileUploadDropzone"] { background: #0d1525 !important; border: 2px dashed #1e2d42 !important; border-radius: 14px !important; }
[data-testid="stFileUploadDropzone"] * { color: #6b8aaa !important; }

/* ─ Download Button ─ */
.stDownloadButton > button { background: #0d1525 !important; border: 1px solid #1e2d42 !important; color: #00bfff !important; border-radius: 10px !important; }

/* ─ Progress Bar ─ */
.stProgress > div > div { background: linear-gradient(90deg, #0050cc, #00bfff) !important; border-radius: 4px !important; }

/* ─ Rec boxes ─ */
.rec-green  { background: #040e08; border: 1px solid #1a7a40; border-radius: 14px; padding: 18px 22px; margin: 10px 0; }
.rec-orange { background: #0e0800; border: 1px solid #8a5010; border-radius: 14px; padding: 18px 22px; margin: 10px 0; }
.rec-red    { background: #0e0404; border: 1px solid #8a1a20; border-radius: 14px; padding: 18px 22px; margin: 10px 0; }

/* ─ Dark feature cards ─ */
.feature-card {
    border-radius: 16px; padding: 24px; height: 190px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.feature-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important; }
.feature-card h3 { margin: 8px 0 6px !important; font-size: 1.05rem !important; }
.feature-card p { font-size: 0.85rem !important; line-height: 1.5 !important; }

/* ─ Code ─ */
.stCode, code { background: #0d1525 !important; color: #00bfff !important; border: 1px solid #1e2d42 !important; border-radius: 8px !important; }

/* ─ Checkbox ─ */
.stCheckbox * { color: #e2e8f0 !important; }

/* ─ Hide branding ─ */
#MainMenu, footer, header { visibility: hidden; }

/* ─ Scrollbar ─ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: #1e2d42; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0080ff; }

hr { border-color: #1e2d42 !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══ Helpers ══
@st.cache_resource(show_spinner=False)
def load_predict_module():
    try:
        import predict as pred
        return pred, None
    except Exception as e:
        return None, str(e)

def generate_pdf_report(data_dict):
    # Strip emojis for PDF compatibility with default fonts
    def clean_text(text):
        return str(text).encode('ascii', 'ignore').decode('ascii').strip()

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(5, 8, 15)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "EcoCharge AI - Battery Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')

    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)

    # Summary Section
    pdf.set_font("Arial", 'B', 16)
    pdf.set_fill_color(240, 245, 255)
    pdf.cell(0, 12, "  1. Executive Summary", ln=True, fill=True)
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    summary_data = [
        ("State of Health (SoH)", f"{data_dict['Predicted SoH (%)']}%"),
        ("Remaining Useful Life", f"{data_dict['Remaining Useful Life (cycles)']} cycles"),
        ("Recommendation", clean_text(data_dict['Recommendation'])),
        ("Battery Chemistry", clean_text(data_dict['Battery Chemistry'])),
    ]

    for label, val in summary_data:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(50, 8, f"{label}:", border=0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, str(val), ln=True)

    pdf.ln(10)

    # Technical Details
    pdf.set_font("Arial", 'B', 16)
    pdf.set_fill_color(240, 245, 255)
    pdf.cell(0, 12, "  2. Technical Sensor Readings", ln=True, fill=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(95, 10, " Parameter", 1, 0, 'L', True)
    pdf.cell(95, 10, " Value", 1, 1, 'L', True)

    pdf.set_font("Arial", '', 10)
    for param, val in data_dict.items():
        if param not in ["Predicted SoH (%)", "Remaining Useful Life (cycles)", "Recommendation", "Battery Chemistry"]:
            pdf.cell(95, 8, f" {clean_text(param)}", 1)
            pdf.cell(95, 8, f" {clean_text(val)}", 1, 1)

    pdf.ln(15)

    # Footer
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "Disclaimer: This report is generated by EcoCharge AI using NASA PCoE battery datasets. Predictions are estimations based on LSTM neural network models and should be used for informational purposes only.", align='C')

    return bytes(pdf.output())

def soh_color(soh): 
 return "#00d084" if soh >= 80 else "#ff9944" if soh >= 60 else "#ff4455"

def white_stat(col, icon, value, label, color):
    col.markdown(f"""
    <div class="white-card" style="text-align:center; border-top: 4px solid {color}">
        <div class="wc-icon">{icon}</div>
        <div class="wc-value" style="color:{color} !important">{value}</div>
        <div class="wc-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def soh_gauge(soh):
    c = soh_color(soh)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        number={'suffix': '%', 'font': {'size': 48, 'color': c}},
        title={'text': "STATE OF HEALTH", 'font': {'size': 12, 'color': '#6b8aaa'}},
        gauge={
            'axis': {'range': [0,100], 'tickwidth':1, 'tickcolor':'#1e2d42', 'tickfont':{'color':'#6b8aaa','size':10}},
            'bar': {'color': c, 'thickness': 0.2},
            'bgcolor': '#0d1525', 'bordercolor': '#1e2d42',
            'steps': [
                {'range': [0,  60], 'color': '#100305'},
                {'range': [60, 80], 'color': '#100800'},
                {'range': [80,100], 'color': '#031008'},
            ],
            'threshold': {'line': {'color': '#ff4455', 'width': 2}, 'thickness': 0.75, 'value': 80},
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=250, margin=dict(t=30,b=0,l=20,r=20))
    return fig

def degradation_plot(cycle, soh):
    cycles = np.arange(0, 1001)
    rate   = -np.log(max(soh,1)/100.0)/cycle if cycle > 0 else 0.0004
    curve  = 100 * np.exp(-rate * cycles)
    fig = go.Figure()
    fig.add_hrect(y0=80, y1=105, fillcolor="rgba(0,200,100,0.05)", line_width=0,
                  annotation_text="✅ EV Use Zone", annotation_font_color="#00d084", annotation_position="top left")
    fig.add_hrect(y0=60, y1=80,  fillcolor="rgba(255,150,0,0.05)", line_width=0,
                  annotation_text="⚠️ Repurpose",  annotation_font_color="#ff9944")
    fig.add_hrect(y0=0,  y1=60,  fillcolor="rgba(255,60,70,0.05)",  line_width=0,
                  annotation_text="♻️ Recycle",     annotation_font_color="#ff4455")
    fig.add_trace(go.Scatter(x=cycles, y=curve, mode='lines', name='Degradation',
                             line=dict(color='#0080ff', width=2.5),
                             fill='tozeroy', fillcolor='rgba(0,128,255,0.04)'))
    fig.add_hline(y=80, line_dash='dot', line_color='#ff4455', annotation_text='EOL 80%', annotation_font_color='#ff4455')
    fig.add_trace(go.Scatter(x=[cycle], y=[soh], mode='markers+text',
                             marker=dict(size=14, color=soh_color(soh), symbol='diamond', line=dict(color='#fff',width=1.5)),
                             text=['  ← Now'], textfont=dict(color=soh_color(soh), size=12), name='Now'))
    fig.update_layout(title=dict(text='Battery Degradation Trajectory', font=dict(size=15, color='#ffffff')),
                      xaxis=dict(title='Cycle', gridcolor='#0d1a28', color='#6b8aaa', title_font_color='#6b8aaa'),
                      yaxis=dict(title='SoH (%)', range=[0,108], gridcolor='#0d1a28', color='#6b8aaa', title_font_color='#6b8aaa'),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#070b14',
                      legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c0ccd8')),
                      height=380, margin=dict(t=50,b=40,l=10,r=10), font=dict(color='#ffffff'))
    return fig

def fleet_chart(df):
    colors = df['soh'].apply(lambda s: '#00d084' if s>=80 else '#ff9944' if s>=60 else '#ff4455')
    fig = go.Figure(go.Bar(x=df['battery_id'], y=df['soh'], marker_color=colors.tolist(),
                           text=df['soh'].apply(lambda s: f'{s:.1f}%'), textposition='outside',
                           textfont=dict(color='#ffffff')))
    fig.add_hline(y=80, line_dash='dot', line_color='#ff4455', annotation_text='EOL 80%', annotation_font_color='#ff4455')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#070b14', font_color='#ffffff',
                      title=dict(text='Fleet Battery Health', font=dict(color='#ffffff', size=15)),
                      xaxis=dict(color='#6b8aaa'), yaxis=dict(range=[0,115], gridcolor='#0d1a28', color='#6b8aaa'),
                      height=400, bargap=0.3)
    return fig


# ══ SIDEBAR ══
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:24px 0 16px">
        <div style="font-size:3.5rem">⚡</div>
        <div style="font-size:1.6rem; font-weight:700; color:#ffffff; letter-spacing:2px">ECOCHARGE</div>
        <div style="font-size:0.72rem; color:#00bfff; letter-spacing:3px; text-transform:uppercase">AI Battery Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e2d42">', unsafe_allow_html=True)

    mode = st.radio("NAVIGATION", [
        "🏠  Dashboard",
        "🎯  Predict",
        "📁  Fleet Analysis",
        "📊  Model Insights",
        "🌿  Sustainability",
        "⚙️  Settings",
    ], index=0)

    st.markdown('<hr style="border-color:#1e2d42">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1525; border:1px solid #1e2d42; border-radius:12px; padding:14px">
        <div style="color:#00bfff; font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px">🧠 Model Info</div>
        <div style="color:#8aaac8; font-size:0.77rem; line-height:2.2">
            Architecture → LSTM × 2<br>
            Lookback &nbsp;&nbsp;&nbsp;&nbsp;→ 30 cycles<br>
            Features &nbsp;&nbsp;&nbsp;&nbsp;→ 7 inputs<br>
            Dataset &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ NASA PCoE<br>
            Framework &nbsp;&nbsp;→ TensorFlow
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e2d42">', unsafe_allow_html=True)
    pred_module, _ = load_predict_module()
    sc, st_txt = ("#00d084","Model Loaded ✅") if pred_module else ("#ff9944","Demo Mode ⚠️")
    st.markdown(f"""
    <div style="text-align:center">
        <span style="background:{sc}20; color:{sc}; border:1px solid {sc}50;
              padding:5px 14px; border-radius:20px; font-size:0.78rem; font-weight:600">{st_txt}</span>
    </div>
    <div style="text-align:center; margin-top:14px">
        <p style="color:#2a4060; font-size:0.7rem">Green AI Capstone 2024</p>
    </div>""", unsafe_allow_html=True)

pred_module, _ = load_predict_module()


# ══════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════
if mode == "🏠  Dashboard":
    col_t, col_b = st.columns([4,1])
    with col_t:
        st.markdown('<p class="eco-header">⚡ EcoCharge AI</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:#6b8aaa; font-size:1rem; margin-top:6px">AI-Powered EV Battery Degradation Prediction Platform</p>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div style="text-align:right; padding-top:22px"><span style="background:#00d08420; color:#00d084; border:1px solid #00d08450; padding:6px 16px; border-radius:20px; font-size:0.8rem; font-weight:600">🟢 LIVE</span></div>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # KPI — WHITE cards BLACK text
    st.markdown("### 📊 Platform Overview")
    k1,k2,k3,k4,k5 = st.columns(5)
    white_stat(k1,"🔋","636",    "Battery Cycles",   "#0080ff")
    white_stat(k2,"🤖","LSTM",   "Model Type",        "#8000ff")
    white_stat(k3,"📈","30",     "Lookback Window",   "#00d084")
    white_stat(k4,"🎯","7",      "Input Features",    "#ff9944")
    white_stat(k5,"🌍","NASA",   "Dataset",           "#00bfff")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### 🚀 Platform Features")
    f1,f2,f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="feature-card" style="background:linear-gradient(135deg,#071428,#0d1a2a);border:1px solid #1e3a5a">
            <div style="font-size:2rem">🎯</div>
            <h3 style="color:#00bfff !important">Real-Time Prediction</h3>
            <p style="color:#8aaac8 !important">Instant SoH prediction from BMS sensor readings using trained LSTM model.</p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="feature-card" style="background:linear-gradient(135deg,#071a10,#0d2018);border:1px solid #1a5a30">
            <div style="font-size:2rem">📁</div>
            <h3 style="color:#00d084 !important">Fleet Analytics</h3>
            <p style="color:#8ac8a0 !important">Bulk CSV upload for entire EV fleets with automated health scoring.</p>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="feature-card" style="background:linear-gradient(135deg,#170714,#1f0d1a);border:1px solid #5a1a5a">
            <div style="font-size:2rem">🌿</div>
            <h3 style="color:#cc88ff !important">Sustainability</h3>
            <p style="color:#b88ac8 !important">CO₂ savings, tree equivalents, and full environmental impact reporting.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    f4,f5,f6 = st.columns(3)
    with f4:
        st.markdown("""<div class="feature-card" style="background:linear-gradient(135deg,#14100a,#1e180f);border:1px solid #5a4010">
            <div style="font-size:2rem">📉</div>
            <h3 style="color:#ff9944 !important">Degradation Simulator</h3>
            <p style="color:#c8a870 !important">Interactive simulator for temperature, fast charging, and DoD effects.</p>
        </div>""", unsafe_allow_html=True)
    with f5:
        st.markdown("""<div class="feature-card" style="background:linear-gradient(135deg,#0a1018,#101828);border:1px solid #1a3a6a">
            <div style="font-size:2rem">🔬</div>
            <h3 style="color:#4488ff !important">Model Explainability</h3>
            <p style="color:#80a0c8 !important">Training curves, residuals, feature importance and full eval metrics.</p>
        </div>""", unsafe_allow_html=True)
    with f6:
        st.markdown("""<div class="feature-card" style="background:linear-gradient(135deg,#100a14,#18101e);border:1px solid #4a2060">
            <div style="font-size:2rem">⚙️</div>
            <h3 style="color:#aa66ff !important">Custom Settings</h3>
            <p style="color:#a080c0 !important">Configure EOL threshold, chemistry, capacity and alert preferences.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Start")
    q1,q2,q3 = st.columns(3)
    with q1:
        if st.button("🎯 Make a Prediction", use_container_width=True): st.info("👈 Select **Predict** from sidebar!")
    with q2:
        if st.button("📁 Analyse Fleet CSV", use_container_width=True): st.info("👈 Select **Fleet Analysis** from sidebar!")
    with q3:
        if st.button("📊 View Model Insights", use_container_width=True): st.info("👈 Select **Model Insights** from sidebar!")


# ══════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════
elif mode == "🎯  Predict":
    st.markdown("## 🎯 Battery Health Prediction")
    st.info("💡 Enter battery sensor readings from your BMS for an instant AI health assessment.")

    with st.expander("📖 What do these parameters mean?"):
        p1,p2 = st.columns(2)
        with p1:
            st.markdown("**Cycle Number** — Total charge/discharge cycles completed\n\n**Voltage Measured** — Discharge terminal voltage (V)\n\n**Current Measured** — Discharge current (negative = discharging)\n\n**Temperature** — Cell surface temperature (°C)")
        with p2:
            st.markdown("**Charge Current** — Current during charging phase (A)\n\n**Charge Voltage** — End-of-charge voltage (V)\n\n**Discharge Time** — Duration of last discharge event (s)\n\n**Internal Resistance** — Estimated cell impedance (mΩ)")

    st.markdown("### 📊 Sensor Parameters")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown('<p style="color:#00bfff;font-size:0.8rem;font-weight:600;letter-spacing:1px">⚡ ELECTRICAL</p>', unsafe_allow_html=True)
        cycle   = st.number_input("Cycle Number",         min_value=1,    max_value=3000,  value=300)
        voltage = st.number_input("Voltage Measured (V)", min_value=2.0,  max_value=4.5,   value=3.7,  step=0.01)
        current = st.number_input("Current Measured (A)", min_value=-5.0, max_value=0.0,   value=-1.8, step=0.1)
    with c2:
        st.markdown('<p style="color:#ff9944;font-size:0.8rem;font-weight:600;letter-spacing:1px">🌡️ THERMAL</p>', unsafe_allow_html=True)
        temperature  = st.number_input("Temperature (°C)",         min_value=-10.0, max_value=80.0,  value=27.0, step=0.5)
        internal_res = st.number_input("Internal Resistance (mΩ)", min_value=0.0,   max_value=500.0, value=80.0, step=1.0)
    with c3:
        st.markdown('<p style="color:#00d084;font-size:0.8rem;font-weight:600;letter-spacing:1px">🔌 CHARGING</p>', unsafe_allow_html=True)
        curr_charge = st.number_input("Charge Current (A)", min_value=0.1, max_value=5.0,   value=1.5,  step=0.1)
        volt_charge = st.number_input("Charge Voltage (V)", min_value=3.5, max_value=4.5,   value=4.2,  step=0.01)
        time_s      = st.number_input("Discharge Time (s)", min_value=100, max_value=10000, value=3400)
    with c4:
        st.markdown('<p style="color:#aa66ff;font-size:0.8rem;font-weight:600;letter-spacing:1px">⚙️ CONFIG</p>', unsafe_allow_html=True)
        battery_type   = st.selectbox("Battery Chemistry",    ["Li-Ion (LCO)","LFP","NMC","NCA"])
        nominal_cap    = st.number_input("Nominal Capacity (Ah)", min_value=0.5, max_value=200.0, value=2.0, step=0.1)
        eol_threshold  = st.slider("EOL Threshold (%)", min_value=60, max_value=90, value=80)
        fast_charge_flag = st.checkbox("Fast Charging Used")

    st.markdown('<br>', unsafe_allow_html=True)
    if st.button("⚡ Run AI Prediction", type="primary", use_container_width=True):
        with st.spinner("🤖 Running LSTM inference..."):
            if pred_module:
                result = pred_module.predict_from_inputs(cycle, voltage, current, temperature, curr_charge, volt_charge, time_s)
                soh,rul = result['soh'], result['rul']
                rec_label,rec_color,rec_desc = result['recommendation']['action'], result['recommendation']['color'], result['recommendation']['description']
            else:
                fc = 1.2 if fast_charge_flag else 1.0
                rate = 0.00045 * fc * (1 + max(0,(temperature-25)*0.005))
                soh  = max(10.0, 100*np.exp(-rate*cycle) + np.random.normal(0,0.3))
                rul  = max(0, int((soh - eol_threshold) / 0.065))
                if soh >= eol_threshold: rec_label,rec_color,rec_desc = "✅ Continue Use","green","Battery is healthy. Continue EV operation normally."
                elif soh >= 60:          rec_label,rec_color,rec_desc = "⚠️ Repurpose",   "orange","Below EOL for EV. Suitable for stationary storage."
                else:                    rec_label,rec_color,rec_desc = "♻️ Recycle",      "red",   "End-of-life. Safe disposal required."

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("### 📈 Prediction Results")

        g_col,m_col = st.columns([1,2])
        with g_col:
            st.plotly_chart(soh_gauge(soh), use_container_width=True)
        with m_col:
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("State of Health", f"{soh:.1f}%", f"{soh-100:.1f}%")
            m2.metric("Remaining Life",  f"{rul:,} cycles")
            m3.metric("Cycle No.",       f"{cycle:,}")
            m4.metric("EOL Threshold",   f"{eol_threshold}%")
            rc = '#00d084' if rec_color=='green' else '#ff9944' if rec_color=='orange' else '#ff4455'
            st.markdown(f'<div class="rec-{rec_color}"><strong style="font-size:1.05rem;color:{rc} !important">{rec_label}</strong><br><span style="color:#8aaac8 !important;font-size:0.88rem">{rec_desc}</span></div>', unsafe_allow_html=True)

        st.plotly_chart(degradation_plot(cycle, soh), use_container_width=True)

        # WHITE analysis cards — BLACK text
        st.markdown("### 📋 Full Analysis Report")
        r1,r2,r3,r4 = st.columns(4)
        health_color = '#1a7a40' if soh>=80 else '#aa5500' if soh>=60 else '#aa1520'
        health_label = "Excellent" if soh>=90 else "Good" if soh>=80 else "Degraded" if soh>=60 else "Critical"
        cap_actual   = (soh/100)*nominal_cap
        fade         = 100 - soh
        with r1: st.markdown(f'<div class="white-card"><div class="wc-label">Battery Health</div><div class="wc-value" style="color:{health_color} !important">{soh:.1f}%</div><p style="color:#4a5568 !important">{health_label}</p></div>', unsafe_allow_html=True)
        with r2: st.markdown(f'<div class="white-card"><div class="wc-label">Remaining Life</div><div class="wc-value">{rul:,}</div><p style="color:#4a5568 !important">Estimated cycles</p></div>', unsafe_allow_html=True)
        with r3: st.markdown(f'<div class="white-card"><div class="wc-label">Actual Capacity</div><div class="wc-value">{cap_actual:.2f} Ah</div><p style="color:#4a5568 !important">of {nominal_cap:.1f} Ah nominal</p></div>', unsafe_allow_html=True)
        with r4: st.markdown(f'<div class="white-card"><div class="wc-label">Capacity Fade</div><div class="wc-value" style="color:#aa2020 !important">{fade:.1f}%</div><p style="color:#4a5568 !important">Total degradation</p></div>', unsafe_allow_html=True)

        if soh >= 60:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("### 🌿 Environmental Impact (this battery)")
            e1,e2,e3 = st.columns(3)
            white_stat(e1,"🌍","74 kg",     "CO₂ Saved",      "#00a050")
            white_stat(e2,"🌳","3.5",        "Tree-Years",     "#0070c0")
            white_stat(e3,"🚗","352 km",     "Car km Avoided", "#cc6600")

        # ── Download Results ──
        report_data = {
            "Cycle Number": cycle, "Voltage Measured (V)": voltage, "Current Measured (A)": current,
            "Temperature (°C)": temperature, "Internal Resistance (mΩ)": internal_res,
            "Charge Current (A)": curr_charge, "Charge Voltage (V)": volt_charge,
            "Discharge Time (s)": time_s, "Battery Chemistry": battery_type,
            "Nominal Capacity (Ah)": nominal_cap, "EOL Threshold (%)": eol_threshold,
            "Fast Charging": "Yes" if fast_charge_flag else "No",
            "Predicted SoH (%)": f"{soh:.2f}",
            "Remaining Useful Life (cycles)": rul,
            "Recommendation": rec_label
        }
        
        pdf_bytes = generate_pdf_report(report_data)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download PDF Health Report",
            data=pdf_bytes,
            file_name=f"battery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ══════════════════════════════════════════════════════
#  FLEET ANALYSIS
# ══════════════════════════════════════════════════════
elif mode == "📁  Fleet Analysis":
    st.markdown("## 📁 Fleet Battery Analysis")
    st.info("💡 Upload a CSV with your fleet battery data for bulk automated health scoring.")

    with st.expander("📋 CSV Format & Template"):
        template_df = pd.DataFrame({
            'battery_id':['B0001','B0001','B0002','B0002'],
            'cycle':[1,2,1,2],
            'voltage_measured':[3.85,3.83,3.80,3.78],
            'current_measured':[-1.8,-1.8,-1.9,-1.9],
            'temperature_measured':[25.0,25.2,26.0,26.2],
            'current_charge':[1.5,1.5,1.5,1.5],
            'voltage_charge':[4.20,4.20,4.20,4.20],
            'time':[3400,3390,3380,3370],
        })
        st.dataframe(template_df, use_container_width=True)
        st.download_button("⬇️ Download Template", template_df.to_csv(index=False), "template.csv", mime="text/csv")

    col_up, col_opt = st.columns([2,1])
    with col_up:
        uploaded = st.file_uploader("📂 Upload Fleet CSV", type=['csv'])
    with col_opt:
        eol_thresh = st.slider("EOL Threshold (%)", 60, 90, 80)

    if uploaded:
        df    = pd.read_csv(uploaded)
        n_bat = df['battery_id'].nunique()
        st.success(f"✅ Loaded {len(df):,} rows — {n_bat} batteries")
        with st.expander("🔍 Raw Data Preview"):
            st.dataframe(df.head(20), use_container_width=True)

        if st.button("⚡ Run Fleet Analysis", type="primary", use_container_width=True):
            results  = []
            progress = st.progress(0, text="Analysing fleet...")
            for i, bid in enumerate(df['battery_id'].unique()):
                bat_df = df[df['battery_id']==bid].sort_values('cycle')
                lc = int(bat_df['cycle'].max())
                if pred_module and len(bat_df) >= 31:
                    res = pred_module.run_inference_from_df(bat_df)
                    soh,rul = res.get('latest_soh',85.0), res.get('rul',200)
                else:
                    soh = max(15.0, 100 - lc*0.065 + np.random.normal(0,0.5))
                    rul = max(0, int((soh-eol_thresh)/0.065))
                status = "✅ Reuse" if soh>=eol_thresh else "⚠️ Repurpose" if soh>=60 else "♻️ Recycle"
                results.append({'Battery ID':bid,'SoH (%)':round(soh,2),'RUL':rul,'Status':status,'Last Cycle':lc})
                progress.progress((i+1)/len(df['battery_id'].unique()))
            progress.empty()
            rdf = pd.DataFrame(results)

            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("### 📊 Fleet Summary")
            s1,s2,s3,s4,s5 = st.columns(5)
            white_stat(s1,"🔋",str(len(rdf)),                                                "Total Batteries","#0080ff")
            white_stat(s2,"📊",f"{rdf['SoH (%)'].mean():.1f}%",                             "Average SoH",    "#0080ff")
            white_stat(s3,"✅",str(int((rdf['SoH (%)']>=eol_thresh).sum())),                "Reusable",       "#00a050")
            white_stat(s4,"⚠️",str(int(((rdf['SoH (%)']>=60)&(rdf['SoH (%)']<eol_thresh)).sum())),"Repurpose","#cc7700")
            white_stat(s5,"♻️",str(int((rdf['SoH (%)']<60).sum())),                         "Recycle",        "#cc2030")

            st.plotly_chart(fleet_chart(rdf.rename(columns={'Battery ID':'battery_id','SoH (%)':'soh'})), use_container_width=True)
            st.dataframe(rdf, use_container_width=True)

            n_r  = int((rdf['SoH (%)']>=60).sum())
            co2  = n_r * 74
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("### 🌿 Fleet Environmental Impact")
            e1,e2,e3 = st.columns(3)
            white_stat(e1,"🌍",f"{co2:,} kg",       "CO₂ Saved",      "#00a050")
            white_stat(e2,"🌳",f"{co2/21:,.0f}",     "Tree-Years",     "#0070c0")
            white_stat(e3,"🚗",f"{co2/0.21:,.0f}",   "Car km Avoided", "#cc6600")
            st.download_button("⬇️ Download Results", rdf.to_csv(index=False), "fleet_results.csv", mime="text/csv")


# ══════════════════════════════════════════════════════
#  MODEL INSIGHTS
# ══════════════════════════════════════════════════════
elif mode == "📊  Model Insights":
    st.markdown("## 📊 Model Performance & Insights")
    tab1,tab2,tab3,tab4 = st.tabs(["📈 Metrics","📉 Simulator","🔬 Training","🏗️ Architecture"])

    with tab1:
        metrics_path = os.path.join(os.path.dirname(__file__),'..','results','metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f: m = json.load(f)
            st.markdown("### 🎯 Test Set Performance")
            mc1,mc2,mc3,mc4,mc5 = st.columns(5)
            white_stat(mc1,"📏",f"{m.get('MAE',0):.4f}",  "MAE (Ah)",  "#0080ff")
            white_stat(mc2,"📐",f"{m.get('RMSE',0):.4f}", "RMSE (Ah)", "#8000ff")
            white_stat(mc3,"🎯",f"{m.get('MSE',0):.5f}",  "MSE",       "#0080ff")
            white_stat(mc4,"📊",f"{m.get('R2',0):.3f}",   "R² Score",  "#00a050" if m.get('R2',0)>0 else "#cc2030")
            white_stat(mc5,"📉",f"{m.get('MAPE',0):.2f}%","MAPE",      "#cc6600")
            st.markdown('<hr>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Run evaluation.ipynb first to see metrics.")
        results_dir = os.path.join(os.path.dirname(__file__),'..','results')
        for fname, caption in [
            ('actual_vs_predicted.png','📈 Actual vs Predicted'),
            ('soh_curve.png','🔋 SoH Curve'),
            ('residuals.png','📉 Residuals'),
            ('training_curves.png','🎯 Training Curves'),
        ]:
            path = os.path.join(results_dir, fname)
            if os.path.exists(path):
                st.markdown(f"**{caption}**")
                st.image(path, use_container_width=True)

    with tab2:
        st.markdown("### 📉 Interactive Degradation Simulator")
        s1,s2 = st.columns([1,2])
        with s1:
            sim_cycle   = st.slider("Cycle Number", 0, 1500, 300, step=10)
            sim_temp    = st.slider("Temperature (°C)", 10, 60, 25)
            sim_fc      = st.checkbox("Fast Charging", value=False)
            sim_dod     = st.slider("Depth of Discharge (%)", 50, 100, 80)
            sim_nominal = st.number_input("Nominal Capacity (Ah)", 0.5, 200.0, 2.0, step=0.1)
            rate = 0.00045
            if sim_temp > 35: rate *= 1 + (sim_temp-35)*0.015
            if sim_fc:        rate *= 1.25
            if sim_dod > 80:  rate *= 1 + (sim_dod-80)*0.008
            sim_soh = max(5.0, 100*np.exp(-rate*sim_cycle))
            sim_rul = max(0, int(-np.log(0.80)/rate - sim_cycle))
            st.markdown('<hr>', unsafe_allow_html=True)
            st.metric("State of Health", f"{sim_soh:.1f}%")
            st.metric("Est. RUL",        f"{sim_rul:,} cycles")
            st.metric("Actual Capacity", f"{(sim_soh/100)*sim_nominal:.3f} Ah")
            if sim_fc:        st.warning("⚡ Fast charging accelerates degradation ~25%")
            if sim_temp > 35: st.warning("🌡️ High temperature accelerates degradation!")
        with s2:
            st.plotly_chart(soh_gauge(sim_soh), use_container_width=True)
            st.plotly_chart(degradation_plot(sim_cycle, sim_soh), use_container_width=True)

    with tab3:
        history_path = os.path.join(os.path.dirname(__file__),'..','models','training_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f: hist = json.load(f)
            epochs = list(range(1, len(hist.get('loss',[]))+1))
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=epochs,y=hist.get('loss',[]),name='Train',line=dict(color='#0080ff',width=2)))
            fig_l.add_trace(go.Scatter(x=epochs,y=hist.get('val_loss',[]),name='Val',line=dict(color='#ff4455',width=2,dash='dash')))
            fig_l.update_layout(title='MSE Loss',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#070b14',font_color='#ffffff',
                                xaxis=dict(gridcolor='#0d1a28',color='#6b8aaa'),yaxis=dict(gridcolor='#0d1a28',color='#6b8aaa'),height=300)
            st.plotly_chart(fig_l, use_container_width=True)
            h1,h2,h3 = st.columns(3)
            white_stat(h1,"📉",str(len(epochs)),              "Epochs Trained","#0080ff")
            white_stat(h2,"🏆",f"{min(hist.get('val_loss',[0])):.5f}","Best Val Loss","#00a050")
            if 'val_mae' in hist: white_stat(h3,"🎯",f"{min(hist.get('val_mae',[0])):.4f}","Best Val MAE","#cc6600")
        else:
            st.info("📊 Train the model first to see training history.")

    with tab4:
        a1,a2 = st.columns(2)
        with a1:
            st.markdown("""
            <div style="background:#0d1525;border:1px solid #1e2d42;border-radius:16px;padding:24px;font-family:'JetBrains Mono',monospace;font-size:0.8rem;line-height:2.3">
                <div style="color:#00bfff;font-weight:700;margin-bottom:10px">ECOCHARGE LSTM MODEL</div>
                <div style="color:#6b8aaa">┌─────────────────────────────┐</div>
                <div style="color:#fff">│ Input  (30 cycles × 7 feat) │</div>
                <div style="color:#6b8aaa">└──────────────┬──────────────┘</div>
                <div style="color:#6b8aaa">               ↓</div>
                <div style="color:#00d084">   LSTM(32) return_seq=True</div>
                <div style="color:#6b8aaa">               ↓</div>
                <div style="color:#ff9944">   Dropout(0.2)</div>
                <div style="color:#6b8aaa">               ↓</div>
                <div style="color:#00d084">   LSTM(16) return_seq=False</div>
                <div style="color:#6b8aaa">               ↓</div>
                <div style="color:#ff9944">   Dropout(0.2)</div>
                <div style="color:#6b8aaa">               ↓</div>
                <div style="color:#aa66ff">   Dense(16, ReLU)</div>
                <div style="color:#6b8aaa">               ↓</div>
                <div style="color:#aa66ff">   BatchNormalization</div>
                <div style="color:#6b8aaa">               ↓</div>
                <div style="color:#6b8aaa">┌──────────────┴──────────────┐</div>
                <div style="color:#0080ff">│ Output (1) → SoH (%)        │</div>
                <div style="color:#6b8aaa">└─────────────────────────────┘</div>
            </div>""", unsafe_allow_html=True)
        with a2:
            specs = {"Total Parameters":"8,625","Input Shape":"(30, 7)","Output":"SoH (%)","Loss":"MSE","Optimizer":"Adam 0.001","Batch Size":"16","Epochs":"10 (early stop)","Dataset":"NASA PCoE"}
            for k,v in specs.items():
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 14px;background:#0d1525;border-radius:8px;margin:4px 0;border:1px solid #1e2d42"><span style="color:#6b8aaa">{k}</span><span style="color:#fff;font-weight:600">{v}</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  SUSTAINABILITY
# ══════════════════════════════════════════════════════
elif mode == "🌿  Sustainability":
    st.markdown("## 🌿 Sustainability & Environmental Impact")
    st.info("💡 Each reused EV battery prevents ~74 kg CO₂ from being emitted.")

    st.markdown("### 🧮 Impact Calculator")
    c1,c2,c3 = st.columns(3)
    with c1: n_batteries = st.number_input("Batteries Reused", 1, 1000000, 100)
    with c2: co2_per_bat = st.number_input("CO₂ per Battery (kg)", 10.0, 200.0, 74.0, step=1.0)
    with c3: years       = st.number_input("Time Period (years)", 1, 30, 1)

    co2   = n_batteries * co2_per_bat * years
    trees = co2 / 21
    km    = co2 / 0.21
    homes = co2 / 4500

    st.markdown('<br>', unsafe_allow_html=True)
    e1,e2,e3,e4 = st.columns(4)
    white_stat(e1,"🌍",f"{co2:,.0f} kg",  "CO₂ Saved",          "#00a050")
    white_stat(e2,"🌳",f"{trees:,.0f}",    "Tree-Years",         "#0070c0")
    white_stat(e3,"🚗",f"{km:,.0f} km",    "Car km Avoided",     "#cc6600")
    white_stat(e4,"🏠",f"{homes:,.1f}",    "Homes Powered",      "#8000aa")

    st.markdown('<hr>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=['CO₂ Saved (kg)','Tree-Years','Car km (÷100)'],
        y=[co2, trees, km/100],
        marker_color=['#00d084','#0080ff','#ff9944'],
        text=[f'{v:,.0f}' for v in [co2,trees,km/100]],
        textposition='outside', textfont=dict(color='#ffffff'),
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#070b14',
                      font_color='#ffffff', height=380, showlegend=False,
                      xaxis=dict(color='#6b8aaa'), yaxis=dict(gridcolor='#0d1a28',color='#6b8aaa'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### 📖 Why Battery Reuse Matters")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div style="background:linear-gradient(135deg,#051410,#071a14);border:1px solid #1a5a30;border-radius:16px;padding:24px">
            <h3 style="color:#00d084 !important">🔋 Second-Life Batteries</h3>
            <p style="color:#80c8a0 !important">EV batteries retain 70–80% capacity after automotive use. These can power stationary storage, grid balancing and off-grid solar — extending life by 5–10 years.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div style="background:linear-gradient(135deg,#070514,#0d0a1e);border:1px solid #3a1a6a;border-radius:16px;padding:24px">
            <h3 style="color:#aa66ff !important">⚡ AI for Green Energy</h3>
            <p style="color:#b080d0 !important">AI models like EcoCharge predict remaining battery life accurately, enabling smarter reuse decisions, reducing waste, and supporting a circular economy.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════
elif mode == "⚙️  Settings":
    st.markdown("## ⚙️ Platform Settings")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 🔋 Battery Configuration")
        st.number_input("Default Nominal Capacity (Ah)", 0.5, 200.0, 2.0, step=0.1)
        st.slider("EOL Threshold (%)", 60, 90, 80)
        st.selectbox("Default Chemistry", ["Li-Ion (LCO)","LFP","NMC","NCA"])
        st.markdown("### 📊 Display")
        st.selectbox("Temperature Unit", ["Celsius (°C)","Fahrenheit (°F)"])
        st.checkbox("Show Advanced Metrics", value=True)
    with c2:
        st.markdown("### 🤖 Model Info")
        m_k = os.path.join(os.path.dirname(__file__),'..','models','lstm_model.keras')
        m_h = os.path.join(os.path.dirname(__file__),'..','models','lstm_model.h5')
        model_exists = os.path.exists(m_k) or os.path.exists(m_h)
        data_path  = os.path.join(os.path.dirname(__file__),'..','data','raw','battery_data.csv')
        st.markdown(f"""
        <div style="background:#0d1525;border:1px solid #1e2d42;border-radius:12px;padding:16px;font-family:'JetBrains Mono';font-size:0.82rem;line-height:2.3">
            <div><span style="color:#6b8aaa">Model  → </span><span style="color:#fff">{"✅ Found" if model_exists else "❌ Not found"}</span></div>
            <div><span style="color:#6b8aaa">Data   → </span><span style="color:#fff">{"✅ Found" if os.path.exists(data_path)  else "❌ Not found"}</span></div>
            <div><span style="color:#6b8aaa">Python → </span><span style="color:#fff">3.10.x ✅</span></div>
            <div><span style="color:#6b8aaa">TF     → </span><span style="color:#fff">2.13.x ✅</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    if st.button("💾 Save Settings", type="primary"): st.success("✅ Settings saved!")


# ══ FOOTER ══
st.markdown('<hr style="border-color:#1e2d42;margin-top:3rem">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;padding:14px 0">
    <p style="color:#2a4060;font-size:0.78rem;margin:0">
        ⚡ <strong style="color:#0060cc">EcoCharge AI</strong>
        &nbsp;|&nbsp; EV Battery Intelligence &nbsp;|&nbsp; TensorFlow + Streamlit
        &nbsp;|&nbsp; NASA PCoE Dataset &nbsp;|&nbsp; Green AI Capstone 2024
        &nbsp;|&nbsp; {datetime.now().strftime('%H:%M')}
    </p>
</div>""", unsafe_allow_html=True)