import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
DB_PATH = "sentiment_system.db"
REFRESH_INTERVAL = 7 

st.set_page_config(
    page_title="BullBear Radar | Professional Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED UI/UX CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #05070a; }
    .stApp { background: radial-gradient(circle at 50% 50%, #11151c 0%, #05070a 100%); }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; transition: transform 0.3s ease; }
    .header-container { display: flex; justify-content: space-between; align-items: center; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 2rem; }
    .live-indicator { background: rgba(0, 255, 150, 0.1); color: #00ff96; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid #00ff96; animation: pulse-glow 2s infinite; }
    @keyframes pulse-glow { 0% { box-shadow: 0 0 0 0 rgba(0, 255, 150, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(0, 255, 150, 0); } 100% { box-shadow: 0 0 0 0 rgba(0, 255, 150, 0); } }
    @keyframes pulse-red { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); } }
    .drift-active { animation: pulse-red 1.5s infinite; border: 2px solid #ff4b4b !important; }
    .metric-title { color: #808495; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #ffffff; font-family: 'JetBrains Mono', monospace; }
    .alert-card { padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid; background: rgba(255,255,255,0.02); }
    .alert-critical { border-left-color: #ff4b4b; box-shadow: 0 0 15px rgba(255, 75, 75, 0.2); }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data(ttl=3)
def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT p.title, p.subreddit, p.sector, p.company, s.final_score, s.timestamp FROM posts p JOIN sentiment_data s ON p.id = s.post_id ORDER BY s.id DESC LIMIT 500", conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_drift = pd.read_sql_query("SELECT * FROM drift_log ORDER BY id DESC LIMIT 50", conn)
        df_drift['timestamp'] = pd.to_datetime(df_drift['timestamp'])
        conn.close()
        return df, df_drift
    except: return pd.DataFrame(), pd.DataFrame()

# --- VIZ HELPERS ---
def plot_hero(df, df_drift):
    df_p = df.sort_values('timestamp')
    fig = go.Figure(go.Scatter(x=df_p['timestamp'], y=df_p['final_score'], name='Sentiment', line=dict(color='#00ff96', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 255, 150, 0.05)'))
    drifts = df_drift[df_drift['drift_detected'] == 1]
    if not drifts.empty:
        fig.add_trace(go.Scatter(x=drifts['timestamp'], y=df_p['final_score'].tail(len(drifts)), mode='markers', name='DRIFT', marker=dict(color='#ff4b4b', size=15, symbol='star-diamond')))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0, r=0, t=20, b=0), transition={'duration': 1000})
    return fig

def plot_gauge(score):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "white"}, 'steps': [{'range': [-1, -0.3], 'color': '#ff4b4b'}, {'range': [-0.3, 0.3], 'color': '#f1c40f'}, {'range': [0.3, 1], 'color': '#00ff96'}]}))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=0), paper_bgcolor='rgba(0,0,0,0)', transition={'duration': 1000})
    return fig

def plot_sparkline(data, color):
    fig = go.Figure(go.Scatter(y=data, mode='lines', line=dict(color=color, width=2)))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0), height=30, width=80, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- COMPONENTS ---
@st.fragment(run_every=REFRESH_INTERVAL)
def render_main():
    df, df_drift = load_data()
    if df.empty: return
    
    st.markdown(f"""<div class="header-container">
        <div style="font-size: 1.5rem; font-weight: 700; color: #fff;">🐂🐻 BullBear Radar <span style="color:#808495; font-weight:400;">| Intelligence Terminal</span></div>
        <div style="display:flex; align-items:center; gap:20px;"><div class="live-indicator">🟢 LIVE</div><div style="color:#808495; font-family:'JetBrains Mono';">{datetime.now().strftime('%H:%M:%S')}</div></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Market Overview", "🧠 Drift Intelligence"])

    with tab1:
        avg_s = df['final_score'].mean()
        drift_act = df_drift.iloc[0]['drift_detected'] == 1 if not df_drift.empty else False
        
        # 1. KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f"""<div class="glass-card"><div class="metric-title">Mood</div><div class="metric-value" style="color:{'#00ff96' if avg_s > 0.05 else ('#ff4b4b' if avg_s < -0.05 else '#f1c40f')}">{'BULLISH' if avg_s > 0.05 else ('BEARISH' if avg_s < -0.05 else 'NEUTRAL')}</div></div>""", unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="glass-card"><div class="metric-title">Score</div><div class="metric-value">{avg_s:.3f}</div></div>""", unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="glass-card {'drift-active' if drift_act else ''}"><div class="metric-title">Drift</div><div class="metric-value" style="color:{'#ff4b4b' if drift_act else '#00ff96'}">{'ALERT' if drift_act else 'STABLE'}</div></div>""", unsafe_allow_html=True)
        
        leading = "N/A"
        if not df['sector'].dropna().empty:
            leading = df.groupby('sector')['final_score'].mean().idxmax().capitalize()
        with c4: st.markdown(f"""<div class="glass-card"><div class="metric-title">Leading</div><div class="metric-value" style="color:#AB63FA">{leading}</div></div>""", unsafe_allow_html=True)

        # 2. Hero
        st.write("")
        col_hero, col_gauge = st.columns([2.2, 1])
        with col_hero: st.plotly_chart(plot_hero(df, df_drift), use_container_width=True, key="hero_chart_v2", config={'displayModeBar': False})
        with col_gauge: st.plotly_chart(plot_gauge(avg_s), use_container_width=True, key="gauge_chart_v2", config={'displayModeBar': False})

        # 3. Ticker Trends
        st.divider()
        tc1, tc2 = st.columns([1.5, 1])
        with tc1:
            st.markdown("### 🏢 TOP TICKER TRENDS")
            top_cos = df['company'].dropna().value_counts().head(5).index
            if len(top_cos) == 0: st.info("No major tickers identified in current stream.")
            for ticker in top_cos:
                tic_data = df[df['company'] == ticker].head(20).sort_values('timestamp')
                t_avg = tic_data['final_score'].mean()
                cols = st.columns([1, 1.5, 1, 1.2])
                with cols[0]: st.markdown(f"**${ticker}**")
                with cols[1]: st.markdown(f"<span style='color:#808495; font-size:0.8rem;'>{len(tic_data)} signals</span>", unsafe_allow_html=True)
                with cols[2]: st.plotly_chart(plot_sparkline(tic_data['final_score'], '#00ff96' if t_avg > 0 else '#ff4b4b'), use_container_width=False, key=f"spark_v2_{ticker}")
                with cols[3]: st.markdown(f"<span style='color:{'#00ff96' if t_avg > 0 else '#ff4b4b'}; font-weight:700;'>{t_avg:+.2f}</span>", unsafe_allow_html=True)
                st.markdown("<div style='border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom:5px;'></div>", unsafe_allow_html=True)
        
        with tc2:
            st.markdown("### 🧠 SENTIMENT DRIVERS")
            words = " ".join(df['title'].astype(str)).lower().split()
            stops = {'the', 'to', 'is', 'in', 'and', 'a', 'of', 'for', 'on', 'with', 'at', 'this', 'that', 'it', 'from', 'stock', 'market'}
            filtered = [w for w in words if w.isalnum() and len(w) > 3 and w not in stops]
            if filtered:
                kws = pd.Series(filtered).value_counts().head(8)
                for w, c in kws.items():
                    st.markdown(f"<div style='display:flex; justify-content:space-between; padding:4px 0;'><span>{w}</span><span style='color:#00ff96;'>{c}</span></div>", unsafe_allow_html=True)
            else: st.info("Analyzing drivers...")

    with tab2:
        st.markdown("### 🧠 DRIFT DIAGNOSTICS & EXPLAINABILITY")
        if df_drift.empty:
            st.info("No drift events logged yet.")
        else:
            # 1. Latest Drift Alert
            latest = df_drift.iloc[0]
            if latest['drift_detected'] == 1:
                st.error(f"⚠️ **REGIME SHIFT DETECTED** at {latest['timestamp']}")
                st.markdown(f"""<div class="glass-card" style="border-left: 5px solid #ff4b4b;">
                    <strong>SHAP/LIME Explanation:</strong><br>{latest.get('explanation', 'Identifying core drivers...')}
                </div>""", unsafe_allow_html=True)
            else:
                st.success("Current sentiment distribution is stable within 1-hour historical bounds.")

            # 2. Metric Breakdown
            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.metric("PSI Stability", f"{latest['psi']:.3f}", delta=f"{latest['psi']-0.1:.2f}", delta_color="inverse")
            m2.metric("KL Divergence", f"{latest['kl_divergence']:.3f}")
            m3.metric("Ensemble Score", f"{latest['ensemble_score']:.2f}")

            # 3. Drift History
            st.write("")
            st.markdown("#### 📜 DRIFT EVENT HISTORY")
            history_df = df_drift[df_drift['drift_detected'] == 1][['timestamp', 'ensemble_score', 'explanation']].head(10)
            if not history_df.empty:
                st.table(history_df)
            else:
                st.info("No historical regime shifts recorded in the current session.")

@st.fragment(run_every=REFRESH_INTERVAL)
def render_sidebar():
    _, df_drift = load_data()
    st.markdown("### 🔔 LIVE SIGNALS")
    if not df_drift.empty and df_drift.iloc[0]['drift_detected'] == 1:
        st.markdown(f"""<div class="alert-card alert-critical"><strong>⚠️ MARKET DRIFT</strong><br>Regime shift detected ({df_drift.iloc[0]['ensemble_score']:.2f}).</div>""", unsafe_allow_html=True)
    else:
        st.success("System stable. Normal volatility.")

# --- APP LAYOUT ---
with st.sidebar:
    render_sidebar()

render_main()