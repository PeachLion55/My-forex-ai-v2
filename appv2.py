import streamlit as st
import pandas as pd
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
import datetime as dt
import os
import json
import hashlib
import requests
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import pytz
import logging
import math
import uuid
import glob
import time
import scipy.stats
import base64
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# === TA_PRO HELPERS START ===
def _ta_safe_lower(s):
    return str(s).strip().lower().replace(" ", "_")

def _ta_human_pct(x, nd=1):
    if pd.isna(x):
        return "—"
    return f"{x*100:.{nd}f}%"

def _ta_human_num(x, nd=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"

def _ta_user_dir(user_id="guest"):
    root = os.path.join(os.path.dirname(__file__), "user_data")
    os.makedirs(root, exist_ok=True)
    d = os.path.join(root, user_id)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "community_images"), exist_ok=True)
    os.makedirs(os.path.join(d, "playbooks"), exist_ok=True)
    return d

def _ta_load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _ta_save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _ta_hash():
    return uuid.uuid4().hex[:12]

def _ta_percent_gain_to_recover(drawdown_pct):
    if drawdown_pct <= 0:
        return 0.0
    if drawdown_pct >= 0.99:
        return float("inf")
    return drawdown_pct / (1 - drawdown_pct)

def _ta_expectancy_by_group(df, group_cols):
    g = df.dropna(subset=["r"]).groupby(group_cols)
    res = g["r"].agg(
        trades="count",
        winrate=lambda s: (s>0).mean(),
        avg_win=lambda s: s[s>0].mean() if (s>0).any() else 0.0,
        avg_loss=lambda s: -s[s<0].mean() if (s<0).any() else 0.0,
        expectancy=lambda s: (s>0).mean()*(s[s>0].mean() if (s>0).any() else 0.0) - (1-(s>0).mean())*(-s[s<0].mean() if (s<0).any() else 0.0)
    ).reset_index()
    return res

def _ta_profit_factor(df):
    if "pnl" not in df.columns:
        return np.nan
    gp = df.loc[df["pnl"]>0, "pnl"].sum()
    gl = -df.loc[df["pnl"]<0, "pnl"].sum()
    if gl == 0:
        return np.nan if gp == 0 else float("inf")
    return gp / gl

def _ta_daily_pnl(df):
    if "datetime" in df.columns and "pnl" in df.columns:
        tmp = df.dropna(subset=["datetime"]).copy()
        tmp["date"] = pd.to_datetime(tmp["datetime"]).dt.date
        return tmp.groupby("date", as_index=False)["pnl"].sum()
    return pd.DataFrame(columns=["date","pnl"])

def _ta_compute_streaks(df):
    d = _ta_daily_pnl(df)
    if d.empty:
        return {"current": 0, "best": 0}
    streak = 0
    best = 0
    for pnl in d["pnl"]:
        if pnl > 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return {"current": streak, "best": best}

def _ta_show_badges(df):
    with st.expander("🏅 Gamification: Streaks & Badges", expanded=False):
        streaks = _ta_compute_streaks(df) if df is not None else {"current":0,"best":0}
        col1, col2 = st.columns(2)
        col1.metric("Current Green-Day Streak", streaks.get("current",0))
        col2.metric("Best Streak", streaks.get("best",0))
        if df is not None and "emotions" in df.columns:
            emo_logged = int((df["emotions"].fillna("").astype(str).str.len()>0).sum())
            st.caption(f"🧠 Emotion-logged trades: {emo_logged}")
# === TA_PRO HELPERS END ===

# Path to SQLite DB
DB_FILE = "users.db"

# Connect to SQLite with error handling
try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS community_data
                 (key TEXT PRIMARY KEY, data TEXT)''')
    conn.commit()
    logging.info("SQLite database initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize SQLite database: {str(e)}")
    st.error(f"Database initialization failed: {str(e)}")

def _ta_load_community(key, default=[]):
    try:
        c.execute("SELECT data FROM community_data WHERE key = ?", (key,))
        result = c.fetchone()
        if result:
            return json.loads(result[0])
        return default
    except Exception as e:
        logging.error(f"Failed to load community data for {key}: {str(e)}")
        return default

def _ta_save_community(key, data):
    try:
        json_data = json.dumps(data)
        c.execute("INSERT OR REPLACE INTO community_data (key, data) VALUES (?, ?)", (key, json_data))
        conn.commit()
        logging.info(f"Community data saved for {key}")
    except Exception as e:
        logging.error(f"Failed to save community data for {key}: {str(e)}")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide")

# ----------------- CUSTOM CSS -----------------
bg_opacity = 0.5
st.markdown(
    f"""
<style>
/* Futuristic dark background with animated grid */
.stApp {{
    background:
        radial-gradient(circle at 15% 20%, rgba(255,215,0,{bg_opacity*0.18}) 0%, transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(0,170,255,{bg_opacity*0.12}) 0%, transparent 25%),
        linear-gradient(135deg, #0b0b0b 0%, #0a0a0a 100%);
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(90deg, rgba(255,255,255,{bg_opacity*0.05}) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,{bg_opacity*0.05}) 1px, transparent 1px);
    background-size: 42px 42px, 42px 42px;
    animation: moveGrid 38s linear infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 1;
}}
@keyframes moveGrid {{
    0% {{ transform: translateY(0px); }}
    100% {{ transform: translateY(42px); }}
}}
/* Content above bg */
.main, .block-container, .stTabs, .stMarkdown, .css-ffhzg2, .css-1d391kg {{ position: relative; z-index: 1; }}
/* Enhanced tab styling for main navigation */
div[data-baseweb="tab-list"] {{
    gap: 12px;
    padding-bottom: 8px;
    background: transparent;
    border-radius: 12px;
    padding: 8px;
}}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background: linear-gradient(45deg, #FFD700, #FFA500) !important;
    color: #000 !important;
    font-weight: 600;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
    background: rgba(255,255,255,0.05) !important;
    color: #fff !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease;
}}
div[data-baseweb="tab-list"] button[aria-selected="false"]:hover {{
    background: rgba(255,255,255,0.15) !important;
    transform: translateY(-2px);
}}
/* Custom button styling */
.stButton > button {{
    background: linear-gradient(45deg, #FFD700, #FFA500);
    color: #000;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
}}
.stButton > button:hover {{
    background: linear-gradient(45deg, #FFA500, #FFD700);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}}
/* Expander styling */
.stExpander {{
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}}
/* Metric card styling */
.stMetric {{
    background: rgba(255,255,255,0.05);
    padding: 16px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}}
/* Input fields */
.stTextInput > div > div > input, .stSelectbox > div > div > select {{
    background: rgba(255,255,255,0.05);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
}}
/* Plotly chart background */
.plotly-graph-div {{
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px;
}}
/* Remove default Streamlit styling */
header, footer, [data-testid="stDecoration"] {{ display: none; }}
</style>
""",
    unsafe_allow_html=True
)

# ----------------- DEMO DATA -----------------
DEMO_CANDLES = [
    {"time": int((dt.datetime.now(pytz.UTC) - dt.timedelta(days=x)).timestamp()), "open": 1.0 + x*0.01, "high": 1.0 + x*0.015, "low": 1.0 + x*0.005, "close": 1.0 + x*0.01, "volume": 1000 + x*10}
    for x in range(100, -1, -1)
]

def update_candles():
    """Simulate live data by adding a new candle every minute"""
    global DEMO_CANDLES
    last_candle = DEMO_CANDLES[-1]
    new_time = last_candle["time"] + 60  # Add 1 minute
    last_close = last_candle["close"]
    # Simulate price movement with small random walk
    change = np.random.normal(0, 0.001)
    new_close = last_close + change
    new_open = last_close
    new_high = max(new_open, new_close) + abs(np.random.normal(0, 0.0005))
    new_low = min(new_open, new_close) - abs(np.random.normal(0, 0.0005))
    new_candle = {
        "time": new_time,
        "open": new_open,
        "high": new_high,
        "low": new_low,
        "close": new_close,
        "volume": last_candle["volume"] + np.random.randint(-5, 5)
    }
    DEMO_CANDLES.append(new_candle)
    if len(DEMO_CANDLES) > 1000:  # Keep reasonable size
        DEMO_CANDLES.pop(0)
    return new_candle

# ----------------- AUTHENTICATION -----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result and result[0] == hash_password(password):
        return True
    return False

def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)",
                  (username, hash_password(password), json.dumps({})))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "login"

# Login/Register UI
if st.session_state.user is None:
    st.session_state.page = "login"
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if check_credentials(login_username, login_password):
                st.session_state.user = login_username
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("New Username", key="reg_username")
        reg_password = st.text_input("New Password", type="password", key="reg_password")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        if st.button("Register"):
            if reg_password != reg_confirm:
                st.error("Passwords do not match")
            elif register_user(reg_username, reg_password):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists")

# ----------------- MAIN APP -----------------
if st.session_state.user is not None:
    st_autorefresh(interval=60*1000, key="datarefresh")  # Refresh every minute
    user_dir = _ta_user_dir(st.session_state.user)
    
    # Load user data
    c.execute("SELECT data FROM users WHERE username = ?", (st.session_state.user,))
    row = c.fetchone()
    user_data = json.loads(row[0]) if row else {}
    
    # Navigation
    tabs = st.tabs(["Dashboard", "Backtesting", "Myfxbook", "Community"])
    
    # ----------------- DASHBOARD -----------------
    with tabs[0]:
        st.header(f"Welcome, {st.session_state.user}")
        
        # Trade Logging
        st.subheader("Log a Trade")
        with st.form("trade_form"):
            col1, col2 = st.columns(2)
            with col1:
                pair = st.text_input("Currency Pair (e.g., EUR/USD)")
                direction = st.selectbox("Direction", ["Long", "Short"])
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001)
                stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.0001)
            with col2:
                take_profit = st.number_input("Take Profit", min_value=0.0, step=0.0001)
                size = st.number_input("Position Size (Lots)", min_value=0.0, step=0.01)
                trade_date = st.date_input("Trade Date", dt.date.today())
                emotions = st.text_area("Emotions/Notes")
            submit = st.form_submit_button("Log Trade")
            
            if submit:
                trade_id = _ta_hash()
                trade = {
                    "id": trade_id,
                    "pair": pair,
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "size": size,
                    "date": str(trade_date),
                    "emotions": emotions,
                    "pnl": 0.0,
                    "r": 0.0
                }
                user_data.setdefault("trades", []).append(trade)
                c.execute("UPDATE users SET data = ? WHERE username = ?",
                         (json.dumps(user_data), st.session_state.user))
                conn.commit()
                st.success("Trade logged!")
        
        # Trade History with Tabs per Trade
        st.subheader("Trade History")
        trades = user_data.get("trades", [])
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trade_tabs = st.tabs([f"Trade {t['id'][-4:]}" for t in trades])
            for i, (trade, tab) in enumerate(zip(trades, trade_tabs)):
                with tab:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Pair**: {trade['pair']}")
                        st.write(f"**Direction**: {trade['direction']}")
                        st.write(f"**Entry Price**: {_ta_human_num(trade['entry_price'])}")
                        st.write(f"**Stop Loss**: {_ta_human_num(trade['stop_loss'])}")
                        st.write(f"**Take Profit**: {_ta_human_num(trade['take_profit'])}")
                        st.write(f"**Size**: {_ta_human_num(trade['size'])} lots")
                        st.write(f"**Date**: {trade['date']}")
                        st.write(f"**Emotions/Notes**: {trade['emotions']}")
                    with col2:
                        st.subheader("Update Trade")
                        with st.form(f"update_trade_{trade['id']}"):
                            new_pnl = st.number_input("PnL ($)", key=f"pnl_{trade['id']}")
                            new_r = st.number_input("R Multiple", key=f"r_{trade['id']}")
                            if st.form_submit_button("Update"):
                                trades[i]["pnl"] = new_pnl
                                trades[i]["r"] = new_r
                                c.execute("UPDATE users SET data = ? WHERE username = ?",
                                         (json.dumps(user_data), st.session_state.user))
                                conn.commit()
                                st.success("Trade updated!")
        
        # Metrics
        if not trades_df.empty:
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", len(trades_df))
            col2.metric("Win Rate", _ta_human_pct(trades_df["r"].gt(0).mean()))
            col3.metric("Profit Factor", _ta_human_num(_ta_profit_factor(trades_df)))
            _ta_show_badges(trades_df)
            
            # Equity Curve
            daily_pnl = _ta_daily_pnl(trades_df)
            if not daily_pnl.empty:
                daily_pnl["cum_pnl"] = daily_pnl["pnl"].cumsum()
                fig = px.line(daily_pnl, x="date", y="cum_pnl", title="Equity Curve")
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    xaxis_title="Date",
                    yaxis_title="Cumulative PnL ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ----------------- BACKTESTING -----------------
    with tabs[1]:
        st.header("Backtesting")
        
        # Lightweight Charts Integration
        st.subheader("Live Price Chart (EUR/USD)")
        chart_container = st.empty()
        
        # Update candles
        new_candle = update_candles()
        
        # Prepare data for Lightweight Charts
        chart_data = json.dumps(DEMO_CANDLES)
        
        # HTML for Lightweight Charts
        chart_html = f"""
        <div id="chart_container" style="width:100%; height:400px;"></div>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
        <script>
        (function() {{{{
            const chart = LightweightCharts.createChart(document.getElementById('chart_container'), {{
                width: document.getElementById('chart_container').offsetWidth,
                height: 400,
                layout: {{ background: {{ color: '#0b0b0b' }}, textColor: '#ffffff' }},
                grid: {{ vertLines: {{ color: 'rgba(255,255,255,0.1)' }}, horzLines: {{ color: 'rgba(255,255,255,0.1)' }} }},
                timeScale: {{ timeVisible: true, secondsVisible: false }},
            }});
            const candleSeries = chart.addCandlestickSeries();
            const data = {chart_data};
            candleSeries.setData(data);
            
            // Auto-resize chart
            window.addEventListener('resize', () => {{
                chart.resize(document.getElementById('chart_container').offsetWidth, 400);
            }});
            
            // Simulate live updates
            setInterval(() => {{
                fetch('/.streamlit/proxy/streamlit_endpoint', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{}})
                }}).then(response => response.json()).then(newCandle => {{
                    if (newCandle) {{
                        candleSeries.update(newCandle);
                    }}
                }}).catch(error => console.error('Error fetching new candle:', error));
            }}, 60000);
        }}}})();
        </script>
        """
        components.html(chart_html, height=420)
        
        # Trade Logging (Moved to Backtesting Tab)
        st.subheader("Log a Trade")
        with st.form("trade_form_backtest"):
            col1, col2 = st.columns(2)
            with col1:
                pair = st.text_input("Currency Pair (e.g., EUR/USD)", key="backtest_pair")
                direction = st.selectbox("Direction", ["Long", "Short"], key="backtest_direction")
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, key="backtest_entry")
                stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.0001, key="backtest_sl")
            with col2:
                take_profit = st.number_input("Take Profit", min_value=0.0, step=0.0001, key="backtest_tp")
                size = st.number_input("Position Size (Lots)", min_value=0.0, step=0.01, key="backtest_size")
                trade_date = st.date_input("Trade Date", dt.date.today(), key="backtest_date")
                emotions = st.text_area("Emotions/Notes", key="backtest_emotions")
            submit = st.form_submit_button("Log Trade")
            
            if submit:
                trade_id = _ta_hash()
                trade = {
                    "id": trade_id,
                    "pair": pair,
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "size": size,
                    "date": str(trade_date),
                    "emotions": emotions,
                    "pnl": 0.0,
                    "r": 0.0,
                    "datetime": str(dt.datetime.combine(trade_date, dt.datetime.now().time()))
                }
                user_data.setdefault("trades", []).append(trade)
                c.execute("UPDATE users SET data = ? WHERE username = ?",
                         (json.dumps(user_data), st.session_state.user))
                conn.commit()
                st.success("Trade logged!")
        
        # Trade History with Tabs per Trade
        st.subheader("Trade History")
        trades = user_data.get("trades", [])
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trade_tabs = st.tabs([f"Trade {t['id'][-4:]}" for t in trades])
            for i, (trade, tab) in enumerate(zip(trades, trade_tabs)):
                with tab:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Pair**: {trade['pair']}")
                        st.write(f"**Direction**: {trade['direction']}")
                        st.write(f"**Entry Price**: {_ta_human_num(trade['entry_price'])}")
                        st.write(f"**Stop Loss**: {_ta_human_num(trade['stop_loss'])}")
                        st.write(f"**Take Profit**: {_ta_human_num(trade['take_profit'])}")
                        st.write(f"**Size**: {_ta_human_num(trade['size'])} lots")
                        st.write(f"**Date**: {trade['date']}")
                        st.write(f"**Emotions/Notes**: {trade['emotions']}")
                    with col2:
                        st.subheader("Update Trade")
                        with st.form(f"update_trade_{trade['id']}"):
                            new_pnl = st.number_input("PnL ($)", key=f"pnl_{trade['id']}")
                            new_r = st.number_input("R Multiple", key=f"r_{trade['id']}")
                            if st.form_submit_button("Update"):
                                trades[i]["pnl"] = new_pnl
                                trades[i]["r"] = new_r
                                c.execute("UPDATE users SET data = ? WHERE username = ?",
                                         (json.dumps(user_data), st.session_state.user))
                                conn.commit()
                                st.success("Trade updated!")
        
        # Backtest Parameters
        st.subheader("Backtest Parameters")
        with st.form("backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                strategy = st.selectbox("Strategy", ["Moving Average Crossover", "RSI", "Bollinger Bands"])
                period = st.number_input("Lookback Period", min_value=1, value=14)
            with col2:
                risk_per_trade = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0)
                account_size = st.number_input("Account Size ($)", min_value=1000, value=10000)
            if st.form_submit_button("Run Backtest"):
                # Simulate backtest
                trades = []
                for i in range(1, len(DEMO_CANDLES)):
                    prev_candle = DEMO_CANDLES[i-1]
                    curr_candle = DEMO_CANDLES[i]
                    if strategy == "Moving Average Crossover":
                        # Simple MA crossover logic
                        ma_short = sum(c["close"] for c in DEMO_CANDLES[max(0, i-5):i])/min(i, 5)
                        ma_long = sum(c["close"] for c in DEMO_CANDLES[max(0, i-14):i])/min(i, 14)
                        if ma_short > ma_long and prev_candle["close"] <= ma_long:
                            direction = "Long"
                            entry = curr_candle["close"]
                            sl = entry * (1 - 0.01)
                            tp = entry * (1 + 0.02)
                            pnl = account_size * (risk_per_trade/100) * (2 if curr_candle["close"] > entry else -1)
                            trades.append({
                                "datetime": dt.datetime.fromtimestamp(curr_candle["time"]).strftime("%Y-%m-%d %H:%M:%S"),
                                "pair": "EUR/USD",
                                "direction": direction,
                                "pnl": pnl,
                                "r": 2 if curr_candle["close"] > entry else -1
                            })
                backtest_df = pd.DataFrame(trades)
                if not backtest_df.empty:
                    st.subheader("Backtest Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Trades", len(backtest_df))
                    col2.metric("Win Rate", _ta_human_pct(backtest_df["r"].gt(0).mean()))
                    col3.metric("Profit Factor", _ta_human_num(_ta_profit_factor(backtest_df)))
                    
                    # Equity Curve
                    daily_pnl = _ta_daily_pnl(backtest_df)
                    if not daily_pnl.empty:
                        daily_pnl["cum_pnl"] = daily_pnl["pnl"].cumsum()
                        fig = px.line(daily_pnl, x="date", y="cum_pnl", title="Backtest Equity Curve")
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="white",
                            xaxis_title="Date",
                            yaxis_title="Cumulative PnL ($)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # ----------------- MYFXBOOK -----------------
    with tabs[2]:
        st.header("Myfxbook Integration")
        myfxbook_id = st.text_input("Enter Myfxbook Widget ID")
        if myfxbook_id:
            myfxbook_widget = f"""
            <div id="myfxbook_widget"></div>
            <script>
            (function() {{{{
                var iframe = document.createElement('iframe');
                iframe.src = 'https://widgets.myfxbook.com/widget/{myfxbook_id}.html';
                iframe.style.width = '100%';
                iframe.style.height = '400px';
                iframe.style.border = 'none';
                document.getElementById('myfxbook_widget').appendChild(iframe);
            }}}})();
            </script>
            """
            components.html(myfxbook_widget, height=420)
    
    # ----------------- COMMUNITY -----------------
    with tabs[3]:
        st.header("Community")
        community_posts = _ta_load_community("posts", [])
        
        # Post Form
        with st.form("community_post"):
            post_content = st.text_area("Share your trade idea or analysis")
            post_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
            if st.form_submit_button("Post"):
                if post_content or post_image:
                    post_id = _ta_hash()
                    post = {
                        "id": post_id,
                        "user": st.session_state.user,
                        "content": post_content,
                        "timestamp": dt.datetime.now(pytz.UTC).isoformat(),
                        "likes": 0,
                        "image": None
                    }
                    if post_image:
                        image_path = os.path.join(user_dir, "community_images", f"{post_id}.png")
                        with open(image_path, "wb") as f:
                            f.write(post_image.read())
                        post["image"] = image_path
                    community_posts.append(post)
                    _ta_save_community("posts", community_posts)
                    st.success("Posted to community!")
        
        # Display Posts
        for post in sorted(community_posts, key=lambda x: x["timestamp"], reverse=True):
            with st.container():
                st.write(f"**{post['user']}** at {post['timestamp']}")
                st.write(post['content'])
                if post.get("image"):
                    try:
                        with open(post["image"], "rb") as f:
                            image_data = base64.b64encode(f.read()).decode()
                            st.image(f"data:image/png;base64,{image_data}")
                    except Exception as e:
                        logging.error(f"Failed to load image {post['image']}: {str(e)}")
                like_key = f"like_{post['id']}"
                if st.button(f"👍 Like ({post['likes']})", key=like_key):
                    post["likes"] += 1
                    _ta_save_community("posts", community_posts)
                    st.rerun()
                st.markdown("---")
```
