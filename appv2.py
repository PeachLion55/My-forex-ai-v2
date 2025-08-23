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
from PIL import Image
import io
import base64
import subprocess
from datetime import datetime, date

# Custom JSON Encoder for handling datetime and other non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if pd.api.types.is_datetime64_any_dtype(obj):
            return obj.isoformat()
        if isinstance(obj, float) and math.isinf(obj):
            return 'inf' if obj > 0 else '-inf'
        if isinstance(obj, float) and math.isnan(obj):
            return 'nan'
        return super(CustomJSONEncoder, self).default(obj)

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add CSS for XP toast
st.markdown("""
<style>
div[data-testid="stToast"] {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    background-color: #4CAF50;
    color: white;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Hide Streamlit elements (consolidated)
st.markdown("""
<style>
/* Hide Streamlit top-right menu */
#MainMenu {visibility: hidden !important;}
/* Hide Streamlit footer (bottom-left) */
footer {visibility: hidden !important;}
/* Hide the GitHub / Share banner (bottom-right) */
[data-testid="stDecoration"] {display: none !important;}
/* Remove top padding and margins for main content */
.css-18e3th9, .css-1d391kg {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}
/* Reduce padding inside Streamlit containers */
.block-container {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

# Gridline background settings
grid_color = "#58b3b1"
grid_opacity = 0.16
grid_size = 40
r = int(grid_color[1:3], 16)
g = int(grid_color[3:5], 16)
b = int(grid_color[5:7], 16)
st.markdown(f"""
<style>
.stApp {{
    background-color: #000000;
    background-image:
        linear-gradient(rgba({r}, {g}, {b}, {grid_opacity}) 1px, transparent 1px),
        linear-gradient(90deg, rgba({r}, {g}, {b}, {grid_opacity}) 1px, transparent 1px);
    background-size: {grid_size}px {grid_size}px;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

# Path to SQLite DB
DB_FILE = "users.db"

# Connect to SQLite with error handling
try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS community_data (key TEXT PRIMARY KEY, data TEXT)''')
    conn.commit()
    logging.info("SQLite database initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize SQLite database: {str(e)}")
    st.error(f"Database initialization failed: {str(e)}")

# Helper Functions
def ta_safe_lower(s):
    return str(s).strip().lower().replace(" ", "")

def ta_human_pct(x, nd=1):
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
    if "r" not in df.columns:
        return pd.DataFrame()
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
        json_data = json.dumps(data, cls=CustomJSONEncoder)
        c.execute("INSERT OR REPLACE INTO community_data (key, data) VALUES (?, ?)", (key, json_data))
        conn.commit()
        logging.info(f"Community data saved for {key}")
    except Exception as e:
        logging.error(f"Failed to save community data for {key}: {str(e)}")

# Gamification helpers
def ta_update_xp(amount):
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result:
            user_data = json.loads(result[0])
            user_data['xp'] = user_data.get('xp', 0) + amount
            level = user_data['xp'] // 100
            if level > user_data.get('level', 0):
                user_data['level'] = level
                user_data['badges'] = user_data.get('badges', []) + [f"Level {level}"]
                st.balloons()
                st.success(f"Level up! You are now level {level}.")
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
            conn.commit()
            st.session_state.xp = user_data['xp']
            st.session_state.level = user_data['level']
            st.session_state.badges = user_data['badges']
            # Show XP toast
            st.markdown(f"""
            <div id="xp-toast">
            Earned {amount} XP!
            </div>
            """, unsafe_allow_html=True)
            components.v1.html("""
            <script>
            setTimeout(() => {{
                document.getElementById('xp-toast').style.display = 'none';
            }}, 6000);
            </script>
            """, height=0)

def ta_update_streak():
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result:
            user_data = json.loads(result[0])
            today = dt.date.today().isoformat()
            last_date = user_data.get('last_journal_date')
            streak = user_data.get('streak', 0)
            if last_date:
                last = dt.date.fromisoformat(last_date)
                if last == dt.date.fromisoformat(today) - dt.timedelta(days=1):
                    streak += 1
                elif last < dt.date.fromisoformat(today) - dt.timedelta(days=1):
                    streak = 1
            else:
                streak = 1
            user_data['streak'] = streak
            user_data['last_journal_date'] = today
            if streak % 7 == 0:
                badge = "Discipline Badge"
                if badge not in user_data.get('badges', []):
                    user_data['badges'] = user_data.get('badges', []) + [badge]
                    st.balloons()
                    st.success(f"Unlocked: {badge} for {streak} day streak!")
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
            conn.commit()
            st.session_state.streak = streak
            st.session_state.badges = user_data['badges']

def ta_check_milestones(journal_df, mt5_df):
    total_trades = len(journal_df)
    if total_trades >= 100:
        st.balloons()
        st.success("Milestone achieved: 100 trades journaled!")
    if not mt5_df.empty:
        daily_pnl = _ta_daily_pnl(mt5_df)
        if not daily_pnl.empty:
            daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
            recent = daily_pnl[daily_pnl['date'] >= pd.to_datetime('today') - pd.Timedelta(days=90)]
            if not recent.empty:
                equity = recent['pnl'].cumsum()
                dd = (equity - equity.cummax()).min() / equity.max() if equity.max() != 0 else 0
                if abs(dd) < 0.1:
                    st.balloons()
                    st.success("Milestone achieved: Survived 3 months without >10% drawdown!")

# Page Config
st.set_page_config(page_title="Forex Dashboard", layout="wide")

# Sidebar CSS
st.markdown("""
<style>
/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #000000 !important;
    overflow: hidden !important;
    max-height: 100vh !important;
}
/* Sidebar buttons default style */
section[data-testid="stSidebar"] div.stButton > button {
    width: 200px !important;
    background: linear-gradient(to right, rgba(88, 179, 177, 0.7), rgba(0, 0, 0, 0.7)) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 10px !important;
    margin: 5px 0 !important;
    font-weight: bold !important;
    font-size: 16px !important;
    text-align: left !important;
    display: block !important;
    box-sizing: border-box !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}
/* Hover effects for sidebar buttons */
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: linear-gradient(to right, rgba(88, 179, 177, 1.0), rgba(0, 0, 0, 1.0)) !important;
    transform: scale(1.05) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    color: #f0f0f0 !important;
    cursor: pointer !important;
}
/* Active page button style */
section[data-testid="stSidebar"] div.stButton > button[data-active="true"] {
    background: rgba(88, 179, 177, 0.7) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}
/* Adjust button size dynamically */
@media (max-height: 800px) {
    section[data-testid="stSidebar"] div.stButton > button {
        font-size: 14px !important;
        padding: 8px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
}
@media (max-height: 600px) {
    section[data-testid="stSidebar"] div.stButton > button {
        font-size: 12px !important;
        padding: 6px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.markdown("""
<style>
.sidebar-content {
    padding-top: 0rem;
}
</style>
""", unsafe_allow_html=True)

logo = Image.open("logo22.png")
logo = logo.resize((60, 50))
buffered = io.BytesIO()
logo.save(buffered, format="PNG")
logo_str = base64.b64encode(buffered.getvalue()).decode()
st.sidebar.markdown(f"""
<div style='text-align: center; margin-bottom: 20px;'>
    <img src="data:image/png;base64,{logo_str}" width="60" height="50"/>
</div>
""", unsafe_allow_html=True)

nav_items = [
    ('fundamentals', 'Forex Fundamentals'),
    ('backtesting', 'Backtesting'),
    ('mt5', 'Performance Dashboard'),
    ('psychology', 'Psychology'),
    ('strategy', 'Manage My Strategy'),
    ('account', 'My Account'),
    ('community', 'Community Trade Ideas'),
    ('tools', 'Tools'),
    ('Zenvo Academy', 'Zenvo Academy')
]
for page_key, page_name in nav_items:
    if st.sidebar.button(page_name, key=f"nav_{page_key}"):
        st.session_state.current_page = page_key
        st.session_state.current_subpage = None
        st.session_state.show_tools_submenu = False
        st.rerun()

# Initialize Session State
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'fundamentals'
if 'current_subpage' not in st.session_state:
    st.session_state.current_subpage = None
if 'show_tools_submenu' not in st.session_state:
    st.session_state.show_tools_submenu = False
if 'drawings' not in st.session_state:
    st.session_state.drawings = {}
if 'tools_trade_journal' not in st.session_state:
    journal_cols = [
        "Date", "Symbol", "Weekly Bias", "Daily Bias", "4H Structure", "1H Structure",
        "Positive Correlated Pair & Bias", "Potential Entry Points", "5min/15min Setup?",
        "Entry Conditions", "Planned R:R", "News Filter", "Alerts", "Concerns", "Emotions",
        "Confluence Score 1-7", "Outcome / R:R Realised", "Notes/Journal",
        "Entry Price", "Stop Loss Price", "Take Profit Price", "Lots", "Tags"
    ]
    journal_dtypes = {
        "Date": "datetime64[ns]", "Symbol": str, "Weekly Bias": str, "Daily Bias": str,
        "4H Structure": str, "1H Structure": str, "Positive Correlated Pair & Bias": str,
        "Potential Entry Points": str, "5min/15min Setup?": str, "Entry Conditions": str,
        "Planned R:R": str, "News Filter": str, "Alerts": str, "Concerns": str, "Emotions": str,
        "Confluence Score 1-7": float, "Outcome / R:R Realised": str, "Notes/Journal": str,
        "Entry Price": float, "Stop Loss Price": float, "Take Profit Price": float, "Lots": float,
        "Tags": str
    }
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
else:
    journal_cols = [
        "Date", "Symbol", "Weekly Bias", "Daily Bias", "4H Structure", "1H Structure",
        "Positive Correlated Pair & Bias", "Potential Entry Points", "5min/15min Setup?",
        "Entry Conditions", "Planned R:R", "News Filter", "Alerts", "Concerns", "Emotions",
        "Confluence Score 1-7", "Outcome / R:R Realised", "Notes/Journal",
        "Entry Price", "Stop Loss Price", "Take Profit Price", "Lots", "Tags"
    ]
    journal_dtypes = {
        "Date": "datetime64[ns]", "Symbol": str, "Weekly Bias": str, "Daily Bias": str,
        "4H Structure": str, "1H Structure": str, "Positive Correlated Pair & Bias": str,
        "Potential Entry Points": str, "5min/15min Setup?": str, "Entry Conditions": str,
        "Planned R:R": str, "News Filter": str, "Alerts": str, "Concerns": str, "Emotions": str,
        "Confluence Score 1-7": float, "Outcome / R:R Realised": str, "Notes/Journal": str,
        "Entry Price": float, "Stop Loss Price": float, "Take Profit Price": float, "Lots": float,
        "Tags": str
    }
    current_journal = st.session_state.tools_trade_journal
    missing_cols = [col for col in journal_cols if col not in current_journal.columns]
    if missing_cols:
        for col in missing_cols:
            current_journal[col] = pd.Series(dtype=journal_dtypes[col])
    st.session_state.tools_trade_journal = current_journal[journal_cols].astype(journal_dtypes, errors='ignore')

# Load community data
if "trade_ideas" not in st.session_state:
    loaded_ideas = _ta_load_community('trade_ideas', [])
    st.session_state.trade_ideas = pd.DataFrame(loaded_ideas, columns=["Username", "Pair", "Direction", "Description", "Timestamp", "IdeaID", "ImagePath"]) if loaded_ideas else pd.DataFrame(columns=["Username", "Pair", "Direction", "Description", "Timestamp", "IdeaID", "ImagePath"])
if "community_templates" not in st.session_state:
    loaded_templates = _ta_load_community('templates', [])
    st.session_state.community_templates = pd.DataFrame(loaded_templates, columns=["Username", "Type", "Name", "Content", "Timestamp", "ID"]) if loaded_templates else pd.DataFrame(columns=["Username", "Type", "Name", "Content", "Timestamp", "ID"])

# Data Helpers
def detect_currency(title: str) -> str:
    t = title.upper()
    currency_map = {
        "USD": ["USD", "US ", " US:", "FED", "FEDERAL RESERVE", "AMERICA", "U.S."],
        "GBP": ["GBP", "UK", " BRITAIN", "BOE", "POUND", "STERLING"],
        "EUR": ["EUR", "EURO", "EUROZONE", "ECB"],
        "JPY": ["JPY", "JAPAN", "BOJ", "YEN"],
        "AUD": ["AUD", "AUSTRALIA", "RBA"],
        "CAD": ["CAD", "CANADA", "BOC"],
        "CHF": ["CHF", "SWITZERLAND", "SNB"],
        "NZD": ["NZD", "NEW ZEALAND", "RBNZ"],
    }
    for curr, kws in currency_map.items():
        for kw in kws:
            if kw in t:
                return curr
    return "Unknown"

def rate_impact(polarity: float) -> str:
    if polarity > 0.5:
        return "Significantly Bullish"
    elif polarity > 0.1:
        return "Bullish"
    elif polarity < -0.5:
        return "Significantly Bearish"
    elif polarity < -0.1:
        return "Bearish"
    else:
        return "Neutral"

@st.cache_data(ttl=600, show_spinner=False)
def get_fxstreet_forex_news() -> pd.DataFrame:
    RSS_URL = "https://www.fxstreet.com/rss/news"
    try:
        feed = feedparser.parse(RSS_URL)
        logging.info("Successfully parsed FXStreet RSS feed")
    except Exception as e:
        logging.error(f"Failed to parse FXStreet RSS feed: {str(e)}")
        return pd.DataFrame(columns=["Date","Currency","Headline","Polarity","Impact","Summary","Link"])
    rows = []
    for entry in getattr(feed, "entries", []):
        title = entry.title
        published = getattr(entry, "published", "")
        date = published[:10] if published else ""
        currency = detect_currency(title)
        polarity = TextBlob(title).sentiment.polarity
        impact = rate_impact(polarity)
        summary = getattr(entry, "summary", "")
        rows.append({
            "Date": date,
            "Currency": currency,
            "Headline": title,
            "Polarity": polarity,
            "Impact": impact,
            "Summary": summary,
            "Link": entry.link
        })
    if rows:
        df = pd.DataFrame(rows)
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=3)
            df = df[df["Date"] >= cutoff]
            logging.info("News dataframe processed successfully")
        except Exception as e:
            logging.error(f"Failed to process news dataframe: {str(e)}")
        return df.reset_index(drop=True)
    return pd.DataFrame(columns=["Date","Currency","Headline","Polarity","Impact","Summary","Link"])

econ_calendar_data = [
    {"Date": "2025-08-22", "Time": "14:30", "Currency": "USD", "Event": "Non-Farm Payrolls", "Actual": "", "Forecast": "200K", "Previous": "185K", "Impact": "High"},
    {"Date": "2025-08-23", "Time": "09:00", "Currency": "EUR", "Event": "CPI Flash Estimate YoY", "Actual": "", "Forecast": "2.2%", "Previous": "2.1%", "Impact": "High"},
    {"Date": "2025-08-24", "Time": "12:00", "Currency": "GBP", "Event": "Bank of England Interest Rate Decision", "Actual": "", "Forecast": "5.00%", "Previous": "5.00%", "Impact": "High"},
    {"Date": "2025-08-25", "Time": "23:50", "Currency": "JPY", "Event": "Trade Balance", "Actual": "", "Forecast": "1000B", "Previous": "950B", "Impact": "Medium"},
    {"Date": "2025-08-26", "Time": "01:30", "Currency": "AUD", "Event": "Retail Sales MoM", "Actual": "", "Forecast": "0.3%", "Previous": "0.2%", "Impact": "Medium"},
    {"Date": "2025-08-27", "Time": "13:30", "Currency": "CAD", "Event": "GDP MoM", "Actual": "", "Forecast": "0.1%", "Previous": "0.0%", "Impact": "Medium"},
    {"Date": "2025-08-28", "Time": "08:00", "Currency": "CHF", "Event": "CPI YoY", "Actual": "", "Forecast": "1.5%", "Previous": "1.4%", "Impact": "Medium"}
]
econ_df = pd.DataFrame(econ_calendar_data)
df_news = get_fxstreet_forex_news()

# Main Application
if st.session_state.current_page == 'fundamentals':
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📅 Forex Fundamentals")
        st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
        st.markdown('---')
    with col2:
        st.info("See the Backtesting tab for live charts + detailed news.")
    st.markdown("### 🗓️ Upcoming Economic Events")
    if 'selected_currency_1' not in st.session_state:
        st.session_state.selected_currency_1 = None
    if 'selected_currency_2' not in st.session_state:
        st.session_state.selected_currency_2 = None
    uniq_ccy = sorted(set(list(econ_df["Currency"].unique()) + list(df_news["Currency"].unique())))
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        currency_filter_1 = st.selectbox("Primary currency to highlight", options=["None"] + uniq_ccy, key="cal_curr_1")
        st.session_state.selected_currency_1 = None if currency_filter_1 == "None" else currency_filter_1
    with col_filter2:
        currency_filter_2 = st.selectbox("Secondary currency to highlight", options=["None"] + uniq_ccy, key="cal_curr_2")
        st.session_state.selected_currency_2 = None if currency_filter_2 == "None" else currency_filter_2
    def highlight_currency(row):
        styles = [''] * len(row)
        if st.session_state.selected_currency_1 and row['Currency'] == st.session_state.selected_currency_1:
            styles = ['background-color: #4c7170; color: white' if col == 'Currency' else 'background-color: #4c7170' for col in row.index]
        if st.session_state.selected_currency_2 and row['Currency'] == st.session_state.selected_currency_2:
            styles = ['background-color: #2e4747; color: white' if col == 'Currency' else 'background-color: #2e4747' for col in row.index]
        return styles
    st.dataframe(econ_df.style.apply(highlight_currency, axis=1), use_container_width=True, height=360)
    st.markdown("### 💹 Major Central Bank Interest Rates")
    st.markdown(""" Interest rates are a key driver in forex markets. Higher rates attract foreign capital, strengthening the currency. Lower rates can weaken it. Monitor changes and forward guidance from central banks for trading opportunities. Below are current rates, with details on recent changes, next meeting dates, and market expectations. """)
    interest_rates = [
        {"Currency": "USD", "Current": "3.78%", "Previous": "4.00%", "Changed": "2025-07-17", "Next Meeting": "2025-09-18"},
        {"Currency": "GBP", "Current": "3.82%", "Previous": "4.00%", "Changed": "2025-08-07", "Next Meeting": "2025-09-19"},
        {"Currency": "EUR", "Current": "1.82%", "Previous": "2.00%", "Changed": "2025-07-10", "Next Meeting": "2025-09-12"},
        {"Currency": "JPY", "Current": "0.50%", "Previous": "0.25%", "Changed": "2025-07-31", "Next Meeting": "2025-09-20"},
        {"Currency": "AUD", "Current": "3.60%", "Previous": "3.85%", "Changed": "2025-08-12", "Next Meeting": "2025-09-24"},
        {"Currency": "CAD", "Current": "2.75%", "Previous": "3.00%", "Changed": "2025-03-12", "Next Meeting": "2025-09-04"},
        {"Currency": "NZD", "Current": "3.25%", "Previous": "3.50%", "Changed": "2025-05-28", "Next Meeting": "2025-10-09"},
        {"Currency": "CHF", "Current": "0.00%", "Previous": "0.25%", "Changed": "2025-06-19", "Next Meeting": "2025-09-26"},
    ]
    boxes_per_row = 4
    colors = ["#2d4646", "#4d7171", "#2d4646", "#4d7171"]
    for i in range(0, len(interest_rates), boxes_per_row):
        cols = st.columns(boxes_per_row)
        for j, rate in enumerate(interest_rates[i:i+boxes_per_row]):
            color = colors[j % len(colors)]
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                    {rate['Currency']}<br>
                    Current: {rate['Current']}<br>
                    Previous: {rate['Previous']}<br>
                    Changed On: {rate['Changed']}<br>
                    Next Meeting: {rate['Next Meeting']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Major High-Impact Forex Events")
    forex_high_impact_events = [
        {
            "event": "Non-Farm Payrolls (NFP)",
            "description": "Monthly report showing U.S. jobs added or lost, excluding farming, households, and non-profits.",
            "why_it_matters": "Indicates economic health; strong jobs → stronger USD, weak jobs → weaker USD.",
            "impact_positive": {
                "USD": "↑ Stronger USD due to strong labor market",
                "EUR/USD": "↓ EUR weakens vs USD",
                "GBP/USD": "↓ GBP weakens vs USD",
                "USD/JPY": "↑ USD strengthens vs JPY",
                "AUD/USD": "↓ AUD weakens vs USD",
                "USD/CAD": "↑ USD strengthens vs CAD"
            },
            "impact_negative": {
                "USD": "↓ Weaker USD due to weak labor market",
                "EUR/USD": "↑ EUR strengthens vs USD",
                "GBP/USD": "↑ GBP strengthens vs USD",
                "USD/JPY": "↓ USD weakens vs JPY",
                "AUD/USD": "↑ AUD strengthens vs USD",
                "USD/CAD": "↓ USD weakens vs CAD"
            },
        },
        {
            "event": "Consumer Price Index (CPI)",
            "description": "Measures changes in consumer prices; gauges inflation.",
            "why_it_matters": "Higher inflation → potential rate hikes → currency strengthens; lower inflation → dovish expectations → currency weakens.",
            "impact_positive": {
                "Currency": "↑ Higher rates likely → currency strengthens",
                "EUR/USD": "↓ Currency strengthens vs EUR",
                "GBP/USD": "↓ Currency strengthens vs GBP",
                "USD/JPY": "↑ USD strengthens vs JPY",
                "AUD/USD": "↓ Currency strengthens vs AUD",
                "USD/CAD": "↑ USD strengthens vs CAD"
            },
            "impact_negative": {
                "Currency": "↓ Lower inflation → dovish → currency weakens",
                "EUR/USD": "↑ Currency weakens vs EUR",
                "GBP/USD": "↑ Currency weakens vs GBP",
                "USD/JPY": "↓ USD weakens vs JPY",
                "AUD/USD": "↑ Currency weakens vs AUD",
                "USD/CAD": "↓ USD weakens vs CAD"
            },
        },
        {
            "event": "Interest Rate Decision",
            "description": "Central bank sets the official interest rate.",
            "why_it_matters": "Rate hikes or hawkish guidance → currency strengthens; rate cuts or dovish guidance → currency weakens.",
            "impact_positive": {
                "Currency": "↑ if hike or hawkish guidance → strengthens vs majors",
                "EUR/USD": "↓ Currency strengthens vs EUR",
                "GBP/USD": "↓ Currency strengthens vs GBP",
                "USD/JPY": "↑ USD strengthens vs JPY",
                "AUD/USD": "↓ Currency strengthens vs AUD",
                "USD/CAD": "↑ USD strengthens vs CAD"
            },
            "impact_negative": {
                "Currency": "↓ if cut or dovish guidance → weakens vs majors",
                "EUR/USD": "↑ Currency weakens vs EUR",
                "GBP/USD": "↑ Currency weakens vs GBP",
                "USD/JPY": "↓ USD weakens vs JPY",
                "AUD/USD": "↑ Currency weakens vs AUD",
                "USD/CAD": "↓ USD weakens vs CAD"
            },
        },
    ]
    for ev in forex_high_impact_events:
        positive_impact = "<br>".join([f"<b>{k}:</b> {v}" for k, v in ev["impact_positive"].items()])
        negative_impact = "<br>".join([f"<b>{k}:</b> {v}" for k, v in ev["impact_negative"].items()])
        st.markdown(
            f"""
            <div style="
                border-radius:12px;
                padding:15px;
                margin-bottom:18px;
                background-color:#12121a;
                color:white;
                box-shadow: 2px 4px 10px rgba(0,0,0,0.4);
            ">
                <h4 style="color:#FFD700; margin:0 0 6px 0;">{ev['event']}</h4>
                <p style="margin:6px 0 6px 0;"><b>What it is:</b> {ev['description']}</p>
                <p style="margin:6px 0 12px 0;"><b>Why it matters:</b> {ev['why_it_matters']}</p>
                <div style="display:flex; gap:12px;">
                    <div style="flex:1; background-color:#0f2b0f; padding:12px; border-radius:10px;">
                        <h5 style="margin:0 0 8px 0; color:#b7f2b7;">Positive →</h5>
                        <div style="font-size:0.95rem;">{positive_impact}</div>
                    </div>
                    <div style="flex:1; background-color:#2b0f0f; padding:12px; border-radius:10px;">
                        <h5 style="margin:0 0 8px 0; color:#f6b3b3;">Negative →</h5>
                        <div style="font-size:0.95rem;">{negative_impact}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

elif st.session_state.current_page == 'backtesting':
    st.title("📊 Backtesting")
    st.caption("Live TradingView chart for backtesting and enhanced trading journal for tracking and analyzing trades.")
    st.markdown('---')
    pairs_map = {
        "EUR/USD": "FX:EURUSD",
        "USD/JPY": "FX:USDJPY",
        "GBP/USD": "FX:GBPUSD",
        "USD/CHF": "OANDA:USDCHF",
        "AUD/USD": "FX:AUDUSD",
        "NZD/USD": "OANDA:NZDUSD",
        "USD/CAD": "CMCMARKETS:USDCAD",
        "EUR/GBP": "FX:EURGBP",
        "EUR/JPY": "FX:EURJPY",
        "GBP/JPY": "FX:GBPJPY",
        "AUD/JPY": "FX:AUDJPY",
        "AUD/NZD": "FX:AUDNZD",
        "AUD/CAD": "FX:AUDCAD",
        "AUD/CHF": "FX:AUDCHF",
        "CAD/JPY": "FX:CADJPY",
        "CHF/JPY": "FX:CHFJPY",
        "EUR/AUD": "FX:EURAUD",
        "EUR/CAD": "FX:EURCAD",
        "EUR/CHF": "FX:EURCHF",
        "GBP/AUD": "FX:GBPAUD",
        "GBP/CAD": "FX:GBPCAD",
        "GBP/CHF": "FX:GBPCHF",
        "NZD/JPY": "FX:NZDJPY",
        "NZD/CAD": "FX:NZDCAD",
        "NZD/CHF": "FX:NZDCHF",
        "CAD/CHF": "FX:CADCHF",
    }
    pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
    tv_symbol = pairs_map[pair]
    if "logged_in_user" in st.session_state and pair not in st.session_state.drawings:
        username = st.session_state.logged_in_user
        try:
            c.execute("SELECT data FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result:
                user_data = json.loads(result[0])
                st.session_state.drawings[pair] = user_data.get("drawings", {}).get(pair, {})
                logging.info(f"Loaded drawings for {pair}: {st.session_state.drawings[pair]}")
        except Exception as e:
            logging.error(f"Error loading drawings for {username}: {str(e)}")
            st.error(f"Failed to load drawings: {str(e)}")
    initial_content = json.dumps(st.session_state.drawings.get(pair, {}))
    tv_html = f"""
    <div id="tradingview_widget"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({{
        "container_id": "tradingview_widget",
        "width": "100%",
        "height": 800,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "studies": [],
        "show_popup_button": true,
        "popup_width": "1000",
        "popup_height": "650"
    }});
    </script>
    """
    st.components.v1.html(tv_html, height=820, scrolling=False)
    if "logged_in_user" in st.session_state:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Save Drawings", key="bt_save_drawings"):
                save_script = f"""
                <script>
                parent.window.postMessage({{action: 'save_drawings', pair: '{pair}'}}, '*');
                </script>
                """
                st.components.v1.html(save_script, height=0)
                st.session_state[f"bt_save_trigger_{pair}"] = True
        with col2:
            if st.button("Load Drawings", key="bt_load_drawings"):
                username = st.session_state.logged_in_user
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result:
                        user_data = json.loads(result[0])
                        content = user_data.get("drawings", {}).get(pair, {})
                        if content:
                            load_script = f"""
                            <script>
                            parent.window.postMessage({{action: 'load_drawings', pair: '{pair}', content: {json.dumps(content, cls=CustomJSONEncoder)}}}, '*');
                            </script>
                            """
                            st.components.v1.html(load_script, height=0)
                            st.success("Drawings loaded successfully!")
                        else:
                            st.info("No saved drawings for this pair.")
                    else:
                        st.error("Failed to load user data.")
                except Exception as e:
                    st.error(f"Failed to load drawings: {str(e)}")
                    logging.error(f"Error loading drawings for {username}: {str(e)}")
        with col3:
            if st.button("Refresh Account", key="bt_refresh_account"):
                username = st.session_state.logged_in_user
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result:
                        user_data = json.loads(result[0])
                        st.session_state.drawings = user_data.get("drawings", {})
                        st.success("Account synced successfully!")
                    else:
                        st.error("Failed to sync account.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to sync account: {str(e)}")
                    logging.error(f"Error syncing account for {username}: {str(e)}")
    drawings_key = f"bt_drawings_key_{pair}"
    if drawings_key in st.session_state and st.session_state.get(f"bt_save_trigger_{pair}", False):
        content = st.session_state[drawings_key]
        if content and isinstance(content, dict) and content:
            username = st.session_state.logged_in_user
            try:
                c.execute("SELECT data FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                user_data = json.loads(result[0]) if result else {}
                user_data.setdefault("drawings", {})[pair] = content
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                conn.commit()
                st.session_state.drawings[pair] = content
                st.success(f"Drawings for {pair} saved successfully!")
                logging.info(f"Drawings saved to database for {pair}: {content}")
            except Exception as e:
                st.error(f"Failed to save drawings: {str(e)}")
                logging.error(f"Database error saving drawings for {pair}: {str(e)}")
            finally:
                del st.session_state[drawings_key]
                del st.session_state[f"bt_save_trigger_{pair}"]
        else:
            st.warning("No valid drawing content received. Ensure you have drawn on the chart.")
            logging.warning(f"No valid drawing content received for {pair}: {content}")
    else:
        st.info("Sign in via the My Account tab to save/load drawings and trading journal.")
    st.markdown("### 📝 Trading Journal")
    st.markdown("Log your trades with detailed analysis, track psychological factors, and review performance with advanced analytics and trade replay.")
    tab_entry, tab_analytics, tab_replay = st.tabs(["📝 Log Trade", "📈 Analytics", "🎥 Trade Replay"])
    with tab_entry:
        st.subheader("Log a New Trade")
        with st.form("trade_entry_form"):
            col1, col2 = st.columns(2)
            with col1:
                trade_date = st.date_input("Date", value=datetime.now().date())
                symbol = st.selectbox("Symbol", list(pairs_map.keys()) + ["Other"], index=0)
                if symbol == "Other":
                    symbol = st.text_input("Custom Symbol")
                weekly_bias = st.selectbox("Weekly Bias", ["Bullish", "Bearish", "Neutral"])
                daily_bias = st.selectbox("Daily Bias", ["Bullish", "Bearish", "Neutral"])
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, format="%.5f")
                stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.0001, format="%.5f")
            with col2:
                take_profit_price = st.number_input("Take Profit Price", min_value=0.0, step=0.0001, format="%.5f")
                lots = st.number_input("Lots", min_value=0.01, step=0.01, format="%.2f")
                entry_conditions = st.text_area("Entry Conditions")
                emotions = st.selectbox("Emotions", ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral"])
                tags = st.multiselect("Tags", ["Setup: Breakout", "Setup: Reversal", "Mistake: Overtrading", "Mistake: No Stop Loss", "Emotion: FOMO", "Emotion: Revenge"])
                notes = st.text_area("Notes/Journal")
            submit_button = st.form_submit_button("Save Trade")
            if submit_button:
                pip_multiplier = 100 if "JPY" in symbol else 10000
                pl = (take_profit_price - entry_price) * lots * pip_multiplier if weekly_bias in ["Bullish", "Neutral"] else (entry_price - take_profit_price) * lots * pip_multiplier
                rr = (take_profit_price - entry_price) / (entry_price - stop_loss_price) if (entry_price - stop_loss_price) != 0 and weekly_bias in ["Bullish", "Neutral"] else (entry_price - take_profit_price) / (stop_loss_price - entry_price) if (stop_loss_price - entry_price) != 0 else 0
                new_trade = {
                    'Date': pd.to_datetime(trade_date),
                    'Symbol': symbol,
                    'Weekly Bias': weekly_bias,
                    'Daily Bias': daily_bias,
                    '4H Structure': '',
                    '1H Structure': '',
                    'Positive Correlated Pair & Bias': '',
                    'Potential Entry Points': '',
                    '5min/15min Setup?': '',
                    'Entry Conditions': entry_conditions,
                    'Planned R:R': f"1:{rr:.2f}",
                    'News Filter': '',
                    'Alerts': '',
                    'Concerns': '',
                    'Emotions': emotions,
                    'Confluence Score 1-7': 0.0,
                    'Outcome / R:R Realised': f"1:{rr:.2f}",
                    'Notes/Journal': notes,
                    'Entry Price': entry_price,
                    'Stop Loss Price': stop_loss_price,
                    'Take Profit Price': take_profit_price,
                    'Lots': lots,
                    'Tags': ','.join(tags)
                }
                st.session_state.tools_trade_journal = pd.concat(
                    [st.session_state.tools_trade_journal, pd.DataFrame([new_trade])],
                    ignore_index=True
                ).astype(journal_dtypes, errors='ignore')
                if 'logged_in_user' in st.session_state:
                    username = st.session_state.logged_in_user
                    user_data = {
                        'xp': st.session_state.get('xp', 0),
                        'level': st.session_state.get('level', 0),
                        'badges': st.session_state.get('badges', []),
                        'streak': st.session_state.get('streak', 0),
                        'last_journal_date': st.session_state.get('last_journal_date', None),
                        'drawings': st.session_state.get('drawings', {}),
                        'tools_trade_journal': st.session_state.tools_trade_journal.to_dict('records'),
                        'strategies': st.session_state.get('strategies', pd.DataFrame()).to_dict('records'),
                        'emotion_log': st.session_state.get('emotion_log', pd.DataFrame()).to_dict('records'),
                        'reflection_log': st.session_state.get('reflection_log', pd.DataFrame()).to_dict('records')
                    }
                    if user_data['last_journal_date'] is not None:
                        if isinstance(user_data['last_journal_date'], (datetime, date, pd.Timestamp)):
                            user_data['last_journal_date'] = user_data['last_journal_date'].isoformat()
                    for key in ['tools_trade_journal', 'strategies', 'emotion_log', 'reflection_log']:
                        user_data[key] = pd.DataFrame(user_data[key]).replace({pd.NA: None, float('nan'): None}).to_dict('records')
                    try:
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                        conn.commit()
                        ta_update_xp(10)
                        ta_update_streak()
                        st.success("Trade saved successfully!")
                        logging.info(f"Trade logged for user {username}")
                    except Exception as e:
                        st.error(f"Failed to save trade: {str(e)}")
                        logging.error(f"Error saving trade for {username}: {str(e)}")
                else:
                    st.success("Trade saved locally (not synced to account, please log in).")
                    logging.info("Trade logged for anonymous user")
                st.rerun()
        st.subheader("Trade Journal")
        column_config = {
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Weekly Bias": st.column_config.SelectboxColumn("Weekly Bias", options=["Bullish", "Bearish", "Neutral"]),
            "Daily Bias": st.column_config.SelectboxColumn("Daily Bias", options=["Bullish", "Bearish", "Neutral"]),
            "4H Structure": st.column_config.TextColumn("4H Structure"),
            "1H Structure": st.column_config.TextColumn("1H Structure"),
            "Positive Correlated Pair & Bias": st.column_config.TextColumn("Positive Correlated Pair & Bias"),
            "Potential Entry Points": st.column_config.TextColumn("Potential Entry Points"),
            "5min/15min Setup?": st.column_config.SelectboxColumn("5min/15min Setup?", options=["Yes", "No"]),
            "Entry Conditions": st.column_config.TextColumn("Entry Conditions"),
            "Planned R:R": st.column_config.TextColumn("Planned R:R"),
            "News Filter": st.column_config.TextColumn("News Filter"),
            "Alerts": st.column_config.TextColumn("Alerts"),
            "Concerns": st.column_config.TextColumn("Concerns"),
            "Emotions": st.column_config.TextColumn("Emotions"),
            "Confluence Score 1-7": st.column_config.NumberColumn("Confluence Score 1-7", min_value=1, max_value=7, format="%d"),
            "Outcome / R:R Realised": st.column_config.TextColumn("Outcome / R:R Realised"),
            "Notes/Journal": st.column_config.TextColumn("Notes/Journal"),
            "Entry Price": st.column_config.NumberColumn("Entry Price", format="%.5f"),
            "Stop Loss Price": st.column_config.NumberColumn("Stop Loss Price", format="%.5f"),
            "Take Profit Price": st.column_config.NumberColumn("Take Profit Price", format="%.5f"),
            "Lots": st.column_config.NumberColumn("Lots", format="%.2f"),
            "Tags": st.column_config.TextColumn("Tags")
        }
        st.dataframe(st.session_state.tools_trade_journal, column_config=column_config, use_container_width=True)
        st.subheader("Export Journal")
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            csv = st.session_state.tools_trade_journal.to_csv(index=False)
            st.download_button("Download CSV", csv, "trade_journal.csv", "text/csv")
        with col_export2:
            if st.button("Generate PDF"):
                latex_content = """
                \\documentclass{article}
                \\usepackage{booktabs}
                \\usepackage{geometry}
                \\geometry{a4paper, margin=1in}
                \\usepackage{pdflscape}
                \\usepackage{xcolor}
                \\definecolor{teal}{RGB}{88,179,177}
                \\begin{document}
                \\section*{\\textcolor{teal}{Trade Journal}}
                \\begin{landscape}
                \\begin{tabular}{llrrll}
                \\toprule
                Date & Symbol & Entry Price & Stop Loss & Take Profit & Outcome / R:R Realised \\\\
                \\midrule
                """
                for _, row in st.session_state.tools_trade_journal.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else ''
                    latex_content += f"{date_str} & {row['Symbol']} & {row['Entry Price']:.5f} & {row['Stop Loss Price']:.5f} & {row['Take Profit Price']:.5f} & {row['Outcome / R:R Realised']} \\\\\n"
                latex_content += """
                \\bottomrule
                \\end{tabular}
                \\end{landscape}
                \\end{document}
                """
                with open("trade_journal.tex", "w") as f:
                    f.write(latex_content)
                try:
                    subprocess.run(["latexmk", "-pdf", "trade_journal.tex"], check=True)
                    with open("trade_journal.pdf", "rb") as f:
                        st.download_button("Download PDF", f, "trade_journal.pdf", "application/pdf")
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
                    logging.error(f"PDF generation error: {str(e)}")
    with tab_analytics:
        st.subheader("Trade Analytics")
        if not st.session_state.tools_trade_journal.empty:
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                symbol_filter = st.multiselect(
                    "Filter by Symbol",
                    options=st.session_state.tools_trade_journal['Symbol'].unique(),
                    default=st.session_state.tools_trade_journal['Symbol'].unique()
                )
            with col_filter2:
                tag_options = [tag for tags in st.session_state.tools_trade_journal['Tags'].str.split(',').explode().unique() if tag and pd.notna(tag)]
                tag_filter = st.multiselect("Filter by Tags", options=tag_options)
            with col_filter3:
                bias_filter = st.selectbox("Filter by Weekly Bias", ["All", "Bullish", "Bearish", "Neutral"])
            filtered_df = st.session_state.tools_trade_journal[
                st.session_state.tools_trade_journal['Symbol'].isin(symbol_filter)
            ]
            if tag_filter and 'Tags' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Tags'].apply(lambda x: any(tag in str(x).split(',') for tag in tag_filter) if x else False)]
            if bias_filter != "All":
                filtered_df = filtered_df[filtered_df['Weekly Bias'] == bias_filter]
            def parse_rr(x):
                try:
                    if isinstance(x, str) and ':' in x:
                        return float(x.split(':')[1])
                    return 0.0
                except (ValueError, IndexError):
                    return 0.0
            win_rate = (filtered_df['Outcome / R:R Realised'].apply(parse_rr) > 0).mean() * 100 if not filtered_df.empty else 0
            avg_pl = filtered_df['Outcome / R:R Realised'].apply(parse_rr).mean() if not filtered_df.empty else 0
            total_trades = len(filtered_df)
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Win Rate (%)", f"{win_rate:.2f}")
            col_metric2.metric("Average R:R", f"{avg_pl:.2f}")
            col_metric3.metric("Total Trades", total_trades)
            st.subheader("Performance Charts")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                rr_values = filtered_df.groupby('Symbol')['Outcome / R:R Realised'].apply(
                    lambda x: pd.Series([parse_rr(r) for r in x]).mean()
                ).reset_index(name='Average R:R')
                fig = px.bar(rr_values, x='Symbol', y='Average R:R', title="Average R:R by Symbol")
                st.plotly_chart(fig, use_container_width=True)
            with col_chart2:
                fig = px.pie(filtered_df, names='Emotions', title="Trades by Emotional State")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades logged yet. Add trades in the 'Log Trade' tab.")
    with tab_replay:
        st.subheader("Trade Replay")
        if not st.session_state.tools_trade_journal.empty:
            trade_id = st.selectbox(
                "Select Trade to Replay",
                options=st.session_state.tools_trade_journal.index,
                format_func=lambda x: f"{st.session_state.tools_trade_journal.loc[x, 'Date'].strftime('%Y-%m-%d') if pd.notna(st.session_state.tools_trade_journal.loc[x, 'Date']) else 'N/A'} - {st.session_state.tools_trade_journal.loc[x, 'Symbol']}"
            )
            selected_trade = st.session_state.tools_trade_journal.loc[trade_id]
            st.write("**Trade Details**")
            st.write(f"Symbol: {selected_trade['Symbol']}")
            st.write(f"Weekly Bias: {selected_trade['Weekly Bias']}")
            st.write(f"Entry Price: {selected_trade['Entry Price']:.5f}")
            st.write(f"Stop Loss Price: {selected_trade['Stop Loss Price']:.5f}")
            st.write(f"Take Profit Price: {selected_trade['Take Profit Price']:.5f}")
            st.write(f"Outcome / R:R Realised: {selected_trade['Outcome / R:R Realised']}")
            st.write(f"Entry Conditions: {selected_trade['Entry Conditions']}")
            st.write(f"Emotions: {selected_trade['Emotions']}")
            st.write(f"Tags: {selected_trade.get('Tags', '')}")
            st.write(f"Notes: {selected_trade['Notes/Journal']}")
            if st.button("Replay Trade"):
                st.warning("Simulated MT5 chart replay. In a real implementation, connect to MT5 API to fetch historical data.")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[selected_trade['Date'], selected_trade['Date'] + pd.Timedelta(minutes=60)],
                    y=[selected_trade['Entry Price'], selected_trade['Take Profit Price']],
                    mode='lines+markers',
                    name='Price Movement'
                ))
                fig.add_hline(y=selected_trade['Stop Loss Price'], line_dash="dash", line_color="red", name="Stop Loss")
                fig.add_hline(y=selected_trade['Take Profit Price'], line_dash="dash", line_color="green", name="Take Profit")
                fig.update_layout(title=f"Trade Replay: {selected_trade['Symbol']}", xaxis_title="Time", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades available for replay.")

elif st.session_state.current_page == 'mt5':
    st.title("📊 Performance Dashboard")
    st.caption("Analyze your MT5 trading history with advanced metrics and visualizations.")
    st.markdown('---')
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload MT5 History CSV",
            type=["csv"],
            help="Export your trading history from MetaTrader 5 as a CSV file."
        )
    if uploaded_file:
        with st.spinner("Processing trading data..."):
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.mt5_df = df
                required_cols = ["Symbol", "Type", "Profit", "Volume", "Open Time", "Close Time"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    logging.error(f"Missing columns in MT5 CSV: {missing_cols}")
                    st.stop()
                df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
                df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")
                df["Trade Duration"] = (df["Close Time"] - df["Open Time"]).dt.total_seconds() / 3600
                tab_summary, tab_charts, tab_edge, tab_export = st.tabs([
                    "📈 Summary Metrics",
                    "📊 Visualizations",
                    "🧭 Edge Finder",
                    "📄 Export Reports"
                ])
                with tab_summary:
                    st.subheader("Key Performance Metrics")
                    total_trades = len(df)
                    wins = df[df["Profit"] > 0]
                    losses = df[df["Profit"] <= 0]
                    win_rate = len(wins) / total_trades if total_trades else 0
                    net_profit = df["Profit"].sum()
                    profit_factor = _ta_profit_factor(df)
                    avg_win = wins["Profit"].mean() if not wins.empty else 0
                    avg_loss = losses["Profit"].mean() if not losses.empty else 0
                    daily_pnl = _ta_daily_pnl(df)
                    max_drawdown = (daily_pnl["pnl"].cumsum() - daily_pnl["pnl"].cumsum().cummax()).min() if not daily_pnl.empty else 0
                    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) if total_trades else 0
                    longest_win_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] > 0) if k), default=0)
                    longest_loss_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] < 0) if k), default=0)
                    metrics = [
                        ("Total Trades", total_trades, "neutral"),
                        ("Win Rate", ta_human_pct(win_rate), "positive" if win_rate >= 0.5 else "negative"),
                        ("Net Profit", f"${net_profit:,.2f}", "positive" if net_profit >= 0 else "negative"),
                        ("Profit Factor", _ta_human_num(profit_factor), "positive" if profit_factor >= 1 else "negative"),
                        ("Max Drawdown", f"${max_drawdown:,.2f}", "negative"),
                        ("Expectancy", f"${expectancy:,.2f}", "positive" if expectancy >= 0 else "negative"),
                        ("Avg Win", f"${avg_win:,.2f}", "positive"),
                        ("Avg Loss", f"${avg_loss:,.2f}", "negative"),
                        ("Longest Win Streak", longest_win_streak, "positive"),
                        ("Longest Loss Streak", longest_loss_streak, "negative"),
                        ("Avg Trade Duration", f"{df['Trade Duration'].mean():.2f}h", "neutral"),
                    ]
                    for row in range(0, len(metrics), 4):
                        row_metrics = metrics[row:row+4]
                        cols = st.columns(4)
                        for i, (title, value, style) in enumerate(row_metrics):
                            with cols[i]:
                                st.markdown(
                                    f"""
                                    <div class="metric-box {style}">
                                        <strong>{title}</strong><br>
                                        {value}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                with tab_charts:
                    st.subheader("Performance Visualizations")
                    col_filter1, col_filter2 = st.columns(2)
                    with col_filter1:
                        symbol_filter = st.multiselect(
                            "Filter by Symbol",
                            options=df["Symbol"].unique(),
                            default=df["Symbol"].unique()
                        )
                    with col_filter2:
                        type_filter = st.multiselect(
                            "Filter by Type",
                            options=df["Type"].unique(),
                            default=df["Type"].unique()
                        )
                    filtered_df = df[
                        (df["Symbol"].isin(symbol_filter)) &
                        (df["Type"].isin(type_filter))
                    ]
                    st.markdown("**Profit by Instrument**")
                    profit_symbol = filtered_df.groupby("Symbol")["Profit"].sum().reset_index()
                    fig_symbol = px.bar(
                        profit_symbol,
                        x="Symbol",
                        y="Profit",
                        color="Profit",
                        title="Profit by Instrument",
                        template="plotly_dark",
                        color_continuous_scale="Tealgrn"
                    )
                    fig_symbol.update_layout(
                        title_font_size=18,
                        title_x=0.5,
                        font_color="#ffffff",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_symbol, use_container_width=True)
                    st.markdown("**Equity Curve**")
                    if not daily_pnl.empty:
                        daily_pnl["Equity"] = daily_pnl["pnl"].cumsum()
                        fig_equity = go.Figure()
                        fig_equity.add_trace(
                            go.Scatter(
                                x=daily_pnl["date"],
                                y=daily_pnl["Equity"],
                                mode="lines",
                                name="Equity",
                                line=dict(color="#58b3b1")
                            )
                        )
                        fig_equity.update_layout(
                            title="Equity Curve",
                            xaxis_title="Date",
                            yaxis_title="Equity ($)",
                            template="plotly_dark",
                            title_font_size=18,
                            title_x=0.5,
                            font_color="#ffffff",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_equity, use_container_width=True)
                    st.markdown("**Trade Distribution**")
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        fig_types = px.pie(
                            filtered_df,
                            names="Type",
                            title="Buy vs Sell Distribution",
                            template="plotly_dark",
                            color_discrete_sequence=["#58b3b1", "#4d7171"]
                        )
                        fig_types.update_layout(title_font_size=16, title_x=0.5)
                        st.plotly_chart(fig_types, use_container_width=True)
                    with col_chart2:
                        filtered_df["Weekday"] = filtered_df["Open Time"].dt.day_name()
                        fig_weekday = px.histogram(
                            filtered_df,
                            x="Weekday",
                            color="Type",
                            title="Trades by Day of Week",
                            template="plotly_dark",
                            color_discrete_sequence=["#58b3b1", "#4d7171"]
                        )
                        fig_weekday.update_layout(title_font_size=16, title_x=0.5)
                        st.plotly_chart(fig_weekday, use_container_width=True)
                with tab_edge:
                    st.subheader("Edge Finder – Highest Expectancy Segments")
                    group_cols = [col for col in ["timeframe", "symbol", "setup"] if col in df.columns]
                    if group_cols:
                        agg = _ta_expectancy_by_group(df, group_cols).sort_values("expectancy", ascending=False)
                        st.dataframe(
                            agg.style.format({
                                "winrate": "{:.2%}",
                                "avg_win": "${:.2f}",
                                "avg_loss": "${:.2f}",
                                "expectancy": "${:.2f}"
                            }),
                            use_container_width=True
                        )
                        top_n = st.slider("Show Top N Segments", 5, 50, 10, key="edge_topn")
                        fig_edge = px.bar(
                            agg.head(top_n),
                            x="expectancy",
                            y=group_cols,
                            orientation="h",
                            title="Top Expectancy Segments",
                            template="plotly_dark",
                            color="expectancy",
                            color_continuous_scale="Tealgrn"
                        )
                        fig_edge.update_layout(
                            title_font_size=18,
                            title_x=0.5,
                            font_color="#ffffff",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_edge, use_container_width=True)
                    else:
                        st.warning("Edge Finder requires columns: timeframe, symbol, or setup.")
                with tab_export:
                    st.subheader("Export Performance Reports")
                    report_types = st.multiselect(
                        "Select Report Formats",
                        ["CSV", "HTML", "PDF"],
                        default=["CSV"]
                    )
                    if st.button("Generate Reports"):
                        if "CSV" in report_types:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="mt5_performance.csv",
                                mime="text/csv"
                            )
                        if "HTML" in report_types:
                            report_html = f"""
                            <html>
                            <head>
                                <style>
                                    body {{ font-family: Arial, sans-serif; background-color: #000000; color: #ffffff; padding: 20px; }}
                                    h2 {{ color: #58b3b1; }}
                                    .metric {{ margin: 10px 0; padding: 10px; background-color: #1a1a1a; border-radius: 5px; }}
                                </style>
                            </head>
                            <body>
                                <h2>MT5 Performance Report</h2>
                                <div class="metric">Total Trades: {total_trades}</div>
                                <div class="metric">Win Rate: {ta_human_pct(win_rate)}</div>
                                <div class="metric">Net Profit: ${net_profit:,.2f}</div>
                                <div class="metric">Profit Factor: {_ta_human_num(profit_factor)}</div>
                                <div class="metric">Max Drawdown: ${max_drawdown:,.2f}</div>
                                <div class="metric">Expectancy: ${expectancy:,.2f}</div>
                            </body>
                            </html>
                            """
                            st.download_button(
                                label="Download HTML Report",
                                data=report_html,
                                file_name="mt5_performance.html",
                                mime="text/html"
                            )
                        if "PDF" in report_types:
                            latex_content = """
                            \\documentclass{article}
                            \\usepackage{booktabs}
                            \\usepackage{geometry}
                            \\geometry{a4paper, margin=1in}
                            \\usepackage{pdflscape}
                            \\usepackage{xcolor}
                            \\definecolor{teal}{RGB}{88,179,177}
                            \\begin{document}
                            \\section*{\\textcolor{teal}{MT5 Performance Report}}
                            \\begin{tabular}{ll}
                            \\toprule
                            \\textbf{Metric} & \\textbf{Value} \\\\
                            \\midrule
                            Total Trades & %s \\\\
                            Win Rate & %s \\\\
                            Net Profit & \\$%s \\\\
                            Profit Factor & %s \\\\
                            Max Drawdown & \\$%s \\\\
                            Expectancy & \\$%s \\\\
                            \\bottomrule
                            \\end{tabular}
                            \\end{document}
                            """ % (
                                total_trades,
                                ta_human_pct(win_rate),
                                f"{net_profit:,.2f}",
                                _ta_human_num(profit_factor),
                                f"{max_drawdown:,.2f}",
                                f"{expectancy:,.2f}"
                            )
                            try:
                                with open("mt5_report.tex", "w") as f:
                                    f.write(latex_content)
                                subprocess.run(["latexmk", "-pdf", "mt5_report.tex"], check=True)
                                with open("mt5_report.pdf", "rb") as f:
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=f,
                                        file_name="mt5_performance.pdf",
                                        mime="application/pdf"
                                    )
                            except Exception as e:
                                st.error(f"PDF generation failed: {str(e)}")
                                logging.error(f"PDF generation error: {str(e)}")
                    st.markdown("**Shareable Insights**")
                    if not daily_pnl.empty:
                        top_symbol = profit_symbol.loc[profit_symbol["Profit"].idxmax(), "Symbol"] if not profit_symbol.empty else "N/A"
                        insight = f"Top performing symbol: {top_symbol} with ${_ta_human_num(profit_symbol['Profit'].max())} profit."
                        st.info(insight)
                        if st.button("Share Insight"):
                            st.success("Insight copied to clipboard! Share with your trading community.")
                            logging.info(f"Shared insight: {insight}")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                logging.error(f"MT5 CSV processing error: {str(e)}")
    else:
        st.info("👆 Upload your MT5 trading history CSV to explore advanced performance metrics.")
    if "mt5_df" in st.session_state and not st.session_state.mt5_df.empty:
        try:
            _ta_show_badges(st.session_state.mt5_df)
        except Exception as e:
            logging.error(f"Error displaying badges: {str(e)}")
elif st.session_state.current_page == 'psychology':
    st.title("🧠 Psychology")
    st.caption("Track and analyze your trading psychology to improve discipline and decision-making.")
    st.markdown('---')
    st.markdown("""
    Trading psychology is critical to success. This section helps you log emotions, reflect on trades, 
    and identify patterns in your psychological state that impact performance.
    """)

    # Initialize emotion log if not present
    if 'emotion_log' not in st.session_state:
        emotion_log_cols = ["Date", "Emotion", "Intensity", "Trigger", "TradeID", "Notes"]
        emotion_log_dtypes = {
            "Date": "datetime64[ns]", "Emotion": str, "Intensity": float,
            "Trigger": str, "TradeID": str, "Notes": str
        }
        st.session_state.emotion_log = pd.DataFrame(columns=emotion_log_cols).astype(emotion_log_dtypes)

    # Initialize reflection log if not present
    if 'reflection_log' not in st.session_state:
        reflection_log_cols = ["Date", "Session", "Reflection", "Lessons", "Goals"]
        reflection_log_dtypes = {
            "Date": "datetime64[ns]", "Session": str, "Reflection": str,
            "Lessons": str, "Goals": str
        }
        st.session_state.reflection_log = pd.DataFrame(columns=reflection_log_cols).astype(reflection_log_dtypes)

    tab_emotions, tab_reflections, tab_analytics = st.tabs(["😊 Emotion Log", "📝 Reflections", "📊 Psychological Analytics"])

    with tab_emotions:
        st.subheader("Log Your Emotions")
        with st.form("emotion_log_form"):
            col1, col2 = st.columns(2)
            with col1:
                emotion_date = st.date_input("Date", value=datetime.now().date(), key="emo_date")
                emotion = st.selectbox("Emotion", ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral", "Other"], key="emo_type")
                if emotion == "Other":
                    emotion = st.text_input("Custom Emotion", key="emo_custom")
                intensity = st.slider("Intensity (1-10)", 1, 10, 5, key="emo_intensity")
            with col2:
                trigger = st.text_area("What triggered this emotion?", key="emo_trigger")
                trade_id = st.selectbox(
                    "Related Trade (optional)",
                    options=["None"] + list(st.session_state.tools_trade_journal.index),
                    format_func=lambda x: "None" if x == "None" else f"{st.session_state.tools_trade_journal.loc[x, 'Date'].strftime('%Y-%m-%d') if pd.notna(st.session_state.tools_trade_journal.loc[x, 'Date']) else 'N/A'} - {st.session_state.tools_trade_journal.loc[x, 'Symbol']}",
                    key="emo_trade_id"
                )
                notes = st.text_area("Additional Notes", key="emo_notes")
            submit_emotion = st.form_submit_button("Log Emotion")
            if submit_emotion:
                new_emotion = {
                    "Date": pd.to_datetime(emotion_date),
                    "Emotion": emotion,
                    "Intensity": intensity,
                    "Trigger": trigger,
                    "TradeID": trade_id if trade_id != "None" else "",
                    "Notes": notes
                }
                st.session_state.emotion_log = pd.concat(
                    [st.session_state.emotion_log, pd.DataFrame([new_emotion])],
                    ignore_index=True
                ).astype(st.session_state.emotion_log.dtypes, errors='ignore')
                if 'logged_in_user' in st.session_state:
                    username = st.session_state.logged_in_user
                    user_data = {
                        'xp': st.session_state.get('xp', 0),
                        'level': st.session_state.get('level', 0),
                        'badges': st.session_state.get('badges', []),
                        'streak': st.session_state.get('streak', 0),
                        'last_journal_date': st.session_state.get('last_journal_date', None),
                        'drawings': st.session_state.get('drawings', {}),
                        'tools_trade_journal': st.session_state.tools_trade_journal.to_dict('records'),
                        'strategies': st.session_state.get('strategies', pd.DataFrame()).to_dict('records'),
                        'emotion_log': st.session_state.emotion_log.to_dict('records'),
                        'reflection_log': st.session_state.reflection_log.to_dict('records')
                    }
                    try:
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                        conn.commit()
                        ta_update_xp(5)
                        ta_update_streak()
                        st.success("Emotion logged successfully!")
                        logging.info(f"Emotion logged for user {username}")
                    except Exception as e:
                        st.error(f"Failed to save emotion: {str(e)}")
                        logging.error(f"Error saving emotion for {username}: {str(e)}")
                else:
                    st.success("Emotion logged locally (not synced, please log in).")
                st.rerun()
        st.subheader("Emotion Log")
        st.dataframe(st.session_state.emotion_log, use_container_width=True)

    with tab_reflections:
        st.subheader("Daily/Weekly Reflections")
        with st.form("reflection_form"):
            col1, col2 = st.columns(2)
            with col1:
                reflection_date = st.date_input("Date", value=datetime.now().date(), key="ref_date")
                session = st.selectbox("Session", ["Daily", "Weekly", "Post-Trade"], key="ref_session")
                reflection = st.text_area("Reflection (What went well? What could be improved?)", key="ref_reflection")
            with col2:
                lessons = st.text_area("Lessons Learned", key="ref_lessons")
                goals = st.text_area("Goals for Next Session", key="ref_goals")
            submit_reflection = st.form_submit_button("Save Reflection")
            if submit_reflection:
                new_reflection = {
                    "Date": pd.to_datetime(reflection_date),
                    "Session": session,
                    "Reflection": reflection,
                    "Lessons": lessons,
                    "Goals": goals
                }
                st.session_state.reflection_log = pd.concat(
                    [st.session_state.reflection_log, pd.DataFrame([new_reflection])],
                    ignore_index=True
                ).astype(st.session_state.reflection_log.dtypes, errors='ignore')
                if 'logged_in_user' in st.session_state:
                    username = st.session_state.logged_in_user
                    user_data = {
                        'xp': st.session_state.get('xp', 0),
                        'level': st.session_state.get('level', 0),
                        'badges': st.session_state.get('badges', []),
                        'streak': st.session_state.get('streak', 0),
                        'last_journal_date': st.session_state.get('last_journal_date', None),
                        'drawings': st.session_state.get('drawings', {}),
                        'tools_trade_journal': st.session_state.tools_trade_journal.to_dict('records'),
                        'strategies': st.session_state.get('strategies', pd.DataFrame()).to_dict('records'),
                        'emotion_log': st.session_state.emotion_log.to_dict('records'),
                        'reflection_log': st.session_state.reflection_log.to_dict('records')
                    }
                    try:
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                        conn.commit()
                        ta_update_xp(10)
                        ta_update_streak()
                        st.success("Reflection saved successfully!")
                        logging.info(f"Reflection saved for user {username}")
                    except Exception as e:
                        st.error(f"Failed to save reflection: {str(e)}")
                        logging.error(f"Error saving reflection for {username}: {str(e)}")
                else:
                    st.success("Reflection saved locally (not synced, please log in).")
                st.rerun()
        st.subheader("Reflection Log")
        st.dataframe(st.session_state.reflection_log, use_container_width=True)

    with tab_analytics:
        st.subheader("Psychological Analytics")
        if not st.session_state.emotion_log.empty:
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                emotion_filter = st.multiselect(
                    "Filter by Emotion",
                    options=st.session_state.emotion_log['Emotion'].unique(),
                    default=st.session_state.emotion_log['Emotion'].unique()
                )
            with col_filter2:
                period_filter = st.selectbox("Time Period", ["All", "Last 7 Days", "Last 30 Days", "Last 90 Days"])
            filtered_emotions = st.session_state.emotion_log[
                st.session_state.emotion_log['Emotion'].isin(emotion_filter)
            ]
            if period_filter != "All":
                days = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days[period_filter])
                filtered_emotions = filtered_emotions[filtered_emotions['Date'] >= cutoff]
            st.markdown("**Emotion Distribution**")
            fig_emotion = px.pie(filtered_emotions, names='Emotion', title="Emotion Distribution", template="plotly_dark")
            st.plotly_chart(fig_emotion, use_container_width=True)
            st.markdown("**Emotion Intensity Over Time**")
            fig_intensity = px.line(
                filtered_emotions,
                x='Date',
                y='Intensity',
                color='Emotion',
                title="Emotion Intensity Over Time",
                template="plotly_dark"
            )
            st.plotly_chart(fig_intensity, use_container_width=True)
            if not st.session_state.tools_trade_journal.empty:
                st.markdown("**Emotion vs Trade Outcome**")
                merged = filtered_emotions.merge(
                    st.session_state.tools_trade_journal,
                    left_on='TradeID',
                    right_index=True,
                    how='left'
                )
                merged['Outcome'] = merged['Outcome / R:R Realised'].apply(lambda x: float(x.split(':')[1]) if isinstance(x, str) and ':' in x else 0.0)
                fig_outcome = px.scatter(
                    merged,
                    x='Intensity',
                    y='Outcome',
                    color='Emotion',
                    title="Emotion Intensity vs Trade Outcome (R:R)",
                    template="plotly_dark",
                    hover_data=['Symbol', 'Date']
                )
                st.plotly_chart(fig_outcome, use_container_width=True)
        else:
            st.info("No emotions logged yet. Add entries in the Emotion Log tab.")

elif st.session_state.current_page == 'strategy':
    st.title("📋 Manage My Strategy")
    st.caption("Create, test, and manage your trading strategies.")
    st.markdown('---')
    if 'strategies' not in st.session_state:
        strategy_cols = ["Name", "Description", "Rules", "Timeframe", "Pairs", "Created", "LastModified", "Performance"]
        strategy_dtypes = {
            "Name": str, "Description": str, "Rules": str, "Timeframe": str,
            "Pairs": str, "Created": "datetime64[ns]", "LastModified": "datetime64[ns]", "Performance": str
        }
        st.session_state.strategies = pd.DataFrame(columns=strategy_cols).astype(strategy_dtypes)

    tab_create, tab_view, tab_backtest = st.tabs(["✍️ Create Strategy", "📋 View Strategies", "🔍 Backtest"])

    with tab_create:
        st.subheader("Create a New Strategy")
        with st.form("strategy_form"):
            col1, col2 = st.columns(2)
            with col1:
                strategy_name = st.text_input("Strategy Name", key="strat_name")
                description = st.text_area("Description", key="strat_desc")
                timeframe = st.selectbox("Timeframe", ["1M", "5M", "15M", "1H", "4H", "D"], key="strat_timeframe")
            with col2:
                rules = st.text_area("Entry/Exit Rules", key="strat_rules")
                pairs = st.multiselect("Currency Pairs", list(pairs_map.keys()), key="strat_pairs")
            submit_strategy = st.form_submit_button("Save Strategy")
            if submit_strategy:
                if strategy_name:
                    new_strategy = {
                        "Name": strategy_name,
                        "Description": description,
                        "Rules": rules,
                        "Timeframe": timeframe,
                        "Pairs": ",".join(pairs),
                        "Created": pd.to_datetime(datetime.now()),
                        "LastModified": pd.to_datetime(datetime.now()),
                        "Performance": ""
                    }
                    st.session_state.strategies = pd.concat(
                        [st.session_state.strategies, pd.DataFrame([new_strategy])],
                        ignore_index=True
                    ).astype(st.session_state.strategies.dtypes, errors='ignore')
                    if 'logged_in_user' in st.session_state:
                        username = st.session_state.logged_in_user
                        user_data = {
                            'xp': st.session_state.get('xp', 0),
                            'level': st.session_state.get('level', 0),
                            'badges': st.session_state.get('badges', []),
                            'streak': st.session_state.get('streak', 0),
                            'last_journal_date': st.session_state.get('last_journal_date', None),
                            'drawings': st.session_state.get('drawings', {}),
                            'tools_trade_journal': st.session_state.tools_trade_journal.to_dict('records'),
                            'strategies': st.session_state.strategies.to_dict('records'),
                            'emotion_log': st.session_state.get('emotion_log', pd.DataFrame()).to_dict('records'),
                            'reflection_log': st.session_state.get('reflection_log', pd.DataFrame()).to_dict('records')
                        }
                        try:
                            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                            conn.commit()
                            ta_update_xp(15)
                            st.success(f"Strategy '{strategy_name}' saved successfully!")
                            logging.info(f"Strategy saved for user {username}: {strategy_name}")
                        except Exception as e:
                            st.error(f"Failed to save strategy: {str(e)}")
                            logging.error(f"Error saving strategy for {username}: {str(e)}")
                    else:
                        st.success("Strategy saved locally (not synced, please log in).")
                    st.rerun()
                else:
                    st.error("Strategy name is required.")

    with tab_view:
        st.subheader("Your Strategies")
        if not st.session_state.strategies.empty:
            st.dataframe(st.session_state.strategies, use_container_width=True)
            strategy_to_edit = st.selectbox(
                "Select Strategy to Edit",
                options=st.session_state.strategies.index,
                format_func=lambda x: st.session_state.strategies.loc[x, 'Name']
            )
            with st.form("edit_strategy_form"):
                selected_strategy = st.session_state.strategies.loc[strategy_to_edit]
                col1, col2 = st.columns(2)
                with col1:
                    edit_name = st.text_input("Strategy Name", value=selected_strategy['Name'], key="edit_strat_name")
                    edit_description = st.text_area("Description", value=selected_strategy['Description'], key="edit_strat_desc")
                    edit_timeframe = st.selectbox("Timeframe", ["1M", "5M", "15M", "1H", "4H", "D"], index=["1M", "5M", "15M", "1H", "4H", "D"].index(selected_strategy['Timeframe']), key="edit_strat_timeframe")
                with col2:
                    edit_rules = st.text_area("Entry/Exit Rules", value=selected_strategy['Rules'], key="edit_strat_rules")
                    edit_pairs = st.multiselect("Currency Pairs", list(pairs_map.keys()), default=selected_strategy['Pairs'].split(','), key="edit_strat_pairs")
                col_submit, col_delete = st.columns(2)
                with col_submit:
                    if st.form_submit_button("Update Strategy"):
                        st.session_state.strategies.loc[strategy_to_edit, 'Name'] = edit_name
                        st.session_state.strategies.loc[strategy_to_edit, 'Description'] = edit_description
                        st.session_state.strategies.loc[strategy_to_edit, 'Rules'] = edit_rules
                        st.session_state.strategies.loc[strategy_to_edit, 'Timeframe'] = edit_timeframe
                        st.session_state.strategies.loc[strategy_to_edit, 'Pairs'] = ",".join(edit_pairs)
                        st.session_state.strategies.loc[strategy_to_edit, 'LastModified'] = pd.to_datetime(datetime.now())
                        if 'logged_in_user' in st.session_state:
                            username = st.session_state.logged_in_user
                            user_data = {
                                'xp': st.session_state.get('xp', 0),
                                'level': st.session_state.get('level', 0),
                                'badges': st.session_state.get('badges', []),
                                'streak': st.session_state.get('streak', 0),
                                'last_journal_date': st.session_state.get('last_journal_date', None),
                                'drawings': st.session_state.get('drawings', {}),
                                'tools_trade_journal': st.session_state.tools_trade_journal.to_dict('records'),
                                'strategies': st.session_state.strategies.to_dict('records'),
                                'emotion_log': st.session_state.get('emotion_log', pd.DataFrame()).to_dict('records'),
                                'reflection_log': st.session_state.get('reflection_log', pd.DataFrame()).to_dict('records')
                            }
                            try:
                                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                                conn.commit()
                                st.success("Strategy updated successfully!")
                                logging.info(f"Strategy updated for user {username}: {edit_name}")
                            except Exception as e:
                                st.error(f"Failed to update strategy: {str(e)}")
                                logging.error(f"Error updating strategy for {username}: {str(e)}")
                            st.rerun()
                with col_delete:
                    if st.form_submit_button("Delete Strategy"):
                        st.session_state.strategies = st.session_state.strategies.drop(strategy_to_edit).reset_index(drop=True)
                        if 'logged_in_user' in st.session_state:
                            username = st.session_state.logged_in_user
                            user_data = {
                                'xp': st.session_state.get('xp', 0),
                                'level': st.session_state.get('level', 0),
                                'badges': st.session_state.get('badges', []),
                                'streak': st.session_state.get('streak', 0),
                                'last_journal_date': st.session_state.get('last_journal_date', None),
                                'drawings': st.session_state.get('drawings', {}),
                                'tools_trade_journal': st.session_state.tools_trade_journal.to_dict('records'),
                                'strategies': st.session_state.strategies.to_dict('records'),
                                'emotion_log': st.session_state.get('emotion_log', pd.DataFrame()).to_dict('records'),
                                'reflection_log': st.session_state.get('reflection_log', pd.DataFrame()).to_dict('records')
                            }
                            try:
                                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                                conn.commit()
                                st.success("Strategy deleted successfully!")
                                logging.info(f"Strategy deleted for user {username}: {edit_name}")
                            except Exception as e:
                                st.error(f"Failed to delete strategy: {str(e)}")
                                logging.error(f"Error deleting strategy for {username}: {str(e)}")
                            st.rerun()
        else:
            st.info("No strategies created yet. Add one in the 'Create Strategy' tab.")

    with tab_backtest:
        st.subheader("Backtest Strategy")
        if not st.session_state.strategies.empty:
            strategy_to_backtest = st.selectbox(
                "Select Strategy to Backtest",
                options=st.session_state.strategies.index,
                format_func=lambda x: st.session_state.strategies.loc[x, 'Name']
            )
            selected_strategy = st.session_state.strategies.loc[strategy_to_backtest]
            st.write(f"**Strategy:** {selected_strategy['Name']}")
            st.write(f"**Timeframe:** {selected_strategy['Timeframe']}")
            st.write(f"**Pairs:** {selected_strategy['Pairs']}")
            st.write(f"**Rules:** {selected_strategy['Rules']}")
            if st.button("Run Backtest"):
                st.warning("Backtesting requires integration with a trading platform (e.g., MT5). This is a simulated result.")
                simulated_results = pd.DataFrame({
                    "Date": pd.date_range(start='2025-01-01', periods=30, freq='D'),
                    "PnL": np.random.normal(0, 50, 30),
                    "Win": np.random.choice([True, False], 30)
                })
                simulated_results['Equity'] = simulated_results['PnL'].cumsum()
                win_rate = simulated_results['Win'].mean()
                net_profit = simulated_results['PnL'].sum()
                st.session_state.strategies.loc[strategy_to_backtest, 'Performance'] = f"Win Rate: {win_rate:.2%}, Net Profit: ${net_profit:.2f}"
                if 'logged_in_user' in st.session_state:
                    username = st.session_state.logged_in_user
                    user_data = {
                        'xp': st.session_state.get('xp', 0),
                        'level': st.session_state.get('level', 0),
                        'badges': st.session_state.get('badges', []),
                        'streak': st.session_state.get('streak', 0),
                        'last_journal_date': st.session_state.get('last_journal_date', None),
                        'drawings': st.session_state.get('drawings', {}),
                        'tools_trade_journal': st.session_state.tools_trade_journal.to_dict('records'),
                        'strategies': st.session_state.strategies.to_dict('records'),
                        'emotion_log': st.session_state.get('emotion_log', pd.DataFrame()).to_dict('records'),
                        'reflection_log': st.session_state.get('reflection_log', pd.DataFrame()).to_dict('records')
                    }
                    try:
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                        conn.commit()
                        st.success("Backtest results saved!")
                    except Exception as e:
                        st.error(f"Failed to save backtest results: {str(e)}")
                        logging.error(f"Error saving backtest results for {username}: {str(e)}")
                fig = px.line(simulated_results, x='Date', y='Equity', title="Simulated Equity Curve", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"**Win Rate:** {win_rate:.2%}")
                st.write(f"**Net Profit:** ${net_profit:.2f}")
        else:
            st.info("No strategies available for backtesting.")

elif st.session_state.current_page == 'account':
    st.title("👤 My Account")
    st.caption("Manage your profile, view progress, and sync data.")
    st.markdown('---')
    
    if 'logged_in_user' not in st.session_state:
        tab_login, tab_register = st.tabs(["Login", "Register"])
        
        with tab_login:
            st.subheader("Login")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.form_submit_button("Login"):
                    try:
                        c.execute("SELECT password, data FROM users WHERE username = ?", (username,))
                        result = c.fetchone()
                        if result:
                            stored_password, user_data = result
                            if stored_password == hashlib.sha256(password.encode()).hexdigest():
                                st.session_state.logged_in_user = username
                                user_data = json.loads(user_data)
                                st.session_state.xp = user_data.get('xp', 0)
                                st.session_state.level = user_data.get('level', 0)
                                st.session_state.badges = user_data.get('badges', [])
                                st.session_state.streak = user_data.get('streak', 0)
                                st.session_state.last_journal_date = user_data.get('last_journal_date', None)
                                st.session_state.drawings = user_data.get('drawings', {})
                                st.session_state.tools_trade_journal = pd.DataFrame(user_data.get('tools_trade_journal', [])).astype(journal_dtypes, errors='ignore')
                                st.session_state.strategies = pd.DataFrame(user_data.get('strategies', [])).astype(st.session_state.strategies.dtypes, errors='ignore')
                                st.session_state.emotion_log = pd.DataFrame(user_data.get('emotion_log', [])).astype(st.session_state.emotion_log.dtypes, errors='ignore')
                                st.session_state.reflection_log = pd.DataFrame(user_data.get('reflection_log', [])).astype(st.session_state.reflection_log.dtypes, errors='ignore')
                                st.success(f"Welcome back, {username}!")
                                logging.info(f"User {username} logged in successfully")
                                st.rerun()
                            else:
                                st.error("Incorrect password")
                                logging.warning(f"Failed login attempt for {username}: incorrect password")
                        else:
                            st.error("Username not found")
                            logging.warning(f"Failed login attempt: username {username} not found")
                    except Exception as e:
                        st.error(f"Login error: {str(e)}")
                        logging.error(f"Login error for {username}: {str(e)}")
        
        with tab_register:
            st.subheader("Register")
            with st.form("register_form"):
                new_username = st.text_input("Username", key="reg_username")
                new_password = st.text_input("Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
                if st.form_submit_button("Register"):
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                        logging.warning(f"Registration failed for {new_username}: passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                        logging.warning(f"Registration failed for {new_username}: password too short")
                    else:
                        try:
                            c.execute("SELECT username FROM users WHERE username = ?", (new_username,))
                            if c.fetchone():
                                st.error("Username already exists")
                                logging.warning(f"Registration failed: username {new_username} already exists")
                            else:
                                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                                user_data = {
                                    'xp': 0,
                                    'level': 0,
                                    'badges': [],
                                    'streak': 0,
                                    'last_journal_date': None,
                                    'drawings': {},
                                    'tools_trade_journal': [],
                                    'strategies': [],
                                    'emotion_log': [],
                                    'reflection_log': []
                                }
                                c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)",
                                        (new_username, hashed_password, json.dumps(user_data, cls=CustomJSONEncoder)))
                                conn.commit()
                                st.session_state.logged_in_user = new_username
                                st.session_state.xp = 0
                                st.session_state.level = 0
                                st.session_state.badges = []
                                st.session_state.streak = 0
                                st.session_state.last_journal_date = None
                                st.session_state.drawings = {}
                                st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
                                st.session_state.strategies = pd.DataFrame(columns=st.session_state.strategies.columns).astype(st.session_state.strategies.dtypes)
                                st.session_state.emotion_log = pd.DataFrame(columns=st.session_state.emotion_log.columns).astype(st.session_state.emotion_log.dtypes)
                                st.session_state.reflection_log = pd.DataFrame(columns=st.session_state.reflection_log.columns).astype(st.session_state.reflection_log.dtypes)
                                st.success(f"Account created for {new_username}!")
                                logging.info(f"User {new_username} registered successfully")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Registration error: {str(e)}")
                            logging.error(f"Registration error for {new_username}: {str(e)}")
    else:
        st.subheader(f"Welcome, {st.session_state.logged_in_user}")
        col1, col2, col3 = st.columns(3)
        col1.metric("XP", st.session_state.get('xp', 0))
        col2.metric("Level", st.session_state.get('level', 0))
        col3.metric("Streak", f"{st.session_state.get('streak', 0)} days")
        st.markdown("**Badges**")
        badges = st.session_state.get('badges', [])
        if badges:
            st.write(", ".join(badges))
        else:
            st.write("No badges yet. Keep trading and logging to earn some!")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                if key not in ['current_page', 'current_subpage', 'show_tools_submenu']:
                    del st.session_state[key]
            st.success("Logged out successfully")
            logging.info(f"User {st.session_state.logged_in_user} logged out")
            st.rerun()
        st.subheader("Account Settings")
        with st.form("update_password_form"):
            current_password = st.text_input("Current Password", type="password", key="current_password")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_new_password")
            if st.form_submit_button("Update Password"):
                try:
                    username = st.session_state.logged_in_user
                    c.execute("SELECT password FROM users WHERE username = ?", (username,))
                    stored_password = c.fetchone()[0]
                    if hashlib.sha256(current_password.encode()).hexdigest() == stored_password:
                        if new_password == confirm_new_password:
                            if len(new_password) >= 6:
                                hashed_new_password = hashlib.sha256(new_password.encode()).hexdigest()
                                c.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_new_password, username))
                                conn.commit()
                                st.success("Password updated successfully!")
                                logging.info(f"Password updated for user {username}")
                            else:
                                st.error("New password must be at least 6 characters")
                                logging.warning(f"Password update failed for {username}: new password too short")
                        else:
                            st.error("New passwords do not match")
                            logging.warning(f"Password update failed for {username}: passwords do not match")
                    else:
                        st.error("Current password is incorrect")
                        logging.warning(f"Password update failed for {username}: incorrect current password")
                except Exception as e:
                    st.error(f"Password update error: {str(e)}")
                    logging.error(f"Password update error for {username}: {str(e)}")
        if st.button("Delete Account"):
            st.warning("Are you sure you want to delete your account? This action is irreversible.")
            if st.button("Confirm Delete"):
                try:
                    username = st.session_state.logged_in_user
                    c.execute("DELETE FROM users WHERE username = ?", (username,))
                    conn.commit()
                    for key in list(st.session_state.keys()):
                        if key not in ['current_page', 'current_subpage', 'show_tools_submenu']:
                            del st.session_state[key]
                    st.success("Account deleted successfully")
                    logging.info(f"Account deleted for user {username}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Account deletion error: {str(e)}")
                    logging.error(f"Account deletion error for {username}: {str(e)}")

elif st.session_state.current_page == 'community':
    st.title("🌐 Community Trade Ideas")
    st.caption("Share and explore trade ideas with the trading community.")
    st.markdown('---')
    
    tab_share, tab_explore = st.tabs(["📤 Share Idea", "📥 Explore Ideas"])

    with tab_share:
        st.subheader("Share a Trade Idea")
        if 'logged_in_user' in st.session_state:
            with st.form("trade_idea_form"):
                col1, col2 = st.columns(2)
                with col1:
                    pair = st.selectbox("Currency Pair", list(pairs_map.keys()), key="idea_pair")
                    direction = st.selectbox("Direction", ["Bullish", "Bearish", "Neutral"], key="idea_direction")
                with col2:
                    description = st.text_area("Trade Idea Description", key="idea_description")
                    image_file = st.file_uploader("Upload Chart (optional)", type=["png", "jpg", "jpeg"], key="idea_image")
                submit_idea = st.form_submit_button("Share Idea")
                if submit_idea:
                    idea_id = _ta_hash()
                    image_path = ""
                    if image_file:
                        user_dir = _ta_user_dir(st.session_state.logged_in_user)
                        image_path = os.path.join(user_dir, "community_images", f"{idea_id}.png")
                        try:
                            with open(image_path, "wb") as f:
                                f.write(image_file.getbuffer())
                            logging.info(f"Image saved for trade idea {idea_id}")
                        except Exception as e:
                            st.error(f"Failed to save image: {str(e)}")
                            logging.error(f"Image save error for idea {idea_id}: {str(e)}")
                    new_idea = {
                        "Username": st.session_state.logged_in_user,
                        "Pair": pair,
                        "Direction": direction,
                        "Description": description,
                        "Timestamp": pd.to_datetime(datetime.now()).isoformat(),
                        "IdeaID": idea_id,
                        "ImagePath": image_path
                    }
                    st.session_state.trade_ideas = pd.concat(
                        [st.session_state.trade_ideas, pd.DataFrame([new_idea])],
                        ignore_index=True
                    )
                    try:
                        _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                        ta_update_xp(20)
                        st.success("Trade idea shared successfully!")
                        logging.info(f"Trade idea shared by {st.session_state.logged_in_user}: {idea_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to share trade idea: {str(e)}")
                        logging.error(f"Error sharing trade idea for {st.session_state.logged_in_user}: {str(e)}")
        else:
            st.info("Please log in to share trade ideas.")

    with tab_explore:
        st.subheader("Explore Community Trade Ideas")
        if not st.session_state.trade_ideas.empty:
            pair_filter = st.multiselect(
                "Filter by Pair",
                options=st.session_state.trade_ideas['Pair'].unique(),
                default=st.session_state.trade_ideas['Pair'].unique()
            )
            direction_filter = st.multiselect(
                "Filter by Direction",
                options=["Bullish", "Bearish", "Neutral"],
                default=["Bullish", "Bearish", "Neutral"]
            )
            filtered_ideas = st.session_state.trade_ideas[
                (st.session_state.trade_ideas['Pair'].isin(pair_filter)) &
                (st.session_state.trade_ideas['Direction'].isin(direction_filter))
            ]
            for _, idea in filtered_ideas.iterrows():
                st.markdown(f"""
                **{idea['Username']}** - {idea['Pair']} ({idea['Direction']})  
                *Posted: {idea['Timestamp']}*  
                {idea['Description']}
                """)
                if idea['ImagePath'] and os.path.exists(idea['ImagePath']):
                    st.image(idea['ImagePath'], caption="Chart", use_column_width=True)
                st.markdown("---")
        else:
            st.info("No trade ideas available. Be the first to share one!")

elif st.session_state.current_page == 'tools':
    st.title("🛠️ Tools")
    st.caption("Useful tools for forex trading analysis and management.")
    st.markdown('---')

    # Tools Submenu
    tools = ["Risk Calculator", "Pip Calculator", "Trade Journal", "Position Sizer"]
    if st.session_state.show_tools_submenu:
        for tool in tools:
            if st.button(tool, key=f"tool_{tool.lower().replace(' ', '_')}"):
                st.session_state.current_subpage = tool
                st.rerun()
    else:
        if st.button("Show Tools", key="show_tools"):
            st.session_state.show_tools_submenu = True
            st.rerun()

    if st.session_state.current_subpage == "Risk Calculator":
        st.subheader("Risk Calculator")
        with st.form("risk_calculator_form"):
            col1, col2 = st.columns(2)
            with col1:
                account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=10000.0, step=100.0)
                risk_percentage = st.number_input("Risk Percentage (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            with col2:
                stop_loss_pips = st.number_input("Stop Loss (Pips)", min_value=0.0, value=50.0, step=1.0)
                pair = st.selectbox("Currency Pair", list(pairs_map.keys()))
            if st.form_submit_button("Calculate"):
                pip_value = 10 if "JPY" in pair else 100
                risk_amount = account_balance * (risk_percentage / 100)
                lot_size = risk_amount / (stop_loss_pips * pip_value)
                st.write(f"**Risk Amount:** ${risk_amount:.2f}")
                st.write(f"**Recommended Lot Size:** {lot_size:.2f} lots")
                ta_update_xp(5)
                logging.info(f"Risk calculation performed for pair {pair}")

    elif st.session_state.current_subpage == "Pip Calculator":
        st.subheader("Pip Calculator")
        with st.form("pip_calculator_form"):
            col1, col2 = st.columns(2)
            with col1:
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, format="%.5f")
                exit_price = st.number_input("Exit Price", min_value=0.0, step=0.0001, format="%.5f")
            with col2:
                lot_size = st.number_input("Lot Size", min_value=0.01, step=0.01, format="%.2f")
                pair = st.selectbox("Currency Pair", list(pairs_map.keys()))
            if st.form_submit_button("Calculate"):
                pip_multiplier = 100 if "JPY" in pair else 10000
                pips = abs(entry_price - exit_price) * pip_multiplier
                profit = pips * lot_size * (10 if "JPY" in pair else 100)
                st.write(f"**Pips Moved:** {pips:.2f}")
                st.write(f"**Estimated Profit/Loss:** ${profit:.2f}")
                ta_update_xp(5)
                logging.info(f"Pip calculation performed for pair {pair}")

    elif st.session_state.current_subpage == "Trade Journal":
        st.subheader("Trade Journal")
        st.write("Access your trade journal from the Backtesting page.")
        if st.button("Go to Trade Journal"):
            st.session_state.current_page = 'backtesting'
            st.session_state.current_subpage = None
            st.rerun()

    elif st.session_state.current_subpage == "Position Sizer":
        st.subheader("Position Sizer")
        with st.form("position_sizer_form"):
            col1, col2 = st.columns(2)
            with col1:
                account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=10000.0, step=100.0)
                risk_percentage = st.number_input("Risk Percentage (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            with col2:
                stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.0001, format="%.5f")
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, format="%.5f")
                pair = st.selectbox("Currency Pair", list(pairs_map.keys()))
            if st.form_submit_button("Calculate Position Size"):
                pip_multiplier = 100 if "JPY" in pair else 10000
                stop_loss_pips = abs(entry_price - stop_loss_price) * pip_multiplier
                risk_amount = account_balance * (risk_percentage / 100)
                lot_size = risk_amount / (stop_loss_pips * (10 if "JPY" in pair else 100))
                st.write(f"**Risk Amount:** ${risk_amount:.2f}")
                st.write(f"**Stop Loss Pips:** {stop_loss_pips:.2f}")
                st.write(f"**Recommended Lot Size:** {lot_size:.2f} lots")
                ta_update_xp(5)
                logging.info(f"Position size calculation performed for pair {pair}")

    else:
        st.subheader("Available Tools")
        st.write("- **Risk Calculator**: Calculate risk amount and lot size based on account balance and risk percentage.")
        st.write("- **Pip Calculator**: Estimate pips moved and profit/loss for a trade.")
        st.write("- **Trade Journal**: Access your detailed trading journal.")
        st.write("- **Position Sizer**: Determine optimal position size based on risk parameters.")
        st.write("Click 'Show Tools' to access these features.")

elif st.session_state.current_page == 'Zenvo Academy':
    st.title("🎓 Zenvo Academy")
    st.caption("Learn forex trading concepts and strategies.")
    st.markdown('---')
    st.markdown("""
    Zenvo Academy provides educational resources to improve your trading skills. 
    Explore tutorials, videos, and quizzes to enhance your knowledge.
    """)

    tab_tutorials, tab_videos, tab_quiz = st.tabs(["📚 Tutorials", "🎥 Videos", "❓ Quiz"])

    with tab_tutorials:
        st.subheader("Tutorials")
        tutorials = [
            {
                "title": "Introduction to Forex Trading",
                "content": """
                Forex trading involves buying and selling currency pairs to profit from exchange rate fluctuations.
                Key concepts include:
                - **Pip**: The smallest price move in a currency pair.
                - **Leverage**: Borrowing capital to increase trade size.
                - **Margin**: Collateral required to open a leveraged position.
                Start with demo accounts to practice without risk.
                """
            },
            {
                "title": "Technical Analysis Basics",
                "content": """
                Technical analysis uses historical price data to predict future movements.
                Common tools:
                - **Moving Averages**: Smooth price data to identify trends.
                - **Support/Resistance**: Levels where price tends to reverse.
                - **Indicators**: RSI, MACD, Bollinger Bands for momentum and volatility.
                Practice identifying patterns like head and shoulders or double tops/bottoms.
                """
            },
            {
                "title": "Risk Management",
                "content": """
                Effective risk management is crucial for long-term success.
                - **Risk per Trade**: Limit to 1-2% of account balance.
                - **Stop Loss**: Set to limit potential losses.
                - **Position Sizing**: Adjust based on stop loss distance and risk tolerance.
                Use the Risk Calculator in the Tools section to practice.
                """
            }
        ]
        for tutorial in tutorials:
            with st.expander(tutorial["title"]):
                st.markdown(tutorial["content"])
                if st.button(f"Mark {tutorial['title']} as Completed", key=f"tutorial_{tutorial['title']}"):
                    ta_update_xp(10)
                    st.success("Tutorial marked as completed!")
                    logging.info(f"Tutorial completed: {tutorial['title']}")

    with tab_videos:
        st.subheader("Video Lessons")
        videos = [
            {"title": "Forex Trading for Beginners", "url": "https://www.youtube.com/watch?v=example1"},
            {"title": "Advanced Technical Analysis", "url": "https://www.youtube.com/watch?v=example2"},
            {"title": "Psychology of Trading", "url": "https://www.youtube.com/watch?v=example3"}
        ]
        for video in videos:
            st.markdown(f"**{video['title']}**")
            st.write("Placeholder for embedded video (requires actual YouTube embed code).")
            st.write(f"Link: {video['url']}")
            if st.button(f"Watch {video['title']}", key=f"video_{video['title']}"):
                ta_update_xp(15)
                st.success("Video marked as watched!")
                logging.info(f"Video watched: {video['title']}")

    with tab_quiz:
        st.subheader("Knowledge Quiz")
        questions = [
            {
                "question": "What is a pip in forex trading?",
                "options": ["A percentage point", "The smallest price move in a currency pair", "A type of order", "A trading strategy"],
                "correct": "The smallest price move in a currency pair"
            },
            {
                "question": "What does leverage allow you to do?",
                "options": ["Trade without a broker", "Increase trade size with borrowed capital", "Avoid setting stop losses", "Trade only major pairs"],
                "correct": "Increase trade size with borrowed capital"
            }
        ]
        score = 0
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                answer = st.radio(f"Select an answer for question {i+1}", q["options"], key=f"quiz_{i}")
                if answer == q["correct"]:
                    score += 1
            if st.form_submit_button("Submit Quiz"):
                st.write(f"**Your Score:** {score}/{len(questions)}")
                if score == len(questions):
                    ta_update_xp(20)
                    st.balloons()
                    st.success("Perfect score! Great job!")
                    logging.info("Quiz completed with perfect score")
                else:
                    ta_update_xp(10)
                    st.info("Quiz submitted. Review the tutorials to improve your score!")
                    logging.info(f"Quiz completed with score {score}/{len(questions)}")

# Auto-refresh for real-time updates (e.g., news)
st_autorefresh(interval=600000, key="news_refresh")  # Refresh every 10 minutes

# Check milestones
if 'logged_in_user' in st.session_state:
    ta_check_milestones(st.session_state.tools_trade_journal, st.session_state.get('mt5_df', pd.DataFrame()))

# Clean up temporary files
for temp_file in glob.glob("*.tex") + glob.glob("*.aux") + glob.glob("*.log") + glob.glob("*.out") + glob.glob("*.fdb_latexmk") + glob.glob("*.fls"):
    try:
        os.remove(temp_file)
    except:
        pass

# Close database connection on app shutdown
import atexit
atexit.register(conn.close)
