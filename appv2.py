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
import streamlit as st
st.markdown(
    """
    <style>
    /* Hide Streamlit top-right menu */
    #MainMenu {visibility: hidden !important;}
    /* Hide Streamlit footer (bottom-left) */
    footer {visibility: hidden !important;}
    /* Hide the GitHub / Share banner (bottom-right) */
    [data-testid="stDecoration"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)
import streamlit as st
st.markdown(
    """
    <style>
    /* Remove top padding and margins for main content */
    .css-18e3th9, .css-1d391kg {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    /* Optional: reduce padding inside Streamlit containers */
    .block-container {
        padding-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Gridline background settings ---
grid_color = "#58b3b1" # gridline color
grid_opacity = 0.16 # 0.0 (transparent) to 1.0 (solid)
grid_size = 40 # distance between gridlines in px
# Convert HEX to RGB
r = int(grid_color[1:3], 16)
g = int(grid_color[3:5], 16)
b = int(grid_color[5:7], 16)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #000000; /* black background */
        background-image:
            linear-gradient(rgba({r}, {g}, {b}, {grid_opacity}) 1px, transparent 1px),
            linear-gradient(90deg, rgba({r}, {g}, {b}, {grid_opacity}) 1px, transparent 1px);
        background-size: {grid_size}px {grid_size}px;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# === TA_PRO HELPERS START ===
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
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS community_data (key TEXT PRIMARY KEY, data TEXT)''')
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
# =========================================================
# CUSTOM CSS + JS
# =========================================================
bg_opacity = 0.5
st.markdown(
    """
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
    }
    /* Active page button style */
    section[data-testid="stSidebar"] div.stButton > button[data-active="true"] {
        background: rgba(88, 179, 177, 0.7) !important;
        color: #ffffff !important;
    }
    /* Adjust button size dynamically */
    @media (max-height: 800px) {
        section[data-testid="stSidebar"] div.stButton > button {
            font-size: 14px !important;
            padding: 8px !important;
        }
    }
    @media (max-height: 600px) {
        section[data-testid="stSidebar"] div.stButton > button {
            font-size: 12px !important;
            padding: 6px !important;
        }
    }
    </style>
    <script>
    // Ensure dynamically loaded buttons get the style applied and set active state
    document.addEventListener("DOMContentLoaded", function() {
        const applyButtonStyles = () => {
            let buttons = document.querySelectorAll('section[data-testid="stSidebar"] div.stButton > button');
            buttons.forEach(btn => {
                btn.style.width = "200px";
                btn.style.background = "linear-gradient(to right, rgba(0, 0, 0, 0.7), rgba(88, 179, 177, 0.7))";
                btn.style.color = "#ffffff";
                btn.style.border = "none";
                btn.style.borderRadius = "5px";
                btn.style.padding = "10px";
                btn.style.margin = "5px 0";
                btn.style.fontWeight = "bold";
                btn.style.fontSize = "16px";
                btn.style.textAlign = "left";
                btn.style.display = "block";
                btn.style.boxSizing = "border-box";
                btn.style.whiteSpace = "nowrap";
                btn.style.overflow = "hidden";
                btn.style.textOverflow = "ellipsis";
                btn.style.transition = "all 0.3s ease";
                // Set active state based on current page
                const page = window.location.hash || '#fundamentals';
                if (btn.textContent.trim() === st.session_state.current_page.replace('_', ' ').replace(/^\w/, c => c.toUpperCase())) {
                    btn.setAttribute('data-active', 'true');
                } else {
                    btn.removeAttribute('data-active');
                }
            });
            // Adjust sidebar height to fit buttons without scrolling
            const sidebar = document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) {
                const buttonHeight = 40; // Approximate height of each button including margin
                const buttonCount = buttons.length;
                const totalHeight = buttonHeight * buttonCount;
                if (totalHeight > window.innerHeight) {
                    sidebar.style.height = `${window.innerHeight}px`;
                    sidebar.style.overflowY = "hidden";
                } else {
                    sidebar.style.height = `${totalHeight}px`;
                    sidebar.style.overflowY = "hidden";
                }
            }
        };
        // Initial apply
        applyButtonStyles();
        // Observe sidebar for dynamically added buttons
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if(sidebar){
            const observer = new MutationObserver(applyButtonStyles);
            observer.observe(sidebar, { childList: true, subtree: true });
        }
        // Handle window resize
        window.addEventListener('resize', applyButtonStyles);
    });
    </script>
    """,
    unsafe_allow_html=True,
)
# =========================================================
# =========================================================
# =========================================================
# =========================================================
# HELPERS / DATA
# =========================================================
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
# Initialize drawings in session_state
if "drawings" not in st.session_state:
    st.session_state.drawings = {}
    logging.info("Initialized st.session_state.drawings")
# Define journal columns and dtypes
journal_cols = [
    "Date", "Symbol", "Weekly Bias", "Daily Bias", "4H Structure", "1H Structure",
    "Positive Correlated Pair & Bias", "Potential Entry Points", "5min/15min Setup?",
    "Entry Conditions", "Planned R:R", "News Filter", "Alerts", "Concerns", "Emotions",
    "Confluence Score 1-7", "Outcome / R:R Realised", "Notes/Journal",
    "Entry Price", "Stop Loss Price", "Take Profit Price", "Lots"
]
journal_dtypes = {
    "Date": "datetime64[ns]", "Symbol": str, "Weekly Bias": str, "Daily Bias": str,
    "4H Structure": str, "1H Structure": str, "Positive Correlated Pair & Bias": str,
    "Potential Entry Points": str, "5min/15min Setup?": str, "Entry Conditions": str,
    "Planned R:R": str, "News Filter": str, "Alerts": str, "Concerns": str, "Emotions": str,
    "Confluence Score 1-7": float, "Outcome / R:R Realised": str, "Notes/Journal": str,
    "Entry Price": float, "Stop Loss Price": float, "Take Profit Price": float, "Lots": float
}
# Initialize trading journal with proper dtypes
if "tools_trade_journal" not in st.session_state or st.session_state.tools_trade_journal.empty:
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
else:
    # Ensure existing journal matches new structure
    current_journal = st.session_state.tools_trade_journal
    missing_cols = [col for col in journal_cols if col not in current_journal.columns]
    if missing_cols:
        for col in missing_cols:
            current_journal[col] = pd.Series(dtype=journal_dtypes[col])
    # Reorder columns and apply dtypes
    st.session_state.tools_trade_journal = current_journal[journal_cols].astype(journal_dtypes, errors='ignore')
# Initialize temporary journal for form
if "temp_journal" not in st.session_state:
    st.session_state.temp_journal = None
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
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
            conn.commit()
            st.session_state.xp = user_data['xp']
            st.session_state.level = user_data['level']
            st.session_state.badges = user_data['badges']
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
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
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
# Load community data
if "trade_ideas" not in st.session_state:
    loaded_ideas = _ta_load_community('trade_ideas', [])
    st.session_state.trade_ideas = pd.DataFrame(loaded_ideas, columns=["Username", "Pair", "Direction", "Description", "Timestamp", "IdeaID", "ImagePath"]) if loaded_ideas else pd.DataFrame(columns=["Username", "Pair", "Direction", "Description", "Timestamp", "IdeaID", "ImagePath"])
if "community_templates" not in st.session_state:
    loaded_templates = _ta_load_community('templates', [])
    st.session_state.community_templates = pd.DataFrame(loaded_templates, columns=["Username", "Type", "Name", "Content", "Timestamp", "ID"]) if loaded_templates else pd.DataFrame(columns=["Username", "Type", "Name", "Content", "Timestamp", "ID"])
# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'fundamentals'
if 'current_subpage' not in st.session_state:
    st.session_state.current_subpage = None
if 'show_tools_submenu' not in st.session_state:
    st.session_state.show_tools_submenu = False
# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
from PIL import Image
import io
import base64
import io
import base64
from PIL import Image
# ---- Reduce top padding in the sidebar ----
st.markdown(
    """
    <style>
    /* Streamlit sidebar: remove top padding to move content up */
    .sidebar-content {
        padding-top: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---- Load and resize the logo ----
logo = Image.open("logo22.png")
logo = logo.resize((60, 50)) # adjust width/height as needed
# ---- Convert logo to base64 ----
buffered = io.BytesIO()
logo.save(buffered, format="PNG")
logo_str = base64.b64encode(buffered.getvalue()).decode()
# ---- Display logo centered in the sidebar ----
st.sidebar.markdown(
    f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <img src="data:image/png;base64,{logo_str}" width="60" height="50"/>
    </div>
    """,
    unsafe_allow_html=True
)
# Navigation items
nav_items = [
    ('fundamentals', 'Forex Fundamentals'),
    ('backtesting', 'Backtesting'),
    ('mt5', 'Performance Dashboard'),
    ('psychology', 'Psychology'),
    ('strategy', 'Manage My Strategy'),
    ('account', 'My Account'),
    ('community', 'Community Trade Ideas'),
    ('tools', 'Tools'),
    #('settings', 'Settings')
]
for page_key, page_name in nav_items:
    if st.sidebar.button(page_name, key=f"nav_{page_key}"):
        st.session_state.current_page = page_key
        st.session_state.current_subpage = None
        st.session_state.show_tools_submenu = False
        st.rerun()
# Logout
if st.sidebar.button("Logout", key="nav_logout"):
    if 'logged_in_user' in st.session_state:
        del st.session_state.logged_in_user
    st.session_state.drawings = {}
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
    st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"])
    st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
    st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
    st.session_state.xp = 0
    st.session_state.level = 0
    st.session_state.badges = []
    st.session_state.streak = 0
    st.success("Logged out successfully!")
    logging.info("User logged out")
    st.rerun()
# =========================================================
# =========================================================
# MAIN APPLICATION
# =========================================================
if st.session_state.current_page == 'fundamentals':
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📅 Forex Fundamentals")
        st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
        st.markdown('---')
    with col2:
        st.info("See the Backtesting tab for live charts + detailed news.")
    # Economic Calendar
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
            styles = ['background-color: #171447; color: white' if col == 'Currency' else 'background-color: #171447' for col in row.index]
        if st.session_state.selected_currency_2 and row['Currency'] == st.session_state.selected_currency_2:
            styles = ['background-color: #471414; color: white' if col == 'Currency' else 'background-color: #471414' for col in row.index]
        return styles
    st.dataframe(econ_df.style.apply(highlight_currency, axis=1), use_container_width=True, height=360)
    # Interest rate tiles
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
    colors = ["#171447", "#471414", "#144714", "#474714"]
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
    # Major High-Impact Events
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
        positive_impact = "<br>".join([f"{k}: {v}" for k, v in ev["impact_positive"].items()])
        negative_impact = "<br>".join([f"{k}: {v}" for k, v in ev["impact_negative"].items()])
        st.markdown(
            f"""
            <div style="background-color: #000000; padding: 10px; border-radius: 5px;">
            <strong>{ev['event']}</strong><br>
            What it is: {ev['description']}<br>
            Why it matters: {ev['why_it_matters']}<br>
            Positive →<br>
            {positive_impact}<br>
            Negative →<br>
            {negative_impact}
            </div>
            """,
            unsafe_allow_html=True,
        )
elif st.session_state.current_page == 'backtesting':
    st.title("📊 Backtesting")
    st.caption("Live TradingView chart for backtesting and trading journal for the selected pair.")
    st.markdown('---')
    # Pair selector & symbol map (28 major & minor pairs)
    pairs_map = {
        # Majors
        "EUR/USD": "FX:EURUSD",
        "USD/JPY": "FX:USDJPY",
        "GBP/USD": "FX:GBPUSD",
        "USD/CHF": "OANDA:USDCHF",
        "AUD/USD": "FX:AUDUSD",
        "NZD/USD": "OANDA:NZDUSD",
        "USD/CAD": "CMCMARKETS:USDCAD",
        # Crosses / Minors
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
    # Load initial drawings if available
    if "logged_in_user" in st.session_state and pair not in st.session_state.drawings:
        username = st.session_state.logged_in_user
        logging.info(f"Loading drawings for user {username}, pair {pair}")
        try:
            c.execute("SELECT data FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result:
                user_data = json.loads(result[0])
                st.session_state.drawings[pair] = user_data.get("drawings", {}).get(pair, {})
                logging.info(f"Loaded drawings for {pair}: {st.session_state.drawings[pair]}")
            else:
                logging.warning(f"No data found for user {username}")
        except Exception as e:
            logging.error(f"Error loading drawings for {username}: {str(e)}")
            st.error(f"Failed to load drawings: {str(e)}")
    initial_content = json.dumps(st.session_state.drawings.get(pair, {}))
    # TradingView widget
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
    components.html(tv_html, height=820, scrolling=False)
    # Save, Load, and Refresh buttons
    if "logged_in_user" in st.session_state:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Save Drawings", key="bt_save_drawings"):
                logging.info(f"Save Drawings button clicked for pair {pair}")
                save_script = f"""
                <script>
                parent.window.postMessage({{action: 'save_drawings', pair: '{pair}'}}, '*');
                </script>
                """
                components.html(save_script, height=0)
                logging.info(f"Triggered save script for {pair}")
                st.session_state[f"bt_save_trigger_{pair}"] = True
        with col2:
            if st.button("Load Drawings", key="bt_load_drawings"):
                username = st.session_state.logged_in_user
                logging.info(f"Load Drawings button clicked for user {username}, pair {pair}")
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result:
                        user_data = json.loads(result[0])
                        content = user_data.get("drawings", {}).get(pair, {})
                        if content:
                            load_script = f"""
                            <script>
                            parent.window.postMessage({{action: 'load_drawings', pair: '{pair}', content: {json.dumps(content)}}}, '*');
                            </script>
                            """
                            components.html(load_script, height=0)
                            st.success("Drawings loaded successfully!")
                            logging.info(f"Successfully loaded drawings for {pair}")
                        else:
                            st.info("No saved drawings for this pair.")
                            logging.info(f"No saved drawings found for {pair}")
                    else:
                        st.error("Failed to load user data.")
                        logging.error(f"No user data found for {username}")
                except Exception as e:
                    st.error(f"Failed to load drawings: {str(e)}")
                    logging.error(f"Error loading drawings for {username}: {str(e)}")
        with col3:
            if st.button("Refresh Account", key="bt_refresh_account"):
                username = st.session_state.logged_in_user
                logging.info(f"Refresh Account button clicked for user {username}")
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result:
                        user_data = json.loads(result[0])
                        st.session_state.drawings = user_data.get("drawings", {})
                        st.success("Account synced successfully!")
                        logging.info(f"Account synced for {username}: {st.session_state.drawings}")
                    else:
                        st.error("Failed to sync account.")
                        logging.error(f"No user data found for {username}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to sync account: {str(e)}")
                    logging.error(f"Error syncing account for {username}: {str(e)}")
    # Check for saved drawings from postMessage
    drawings_key = f"bt_drawings_key_{pair}"
    if drawings_key in st.session_state and st.session_state.get(f"bt_save_trigger_{pair}", False):
        content = st.session_state[drawings_key]
        logging.info(f"Received drawing content for {pair}: {content}")
        if content and isinstance(content, dict) and content:
            username = st.session_state.logged_in_user
            try:
                c.execute("SELECT data FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                user_data = json.loads(result[0]) if result else {}
                user_data.setdefault("drawings", {})[pair] = content
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
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
        logging.info("User not logged in, save/load drawings disabled")
    # Backtesting Journal
    st.markdown("### 📝 Trading Journal")
    # Configure column settings for data editor
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
        "Lots": st.column_config.NumberColumn("Lots", format="%.2f")
    }
    # Prepare transposed journal for display
    if st.session_state.tools_trade_journal.empty:
        # Initialize with one trade column if empty
        transposed_journal = pd.DataFrame(index=journal_cols, columns=["Trade 1"]).astype(object)
    else:
        # Transpose the journal: fields as rows, trades as columns
        transposed_journal = st.session_state.tools_trade_journal.transpose()
        # Rename columns to "Trade 1", "Trade 2", etc.
        transposed_journal.columns = [f"Trade {i+1}" for i in range(len(transposed_journal.columns))]
    # Button to add new trade column
    if st.button("➕ Add New Trade", key="bt_add_trade_button"):
        current_trades = transposed_journal.columns.tolist()
        next_trade_num = len(current_trades) + 1
        new_trade = f"Trade {next_trade_num}"
        transposed_journal[new_trade] = pd.Series(index=journal_cols, dtype=object)
        updated_journal = transposed_journal.transpose().reset_index(drop=True)
        updated_journal.columns = journal_cols
        st.session_state.tools_trade_journal = updated_journal.astype(journal_dtypes, errors='ignore')
        st.session_state.temp_journal = None
        st.rerun()
    # Dynamically configure columns for trades
    transposed_column_config = {}
    for col in transposed_journal.columns:
        transposed_column_config[col] = column_config
    # Use form to stabilize data editor
    old_num_trades = len(st.session_state.tools_trade_journal)
    with st.form(key="bt_journal_form"):
        updated_journal_tools = st.data_editor(
            data=transposed_journal.copy(),
            num_rows="dynamic",
            column_config=transposed_column_config,
            key="bt_backtesting_journal_static",
            use_container_width=True,
            height=800
        )
        submit_button = st.form_submit_button("Submit Journal Changes")
        if submit_button:
            if not updated_journal_tools.empty:
                updated_journal_tools = updated_journal_tools.transpose()
                updated_journal_tools.columns = journal_cols
                updated_journal_tools = updated_journal_tools.reset_index(drop=True)
                st.session_state.tools_trade_journal = updated_journal_tools.astype(journal_dtypes, errors='ignore')
                st.session_state.temp_journal = None
                new_num_trades = len(st.session_state.tools_trade_journal)
                if new_num_trades > old_num_trades:
                    added = new_num_trades - old_num_trades
                    ta_update_xp(added * 10)
                    ta_update_streak()
                st.success("Journal changes submitted successfully!")
            else:
                st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
                st.session_state.temp_journal = None
                st.rerun()
    if "logged_in_user" in st.session_state:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💾 Save to My Account", key="bt_save_journal_button"):
                username = st.session_state.logged_in_user
                journal_data = st.session_state.tools_trade_journal.to_dict(orient="records")
                logging.info(f"Saving journal for user {username}")
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    user_data = json.loads(result[0]) if result else {}
                    user_data["tools_trade_journal"] = journal_data
                    c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
                    conn.commit()
                    st.success("Trading journal saved to your account!")
                    logging.info(f"Journal saved for {username}")
                except Exception as e:
                    st.error(f"Failed to save journal: {str(e)}")
                    logging.error(f"Error saving journal for {username}: {str(e)}")
        with col2:
            if st.button("📂 Load Journal", key="bt_load_journal_button"):
                username = st.session_state.logged_in_user
                logging.info(f"Loading journal for user {username}")
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result:
                        user_data = json.loads(result[0])
                        saved_journal = user_data.get("tools_trade_journal", [])
                        if saved_journal:
                            loaded_df = pd.DataFrame(saved_journal)
                            for col in journal_cols:
                                if col not in loaded_df.columns:
                                    loaded_df[col] = pd.Series(dtype=journal_dtypes[col])
                            loaded_df = loaded_df[journal_cols].astype(journal_dtypes, errors='ignore')
                            st.session_state.tools_trade_journal = loaded_df
                            st.session_state.temp_journal = None
                            st.success("Trading journal loaded from your account!")
                            logging.info(f"Journal loaded for {username}")
                        else:
                            st.info("No saved journal found in your account.")
                            logging.info(f"No journal found for {username}")
                    else:
                        st.error("Failed to load user data.")
                        logging.error(f"No user data found for {username}")
                except Exception as e:
                    st.error(f"Failed to load journal: {str(e)}")
                    logging.error(f"Error loading journal for {username}: {str(e)}")
elif st.session_state.current_page == 'mt5':
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Performance Dashboard")
        st.caption("Analyze your trading performance by uploading your MT5 trading history CSV.")
        st.markdown('---')
    with col2:
        st.info("See the Backtesting tab for live charts + detailed news.")
    st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .metric-box.positive {
        background-color: #d4edda;
        color: #155724;
    }
    .metric-box.negative {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<br>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose your MT5 History CSV file",
            type=["csv"],
            help="Upload a CSV file exported from MetaTrader 5 containing your trading history."
        )
        st.markdown('<br>', unsafe_allow_html=True)
        if uploaded_file:
            with st.spinner("Processing your trading data..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.mt5_df = df
                    required_cols = ["Symbol", "Type", "Profit", "Volume", "Open Time", "Close Time", "Balance"]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}.")
                    else:
                        df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
                        df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")
                        # Metrics calculations
                        total_trades = len(df)
                        wins = df[df["Profit"] > 0]
                        losses = df[df["Profit"] <= 0]
                        win_rate = (len(wins) / total_trades * 100) if total_trades else 0
                        avg_win = wins["Profit"].mean() if not wins.empty else 0
                        avg_loss = losses["Profit"].mean() if not losses.empty else 0
                        profit_factor = round((wins["Profit"].sum() / abs(losses["Profit"].sum())) if not losses.empty else np.inf, 2)
                        net_profit = df["Profit"].sum()
                        biggest_win = df["Profit"].max()
                        biggest_loss = df["Profit"].min()
                        longest_win_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] > 0) if k), default=0)
                        longest_loss_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] < 0) if k), default=0)
                        total_volume = df["Volume"].sum()
                        avg_volume = df["Volume"].mean()
                        largest_volume_trade = df["Volume"].max()
                        profit_per_trade = net_profit / total_trades if total_trades else 0
                        avg_trade_duration = ((df["Close Time"] - df["Open Time"]).dt.total_seconds() / 3600).mean()
                        # Metrics list
                        metrics = [
                            ("📊 Total Trades", total_trades, "neutral"),
                            ("✅ Win Rate", f"{win_rate:.2f}%", "positive" if win_rate >= 50 else "negative"),
                            ("💰 Net Profit", f"${net_profit:,.2f}", "positive" if net_profit >= 0 else "negative"),
                            ("⚡ Profit Factor", profit_factor, "positive" if profit_factor >= 1 else "negative"),
                            ("🏆 Biggest Win", f"${biggest_win:,.2f}", "positive"),
                            ("💀 Biggest Loss", f"${biggest_loss:,.2f}", "negative"),
                            ("🔥 Longest Win Streak", longest_win_streak, "positive"),
                            ("❌ Longest Loss Streak", longest_loss_streak, "negative"),
                            ("⏱️ Avg Trade Duration", f"{avg_trade_duration:.2f}h", "neutral"),
                            ("📦 Total Volume", f"{total_volume:,.2f}", "neutral"),
                            ("📊 Avg Volume", f"{avg_volume:.2f}", "neutral"),
                            ("💵 Profit / Trade", f"${profit_per_trade:.2f}", "positive" if profit_per_trade >= 0 else "negative"),
                        ]
                        # Display metrics in three rows of four
                        for row in range(3):
                            row_metrics = metrics[row * 4:(row + 1) * 4]
                            cols = st.columns(4)
                            for i, (title, value, style) in enumerate(row_metrics):
                                with cols[i]:
                                    st.markdown(
                                        f"""
                                        <div class="metric-box {style}">
                                        {title}<br>
                                        <strong>{value}</strong>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        # Visualizations
                        st.markdown('<br>📊 Profit by Instrument<br>', unsafe_allow_html=True)
                        profit_symbol = df.groupby("Symbol")["Profit"].sum().reset_index()
                        fig_symbol = px.bar(
                            profit_symbol,
                            x="Symbol",
                            y="Profit",
                            color="Profit",
                            title="Profit by Instrument",
                            template="plotly_white",
                            color_continuous_scale=px.colors.diverging.Tealrose
                        )
                        fig_symbol.update_layout(
                            title_font_size=18,
                            title_x=0.5,
                            font_color="#333333"
                        )
                        st.plotly_chart(fig_symbol, use_container_width=True)
                        st.markdown('<br>🔎 Trade Distribution<br>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_types = px.pie(df, names="Type", title="Buy vs Sell Distribution", template="plotly_white")
                            fig_types.update_layout(title_font_size=16, title_x=0.5)
                            st.plotly_chart(fig_types, use_container_width=True)
                        with col2:
                            df["Weekday"] = df["Open Time"].dt.day_name()
                            fig_weekday = px.histogram(df, x="Weekday", color="Type", title="Trades by Day of Week", template="plotly_white")
                            fig_weekday.update_layout(title_font_size=16, title_x=0.5)
                            st.plotly_chart(fig_weekday, use_container_width=True)
                        st.success("✅ Performance Dashboard Loaded Successfully!")
                        ta_update_xp(50)
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")
                finally:
                    pass
        else:
            st.info("👆 Upload your MT5 trading history CSV to explore your performance metrics.")
    st.markdown("### 🧭 Edge Finder – Highest Expectancy Segments")
    df = st.session_state.get("mt5_df", pd.DataFrame())
    if df.empty:
        st.info("Upload trades with at least one of: timeframe, symbol, setup and 'r' (R-multiple).")
    else:
        group_cols = []
        if "timeframe" in df.columns:
            group_cols.append("timeframe")
        if "symbol" in df.columns:
            group_cols.append("symbol")
        if "setup" in df.columns:
            group_cols.append("setup")
        if group_cols:
            agg = _ta_expectancy_by_group(df, group_cols).sort_values("expectancy", ascending=False)
            st.dataframe(agg, use_container_width=True)
            top_n = st.slider("Show Top N", 5, 50, 15, key="edge_topn")
            st.plotly_chart(px.bar(agg.head(top_n), x="expectancy", y=group_cols, orientation="h"), use_container_width=True)
        else:
            st.warning("Edge Finder needs timeframe/symbol/setup columns.")
    st.markdown("### 🧩 Customisable Dashboard")
    if df.empty:
        st.info("Upload trades to customise KPIs.")
    else:
        all_kpis = [
            "Total Trades", "Win Rate", "Avg R", "Profit Factor", "Max Drawdown (PnL)",
            "Best Symbol", "Worst Symbol", "Best Timeframe", "Worst Timeframe"
        ]
        chosen = st.multiselect("Select KPIs to display", all_kpis, default=["Total Trades","Win Rate","Avg R","Profit Factor"], key="mt5_kpis")
        cols = st.columns(4)
        i = 0
        best_sym = df.groupby("symbol")["r"].mean().sort_values(ascending=False).index[0] if "symbol" in df.columns and "r" in df.columns and not df["r"].isna().all() else "—"
        worst_sym = df.groupby("symbol")["r"].mean().sort_values(ascending=True).index[0] if "symbol" in df.columns and "r" in df.columns and not df["r"].isna().all() else "—"
        best_tf = df.groupby("timeframe")["r"].mean().sort_values(ascending=False).index[0] if "timeframe" in df.columns and "r" in df.columns and not df["r"].isna().all() else "—"
        worst_tf = df.groupby("timeframe")["r"].mean().sort_values(ascending=True).index[0] if "timeframe" in df.columns and "r" in df.columns and not df["r"].isna().all() else "—"
        def _metric_map():
            return {
                "Total Trades": len(df),
                "Win Rate": ta_human_pct((df["r"]>0).mean()) if "r" in df.columns else "—",
                "Avg R": _ta_human_num(df["r"].mean()) if "r" in df.columns else "—",
                "Profit Factor": _ta_human_num(_ta_profit_factor(df)) if "pnl" in df.columns else "—",
                "Max Drawdown (PnL)": _ta_human_num((df["pnl"].fillna(0).cumsum() - df["pnl"].fillna(0).cumsum().cummax()).min()) if "pnl" in df.columns else "—",
                "Best Symbol": best_sym,
                "Worst Symbol": worst_sym,
                "Best Timeframe": best_tf,
                "Worst Timeframe": worst_tf,
            }
        for k in chosen:
            with cols[i % 4]:
                st.metric(k, _metric_map().get(k, "—"))
            i += 1
        try:
            _ta_show_badges(df)
        except Exception:
            pass
    # Dynamic Performance Reports
    st.subheader("📈 Dynamic Performance Reports")
    if not df.empty:
        group_cols = []
        if "timeframe" in df.columns:
            group_cols.append("timeframe")
        if "symbol" in df.columns:
            group_cols.append("symbol")
        if "setup" in df.columns:
            group_cols.append("setup")
        if group_cols:
            agg = _ta_expectancy_by_group(df, group_cols).sort_values("winrate", ascending=False)
            if not agg.empty:
                top_row = agg.iloc[0]
                insight = f"This month your highest probability setup was {' '.join([str(top_row[col]) for col in group_cols])} with {top_row['winrate']*100:.1f}% winrate."
                st.info(insight)
        else:
            st.info("Upload trades to generate insights.")
    # Report Export & Sharing
    if not df.empty:
        if st.button("📄 Generate Performance Report"):
            report_html = f"""
            <html>
            <body>
            <h2>Performance Report</h2>
            <p>Total Trades: {total_trades}</p>
            <p>Win Rate: {win_rate:.2f}%</p>
            <p>Net Profit: ${net_profit:,.2f}</p>
            <p>Profit Factor: {profit_factor}</p>
            <p>Biggest Win: ${biggest_win:,.2f}</p>
            <p>Biggest Loss: ${biggest_loss:,.2f}</p>
            <p>Longest Win Streak: {longest_win_streak}</p>
            <p>Longest Loss Streak: {longest_loss_streak}</p>
            <p>Avg Trade Duration: {avg_trade_duration:.2f}h</p>
            <p>Total Volume: {total_volume:,.2f}</p>
            <p>Avg Volume: {avg_volume:.2f}</p>
            <p>Profit / Trade: ${profit_per_trade:.2f}</p>
            </body>
            </html>
            """
            st.download_button(
                label="Download HTML Report",
                data=report_html,
                file_name="performance_report.html",
                mime="text/html"
            )
            st.info("Download the HTML report and share it with mentors or communities. You can print it to PDF in your browser.")
elif st.session_state.current_page == 'psychology':
    st.title("🧠 Psychology")
    st.caption("Track your emotions, reflect on your mindset, and maintain discipline.")
    st.markdown('---')
    st.subheader("📝 Emotion Tracker")
    with st.form("emotion_form"):
        emotion = st.selectbox("Current Emotion", ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral"])
        notes = st.text_area("Notes on Your Mindset")
        submit_emotion = st.form_submit_button("Log Emotion")
        if submit_emotion:
            log_entry = {
                "Date": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Emotion": emotion,
                "Notes": notes
            }
            if "emotion_log" not in st.session_state:
                st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
            st.session_state.emotion_log = pd.concat([st.session_state.emotion_log, pd.DataFrame([log_entry])], ignore_index=True)
            if "logged_in_user" in st.session_state:
                username = st.session_state.logged_in_user
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    user_data = json.loads(result[0]) if result else {}
                    user_data["emotion_log"] = st.session_state.emotion_log.to_dict(orient="records")
                    c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
                    conn.commit()
                except Exception as e:
                    logging.error(f"Error saving emotion log: {str(e)}")
            st.success("Emotion logged successfully!")
            logging.info(f"Emotion logged: {emotion}")
    if "emotion_log" in st.session_state and not st.session_state.emotion_log.empty:
        st.subheader("Your Emotion Log")
        st.dataframe(st.session_state.emotion_log, use_container_width=True)
        fig = px.histogram(st.session_state.emotion_log, x="Emotion", title="Emotion Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No emotions logged yet. Use the form above to start tracking.")
    st.subheader("🧘 Mindset Tips")
    tips = [
        "Stick to your trading plan to avoid impulsive decisions.",
        "Take breaks after losses to reset your mindset.",
        "Focus on process, not profits, to stay disciplined.",
        "Journal every trade to identify emotional patterns.",
        "Practice mindfulness to manage stress during volatile markets."
    ]
    for tip in tips:
        st.markdown(f"- {tip}")
    # Curated Education Feeds
    st.subheader("📚 Curated Trading Insights")
    insights = [
        "Risk Management: Always risk no more than 1-2% of your account per trade to preserve capital.",
        "Psychology: Master your emotions; fear and greed are the biggest enemies of traders.",
        "Setups: Focus on high-probability patterns like pin bars and engulfing candles in trending markets."
    ]
    week_num = dt.datetime.now().isocalendar()[1]
    current_insight = insights[week_num % len(insights)]
    st.info(f"Insight of the Week: {current_insight}")
    # Challenge Mode
    st.subheader("🏅 Challenge Mode")
    st.write("30-Day Journaling Discipline Challenge")
    streak = st.session_state.get('streak', 0)
    progress = min(streak / 30.0, 1.0)
    st.progress(progress)
    if progress >= 1.0:
        st.success("Challenge completed! Great job on your consistency.")
        ta_update_xp(100) # Bonus XP for completion
elif st.session_state.current_page == 'strategy':
    st.title("📈 Manage My Strategy")
    st.caption("Define, refine, and track your trading strategies.")
    st.markdown('---')
    st.subheader("➕ Add New Strategy")
    with st.form("strategy_form"):
        strategy_name = st.text_input("Strategy Name")
        description = st.text_area("Strategy Description")
        entry_rules = st.text_area("Entry Rules")
        exit_rules = st.text_area("Exit Rules")
        risk_management = st.text_area("Risk Management Rules")
        submit_strategy = st.form_submit_button("Save Strategy")
        if submit_strategy:
            strategy_data = {
                "Name": strategy_name,
                "Description": description,
                "Entry Rules": entry_rules,
                "Exit Rules": exit_rules,
                "Risk Management": risk_management,
                "Date Added": dt.datetime.now().strftime("%Y-%m-%d")
            }
            if "strategies" not in st.session_state:
                st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"])
            st.session_state.strategies = pd.concat([st.session_state.strategies, pd.DataFrame([strategy_data])], ignore_index=True)
            if "logged_in_user" in st.session_state:
                username = st.session_state.logged_in_user
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    user_data = json.loads(result[0]) if result else {}
                    user_data["strategies"] = st.session_state.strategies.to_dict(orient="records")
                    c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
                    conn.commit()
                    st.success("Strategy saved to your account!")
                    logging.info(f"Strategy saved for {username}: {strategy_name}")
                except Exception as e:
                    st.error(f"Failed to save strategy: {str(e)}")
                    logging.error(f"Error saving strategy for {username}: {str(e)}")
            st.success(f"Strategy '{strategy_name}' added successfully!")
    if "strategies" in st.session_state and not st.session_state.strategies.empty:
        st.subheader("Your Strategies")
        for idx, row in st.session_state.strategies.iterrows():
            with st.expander(f"Strategy: {row['Name']} (Added: {row['Date Added']})"):
                st.markdown(f"Description: {row['Description']}")
                st.markdown(f"Entry Rules: {row['Entry Rules']}")
                st.markdown(f"Exit Rules: {row['Exit Rules']}")
                st.markdown(f"Risk Management: {row['Risk Management']}")
                if st.button("Delete Strategy", key=f"delete_strategy_{idx}"):
                    st.session_state.strategies = st.session_state.strategies.drop(idx).reset_index(drop=True)
                    if "logged_in_user" in st.session_state:
                        username = st.session_state.logged_in_user
                        try:
                            c.execute("SELECT data FROM users WHERE username = ?", (username,))
                            result = c.fetchone()
                            user_data = json.loads(result[0]) if result else {}
                            user_data["strategies"] = st.session_state.strategies.to_dict(orient="records")
                            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
                            conn.commit()
                            st.success("Strategy deleted and account updated!")
                            logging.info(f"Strategy deleted for {username}")
                        except Exception as e:
                            st.error(f"Failed to delete strategy: {str(e)}")
                            logging.error(f"Error deleting strategy for {username}: {str(e)}")
                    st.rerun()
    else:
        st.info("No strategies defined yet. Add one above.")
    # Evolving Playbook
    st.subheader("📖 Evolving Playbook")
    journal_df = st.session_state.tools_trade_journal
    mt5_df = st.session_state.get('mt5_df', pd.DataFrame())
    combined_df = pd.concat([journal_df, mt5_df], ignore_index=True) if not mt5_df.empty else journal_df
    group_cols = ["Symbol"] if "Symbol" in combined_df.columns else []
    if "Outcome / R:R Realised" in combined_df.columns:
        combined_df['r'] = combined_df["Outcome / R:R Realised"].apply(lambda x: float(x.split(':')[1]) if isinstance(x, str) and ':' in x else
