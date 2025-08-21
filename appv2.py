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
st.set_page_config(page_title="Forex Dashboard", layout="wide", initial_sidebar_state="expanded")
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
    font-size: 16px;
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
    background: #2a2a2a !important;
    color: #ccc !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: 1px solid #3a3a3a !important;
    font-size: 16px;
}}
div[data-baseweb="tab-list"] button:hover {{
    background: #3a3a3a !important;
    color: #fff !important;
    transform: translateY(-2px);
}}
/* Button styling */
.stButton button {{
    background: linear-gradient(45deg, rgba(255,215,0,0.9), rgba(255,165,0,0.9)) !important;
    color: #000 !important;
    font-weight: 600;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.3);
    border: none !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    font-size: 14px;
    line-height: 1.4;
    min-height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}
.stButton button:hover {{
    background: linear-gradient(45deg, rgba(230,194,0,0.9), rgba(255,140,0,0.9)) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}}
/* Form submit button styling */
.stFormSubmitButton button {{
    background: linear-gradient(45deg, rgba(255,215,0,0.9), rgba(255,165,0,0.9)) !important;
    color: #000 !important;
    font-weight: 600;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.3);
    border: none !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    font-size: 14px;
    line-height: 1.4;
    min-height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}
.stFormSubmitButton button:hover {{
    background: linear-gradient(45deg, rgba(230,194,0,0.9), rgba(255,140,0,0.9)) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}}
/* Card look */
.card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    transition: transform 0.3s ease;
}}
.card:hover {{
    transform: translateY(-4px);
}}
/* Dataframe styling */
.dataframe th {{
    background-color: #1f1f1f;
    color: #FFD700;
}}
.dataframe td {{
    background-color: #121212;
    color: white;
}}
/* Selectbox and input styling */
.stSelectbox, .stNumberInput, .stTextInput, .stRadio {{
    background-color: #1b1b1b;
    border-radius: 8px;
    padding: 8px;
}}
/* Expander styling */
.stExpander {{
    border: 1px solid #242424;
    border-radius: 8px;
    background-color: #1b1b1b;
}}
/* Small utility */
.small-muted {{ color:#9e9e9e; font-size:0.9rem; }}
/* Enhanced data editor input styling */
.stDataFrame .stTextInput input, .stDataFrame .stSelectbox select, .stDataFrame .stNumberInput input, .stDataFrame .stDateInput input {{
    background-color: #1b1b1b;
    color: white;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px;
    font-size: 14px;
    transition: all 0.2s ease;
}}
.stDataFrame .stTextInput input:focus, .stDataFrame .stSelectbox select:focus, .stDataFrame .stNumberInput input:focus, .stDataFrame .stDateInput input:focus {{
    background-color: #2a2a2a;
    border-color: #FFD700;
    box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
    outline: none;
}}
/* Sidebar styling for launch ready */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
    border-right: 2px solid #333;
}}
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    background: linear-gradient(45deg, #ff6b35, #f7931e);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-weight: 600;
    transition: all 0.3s ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: linear-gradient(45deg, #e55a2b, #d87419);
    transform: translateX(5px);
}}
section[data-testid="stSidebar"] .stExpander {{
    background: transparent;
}}
/* Main content padding */
div.block-container {{
    padding-top: 2rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

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
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP Price Index y/y", "Actual": "3.0%", "Forecast": "3.1%", "Previous": "3.3%", "Impact": ""},
    {"Date": "2025-08-22", "Time": "09:30", "Currency": "GBP", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.3%", "Previous": "0.2%", "Impact": "Medium"},
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
    "Entry Conditions", "Planned R:R", "News Filter", "Alerts", "Concerns",
    "Emotions", "Confluence Score 1-7", "Outcome / R:R Realised", "Notes/Journal",
    "Entry Price", "Stop Loss Price", "Take Profit Price", "Lots"
]
journal_dtypes = {
    "Date": "datetime64[ns]",
    "Symbol": str,
    "Weekly Bias": str,
    "Daily Bias": str,
    "4H Structure": str,
    "1H Structure": str,
    "Positive Correlated Pair & Bias": str,
    "Potential Entry Points": str,
    "5min/15min Setup?": str,
    "Entry Conditions": str,
    "Planned R:R": str,
    "News Filter": str,
    "Alerts": str,
    "Concerns": str,
    "Emotions": str,
    "Confluence Score 1-7": float,
    "Outcome / R:R Realised": str,
    "Notes/Journal": str,
    "Entry Price": float,
    "Stop Loss Price": float,
    "Take Profit Price": float,
    "Lots": float
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
def _ta_update_xp(amount):
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
def _ta_update_streak():
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
def _ta_check_milestones(journal_df, mt5_df):
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
st.sidebar.title("Forex Dashboard")
# Navigation items
nav_items = [
    ('fundamentals', 'Forex Fundamentals', '📅'),
    ('backtesting', 'Backtesting', '📊'),
    ('mt5', 'MT5 Performance Dashboard', '📈'),
    ('psychology', 'Psychology', '🧠'),
    ('strategy', 'Manage My Strategy', '📈'),
    ('account', 'My Account', '👤'),
    ('community', 'Community Trade Ideas', '🌐')
]
for page_key, page_name, icon in nav_items:
    if st.sidebar.button(f"{icon} {page_name}", key=f"nav_{page_key}"):
        st.session_state.current_page = page_key
        st.session_state.current_subpage = None
        st.session_state.show_tools_submenu = False
        st.experimental_rerun()
# Tools submenu
if st.sidebar.button("🛠 Tools", key="nav_tools"):
    st.session_state.show_tools_submenu = not st.session_state.show_tools_submenu
    st.session_state.current_page = 'tools'
    if not st.session_state.show_tools_submenu:
        st.session_state.current_subpage = None
    st.experimental_rerun()
# Tools submenu items
if st.session_state.show_tools_submenu:
    tools_subitems = [
        ('profit_loss', 'Profit/Loss Calculator'),
        ('alerts', 'Price Alerts'),
        ('correlation', 'Currency Correlation Heatmap'),
        ('risk_mgmt', 'Risk Management Calculator'),
        ('sessions', 'Trading Session Tracker'),
        ('drawdown', 'Drawdown Recovery Planner'),
        ('checklist', 'Pre-Trade Checklist'),
        ('premarket', 'Pre-Market Checklist')
    ]
    for sub_key, sub_name in tools_subitems:
        if st.sidebar.button(sub_name, key=f"sub_{sub_key}"):
            st.session_state.current_subpage = sub_key
            st.experimental_rerun()
# Settings
if st.sidebar.button("⚙ Settings", key="nav_settings"):
    st.session_state.current_page = 'settings'
    st.experimental_rerun()
# Logout
if st.sidebar.button("🚪 Logout", key="nav_logout"):
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
        st.experimental_rerun()
# =========================================================
# MAIN APPLICATION
# =========================================================
def show_fundamentals():
    st.write("Fundamentals page under construction.")
elif st.session_state.current_page == 'backtesting':
    show_backtesting()
elif st.session_state.current_page == 'mt5':
    show_mt5()
elif st.session_state.current_page == 'tools':
    show_tools()
elif st.session_state.current_page == 'psychology':
    show_psychology()
elif st.session_state.current_page == 'strategy':
    show_strategy()
elif st.session_state.current_page == 'account':
    show_account()
elif st.session_state.current_page == 'community':
    show_community()
elif st.session_state.current_page == 'settings':
    show_settings()
# Close database connection
conn.close()
