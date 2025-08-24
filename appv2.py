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
    
    #/* Optional: remove extra padding/margin from main page */
    #.css-1d391kg {padding-top: 0rem !important;}
    #</style>
    """,
    unsafe_allow_html=True
)

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
    with st.expander("🎮 Gamification: Streaks & Badges", expanded=False):
        streaks = _ta_compute_streaks(df) if df is not None else {"current":0,"best":0}
        col1, col2 = st.columns(2)
        col1.metric("Current Green-Day Streak", streaks.get("current",0))
        col2.metric("Best Streak", streaks.get("best",0))
        if df is not None and "emotions" in df.columns:
            emo_logged = int((df["emotions"].fillna("").astype(str).str.len()>0).sum())
            st.caption(f"💭 Emotion-logged trades: {emo_logged}")

def _ta_save_journal(username, journal_df):
    try:
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        user_data = json.loads(result[0]) if result else {}
        user_data["tools_trade_journal"] = journal_df.replace({pd.NA: None, float('nan'): None}).to_dict('records')
        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
        conn.commit()
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    except Exception as e:
        logging.error(f"Failed to save journal for {username}: {str(e)}")
        st.error(f"Failed to save journal: {str(e)}")
        return False

# === TA_PRO HELPERS END ===

# Path to SQLite DB
DB_FILE = "users.db"

# XP notification system
def show_xp_notification(xp_gained):
    """Show a visually appealing XP notification"""
    notification_html = f"""
    <div id="xp-notification" style="
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #58b3b1, #4d7171);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(88, 179, 177, 0.3);
        z-index: 9999;
        animation: slideInRight 0.5s ease-out, fadeOut 0.5s ease-out 3s forwards;
        font-weight: bold;
        border: 2px solid #fff;
        backdrop-filter: blur(10px);
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 24px;">⭐</div>
            <div>
                <div style="font-size: 16px;">+{xp_gained} XP Earned!</div>
                <div style="font-size: 12px; opacity: 0.8;">Keep up the great work!</div>
            </div>
        </div>
    </div>
    <style>
        @keyframes slideInRight {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        @keyframes fadeOut {{
            from {{ opacity: 1; }}
            to {{ opacity: 0; }}
        }}
    </style>
    """
    st.components.v1.html(notification_html, height=0)

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

# Custom JSON encoder for handling datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)

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
            old_xp = user_data.get('xp', 0)
            user_data['xp'] = old_xp + amount
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
            
            # Show XP notification
            show_xp_notification(amount)

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
            styles = ['background-color: #4c7170; color: white' if col == 'Currency' else 'background-color: #4c7170' for col in row.index]
        if st.session_state.selected_currency_2 and row['Currency'] == st.session_state.selected_currency_2:
            styles = ['background-color: #2e4747; color: white' if col == 'Currency' else 'background-color: #2e4747' for col in row.index]
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
    st.title("📈 Backtesting")
    st.caption("Live TradingView chart for backtesting and enhanced trading journal for tracking and analyzing trades.")
    st.markdown('---')
    # Pair selector & symbol map
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
    # Initialize drawings in session state if not present
    if 'drawings' not in st.session_state:
        st.session_state.drawings = {}
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
    st.components.v1.html(tv_html, height=820, scrolling=False)
    # Save, Load, and Refresh buttons
    if "logged_in_user" in st.session_state:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Save Drawings", key="bt_save_drawings"):
                logging.info(f"Save Drawings button clicked for pair {pair}")
                save_script = f"""
                <script>
                parent.window.postMessage({{action: 'save_drawings', pair: '{pair}'}}, '');
                </script>
                """
                st.components.v1.html(save_script, height=0)
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
                            parent.window.postMessage({{action: 'load_drawings', pair: '{pair}', content: {json.dumps(content)}}}, '');
                            </script>
                            """
                            st.components.v1.html(load_script, height=0)
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
        logging.info("User not logged in, save/load drawings disabled")
    # Backtesting Journal
    st.markdown("### 📝 Trading Journal")
    st.markdown(
        """
        Log your trades with detailed analysis, track psychological factors, and review performance with advanced analytics and trade replay.
        """
    )
    # Ensure journal DataFrame is initialized with all columns, including 'Tags'
    if 'tools_trade_journal' not in st.session_state or st.session_state.tools_trade_journal.empty:
        st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
    else:
        current_journal = st.session_state.tools_trade_journal
        missing_cols = [col for col in journal_cols if col not in current_journal.columns]
        if missing_cols:
            for col in missing_cols:
                current_journal[col] = pd.Series(dtype=journal_dtypes[col])
        if 'Tags' not in current_journal.columns:
            current_journal['Tags'] = ''
        current_journal['Tags'] = current_journal['Tags'].astype(str).fillna('')
        st.session_state.tools_trade_journal = current_journal[journal_cols].astype(journal_dtypes, errors='ignore')
    # Tabs for Journal Entry, Analytics, and Replay (renamed)
    tab_entry, tab_analytics, tab_history = st.tabs(["📝 Log Trade", "📊 Analytics", "📜 Trade History"])
    # Log Trade Tab
    with tab_entry:
        st.subheader("Log a New Trade")
    
        with st.form("trade_entry_form"):
            col1, col2 = st.columns(2)
        
            with col1:
                trade_date = st.date_input("Date", value=dt.datetime.now().date())
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
                rr = (take_profit_price - entry_price) / (entry_price - stop_loss_price) if stop_loss_price != 0 and weekly_bias in ["Bullish", "Neutral"] else (entry_price - take_profit_price) / (stop_loss_price - entry_price) if stop_loss_price != 0 else 0
            
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
            
                # Append new trade to journal
                st.session_state.tools_trade_journal = pd.concat(
                    [st.session_state.tools_trade_journal, pd.DataFrame([new_trade])],
                    ignore_index=True
                ).astype(journal_dtypes, errors='ignore')
            
                # Save to database if user is logged in
                if 'logged_in_user' in st.session_state:
                    username = st.session_state.logged_in_user
                    if _ta_save_journal(username, st.session_state.tools_trade_journal):
                        ta_update_xp(10)
                        ta_update_streak()
                        st.success("Trade saved successfully!")
                        logging.info(f"Trade logged and saved to database for user {username}")
                    else:
                        st.error("Failed to save trade to account. Saved locally only.")
                else:
                    st.success("Trade saved locally (not synced to account, please log in).")
                    logging.info("Trade logged for anonymous user")
            
                st.rerun()

        # Display current journal
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
        # Export options
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
                \\begin{document}
                \\section*{Trade Journal}
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
                    import subprocess
                    subprocess.run(["latexmk", "-pdf", "trade_journal.tex"], check=True)
                    with open("trade_journal.pdf", "rb") as f:
                        st.download_button("Download PDF", f, "trade_journal.pdf", "application/pdf")
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
                    logging.error(f"PDF generation error: {str(e)}")

    # Analytics Tab
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
                tag_options = []
                if 'Tags' in st.session_state.tools_trade_journal.columns:
                    tag_options = [tag for tags in st.session_state.tools_trade_journal['Tags'].str.split(',').explode().unique() if tag and pd.notna(tag)]
                tag_filter = st.multiselect("Filter by Tags", options=tag_options)
        
            with col_filter3:
                bias_filter = st.selectbox("Filter by Weekly Bias", ["All", "Bullish", "Bearish", "Neutral"])
            
            filtered_df = st.session_state.tools_trade_journal[
                st.session_state.tools_trade_journal['Symbol'].isin(symbol_filter)
            ]
        
            if tag_filter and 'Tags' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Tags'].apply(lambda x: any(tag in x.split(',') for tag in tag_filter) if isinstance(x, str) and x else False)]
        
            if bias_filter != "All":
                filtered_df = filtered_df[filtered_df['Weekly Bias'] == bias_filter]
            # Metrics
            win_rate = (filtered_df['Outcome / R:R Realised'].apply(lambda x: float(x.split(':')[1]) > 0 if isinstance(x, str) and ':' in x else False)).mean() * 100 if not filtered_df.empty else 0
            avg_pl = filtered_df['Outcome / R:R Realised'].apply(lambda x: float(x.split(':')[1]) if isinstance(x, str) and ':' in x else 0).mean() if not filtered_df.empty else 0
            total_trades = len(filtered_df)
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Win Rate (%)", f"{win_rate:.2f}")
            col_metric2.metric("Average R:R", f"{avg_pl:.2f}")
            col_metric3.metric("Total Trades", total_trades)
            # Visualizations
            st.subheader("Performance Charts")
            col_chart1, col_chart2 = st.columns(2)
        
            with col_chart1:
                def parse_rr(x):
                    try:
                        if isinstance(x, str) and ':' in x:
                            return float(x.split(':')[1])
                        return 0.0
                    except (ValueError, IndexError):
                        return 0.0
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
            
    # Trade History Tab (renamed from Trade Replay)
    with tab_history:
        st.subheader("Trade History")
    
        if not st.session_state.tools_trade_journal.empty:
            trade_id = st.selectbox(
                "Select Trade to Review",
                options=st.session_state.tools_trade_journal.index,
                format_func=lambda x: f"{st.session_state.tools_trade_journal.loc[x, 'Date'].strftime('%Y-%m-%d')} - {st.session_state.tools_trade_journal.loc[x, 'Symbol']}"
            )
        
            selected_trade = st.session_state.tools_trade_journal.loc[trade_id]
        
            st.write("Trade Details")
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
            st.info("No trades available for review.")
                # Challenge Mode
    st.subheader("🏅 Challenge Mode")
    st.write("30-Day Journaling Discipline Challenge")
    streak = st.session_state.get('streak', 0)
    progress = min(streak / 30.0, 1.0)
    st.progress(progress)
    if progress >= 1.0:
        st.success("Challenge completed! Great job on your consistency.")
        ta_update_xp(100) # Bonus XP for completion

# CORRECTED INDENTATION FOR THE 'mt5' BLOCK
elif st.session_state.current_page == 'mt5':
    st.title("📊 Performance Dashboard")
    st.caption("Analyze your MT5 trading history with advanced metrics and visualizations.")
    st.markdown('---')
    # ... (The rest of your code for the MT5 page goes here) ...

    # Custom CSS for theme consistency
    st.markdown(
        """
        <style>
        .metric-box { background-color: #2d4646; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #58b3b1; color: #ffffff; transition: all 0.3s ease-in-out; margin: 5px 0; }
        .metric-box:hover { transform: translateY(-3px); box-shadow: 0 6px 12px rgba(88, 179, 177, 0.3); }
        .metric-box.positive { background-color: #0f2b0f; border-color: #58b3b1; }
        .metric-box.negative { background-color: #2b0f0f; border-color: #a94442; }
        .stTabs [data-baseweb="tab"] { color: #ffffff !important; background-color: #2d4646 !important; border-radius: 8px 8px 0 0; padding: 10px 20px; margin-right: 5px; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #58b3b1 !important; color: #ffffff !important; font-weight: 600; border-bottom: 2px solid #4d7171 !important; }
        .stTabs [data-baseweb="tab"]:hover { background-color: #4d7171 !important; color: #ffffff !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --------------------------
    # Helper functions
    # --------------------------
    def _ta_human_pct(value):
        try:
            return f"{value * 100:.2f}%" if value is not None else "N/A"
        except Exception:
            return "N/A"

    def _ta_human_num(value):
        try:
            return f"{value:,.2f}" if value is not None else "N/A"
        except Exception:
            return "N/A"

    def _ta_compute_sharpe(df, risk_free_rate=0.02):
        if "Profit" not in df.columns:
            return np.nan
        daily_pnl = _ta_daily_pnl(df)
        if daily_pnl.empty:
            return np.nan
        returns = daily_pnl["Profit"].pct_change().dropna()
        if len(returns) < 2:
            return np.nan
        mean_return = returns.mean() * 252  # Annualized
        std_return = returns.std() * np.sqrt(252)  # Annualized
        return (mean_return - risk_free_rate) / std_return if std_return != 0 else np.nan

    # Placeholder for daily PnL computation
    def _ta_daily_pnl(df):
        if "Close Time" in df.columns and "Profit" in df.columns:
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["Close Time"]).dt.date
            return df_copy.groupby("date")["Profit"].sum().reset_index()
        return pd.DataFrame(columns=["date", "Profit"])

    # Placeholder for profit factor computation
    def _ta_profit_factor(df):
        wins = df[df["Profit"] > 0]["Profit"].sum()
        losses = abs(df[df["Profit"] < 0]["Profit"].sum())
        return wins / losses if losses != 0 else np.nan

    # --------------------------
    # File Uploader
    # --------------------------
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
                    st.stop()

                df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
                df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")
                df["Trade Duration"] = (df["Close Time"] - df["Open Time"]).dt.total_seconds() / 3600

                # ---------- Tabs ----------
                tab_summary, tab_charts, tab_edge, tab_export = st.tabs([
                    "📈 Summary Metrics",
                    "📊 Visualizations", 
                    "🔍 Edge Finder",
                    "📤 Export Reports"
                ])

                # ---------- Summary Metrics Tab ----------
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
                    max_drawdown = (daily_pnl["Profit"].cumsum() - daily_pnl["Profit"].cumsum().cummax()).min() if not daily_pnl.empty else 0
                    sharpe_ratio = _ta_compute_sharpe(df)
                    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) if total_trades else 0
                    longest_win_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] > 0) if k), default=0)
                    longest_loss_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] < 0) if k), default=0)

                    metrics = [
                        ("Total Trades", total_trades, "neutral"),
                        ("Win Rate", _ta_human_pct(win_rate), "positive" if win_rate >= 0.5 else "negative"),
                        ("Net Profit", f"${net_profit:,.2f}", "positive" if net_profit >= 0 else "negative"),
                        ("Profit Factor", _ta_human_num(profit_factor), "positive" if profit_factor >= 1 else "negative"),
                        ("Max Drawdown", f"${max_drawdown:,.2f}", "negative"),
                        ("Sharpe Ratio", _ta_human_num(sharpe_ratio), "positive" if sharpe_ratio >= 1 else "negative"),
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
                                    f"<div class='metric-box {style}'><strong>{title}</strong><br>{value}</div>",
                                    unsafe_allow_html=True
                                )

                # ---------- The rest of your tabs (charts, edge, export) remain unchanged ----------
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    else:
        st.info("👆 Upload your MT5 trading history CSV to explore advanced performance metrics.")

    # Gamification Badges
    if "mt5_df" in st.session_state and not st.session_state.mt5_df.empty:
        try:
            _ta_show_badges(st.session_state.mt5_df)
        except Exception as e:
            logging.error(f"Error displaying badges: {str(e)}")

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
elif st.session_state.current_page == 'strategy':
    st.title("📈 Manage My Strategy")
    st.markdown(""" Define, refine, and track your trading strategies. Save your setups and review performance to optimize your edge. """)
    st.write('---')
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
        combined_df['r'] = combined_df["Outcome / R:R Realised"].apply(lambda x: float(x.split(':')[1]) if isinstance(x, str) and ':' in x else np.nan)
    if group_cols and 'r' in combined_df.columns:
        agg = _ta_expectancy_by_group(combined_df, group_cols).sort_values("expectancy", ascending=False)
        st.write("Your refined edge profile based on logged trades:")
        st.dataframe(agg)
    else:
        st.info("Log more trades with symbols and outcomes to evolve your playbook.")
elif st.session_state.current_page == 'account':
    st.title("👤 My Account")
    st.markdown(
        """
        Manage your account, save your data, and sync your trading journal and drawings. Signing in lets you:
        - Keep your trading journal and strategies backed up.
        - Track your progress and gamification stats.
        - Sync across devices.
        - Import/export your account data easily.
        """
    )
    st.write('---')
    if "logged_in_user" not in st.session_state:
        # Tabs for Sign In and Sign Up
        tab_signin, tab_signup, tab_debug = st.tabs(["🔑 Sign In", "📝 Sign Up", "🛠 Debug"])
        # --------------------------
        # SIGN IN TAB
        # --------------------------
        with tab_signin:
            st.subheader("Welcome back! Please sign in to access your account.")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                if login_button:
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()
                    c.execute("SELECT password, data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result and result[0] == hashed_password:
                        st.session_state.logged_in_user = username
                        user_data = json.loads(result[1]) if result[1] else {}
                        st.session_state.drawings = user_data.get("drawings", {})
                        if "tools_trade_journal" in user_data:
                            loaded_df = pd.DataFrame(user_data["tools_trade_journal"])
                            for col in journal_cols:
                                if col not in loaded_df.columns:
                                    loaded_df[col] = pd.Series(dtype=journal_dtypes[col])
                            st.session_state.tools_trade_journal = loaded_df[journal_cols].astype(journal_dtypes, errors='ignore')
                        if "strategies" in user_data:
                            st.session_state.strategies = pd.DataFrame(user_data["strategies"])
                        if "emotion_log" in user_data:
                            st.session_state.emotion_log = pd.DataFrame(user_data["emotion_log"])
                        if "reflection_log" in user_data:
                            st.session_state.reflection_log = pd.DataFrame(user_data["reflection_log"])
                        st.session_state.xp = user_data.get('xp', 0)
                        st.session_state.level = user_data.get('level', 0)
                        st.session_state.badges = user_data.get('badges', [])
                        st.session_state.streak = user_data.get('streak', 0)
                        st.session_state.last_journal_date = user_data.get('last_journal_date', None)
                        st.success(f"Welcome back, {username}!")
                        logging.info(f"User {username} logged in successfully")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                        logging.warning(f"Failed login attempt for {username}")
        # --------------------------
        # SIGN UP TAB
        # --------------------------
        with tab_signup:
            st.subheader("Create a new account to start tracking your trades and progress.")
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Register")
                if register_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                        logging.warning(f"Registration failed for {new_username}: Passwords do not match")
                    elif not new_username or not new_password:
                        st.error("Username and password cannot be empty.")
                        logging.warning(f"Registration failed: Empty username or password")
                    else:
                        c.execute("SELECT username FROM users WHERE username = ?", (new_username,))
                        if c.fetchone():
                            st.error("Username already exists.")
                            logging.warning(f"Registration failed: Username {new_username} already exists")
                        else:
                            hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                            initial_data = json.dumps({"xp": 0, "level": 0, "badges": [], "streak": 0, "drawings": {}, "tools_trade_journal": [], "strategies": [], "emotion_log": [], "reflection_log": []})
                            try:
                                c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", (new_username, hashed_password, initial_data))
                                conn.commit()
                                st.session_state.logged_in_user = new_username
                                st.session_state.drawings = {}
                                st.session_state.tools_st.rerunurnal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
                                st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"])
                                st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
                                st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
                                st.session_state.xp = 0
                                st.session_state.level = 0
                                st.session_state.badges = []
                                st.session_state.streak = 0
                                st.success(f"Account created for {new_username}!")
                                logging.info(f"User {new_username} registered successfully")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to create account: {str(e)}")
                                logging.error(f"Registration error for {new_username}: {str(e)}")
        # --------------------------
        # DEBUG TAB
        # --------------------------
        with tab_debug:
            st.subheader("Debug: Inspect Users Database")
            st.warning("This is for debugging only. Remove in production.")
            try:
                c.execute("SELECT username, password, data FROM users")
                users = c.fetchall()
                if users:
                    debug_df = pd.DataFrame(users, columns=["Username", "Password (Hashed)", "Data"])
                    st.dataframe(debug_df, use_container_width=True)
                else:
                    st.info("No users found in the database.")
                c.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = c.fetchall()
                st.write("Database Tables:", tables)
            except Exception as e:
                st.error(f"Error accessing database: {str(e)}")
                logging.error(f"Debug error: {str(e)}")
        # --------------------------
        # LOGGED-IN USER VIEW
        # --------------------------
import streamlit as st
import pandas as pd
import logging

# --- (This section is for demonstration if you run the script standalone) ---
# In your actual app, this data would already be in the session_state from login.
def initialize_session_for_demo():
    """Initializes session state with sample data for demonstration."""
    # Assume the app starts on the 'account' page for this demo
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'account'
    if 'logged_in_user' not in st.session_state:
        st.session_state.logged_in_user = "TraderPro"
    if 'xp' not in st.session_state:
        st.session_state.xp = 1750
    if 'level' not in st.session_state:
        st.session_state.level = 1
    if 'badges' not in st.session_state:
        st.session_state.badges = ["First Trade", "Risk Manager"]
    if 'streak' not in st.session_state:
        st.session_state.streak = 14
    # Define dummy dataframes for logout functionality
    global journal_cols, journal_dtypes
    journal_cols = ["Date", "Symbol", "Entry", "Exit", "PnL"]
    journal_dtypes = {"Date": "datetime64[ns]", "Symbol": "object", "Entry": "float64", "Exit": "float64", "PnL": "float64"}

# Call the function to set up the demo state
initialize_session_for_demo()
# ------------------------- (End of demonstration section) -------------------------


# --- Refactored Logout Logic ---
def handle_logout():
    """
    Clears all user-specific data from the session state upon logout.
    This modular function makes the main code cleaner and the logic reusable.
    """
    user_session_keys = [
        'logged_in_user', 'drawings', 'tools_trade_journal', 'strategies',
        'emotion_log', 'reflection_log', 'xp', 'level', 'badges', 'streak'
    ]
    for key in user_session_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Re-initialize core data structures to their empty state
    st.session_state.drawings = {}
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
    st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Date Added"])
    st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
    st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
    st.session_state.xp = 0
    st.session_state.level = 0
    st.session_state.badges = []
    st.session_state.streak = 0

    logging.info("User logged out")
    st.session_state.current_page = "account" # Ensure redirection to the same page
    st.rerun()

# ==============================================================================
# --- MAIN PAGE ROUTER ---
# This is the correct structure for managing different pages in your app.
# ==============================================================================

# --- ACCOUNT PAGE ---
if st.session_state.current_page == 'account':

    # --- Header ---
    st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")
    st.markdown("This is your personal dashboard. Track your progress and manage your account.")
    st.markdown("---")

    # --- Main Dashboard Layout using Columns ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Progress Snapshot")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric(label="Current Level", value=f"LVL {st.session_state.get('level', 0)}")
        metric2.metric(label="Journaling Streak", value=f"{st.session_state.get('streak', 0)} Days 🔥")
        metric3.metric(label="Total XP", value=f"{st.session_state.get('xp', 0)}")

        st.markdown("#### **Level Progress**")
        current_level = st.session_state.get('level', 0)
        xp_for_next_level = (current_level + 1) * 1000
        xp_in_current_level = st.session_state.get('xp', 0) - (current_level * 1000)
        progress_percentage = xp_in_current_level / 1000
        st.progress(progress_percentage)
        st.caption(f"{xp_in_current_level} / 1000 XP to the next level.")

    with col2:
        st.subheader("🏆 Badges")
        badges = st.session_state.get('badges', [])
        if badges:
            for badge in badges:
                st.markdown(f"- 🏅 {badge}")
        else:
            st.info("No badges earned yet. Keep up the great work to unlock them!")

    st.markdown("---")

    # --- Account Details and Actions using an Expander ---
    with st.expander("⚙️ Manage Account"):
        st.write(f"**Username**: `{st.session_state.logged_in_user}`")
        st.write("**Email**: `trader.pro@email.com` (example)")
        if st.button("Log Out", key="logout_account_page", type="primary"):
            handle_logout()

# --- COMMUNITY PAGE ---
elif st.session_state.current_page == 'community':
    st.title("🌐 Community Trade Ideas")
    st.markdown(""" Share and explore trade ideas with the community. Upload your chart screenshots and discuss strategies with other traders. """)
    st.write('---')
    # ... (Add the rest of your community page code here)

# --- You can add other pages below ---
# elif st.session_state.current_page == 'settings':
#     st.title("Settings")
#     # ...
    st.subheader("➕ Share a Trade Idea")
    with st.form("trade_idea_form"):
        trade_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP", "EUR/JPY"])
        trade_direction = st.radio("Direction", ["Long", "Short"])
        trade_description = st.text_area("Trade Description")
        uploaded_image = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg"])
        submit_idea = st.form_submit_button("Share Idea")
        if submit_idea:
            if "logged_in_user" in st.session_state:
                username = st.session_state.logged_in_user
                user_dir = _ta_user_dir(username)
                idea_id = _ta_hash()
                idea_data = {
                    "Username": username,
                    "Pair": trade_pair,
                    "Direction": trade_direction,
                    "Description": trade_description,
                    "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "IdeaID": idea_id
                }
                if uploaded_image:
                    image_path = os.path.join(user_dir, "community_images", f"{idea_id}.png")
                    with open(image_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    idea_data["ImagePath"] = image_path
                st.session_state.trade_ideas = pd.concat([st.session_state.trade_ideas, pd.DataFrame([idea_data])], ignore_index=True)
                _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                st.success("Trade idea shared successfully!")
                logging.info(f"Trade idea shared by {username}: {idea_id}")
                st.rerun()
            else:
                st.error("Please log in to share trade ideas.")
                logging.warning("Attempt to share trade idea without login")
    st.subheader("📈 Community Trade Ideas")
    if not st.session_state.trade_ideas.empty:
        for idx, idea in st.session_state.trade_ideas.iterrows():
            with st.expander(f"{idea['Pair']} - {idea['Direction']} by {idea['Username']} ({idea['Timestamp']})"):
                st.markdown(f"Description: {idea['Description']}")
                if "ImagePath" in idea and os.path.exists(idea['ImagePath']):
                    st.image(idea['ImagePath'], caption="Chart Screenshot", use_column_width=True)
                if st.button("Delete Idea", key=f"delete_idea_{idea['IdeaID']}"):
                    if "logged_in_user" in st.session_state and st.session_state.logged_in_user == idea["Username"]:
                        st.session_state.trade_ideas = st.session_state.trade_ideas.drop(idx).reset_index(drop=True)
                        _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                        st.success("Trade idea deleted successfully!")
                        logging.info(f"Trade idea {idea['IdeaID']} deleted by {st.session_state.logged_in_user}")
                        st.rerun()
                    else:
                        st.error("You can only delete your own trade ideas.")
                        logging.warning(f"Unauthorized attempt to delete trade idea {idea['IdeaID']}")
    else:
        st.info("No trade ideas shared yet. Be the first to contribute!")
    # Community Templates
    st.subheader("📄 Community Templates")
    with st.form("template_form"):
        template_type = st.selectbox("Template Type", ["Journaling Template", "Checklist", "Strategy Playbook"])
        template_name = st.text_input("Template Name")
        template_content = st.text_area("Template Content")
        submit_template = st.form_submit_button("Share Template")
        if submit_template:
            if "logged_in_user" in st.session_state:
                username = st.session_state.logged_in_user
                template_id = _ta_hash()
                template_data = {
                    "Username": username,
                    "Type": template_type,
                    "Name": template_name,
                    "Content": template_content,
                    "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ID": template_id
                }
                st.session_state.community_templates = pd.concat([st.session_state.community_templates, pd.DataFrame([template_data])], ignore_index=True)
                _ta_save_community('templates', st.session_state.community_templates.to_dict('records'))
                st.success("Template shared successfully!")
                logging.info(f"Template shared by {username}: {template_id}")
                st.rerun()
            else:
                st.error("Please log in to share templates.")
    if not st.session_state.community_templates.empty:
        for idx, template in st.session_state.community_templates.iterrows():
            with st.expander(f"{template['Type']} - {template['Name']} by {template['Username']} ({template['Timestamp']})"):
                st.markdown(template['Content'])
                if st.button("Delete Template", key=f"delete_template_{template['ID']}"):
                    if "logged_in_user" in st.session_state and st.session_state.logged_in_user == template["Username"]:
                        st.session_state.community_templates = st.session_state.community_templates.drop(idx).reset_index(drop=True)
                        _ta_save_community('templates', st.session_state.community_templates.to_dict('records'))
                        st.success("Template deleted successfully!")
                        logging.info(f"Template {template['ID']} deleted by {st.session_state.logged_in_user}")
                        st.rerun()
                    else:
                        st.error("You can only delete your own templates.")
    else:
        st.info("No templates shared yet. Share one above!")
    # Leaderboard / Self-Competition
    st.subheader("🏆 Leaderboard - Consistency")
    users = c.execute("SELECT username, data FROM users").fetchall()
    leader_data = []
    for u, d in users:
        user_d = json.loads(d) if d else {}
        trades = len(user_d.get("tools_trade_journal", []))
        leader_data.append({"Username": u, "Journaled Trades": trades})
    if leader_data:
        leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
        leader_df["Rank"] = leader_df.index + 1
        st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]])
    else:
        st.info("No leaderboard data yet.")
# Tools
elif st.session_state.current_page == 'tools':
    st.title("🛠 Tools")
    st.markdown("""
    ### Available Tools
    - **Profit/Loss Calculator**: Calculate potential profits or losses based on trade size, entry, and exit prices.
    - **Price Alerts**: Set custom price alerts for key levels on your chosen currency pairs.
    - **Currency Correlation Heatmap**: Visualize correlations between currency pairs to identify hedging opportunities.
    - **Risk Management Calculator**: Determine optimal position sizes based on risk tolerance and stop-loss levels.
    - **Trading Session Tracker**: Monitor active trading sessions (e.g., London, New York) to align with market hours.
    - **Drawdown Recovery Planner**: Plan recovery strategies for account drawdowns with calculated targets.
    - **Pre-Trade Checklist**: Follow a structured checklist to ensure disciplined trade entries.
    - **Pre-Market Checklist**: Prepare for the trading day with a comprehensive market analysis checklist.
    """, unsafe_allow_html=True)
    st.markdown('---')
    st.markdown("""
    <style>
    div[data-testid="stTabs"] div[role="tablist"] > div {
        background-color: #5bb4b0 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #ffffff !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        color: #5bb4b0 !important;
        background-color: rgba(91, 180, 176, 0.2) !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #5bb4b0 !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)
    tools_options = [
        'Profit/Loss Calculator',
        'Price Alerts',
        'Currency Correlation Heatmap',
        'Risk Management Calculator',
        'Trading Session Tracker',
        'Drawdown Recovery Planner',
        'Pre-Trade Checklist',
        'Pre-Market Checklist'
    ]
    tabs = st.tabs(tools_options)
    with tabs[0]:
        st.header("💰 Profit / Loss Calculator")
        st.markdown("Calculate your potential profit or loss for a trade.")
        st.markdown('---')
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            currency_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY"], key="pl_currency_pair")
            position_size = st.number_input("Position Size (lots)", min_value=0.01, value=0.1, step=0.01, key="pl_position_size")
            close_price = st.number_input("Close Price", value=1.1050, step=0.0001, key="pl_close_price")
        with col_calc2:
            account_currency = st.selectbox("Account Currency", ["USD", "EUR", "GBP", "JPY"], index=0, key="pl_account_currency")
            open_price = st.number_input("Open Price", value=1.1000, step=0.0001, key="pl_open_price")
            trade_direction = st.radio("Trade Direction", ["Long", "Short"], key="pl_trade_direction")
        pip_multiplier = 100 if "JPY" in currency_pair else 10000
        pip_movement = abs(close_price - open_price) * pip_multiplier
        exchange_rate = 1.1000
        pip_value = (
            (0.0001 / exchange_rate) * position_size * 100000 if "JPY" not in currency_pair else (0.01 / exchange_rate) * position_size * 100000
        )
        profit_loss = pip_movement * pip_value
        st.write(f"Pip Movement: {pip_movement:.2f} pips")
        st.write(f"Pip Value: {pip_value:.2f} {account_currency}")
        st.write(f"Potential Profit/Loss: {profit_loss:.2f} {account_currency}")
    with tabs[1]:
        st.header("⏰ Price Alerts")
        st.markdown("Set price alerts for your favourite forex pairs and get notified when the price hits your target.")
        st.markdown('---')
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY"]
        if "price_alerts" not in st.session_state:
            st.session_state.price_alerts = pd.DataFrame(columns=["Pair", "Target Price", "Triggered"])
        with st.form("add_alert_form"):
            col1, col2 = st.columns([2, 2])
            with col1:
                pair = st.selectbox("Currency Pair", forex_pairs)
            with col2:
                price = st.number_input("Target Price", min_value=0.0, format="%.5f")
            submitted = st.form_submit_button("➕ Add Alert")
            if submitted:
                new_alert = {"Pair": pair, "Target Price": price, "Triggered": False}
                st.session_state.price_alerts = pd.concat([st.session_state.price_alerts, pd.DataFrame([new_alert])], ignore_index=True)
                st.success(f"Alert added: {pair} at {price}")
                logging.info(f"Alert added: {pair} at {price}")
        st.subheader("Your Alerts")
        st.dataframe(st.session_state.price_alerts, use_container_width=True, height=220)
        if st.session_state.get("price_alert_refresh", False):
            st_autorefresh(interval=5000, key="price_alert_autorefresh")
        active_pairs = st.session_state.price_alerts["Pair"].unique().tolist()
        live_prices = {}
        for p in active_pairs:
            if not p:
                continue
            base, quote = p[:3], p[3:]
            try:
                r = requests.get(f"https://api.exchangerate.host/latest?base={base}&symbols={quote}", timeout=6)
                data = r.json()
                price_val = data.get("rates", {}).get(quote)
                live_prices[p] = float(price_val) if price_val is not None else None
                logging.info(f"Fetched price for {p}: {live_prices[p]}")
            except Exception as e:
                live_prices[p] = None
                logging.error(f"Error fetching price for {p}: {str(e)}")
        triggered_alerts = []
        for idx, row in st.session_state.price_alerts.iterrows():
            pair = row["Pair"]
            target = row["Target Price"]
            current_price = live_prices.get(pair)
            if isinstance(current_price, (int, float)):
                if not row["Triggered"] and abs(current_price - target) < (0.0005 if "JPY" not in pair else 0.01):
                    st.session_state.price_alerts.at[idx, "Triggered"] = True
                    triggered_alerts.append((idx, f"{pair} reached {target} (Current: {current_price:.5f})"))
                    logging.info(f"Alert triggered: {pair} at {target}")
        if triggered_alerts:
            for idx, alert_msg in triggered_alerts:
                st.balloons()
                st.success(f"⚡ {alert_msg}")
        st.markdown("### 📊 Active Alerts")
        if not st.session_state.price_alerts.empty:
            for idx, row in st.session_state.price_alerts.iterrows():
                pair = row["Pair"]
                target = row["Target Price"]
                triggered = row["Triggered"]
                current_price = live_prices.get(pair)
                current_price_display = f"{current_price:.5f}" if isinstance(current_price, (int, float)) else "N/A"
                color = "#2ecc71" if triggered else "#f4a261"
                status = "Triggered" if triggered else "Pending"
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(
                        f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                        {pair} {status}<br>
                        Current: {current_price_display} &nbsp;&nbsp;&nbsp; Target: {target}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    if st.button("❌ Cancel", key=f"cancel_{idx}"):
                        st.session_state.price_alerts = st.session_state.price_alerts.drop(idx).reset_index(drop=True)
                        st.rerun()
                        logging.info(f"Cancelled alert at index {idx}")
        else:
            st.info("No price alerts set. Add one above to start monitoring prices.")
    with tabs[2]:
        st.header("📊 Currency Correlation Heatmap")
        st.markdown("Understand how forex pairs move relative to each other.")
        st.markdown('---')
        pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF"]
        data = np.array([
            [1.00, 0.87, -0.72, 0.68, -0.55, -0.60],
            [0.87, 1.00, -0.65, 0.74, -0.58, -0.62],
            [-0.72, -0.65, 1.00, -0.55, 0.69, 0.71],
            [0.68, 0.74, -0.55, 1.00, -0.61, -0.59],
            [-0.55, -0.58, 0.69, -0.61, 1.00, 0.88],
            [-0.60, -0.62, 0.71, -0.59, 0.88, 1.00],
        ])
        corr_df = pd.DataFrame(data, columns=pairs, index=pairs)
        fig = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="Forex Pair Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        st.header("🛡️ Risk Management Calculator")
        st.markdown(""" Proper position sizing keeps your account safe. Risk management is crucial to long-term trading success. It helps prevent large losses, preserves capital, and allows you to stay in the game during drawdowns. Always risk no more than 1-2% per trade, use stop losses, and calculate position sizes based on your account size and risk tolerance. """)
        st.markdown('---')
        # 📊 Lot Size Calculator
        st.subheader('📊 Lot Size Calculator')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=10000.0)
        with col2:
            risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0)
        with col3:
            stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=20.0)
        with col4:
            pip_value = st.number_input("Pip Value per Lot ($)", min_value=0.01, value=10.0)
        if st.button("Calculate Lot Size"):
            risk_amount = balance * (risk_percent / 100)
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            st.success(f"✅ Recommended Lot Size: {lot_size:.2f} lots")
            logging.info(f"Calculated lot size: {lot_size}")
        # 🔄 What-If Analyzer
        st.subheader('🔄 What-If Analyzer')
        base_equity = st.number_input('Starting Equity', value=10000.0, min_value=0.0, step=100.0, key='whatif_equity')
        risk_pct = st.slider('Risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key='whatif_risk') / 100.0
        winrate = st.slider('Win rate (%)', 10.0, 90.0, 50.0, 1.0, key='whatif_wr') / 100.0
        avg_r = st.slider('Average R multiple', 0.5, 5.0, 1.5, 0.1, key='whatif_avg_r')
        trades = st.slider('Number of trades', 10, 500, 100, 10, key='whatif_trades')
        E_R = winrate * avg_r - (1 - winrate) * 1.0
        exp_growth = (1 + risk_pct * E_R) ** trades
        st.metric('Expected Growth Multiplier', f"{exp_growth:.2f}x")
        alt_risk = st.slider('What if risk per trade was (%)', 0.1, 5.0, 0.5, 0.1, key='whatif_alt') / 100.0
        alt_growth = (1 + alt_risk * E_R) ** trades
        st.metric('Alt Growth Multiplier', f"{alt_growth:.2f}x")
        # 📈 Equity Projection
        sim = pd.DataFrame({
            'trade': list(range(trades + 1)),
            'equity_base': base_equity * (1 + risk_pct * E_R) ** np.arange(trades + 1),
            'equity_alt': base_equity * (1 + alt_risk * E_R) ** np.arange(trades + 1),
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim['trade'], y=sim['equity_base'], mode='lines', name=f'Risk {risk_pct*100:.1f}%'))
        fig.add_trace(go.Scatter(x=sim['trade'], y=sim['equity_alt'], mode='lines', name=f'What-If {alt_risk*100:.1f}%'))
        fig.update_layout(title='Equity Projection – Base vs What-If', xaxis_title='Trade #', yaxis_title='Equity')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[4]:
        st.header("🕒 Forex Market Sessions")
        st.markdown(""" Stay aware of active trading sessions to trade when volatility is highest. Each session has unique characteristics: Sydney/Tokyo for Asia-Pacific news, London for Europe, New York for US data. Overlaps like London/New York offer highest liquidity and volatility, ideal for major pairs. Track your performance per session to identify your edge. """)
        st.markdown('---')
        st.subheader('📊 Session Statistics')
        mt5_df = st.session_state.get('mt5_df', pd.DataFrame())
        df = mt5_df if not mt5_df.empty else st.session_state.tools_trade_journal
        if not df.empty and 'session' in df.columns:
            by_sess = df.groupby(['session']).agg(
                trades=('r', 'count') if 'r' in df.columns else ('session', 'count'),
                winrate=('r', lambda s: (s > 0).mean()) if 'r' in df.columns else ('session', 'count'),
                avg_r=('r', 'mean') if 'r' in df.columns else ('session', 'count')
            ).reset_index()
            st.dataframe(by_sess, use_container_width=True)
            if 'r' in df.columns:
                fig = px.bar(by_sess, x='session', y='winrate', title='Win Rate by Session', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload trades with a 'session' column to analyze performance by trading session.")
        st.subheader('🕒 Current Market Sessions')
        now = dt.datetime.now(pytz.UTC)
        sessions = [
            {"name": "Sydney", "start": 22, "end": 7, "tz": "Australia/Sydney"},
            {"name": "Tokyo", "start": 0, "end": 9, "tz": "Asia/Tokyo"},
            {"name": "London", "start": 8, "end": 17, "tz": "Europe/London"},
            {"name": "New York", "start": 13, "end": 22, "tz": "America/New_York"},
        ]
        session_status = []
        for session in sessions:
            tz = pytz.timezone(session["tz"])
            local_time = now.astimezone(tz)
            local_hour = local_time.hour + local_time.minute / 60
            start = session["start"]
            end = session["end"]
            is_open = (start <= local_hour < end) if start <= end else (start <= local_hour or local_hour < end)
            session_status.append({
                "Session": session["name"],
                "Status": "Open" if is_open else "Closed",
                "Local Time": local_time.strftime("%H:%M"),
                "Time Until": (start - local_hour) % 24 if not is_open else (end - local_hour) % 24
            })
        session_df = pd.DataFrame(session_status)
        st.dataframe(session_df, use_container_width=True)
        for session in session_status:
            color = "#2ecc71" if session["Status"] == "Open" else "#e74c3c"
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                {session['Session']} Session: {session['Status']} (Local: {session['Local Time']}, {'Closes in' if session['Status'] == 'Open' else 'Opens in'} {session['Time Until']:.1f} hours)
                </div>
                """,
                unsafe_allow_html=True
            )
    with tabs[5]:
        st.header("📉 Drawdown Recovery Planner")
        st.markdown(""" Plan your recovery from a drawdown. Understand the percentage gain required to recover losses and simulate recovery based on your trading parameters. """)
        st.markdown('---')
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 50.0, 10.0) / 100
        recovery_pct = _ta_percent_gain_to_recover(drawdown_pct)
        st.metric("Required Gain to Recover", f"{recovery_pct*100:.2f}%")
        st.subheader("📈 Recovery Simulation")
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_equity = st.number_input("Initial Equity ($)", min_value=100.0, value=10000.0)
        with col2:
            win_rate = st.slider("Expected Win Rate (%)", 10, 90, 50) / 100
        with col3:
            avg_rr = st.slider("Average R:R", 0.5, 5.0, 1.5, 0.1)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0) / 100
        trades_needed = math.ceil(math.log(1 / (1 - drawdown_pct)) / math.log(1 + risk_per_trade * (win_rate * avg_rr - (1 - win_rate))))
        st.write(f"Estimated Trades to Recover: {trades_needed}")
        sim_equity = [initial_equity * (1 - drawdown_pct)]
        for _ in range(min(trades_needed + 10, 100)):
            sim_equity.append(sim_equity[-1] * (1 + risk_per_trade * (win_rate * avg_rr - (1 - win_rate))))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(sim_equity))), y=sim_equity, mode='lines', name='Equity'))
        fig.add_hline(y=initial_equity, line_dash="dash", line_color="green", annotation_text="Initial Equity")
        fig.update_layout(title='Drawdown Recovery Simulation', xaxis_title='Trade #', yaxis_title='Equity ($)')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[6]:
        st.header("✅ Pre-Trade Checklist")
        st.markdown(""" Ensure discipline by running through this checklist before every trade. A structured approach reduces impulsive decisions and aligns trades with your strategy. """)
        st.markdown('---')
        checklist_items = [
            "Market structure aligns with my bias",
            "Key levels (S/R) identified",
            "Entry trigger confirmed",
            "Risk-reward ratio ≥ 1:2",
            "No high-impact news imminent",
            "Position size calculated correctly",
            "Stop loss set",
            "Take profit set",
            "Trade aligns with my edge",
            "Emotionally calm and focused"
        ]
        checklist_state = {item: st.checkbox(item, key=f"checklist_{i}") for i, item in enumerate(checklist_items)}
        checked_count = sum(1 for v in checklist_state.values() if v)
        st.metric("Checklist Completion", f"{checked_count}/{len(checklist_items)}")
        if checked_count == len(checklist_items):
            st.success("✅ All checks passed! Ready to trade.")
        else:
            st.warning(f"⚠ Complete all {len(checklist_items)} checklist items before trading.")
    with tabs[7]:
        st.header("📅 Pre-Market Checklist")
        st.markdown(""" Build consistent habits with pre-market checklists and end-of-day reflections. These rituals help maintain discipline and continuous improvement. """)
        st.markdown('---')
        st.subheader("Pre-Market Routine Checklist")
        pre_market_items = [
            "Reviewed economic calendar",
            "Analyzed major news events",
            "Set weekly/daily biases",
            "Identified key levels on charts",
            "Prepared watchlist of pairs",
            "Checked correlations",
            "Reviewed previous trades"
        ]
        pre_checklist = {item: st.checkbox(item, key=f"pre_{i}") for i, item in enumerate(pre_market_items)}
        pre_checked = sum(1 for v in pre_checklist.values() if v)
        st.metric("Pre-Market Completion", f"{pre_checked}/{len(pre_market_items)}")
        if pre_checked == len(pre_market_items):
            st.success("✅ Pre-market routine complete!")
        st.subheader("End-of-Day Reflection")
        with st.form("reflection_form"):
            reflection = st.text_area("What went well today? What can be improved?")
            submit_reflection = st.form_submit_button("Log Reflection")
            if submit_reflection:
                log_entry = {
                    "Date": dt.datetime.now().strftime("%Y-%m-%d"),
                    "Reflection": reflection
                }
                if "reflection_log" not in st.session_state:
                    st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
                st.session_state.reflection_log = pd.concat([st.session_state.reflection_log, pd.DataFrame([log_entry])], ignore_index=True)
                if "logged_in_user" in st.session_state:
                    username = st.session_state.logged_in_user
                    try:
                        c.execute("SELECT data FROM users WHERE username = ?", (username,))
                        result = c.fetchone()
                        user_data = json.loads(result[0]) if result else {}
                        user_data["reflection_log"] = st.session_state.reflection_log.to_dict(orient="records")
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
                        conn.commit()
                    except Exception as e:
                        logging.error(f"Error saving reflection: {str(e)}")
                st.success("Reflection logged!")
        if "reflection_log" in st.session_state and not st.session_state.reflection_log.empty:
            st.dataframe(st.session_state.reflection_log)

elif st.session_state.current_page == "Zenvo Academy":
    st.title("Zenvo Academy")
    st.caption("Explore experimental features and tools for your trading journey.")
    st.markdown('---')
    st.markdown("### Welcome to Zenvo Academy")
    st.write("Our Academy provides beginner traders with a clear learning path – covering Forex basics, chart analysis, risk management, and trading psychology. Build a solid foundation before stepping into live markets.")
    st.info("This page is under development. Stay tuned for new features!")
    if st.button("Log Out", key="logout_test_page"):
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
        st.session_state.current_page = "login"
        st.rerun()
