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
from PIL import Image
import io
import base64

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
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    components.html(notification_html, height=0)

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

# --- [FIXED] Gamification helpers ---
def ta_update_xp(amount):
    """
    Awards XP to the logged-in user, saves it to the DB, and handles level-ups.
    """
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result:
            user_data = json.loads(result[0])
            old_xp = user_data.get('xp', 0)
            new_xp = old_xp + amount
            user_data['xp'] = new_xp

            # --- LEVEL UP LOGIC (1000 XP per level) ---
            current_level = user_data.get('level', 0)
            new_level = new_xp // 1000

            if new_level > current_level:
                user_data['level'] = new_level
                st.balloons()
                st.success(f"LEVEL UP! You have reached Level {new_level}.")
                # Update badges for the new level
                badges = user_data.get('badges', [])
                level_badge = f"Level {new_level}"
                if level_badge not in badges:
                    badges.append(level_badge)
                user_data['badges'] = badges
            
            # Save data back to the database
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
            conn.commit()

            # Update session state to reflect changes immediately
            st.session_state.xp = user_data['xp']
            st.session_state.level = user_data.get('level', 0)
            st.session_state.badges = user_data.get('badges', [])
            
            # Show XP notification
            show_xp_notification(amount)
            logging.info(f"User {username} awarded {amount} XP. Total XP: {new_xp}. Level: {user_data.get('level', 0)}.")

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
                if last == dt.date.today() - dt.timedelta(days=1):
                    streak += 1
                elif last < dt.date.today() - dt.timedelta(days=1):
                    streak = 1
                # If it's the same day, do nothing
                elif last == dt.date.today():
                    pass
            else:
                streak = 1
                
            user_data['streak'] = streak
            user_data['last_journal_date'] = today
            
            if streak > 0 and streak % 7 == 0:
                badge = f"{streak}-Day Discipline Badge"
                if badge not in user_data.get('badges', []):
                    user_data['badges'] = user_data.get('badges', []) + [badge]
                    st.balloons()
                    st.success(f"Unlocked: {badge} for maintaining a {streak} day streak!")
            
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
            conn.commit()
            st.session_state.streak = streak
            st.session_state.badges = user_data.get('badges', [])
            logging.info(f"User {username} streak updated to {streak}.")

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
                if not equity.empty and equity.max() > 0:
                    dd = (equity - equity.cummax()).min() / equity.max()
                    if abs(dd) < 0.1:
                        st.balloons()
                        st.success("Milestone achieved: Survived 3 months without >10% drawdown!")

# --- [NEW] Centralized Logout Handler ---
def handle_logout():
    """
    Clears all user-specific data from the session state upon logout.
    """
    user_session_keys = [
        'logged_in_user', 'drawings', 'tools_trade_journal', 'strategies',
        'emotion_log', 'reflection_log', 'xp', 'level', 'badges', 'streak',
        'last_journal_date'
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
    st.session_state.last_journal_date = None

    logging.info("User logged out successfully")
    st.session_state.current_page = "account"
    st.rerun()

# Load community data
if "trade_ideas" not in st.session_state:
    loaded_ideas = _ta_load_community('trade_ideas', [])
    st.session_state.trade_ideas = pd.DataFrame(loaded_ideas) if loaded_ideas else pd.DataFrame(columns=["Username", "Pair", "Direction", "Description", "Timestamp", "IdeaID", "ImagePath"])

if "community_templates" not in st.session_state:
    loaded_templates = _ta_load_community('templates', [])
    st.session_state.community_templates = pd.DataFrame(loaded_templates) if loaded_templates else pd.DataFrame(columns=["Username", "Type", "Name", "Content", "Timestamp", "ID"])

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
try:
    logo = Image.open("logo22.png")
    logo = logo.resize((60, 50))
    buffered = io.BytesIO()
    logo.save(buffered, format="PNG")
    logo_str = base64.b64encode(buffered.getvalue()).decode()
    st.sidebar.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src="data:image/png;base64,{logo_str}" width="60" height="50"/>
        </div>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.sidebar.warning("logo22.png not found.")

nav_items = [
    ('fundamentals', 'Forex Fundamentals'),
    ('backtesting', 'Backtesting'),
    ('mt5', 'Performance Dashboard'),
    ('strategy', 'Manage My Strategy'),
    ('community', 'Community Trade Ideas'),
    ('tools', 'Tools'),
    ('Zenvo Academy', 'Zenvo Academy'),
    ('account', 'My Account')
]
# Move 'My Account' to the end for better flow

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
        "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD", "USD/CHF": "OANDA:USDCHF",
        "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD", "USD/CAD": "CMCMARKETS:USDCAD", "EUR/GBP": "FX:EURGBP",
        "EUR/JPY": "FX:EURJPY", "GBP/JPY": "FX:GBPJPY", "AUD/JPY": "FX:AUDJPY", "AUD/NZD": "FX:AUDNZD",
        "AUD/CAD": "FX:AUDCAD", "AUD/CHF": "FX:AUDCHF", "CAD/JPY": "FX:CADJPY", "CHF/JPY": "FX:CHFJPY",
        "EUR/AUD": "FX:EURAUD", "EUR/CAD": "FX:EURCAD", "EUR/CHF": "FX:EURCHF", "GBP/AUD": "FX:GBPAUD",
        "GBP/CAD": "FX:GBPCAD", "GBP/CHF": "FX:GBPCHF", "NZD/JPY": "FX:NZDJPY", "NZD/CAD": "FX:NZDCAD",
        "NZD/CHF": "FX:NZDCHF", "CAD/CHF": "FX:CADCHF",
    }
    pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
    tv_symbol = pairs_map[pair]

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
        "container_id": "tradingview_widget", "width": "100%", "height": 800, "symbol": "{tv_symbol}",
        "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
        "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true,
        "studies": [], "show_popup_button": true, "popup_width": "1000", "popup_height": "650"
    }});
    </script>
    """
    components.html(tv_html, height=820, scrolling=False)
    
    if "logged_in_user" in st.session_state:
        # User is logged in, show save/load controls
        st.info("The drawing tools functionality with save/load is currently under maintenance. We appreciate your patience.")
    else:
        st.warning("Sign in via the My Account tab to save your trading journal.")
        logging.info("User not logged in, save/load drawings disabled")
    # Backtesting Journal
    st.markdown("### 📝 Trading Journal")
    st.markdown(
        """
        Log your trades with detailed analysis, track psychological factors, and review performance with advanced analytics and trade replay.
        """
    )
    
    if 'tools_trade_journal' not in st.session_state or st.session_state.tools_trade_journal.empty:
        st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
    # Add 'Tags' if it's missing for backward compatibility
    if 'Tags' not in st.session_state.tools_trade_journal.columns:
        st.session_state.tools_trade_journal['Tags'] = ''
        
    tab_entry, tab_analytics, tab_history = st.tabs(["📝 Log Trade", "📊 Analytics", "📜 Trade History"])
    # Log Trade Tab
    with tab_entry:
        st.subheader("Log a New Trade")
    
        with st.form("trade_entry_form"):
            col1, col2 = st.columns(2)
        
            with col1:
                trade_date = st.date_input("Date", value=dt.datetime.now(dt.timezone.utc).date())
                symbol = st.selectbox("Symbol", list(pairs_map.keys()) + ["Other"], index=0)
                if symbol == "Other": symbol = st.text_input("Custom Symbol")
                weekly_bias = st.selectbox("Weekly Bias", ["Bullish", "Bearish", "Neutral"])
                daily_bias = st.selectbox("Daily Bias", ["Bullish", "Bearish", "Neutral"])
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, format="%.5f")
                stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.0001, format="%.5f")
        
            with col2:
                take_profit_price = st.number_input("Take Profit Price", min_value=0.0, step=0.0001, format="%.5f")
                lots = st.number_input("Lots", min_value=0.01, step=0.01, format="%.2f")
                emotions = st.selectbox("Emotions", ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral"])
                tags = st.multiselect("Tags", ["Setup: Breakout", "Setup: Reversal", "Mistake: Overtrading", "Mistake: No Stop Loss", "Emotion: FOMO", "Emotion: Revenge"])
            entry_conditions = st.text_area("Entry Conditions")
            notes = st.text_area("Notes/Journal")
        
            submit_button = st.form_submit_button("Save Trade")
        
            if submit_button:
                rr = 0
                if entry_price > 0 and stop_loss_price > 0 and take_profit_price > 0:
                    risk = abs(entry_price - stop_loss_price)
                    reward = abs(take_profit_price - entry_price)
                    rr = reward / risk if risk != 0 else 0
            
                new_trade = {
                    'Date': pd.to_datetime(trade_date), 'Symbol': symbol, 'Weekly Bias': weekly_bias, 'Daily Bias': daily_bias,
                    '4H Structure': '', '1H Structure': '', 'Positive Correlated Pair & Bias': '', 'Potential Entry Points': '',
                    '5min/15min Setup?': '', 'Entry Conditions': entry_conditions, 'Planned R:R': f"1:{rr:.2f}",
                    'News Filter': '', 'Alerts': '', 'Concerns': '', 'Emotions': emotions, 'Confluence Score 1-7': 0.0,
                    'Outcome / R:R Realised': f"1:{rr:.2f}", 'Notes/Journal': notes, 'Entry Price': entry_price,
                    'Stop Loss Price': stop_loss_price, 'Take Profit Price': take_profit_price, 'Lots': lots, 'Tags': ','.join(tags)
                }
                
                st.session_state.tools_trade_journal = pd.concat(
                    [st.session_state.tools_trade_journal, pd.DataFrame([new_trade])], ignore_index=True
                )
            
                if 'logged_in_user' in st.session_state:
                    username = st.session_state.logged_in_user
                    if _ta_save_journal(username, st.session_state.tools_trade_journal):
                        ta_update_xp(10) # Award 10 XP per trade
                        ta_update_streak() # Check and update streak
                        st.success("Trade saved successfully to your account!")
                        logging.info(f"Trade logged and saved to database for user {username}")
                    else:
                        st.error("Failed to save trade to account. Saved locally for this session.")
                else:
                    st.warning("Trade saved locally. Please log in to save to your account.")
                    logging.info("Trade logged for anonymous user")
            
                st.rerun()

        # Display current journal
        st.subheader("Most Recent Trades")
        st.dataframe(st.session_state.tools_trade_journal.tail(), use_container_width=True)

    # Analytics Tab
    with tab_analytics:
        st.subheader("Trade Analytics")
        df_journal = st.session_state.tools_trade_journal
        if not df_journal.empty:
            df_journal['pnl'] = df_journal.apply(
                lambda row: (row['Take Profit Price'] - row['Entry Price']) * row['Lots'] if row['Weekly Bias'] == 'Bullish'
                else (row['Entry Price'] - row['Take Profit Price']) * row['Lots'],
                axis=1
            )
            df_journal['win'] = df_journal['pnl'] > 0

            win_rate = df_journal['win'].mean() * 100
            total_trades = len(df_journal)
            avg_pnl = df_journal['pnl'].mean()

            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Win Rate", f"{win_rate:.2f}%")
            col_metric2.metric("Average PnL", f"${avg_pnl:.2f}")
            col_metric3.metric("Total Trades", total_trades)
            
            st.subheader("Performance Charts")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                pnl_by_symbol = df_journal.groupby('Symbol')['pnl'].sum().reset_index()
                fig = px.bar(pnl_by_symbol, x='Symbol', y='pnl', title="Total PnL by Symbol")
                st.plotly_chart(fig, use_container_width=True)
            with col_chart2:
                emotion_counts = df_journal['Emotions'].value_counts().reset_index()
                fig = px.pie(emotion_counts, names='Emotions', values='count', title="Trades by Emotional State")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades logged yet. Add trades in the '📝 Log Trade' tab to see analytics.")

    with tab_history:
        st.subheader("Complete Trade History")
        if not st.session_state.tools_trade_journal.empty:
            st.dataframe(st.session_state.tools_trade_journal, use_container_width=True)
        else:
            st.info("No trades available in history.")

elif st.session_state.current_page == 'mt5':
    st.title("📊 Performance Dashboard")
    st.caption("Analyze your MT5 trading history with advanced metrics and visualizations.")
    st.markdown('---')
    uploaded_file = st.file_uploader("Upload MT5 History (HTML file)", type=["htm", "html"])
    st.info("To get your MT5 history, open MT5, go to the 'History' tab, right-click, choose 'Report', and save as 'HTML File'.")

    if uploaded_file is not None:
        try:
            dfs = pd.read_html(uploaded_file.getvalue(), encoding='utf-16')
            df_deals = dfs[2] 
            df_deals.columns = df_deals.iloc[0]
            df_deals = df_deals[1:]
            df_deals.rename(columns={
                'Time': 'close_time', 'Order': 'order_id', 'Type': 'type',
                'Size': 'size', 'Price': 'close_price', 'Profit': 'pnl'
            }, inplace=True)
            
            df_deals['pnl'] = pd.to_numeric(df_deals['pnl'].str.replace(' ', ''), errors='coerce')
            st.session_state.mt5_df = df_deals

            st.success("Successfully processed MT5 report!")
            
            total_trades = len(df_deals)
            net_profit = df_deals['pnl'].sum()
            wins = df_deals[df_deals['pnl'] > 0]
            losses = df_deals[df_deals['pnl'] <= 0]
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
            
            st.subheader("Overall Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", total_trades)
            col2.metric("Net Profit", f"${net_profit:,.2f}")
            col3.metric("Win Rate", f"{win_rate:.2f}%")

            st.subheader("Equity Curve")
            df_deals['equity'] = df_deals['pnl'].cumsum()
            fig = px.line(df_deals, x=pd.to_datetime(df_deals['close_time']), y='equity', title='Account Equity Over Time')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to process the report. Ensure it's a valid MT5 HTML report. Error: {e}")
            logging.error(f"Error processing MT5 report: {e}")
    else:
        st.info("Upload a report to see your performance dashboard.")

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
                "Name": strategy_name, "Description": description, "Entry Rules": entry_rules,
                "Exit Rules": exit_rules, "Risk Management": risk_management, "Date Added": dt.datetime.now().strftime("%Y-%m-%d")
            }
            if "strategies" not in st.session_state:
                st.session_state.strategies = pd.DataFrame(columns=strategy_data.keys())
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
                except Exception as e:
                    st.error(f"Failed to save strategy: {str(e)}")
            else:
                st.success(f"Strategy '{strategy_name}' saved locally.")
    
    if "strategies" in st.session_state and not st.session_state.strategies.empty:
        st.subheader("Your Strategies")
        for idx, row in st.session_state.strategies.iterrows():
            with st.expander(f"Strategy: {row['Name']} (Added: {row['Date Added']})"):
                st.markdown(f"**Description:** {row['Description']}")
                st.markdown(f"**Entry Rules:** {row['Entry Rules']}")
                st.markdown(f"**Exit Rules:** {row['Exit Rules']}")
                st.markdown(f"**Risk Management:** {row['Risk Management']}")

# --- [IMPROVED] My Account Page ---
elif st.session_state.current_page == 'account':
    
    # --- View for Logged-Out Users ---
    if "logged_in_user" not in st.session_state:
        st.title("👤 My Account")
        st.markdown("Manage your account, save your data, and sync your trading journal and drawings.")
        st.write('---')
        tab_signin, tab_signup = st.tabs(["🔑 Sign In", "📝 Sign Up"])

        with tab_signin:
            st.subheader("Welcome back! Please sign in.")
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
                            st.session_state.tools_trade_journal = pd.DataFrame(user_data["tools_trade_journal"])
                        if "strategies" in user_data:
                            st.session_state.strategies = pd.DataFrame(user_data["strategies"])
                        
                        # Load gamification data
                        st.session_state.xp = user_data.get('xp', 0)
                        st.session_state.level = user_data.get('level', 0)
                        st.session_state.badges = user_data.get('badges', [])
                        st.session_state.streak = user_data.get('streak', 0)
                        st.session_state.last_journal_date = user_data.get('last_journal_date')
                        
                        st.success(f"Welcome back, {username}!")
                        logging.info(f"User {username} logged in successfully")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_signup:
            st.subheader("Create a new account.")
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Register")
                if register_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif not new_username or not new_password:
                        st.error("Username and password cannot be empty.")
                    else:
                        c.execute("SELECT username FROM users WHERE username = ?", (new_username,))
                        if c.fetchone():
                            st.error("Username already exists.")
                        else:
                            hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                            initial_data = json.dumps({"xp": 0, "level": 0, "badges": [], "streak": 0, "drawings": {}, "tools_trade_journal": [], "strategies": []})
                            c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", (new_username, hashed_password, initial_data))
                            conn.commit()
                            st.success(f"Account created for {new_username}! Please log in.")
                            logging.info(f"User {new_username} registered successfully")
                            
    # --- [NEW] View for Logged-In Users ---
    else:
        st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")
        st.markdown("This is your personal dashboard. Track your progress and manage your account.")
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Progress Snapshot")
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(label="Current Level", value=f"LVL {st.session_state.get('level', 0)}")
            metric2.metric(label="Journaling Streak", value=f"{st.session_state.get('streak', 0)} Days 🔥")
            metric3.metric(label="Total XP", value=f"{st.session_state.get('xp', 0)}")

            st.markdown("#### **Level Progress**")
            current_level = st.session_state.get('level', 0)
            xp_in_current_level = st.session_state.get('xp', 0) - (current_level * 1000)
            progress_percentage = xp_in_current_level / 1000
            
            st.progress(progress_percentage)
            st.caption(f"{xp_in_current_level} / 1000 XP to the next level.")

        with col2:
            st.subheader("🏆 Badges")
            badges = st.session_state.get('badges', [])
            if badges:
                badge_str = "".join([f"<span style='font-size: 1.1em; margin: 4px; padding: 5px 10px; background-color: #4d7171; border-radius: 15px;'>🏅 {badge}</span>" for badge in badges])
                st.markdown(badge_str, unsafe_allow_html=True)
            else:
                st.info("No badges earned yet. Keep up the great work to unlock them!")

        st.markdown("---")

        with st.expander("⚙️ Manage Account & Data"):
            st.write(f"**Username**: `{st.session_state.logged_in_user}`")
            if st.button("Log Out", key="logout_account_page", type="primary"):
                handle_logout()

elif st.session_state.current_page == 'community':
    st.title("🌐 Community Trade Ideas")
    st.markdown(""" Share and explore trade ideas with the community. Upload your chart screenshots and discuss strategies with other traders. """)
    st.write('---')
    if "logged_in_user" in st.session_state:
        with st.expander("➕ Share a Trade Idea"):
            with st.form("trade_idea_form"):
                trade_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
                trade_direction = st.radio("Direction", ["Long", "Short"])
                trade_description = st.text_area("Trade Description")
                uploaded_image = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg"])
                submit_idea = st.form_submit_button("Share Idea")
                if submit_idea:
                    username = st.session_state.logged_in_user
                    idea_id = _ta_hash()
                    idea_data = {
                        "Username": username, "Pair": trade_pair, "Direction": trade_direction,
                        "Description": trade_description, "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "IdeaID": idea_id
                    }
                    if uploaded_image:
                        image_path = os.path.join(_ta_user_dir(username), "community_images", f"{idea_id}.png")
                        with open(image_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())
                        idea_data["ImagePath"] = image_path
                    
                    # Add new idea to a DataFrame before saving
                    new_idea_df = pd.DataFrame([idea_data])
                    st.session_state.trade_ideas = pd.concat([st.session_state.trade_ideas, new_idea_df], ignore_index=True)
                    _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                    st.success("Trade idea shared successfully!")
                    st.rerun()
    else:
        st.warning("Please log in to share trade ideas.")

    st.subheader("📈 Recent Trade Ideas")
    if not st.session_state.trade_ideas.empty:
        # Sort by timestamp to show recent first
        df_ideas = st.session_state.trade_ideas
        df_ideas['Timestamp'] = pd.to_datetime(df_ideas['Timestamp'])
        df_ideas = df_ideas.sort_values(by='Timestamp', ascending=False)
        for idx, idea in df_ideas.iterrows():
            with st.container():
                st.markdown(f"**{idea['Pair']} ({idea['Direction']})** by *{idea['Username']}* at {idea['Timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown(idea['Description'])
                if "ImagePath" in idea and pd.notna(idea['ImagePath']) and os.path.exists(idea['ImagePath']):
                    st.image(idea['ImagePath'])
                st.markdown("---")
    else:
        st.info("No trade ideas shared yet. Be the first to contribute!")
    
elif st.session_state.current_page == 'tools':
    st.title("🛠 Tools")
    st.markdown("A suite of calculators and planners to support your trading.")
    st.markdown('---')
    tools_options = [
        'Profit/Loss Calculator', 'Risk Management Calculator', 'Drawdown Recovery Planner',
        'Pre-Trade Checklist', 'Trading Session Tracker'
    ]
    tabs = st.tabs(tools_options)
    with tabs[0]:
        st.header("💰 Profit / Loss Calculator")
        col1, col2 = st.columns(2)
        with col1:
            pair = st.selectbox("Pair", ["EUR/USD", "USD/JPY"], key="pl_pair")
            pos_size = st.number_input("Position Size (lots)", 0.01, value=0.1)
        with col2:
            open_price = st.number_input("Open Price", format="%.5f", value=1.08000)
            close_price = st.number_input("Close Price", format="%.5f", value=1.08500)
        
        pips = (close_price - open_price) * (10000 if 'JPY' not in pair else 100)
        pnl = pips * pos_size * (10 if 'JPY' not in pair else 0.8) # Approx USD pip value
        st.metric("Potential Profit/Loss", f"${pnl:.2f}", f"{pips:.1f} pips")

    with tabs[1]:
        st.header("🛡️ Risk Management Calculator")
        st.markdown("Calculate the appropriate lot size based on your risk tolerance.")
        col1, col2, col3 = st.columns(3)
        with col1:
            balance = st.number_input("Account Balance ($)", 1.0, value=10000.0)
        with col2:
            risk_pct = st.number_input("Risk per Trade (%)", 0.1, 10.0, value=1.0)
        with col3:
            stop_pips = st.number_input("Stop Loss (pips)", 1.0, value=20.0)

        risk_amount = balance * (risk_pct / 100)
        lot_size = risk_amount / (stop_pips * 10) # Assuming $10 pip value per standard lot
        st.metric("Recommended Lot Size", f"{lot_size:.2f} lots", f"Risking ${risk_amount:.2f}")

    with tabs[2]:
        st.header("📉 Drawdown Recovery Planner")
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 75.0, 10.0) / 100
        recovery_pct = _ta_percent_gain_to_recover(drawdown_pct)
        st.metric("Required Gain to Recover", f"{recovery_pct*100:.2f}%")
        st.info("A 10% loss requires an 11.11% gain to recover. A 50% loss requires a 100% gain.")

    with tabs[3]:
        st.header("✅ Pre-Trade Checklist")
        st.markdown("Ensure discipline by running through this checklist before every trade.")
        checklist_items = [
            "Market structure aligns with my bias?",
            "Key levels (S/R) identified?",
            "Entry trigger confirmed?",
            "Risk-reward ratio is favorable (e.g., ≥ 1:2)?",
            "No high-impact news imminent?",
            "Position size calculated correctly?",
        ]
        all_checked = all(st.checkbox(item, key=f"checklist_{i}") for i, item in enumerate(checklist_items))
        if all_checked:
            st.success("✅ All checks passed! Ready to trade.")
        else:
            st.warning("Complete all checklist items before executing a trade.")

    with tabs[4]:
        st.header("🕒 Forex Market Sessions")
        now_utc = dt.datetime.now(pytz.utc)
        sessions = {
            "London": (7, 16), "New York": (12, 21), "Tokyo": (0, 9), "Sydney": (22, 7)
        }
        active_sessions = []
        for name, (start, end) in sessions.items():
            if start < end:
                is_active = start <= now_utc.hour < end
            else: # Handles overnight sessions like Sydney
                is_active = now_utc.hour >= start or now_utc.hour < end
            
            if is_active:
                active_sessions.append(name)
        
        st.write(f"**Current UTC Time:** {now_utc.strftime('%H:%M:%S')}")
        if active_sessions:
            st.success(f"**Active Sessions:** {', '.join(active_sessions)}")
        else:
            st.info("Markets are currently closed or in a quiet period.")

elif st.session_state.current_page == "Zenvo Academy":
    st.title("🎓 Zenvo Academy")
    st.caption("Learn, Grow, Succeed.")
    st.markdown('---')
    st.write("Our Academy provides beginner traders with a clear learning path – covering Forex basics, chart analysis, risk management, and trading psychology. Build a solid foundation before stepping into live markets.")
    st.info("This page is under construction. More content and features coming soon!")
