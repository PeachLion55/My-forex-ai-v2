# =========================================================
# IMPORTS
# =========================================================
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
#import scipy.stats
import streamlit as st
from PIL import Image
import io
import base64
import calendar
from datetime import datetime, date, timedelta

# =========================================================
# GLOBAL CSS & GRIDLINE SETTINGS
# =========================================================
st.markdown(
    """
    <style>
    /* --- Global Horizontal Line Style --- */
    hr {
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        border-top: 1px solid #4d7171 !important;
        border-bottom: none !important; /* Remove any bottom border */
        background-color: transparent !important; /* Ensure no background color interferes */
        height: 1px !important; /* Set a specific height */
    }

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

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================
# TA_PRO HELPER FUNCTIONS
# =========================================================
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

# =========================================================
# DATABASE & XP SYSTEM SETUP
# =========================================================
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
        # Handle datetime and date objects by returning ISO format. [6, 7, 8, 9, 10]
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        # Handle Pandas NA values by returning None, ensuring JSON compatibility. [7]
        if pd.isna(obj):
            return None
        return super().default(obj)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide") # [1, 2, 3, 4, 5]

# =========================================================
# CUSTOM SIDEBAR CSS
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
# NEWS & ECONOMIC CALENDAR DATA / HELPERS
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

# =========================================================
# JOURNAL & DRAWING INITIALIZATION
# =========================================================
# Initialize drawings in session_state
if "drawings" not in st.session_state:
    st.session_state.drawings = {}
    logging.info("Initialized st.session_state.drawings")

# Define journal columns and dtypes (UPDATED GLOBAL DEFINITIONS)
journal_cols = [
    "Trade ID", "Date", "Entry Time", "Exit Time", "Symbol", "Trade Type", "Lots",
    "Entry Price", "Stop Loss Price", "Take Profit Price", "Final Exit Price",
    "Win/Loss", "PnL ($)", "Pips", "Initial R", "Realized R",
    "Strategy Used",
    "Weekly Bias", "Daily Bias", "4H Structure", "1H Structure", "Market State (HTF)",
    "Primary Correlation", "Secondary Correlation", "News Event Impact",
    "Setup Name", "Indicators Used", "Entry Trigger", "Reasons for Entry",
    "Order Type", "Partial Exits", "Reasons for Exit",
    "Pre-Trade Mindset", "In-Trade Emotions", "Emotional Triggers", "Discipline Score 1-5",
    "Post-Trade Analysis", "Lessons Learned", "Adjustments", "Journal Notes",
    "Entry Screenshot", "Exit Screenshot", "Tags"
]

journal_dtypes = {
    "Trade ID": str, "Date": "datetime64[ns]", "Entry Time": "datetime64[ns]", "Exit Time": "datetime64[ns]", "Symbol": str,
    "Trade Type": str, "Lots": float,
    "Entry Price": float, "Stop Loss Price": float, "Take Profit Price": float, "Final Exit Price": float,
    "Win/Loss": str, "PnL ($)": float, "Pips": float, "Initial R": float, "Realized R": float,
    "Strategy Used": str,
    "Weekly Bias": str, "Daily Bias": str, "4H Structure": str, "1H Structure": str, "Market State (HTF)": str,
    "Primary Correlation": str, "Secondary Correlation": str, "News Event Impact": str,
    "Setup Name": str, "Indicators Used": str, "Entry Trigger": str, "Reasons for Entry": str,
    "Order Type": str, "Partial Exits": bool, "Reasons for Exit": str,
    "Pre-Trade Mindset": str, "In-Trade Emotions": str, "Emotional Triggers": str, "Discipline Score 1-5": float,
    "Post-Trade Analysis": str, "Lessons Learned": str, "Adjustments": str, "Journal Notes": str,
    "Entry Screenshot": str, "Exit Screenshot": str, "Tags": str
}

# This robust initialization logic is key. It ensures the session state DataFrame always matches the current schema.
if "tools_trade_journal" not in st.session_state:
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes, errors='ignore')
    for col, dtype in journal_dtypes.items():
        if dtype == str:
            st.session_state.tools_trade_journal[col] = ''
        elif 'datetime' in str(dtype):
            st.session_state.tools_trade_journal[col] = pd.NaT
        elif dtype == float:
            st.session_state.tools_trade_journal[col] = 0.0
        elif dtype == bool:
            st.session_state.tools_trade_journal[col] = False
else:
    # Migrate existing journal data to new schema on app load if needed.
    # This ensures consistency even if column definitions change across app versions.
    current_journal_df = st.session_state.tools_trade_journal.copy()
    reindexed_journal = pd.DataFrame(index=current_journal_df.index, columns=journal_cols)

    for col in journal_cols:
        if col in current_journal_df.columns:
            reindexed_journal[col] = current_journal_df[col]
        else:
            if journal_dtypes[col] == str:
                reindexed_journal[col] = ""
            elif 'datetime' in str(journal_dtypes[col]):
                reindexed_journal[col] = pd.NaT
            elif journal_dtypes[col] == float:
                reindexed_journal[col] = 0.0
            elif journal_dtypes[col] == bool:
                reindexed_journal[col] = False
            else:
                reindexed_journal[col] = np.nan

    for col, dtype in journal_dtypes.items():
        if dtype == str:
            reindexed_journal[col] = reindexed_journal[col].fillna('').astype(str)
        elif 'datetime' in str(dtype):
            reindexed_journal[col] = pd.to_datetime(reindexed_journal[col], errors='coerce')
        elif dtype == float:
            reindexed_journal[col] = pd.to_numeric(reindexed_journal[col], errors='coerce').fillna(0.0).astype(float)
        elif dtype == bool:
            reindexed_journal[col] = reindexed_journal[col].fillna(False).astype(bool)
        else:
            reindexed_journal[col] = reindexed_journal[col].astype(dtype, errors='ignore')

    st.session_state.tools_trade_journal = reindexed_journal[journal_cols] # Ensure column order
    
# Initialize temporary journal for form (remains the same)
if "temp_journal" not in st.session_state:
    st.session_state.temp_journal = None

# =========================================================
# GAMIFICATION HELPERS
# =========================================================
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

# =========================================================
# COMMUNITY DATA LOADING
# =========================================================
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
    ('tools', 'Tools'),
    ('strategy', 'Manage My Strategy'),
    ('community', 'Community Trade Ideas'),
    ('Zenvo Academy', 'Zenvo Academy'),
    ('account', 'My Account')
]

for page_key, page_name in nav_items:
    if st.sidebar.button(page_name, key=f"nav_{page_key}"):
        st.session_state.current_page = page_key
        st.session_state.current_subpage = None
        st.session_state.show_tools_submenu = False
        st.rerun()

# =========================================================
# MAIN APPLICATION LOGIC
# =========================================================
# =========================================================
# FUNDAMENTALS PAGE
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

# =========================================================
# BACKTESTING PAGE
# =========================================================
elif st.session_state.current_page == 'backtesting':
    st.title("📈 Backtesting")
    st.caption("Live TradingView chart for backtesting and enhanced trading journal for tracking and analyzing trades.")
    st.markdown('---')

    # Pair selector & symbol map (remains unchanged)
    pairs_map = {
        "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD",
        "USD/CHF": "OANDA:USDCHF", "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD",
        "USD/CAD": "CMCMARKETS:USDCAD", "EUR/GBP": "FX:EURGBP", "EUR/JPY": "FX:EURJPY",
        "GBP/JPY": "FX:GBPJPY", "AUD/JPY": "FX:AUDJPY", "AUD/NZD": "FX:AUDNZD",
        "AUD/CAD": "FX:AUDCAD", "AUD/CHF": "FX:AUDCHF", "CAD/JPY": "FX:CADJPY",
        "CHF/JPY": "FX:CHFJPY", "EUR/AUD": "FX:EURAUD", "EUR/CAD": "FX:EURCAD",
        "EUR/CHF": "FX:EURCHF", "GBP/AUD": "FX:GBPAUD", "GBP/CAD": "FX:GBPCAD",
        "GBP/CHF": "FX:GBPCHF", "NZD/JPY": "FX:NZDJPY", "NZD/CAD": "FX:NZDCAD",
        "NZD/CHF": "FX:NZDCHF", "CAD/CHF": "FX:CADCHF",
    }
    pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
    tv_symbol = pairs_map[pair]

    # Initialize drawings in session state if not present
    if 'drawings' not in st.session_state:
        st.session_state.drawings = {}
    
    # Load initial drawings if available (requires login)
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
    
    # TradingView widget (remains unchanged)
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

    # Save, Load, and Refresh buttons for drawings (minor update for journal sync)
    if "logged_in_user" in st.session_state:
        col1_draw, col2_draw, col3_draw = st.columns([1, 1, 1])
        with col1_draw:
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
        with col2_draw:
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
        with col3_draw:
            if st.button("Refresh Account (Drawings/Journal Sync)", key="bt_refresh_account"):
                username = st.session_state.logged_in_user
                logging.info(f"Refresh Account button clicked for user {username}")
                try:
                    c.execute("SELECT data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result:
                        user_data = json.loads(result[0])
                        st.session_state.drawings = user_data.get("drawings", {})
                        
                        # Robust re-processing of journal data for new schema.
                        loaded_journal_raw = user_data.get("tools_trade_journal", [])
                        loaded_journal_df = pd.DataFrame(loaded_journal_raw)

                        # Create an empty, fully-structured DataFrame with all current journal_cols
                        master_journal_df = pd.DataFrame(columns=journal_cols)

                        # Copy existing data, fill missing with defaults based on dtype
                        for col in journal_cols:
                            if col in loaded_journal_df.columns:
                                master_journal_df[col] = loaded_journal_df[col]
                            else:
                                if journal_dtypes[col] == str:
                                    master_journal_df[col] = ""
                                elif 'datetime' in str(journal_dtypes[col]):
                                    master_journal_df[col] = pd.NaT # pandas Not-a-Time for missing datetimes
                                elif journal_dtypes[col] == float:
                                    master_journal_df[col] = 0.0
                                elif journal_dtypes[col] == bool:
                                    master_journal_df[col] = False
                                else:
                                    master_journal_df[col] = np.nan

                        # Enforce dtypes and fill any remaining NaNs in string columns
                        for col, dtype in journal_dtypes.items():
                            if dtype == str:
                                master_journal_df[col] = master_journal_df[col].fillna('').astype(str)
                            elif 'datetime' in str(dtype):
                                master_journal_df[col] = pd.to_datetime(master_journal_df[col], errors='coerce')
                            elif dtype == float:
                                master_journal_df[col] = pd.to_numeric(master_journal_df[col], errors='coerce').fillna(0.0).astype(float)
                            elif dtype == bool:
                                master_journal_df[col] = master_journal_df[col].fillna(False).astype(bool)
                            else: # Fallback for other dtypes
                                master_journal_df[col] = master_journal_df[col].astype(dtype, errors='ignore')

                        st.session_state.tools_trade_journal = master_journal_df
                        
                        st.success("Account synced successfully!")
                        logging.info(f"Account synced for {username}. Drawings and Journal updated.")
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
                    # Clean up the session state flags
                    if drawings_key in st.session_state: del st.session_state[drawings_key]
                    if f"bt_save_trigger_{pair}" in st.session_state: del st.session_state[f"bt_save_trigger_{pair}"]
            else:
                st.warning("No valid drawing content received. Ensure you have drawn on the chart.")
                logging.warning(f"No valid drawing content received for {pair}: {content}")
    else:
        st.info("Sign in via the My Account tab to save/load drawings and trading journal.")
        logging.info("User not logged in, save/load drawings disabled")
    
    # Backtesting Journal (Restructured for better utility and insights)
    st.markdown("### 📝 Advanced Trading Journal")
    st.markdown(
        """
        Log your trades with detailed analysis, track psychological factors, market context, and strategy performance.
        Leverage rich text fields to enhance your journaling experience!
        """
    )

    # Add custom CSS for enhanced Markdown display of text areas
    st.markdown("""
        <style>
        div.stText.css-1r0sq94 p, /* For direct st.markdown of raw text_area output */
        .st-br p, .st-cw p {
             margin-bottom: 0.5rem; /* Better spacing for paragraphs in notes */
             line-height: 1.5;
        }
        .styled-text-area-display p {
            margin-bottom: 0.5rem; /* Better spacing for paragraphs in notes */
            line-height: 1.5;
        }
        .styled-text-area-display ul, .styled-text-area-display ol {
            margin-top: 0;
            margin-bottom: 0.5rem;
            padding-left: 20px;
        }
        .styled-text-area-display h1, .styled-text-area-display h2, .styled-text-area-display h3,
        .styled-text-area-display h4, .styled-text-area-display h5, .styled-text-area-display h6 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(88, 179, 177, 0.5);
            padding-bottom: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Function to apply user-chosen styles to text (used for display)
    def apply_text_styles(text_content, font_size="14px", font_color="white", bold=False, italic=False, underline=False):
        style = f"font-size: {font_size}; color: {font_color};"
        if bold: style += "font-weight: bold;"
        if italic: style += "font-style: italic;"
        if underline: style += "text-decoration: underline;"
        # Use a div to apply overall styles and markdown for content within it
        return f'<div class="styled-text-area-display" style="{style}">{text_content}</div>'

    # Tabs for Journal Entry, Analytics, and History
    tab_entry, tab_analytics, tab_history = st.tabs(["📝 Log Trade", "📊 Analytics", "📜 Trade History"])

    # =========================================================
    # LOG TRADE TAB (UX Friendly Redesign)
    # =========================================================
    with tab_entry:
        st.subheader("Log a New Trade (Comprehensive & Intuitive)")
        # Pre-fill form if 'edit_trade_data' exists (from Trade History's Edit button)
        initial_data = st.session_state.get('edit_trade_data', {})
        is_editing = bool(initial_data)

        with st.form("trade_entry_form", clear_on_submit=not is_editing):
            # Section: General Trade Details
            st.markdown("### 🏷️ Trade Overview")
            
            cols_id_strategy, cols_datetime = st.columns(2)
            with cols_id_strategy:
                trade_id_input = st.text_input("Trade ID", value=initial_data.get("Trade ID", f"TRD-{_ta_hash()}"), disabled=is_editing, help="A unique identifier for your trade. Auto-generated if left empty.")
                # Add default strategy list from current user's saved strategies
                user_strategies = ["(Select One)"] + sorted([s['Name'] for s in st.session_state.strategies.to_dict('records')] if "strategies" in st.session_state and not st.session_state.strategies.empty else [])
                default_strategy_idx = user_strategies.index(initial_data.get("Strategy Used", "(Select One)")) if initial_data.get("Strategy Used", "(Select One)") in user_strategies else 0
                selected_strategy = st.selectbox("Strategy Used", options=user_strategies, index=default_strategy_idx, help="Link this trade to one of your defined strategies.", key="strategy_used_input")
            
            with cols_datetime:
                trade_date_val = initial_data.get("Date", dt.datetime.now())
                entry_time_val = initial_data.get("Entry Time", dt.datetime.now())
                exit_time_val = initial_data.get("Exit Time", dt.datetime.now())

                trade_date = st.date_input("Date", value=trade_date_val.date() if isinstance(trade_date_val, dt.datetime) else trade_date_val, help="The calendar date of your trade.", key="trade_date_input")
                entry_time = st.time_input("Entry Time", value=entry_time_val.time() if isinstance(entry_time_val, dt.datetime) else entry_time_val, help="The time you entered the trade.", key="entry_time_input")
                exit_time = st.time_input("Exit Time", value=exit_time_val.time() if isinstance(exit_time_val, dt.datetime) else exit_time_val, help="The time you exited the trade.", key="exit_time_input")

            cols_symbol_type, cols_sizing_action = st.columns(2)
            with cols_symbol_type:
                symbol_options = list(pairs_map.keys()) + ["Other"]
                default_symbol = initial_data.get("Symbol", pair)
                default_symbol_idx = symbol_options.index(default_symbol) if default_symbol in symbol_options else (symbol_options.index("Other") if default_symbol != "" else 0)
                symbol = st.selectbox("Currency Pair / Asset", symbol_options, index=default_symbol_idx, help="The asset you traded.", key="symbol_input")
                if symbol == "Other":
                    symbol = st.text_input("Specify Custom Asset", value=initial_data.get("Symbol", ""), help="Enter asset name if not in list.", key="custom_symbol_input")
                
                trade_type = st.radio("Trade Direction", ["Long", "Short", "Breakeven", "No-Trade (Study)"], horizontal=True, 
                                      index=["Long", "Short", "Breakeven", "No-Trade (Study)"].index(initial_data.get("Trade Type", "Long")), 
                                      help="Was this a buy, sell, a trade that broke even, or just a study/watch entry?", key="trade_type_input")
            
            with cols_sizing_action:
                lots = st.number_input("Position Size (Lots)", min_value=0.01, step=0.01, format="%.2f", value=float(initial_data.get("Lots", 0.1)), help="The size of your trade in lots.", key="lots_input")
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Entry Price", 0.0)), help="The price you entered the market.", key="entry_price_input")
                stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Stop Loss Price", 0.0)), help="The price where your stop loss was placed.", key="stop_loss_price_input")
                take_profit_price = st.number_input("Take Profit Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Take Profit Price", 0.0)), help="The price where your take profit was placed.", key="take_profit_price_input")
                final_exit_price = st.number_input("Final Exit Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Final Exit Price", 0.0)), help="The actual price your trade was closed.", key="final_exit_price_input")
            
            st.markdown("---")

            # --- Section: Market Context ---
            with st.expander("🌍 Market Context & Bias", expanded=False):
                st.markdown("Assess the broader market conditions at the time of your trade.")
                cols_bias, cols_structure, cols_influencers = st.columns(3)
                with cols_bias:
                    st.markdown("##### Longer Term Bias")
                    weekly_bias = st.selectbox("Weekly Bias", ["Bullish", "Bearish", "Neutral"], index=["Bullish", "Bearish", "Neutral"].index(initial_data.get("Weekly Bias", "Neutral")), help="Overall market direction on the weekly timeframe.", key="weekly_bias_input")
                    daily_bias = st.selectbox("Daily Bias", ["Bullish", "Bearish", "Neutral"], index=["Bullish", "Bearish", "Neutral"].index(initial_data.get("Daily Bias", "Neutral")), help="Overall market direction on the daily timeframe.", key="daily_bias_input")
                with cols_structure:
                    st.markdown("##### Price Structure")
                    h4_structure = st.selectbox("4H Structure", ["Impulsive", "Corrective", "Consolidating", "None"], index=["Impulsive", "Corrective", "Consolidating", "None"].index(initial_data.get("4H Structure", "None")), help="Current price movement characteristic on 4H.", key="4h_structure_input")
                    h1_structure = st.selectbox("1H Structure", ["Impulsive", "Corrective", "Consolidating", "None"], index=["Impulsive", "Corrective", "Consolidating", "None"].index(initial_data.get("1H Structure", "None")), help="Current price movement characteristic on 1H.", key="1h_structure_input")
                with cols_influencers:
                    st.markdown("##### Influencing Factors")
                    market_state = st.selectbox("Market State (Higher Timeframe)", ["Trend (Bullish)", "Trend (Bearish)", "Range", "Complex Pullback", "Choppy", "Undefined"], 
                                                index=["Trend (Bullish)", "Trend (Bearish)", "Range", "Complex Pullback", "Choppy", "Undefined"].index(initial_data.get("Market State (HTF)", "Undefined")), help="Dominant market condition.", key="market_state_input")
                    primary_corr = st.text_input("Primary Correlated Pair & Bias", value=initial_data.get("Primary Correlation", ""), help="E.g., EUR/JPY (Bullish) - a strongly correlated pair and its current bias.", key="primary_corr_input")
                    secondary_corr = st.text_input("Secondary Correlated Pair & Bias", value=initial_data.get("Secondary Correlation", ""), help="A second correlated pair, if applicable.", key="secondary_corr_input")
                    news_impact_options = ["High Impact (Positive)", "High Impact (Negative)", "Medium Impact", "Low Impact", "None"]
                    initial_news_impact = initial_data.get("News Event Impact", "").split(',') if initial_data.get("News Event Impact") else ["None"]
                    news_impact = st.multiselect("News Event Impact", news_impact_options, 
                                                 default=[ni for ni in initial_news_impact if ni in news_impact_options], help="How upcoming or recent news events influenced your trade decision.", key="news_impact_input")
            
            st.markdown("---")

            # --- Section: Setup & Entry Details ---
            with st.expander("🎯 Setup & Entry Plan", expanded=False):
                st.markdown("Document your specific setup, entry criteria, and rationale.")
                cols_setup, cols_entry_notes = st.columns(2)
                with cols_setup:
                    setup_name = st.text_input("Setup Name", value=initial_data.get("Setup Name", ""), help="Name of your recognized trade setup (e.g., 'Double Bottom Reversal').", key="setup_name_input")
                    indicators_used_options = ["Moving Averages", "RSI", "MACD", "Fibonacci", "Support/Resistance", "Trendlines", "Chart Patterns", "Order Blocks", "Liquidity Concepts", "Supply/Demand Zones", "Volume"]
                    initial_indicators = initial_data.get("Indicators Used", "").split(',') if initial_data.get("Indicators Used") else []
                    indicators_used = st.multiselect("Indicators & Tools Used", indicators_used_options, 
                                                     default=[ind for ind in initial_indicators if ind in indicators_used_options], help="Which technical tools did you use for this setup?", key="indicators_input")
                    entry_trigger = st.text_input("Specific Entry Trigger", value=initial_data.get("Entry Trigger", ""), help="The exact condition that triggered your entry (e.g., 'Pin bar close above support', 'Break of trendline on 5min').", key="entry_trigger_input")

                with cols_entry_notes:
                    st.markdown("##### Reasons for Entry Notes (Use Markdown: `**bold**`, `*italic*`, `- list`)")
                    initial_entry_reasons_json = initial_data.get("Reasons for Entry", json.dumps({'text': '', 'style': {'font_size': '14px', 'font_color': '#FFFFFF', 'bold': False, 'italic': False, 'underline': False}}))
                    initial_entry_reasons_dict = json.loads(initial_entry_reasons_json)
                    reasons_for_entry_text = initial_entry_reasons_dict.get('text', '')
                    reasons_entry_style = initial_entry_reasons_dict.get('style', {})

                    cols_style_entry_text_1, cols_style_entry_text_2 = st.columns([1,1])
                    with cols_style_entry_text_1:
                        reasons_entry_font_size = st.selectbox("Font Size", ["12px", "14px", "16px", "18px"], index=["12px", "14px", "16px", "18px"].index(reasons_entry_style.get('font_size', '14px')), key="reasons_entry_font_size_sel")
                        reasons_entry_font_color = st.color_picker("Font Color", reasons_entry_style.get('font_color', '#FFFFFF'), key="reasons_entry_font_color_cp")
                    with cols_style_entry_text_2:
                        reasons_entry_bold = st.checkbox("Bold Text", value=reasons_entry_style.get('bold', False), key="reasons_entry_bold_cb")
                        reasons_entry_italic = st.checkbox("Italic Text", value=reasons_entry_style.get('italic', False), key="reasons_entry_italic_cb")
                        reasons_entry_underline = st.checkbox("Underline Text", value=reasons_entry_style.get('underline', False), key="reasons_entry_underline_cb")
                    reasons_for_entry = st.text_area(
                        "Describe your comprehensive entry reasons here.",
                        value=reasons_for_entry_text, height=200, help="Why did you take this trade? What were the confluence factors?", key="reasons_for_entry_input"
                    )

            st.markdown("---")

            # --- Section: Trade Execution & Management ---
            with st.expander("🛠️ Execution & Management", expanded=False):
                st.markdown("How was the trade managed once entered?")
                cols_order_exit, cols_screenshots = st.columns(2)
                with cols_order_exit:
                    order_type = st.radio("Order Type", ["Market Order", "Limit Order", "Stop Order"], horizontal=True,
                                          index=["Market Order", "Limit Order", "Stop Order"].index(initial_data.get("Order Type", "Market Order")), help="Type of order used to enter the market.", key="order_type_input")
                    partial_exits = st.checkbox("Were there partial exits taken?", value=initial_data.get("Partial Exits", False), help="Indicate if you closed portions of your position at different times/prices.", key="partial_exits_cb")
                    
                    st.markdown("##### Reasons for Exit Notes (Use Markdown)")
                    initial_exit_reasons_json = initial_data.get("Reasons for Exit", json.dumps({'text': '', 'style': {'font_size': '14px', 'font_color': '#FFFFFF', 'bold': False, 'italic': False, 'underline': False}}))
                    initial_exit_reasons_dict = json.loads(initial_exit_reasons_json)
                    reasons_for_exit_text = initial_exit_reasons_dict.get('text', '')
                    reasons_exit_style = initial_exit_reasons_dict.get('style', {})

                    cols_style_exit_text_1, cols_style_exit_text_2 = st.columns([1,1])
                    with cols_style_exit_text_1:
                        reasons_exit_font_size = st.selectbox("Font Size ", ["12px", "14px", "16px", "18px"], index=["12px", "14px", "16px", "18px"].index(reasons_exit_style.get('font_size', '14px')), key="reasons_exit_font_size_sel")
                        reasons_exit_font_color = st.color_picker("Font Color ", reasons_exit_style.get('font_color', '#FFFFFF'), key="reasons_exit_font_color_cp")
                    with cols_style_exit_text_2:
                        reasons_exit_bold = st.checkbox("Bold Text ", value=reasons_exit_style.get('bold', False), key="reasons_exit_bold_cb")
                        reasons_exit_italic = st.checkbox("Italic Text ", value=reasons_exit_style.get('italic', False), key="reasons_exit_italic_cb")
                        reasons_exit_underline = st.checkbox("Underline Text ", value=reasons_exit_style.get('underline', False), key="reasons_exit_underline_cb")
                    reasons_for_exit = st.text_area(
                        "Detailed Reasons for Exit (Why did you close the trade?)",
                        value=reasons_for_exit_text, height=150, help="What caused your exit (hit TP, SL, discretionary, etc.)?", key="reasons_for_exit_input"
                    )

                with cols_screenshots:
                    st.markdown("##### Visual Documentation")
                    # Initial image paths might come from an existing trade being edited.
                    # This logic saves uploaded files to the user's local directory and stores the path.
                    user_journal_images_dir = os.path.join(_ta_user_dir(st.session_state.get("logged_in_user", "guest")), "journal_images")
                    os.makedirs(user_journal_images_dir, exist_ok=True) # Ensure directory exists
                    
                    # Store current screenshot links/hashes to enable editing existing entries
                    entry_ss_initial_val = initial_data.get("Entry Screenshot", "")
                    exit_ss_initial_val = initial_data.get("Exit Screenshot", "")
                    
                    uploaded_entry_image = st.file_uploader("Upload Entry Screenshot (Optional)", type=["png", "jpg", "jpeg"], help="Upload an image of your chart at entry.", key="upload_entry_screenshot")
                    if uploaded_entry_image:
                        image_filename = f"{trade_id_input}_entry.png"
                        image_file_path = os.path.join(user_journal_images_dir, image_filename)
                        with open(image_file_path, "wb") as f:
                            f.write(uploaded_entry_image.getbuffer())
                        entry_ss_initial_val = image_file_path # Update the value to be saved
                        st.success("Entry screenshot uploaded successfully!")

                    uploaded_exit_image = st.file_uploader("Upload Exit Screenshot (Optional)", type=["png", "jpg", "jpeg"], help="Upload an image of your chart at exit.", key="upload_exit_screenshot")
                    if uploaded_exit_image:
                        image_filename = f"{trade_id_input}_exit.png"
                        image_file_path = os.path.join(user_journal_images_dir, image_filename)
                        with open(image_file_path, "wb") as f:
                            f.write(uploaded_exit_image.getbuffer())
                        exit_ss_initial_val = image_file_path # Update the value to be saved
                        st.success("Exit screenshot uploaded successfully!")

                    # Hidden inputs to store the file paths/hashes which will be saved to the DataFrame
                    # The value passed to st.text_input for 'value' must be str, so ensure it.
                    entry_screenshot_storage = st.text_input("Entry Screenshot (Stored Path)", value=entry_ss_initial_val, key="entry_screenshot_storage_hidden", label_visibility="hidden")
                    exit_screenshot_storage = st.text_input("Exit Screenshot (Stored Path)", value=exit_ss_initial_val, key="exit_screenshot_storage_hidden", label_visibility="hidden")

            st.markdown("---")

            # --- Section: Psychology & Discipline ---
            with st.expander("🧠 Psychology & Discipline", expanded=False):
                st.markdown("Reflect on your mental state before and during the trade.")
                cols_mindset, cols_emotions = st.columns(2)
                with cols_mindset:
                    pre_trade_mindset = st.text_area("Pre-Trade Mindset / Plan", value=initial_data.get("Pre-Trade Mindset", ""), help="How were you feeling, what was your plan going into the trade?", key="pre_trade_mindset_input")
                with cols_emotions:
                    in_trade_emotions_options = ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral", "FOMO", "Greedy", "Revenge", "Impulsive", "Disciplined", "Overconfident", "Patient", "Irritable"]
                    initial_in_trade_emotions = initial_data.get("In-Trade Emotions", "").split(',') if initial_data.get("In-Trade Emotions") else []
                    in_trade_emotions = st.multiselect(
                        "In-Trade Emotions During Trade",
                        in_trade_emotions_options, default=[e for e in initial_in_trade_emotions if e in in_trade_emotions_options],
                        help="Select emotions experienced during the trade.", key="in_trade_emotions_input"
                    )
                    emotional_triggers = st.text_area("Emotional Triggers Noted", value=initial_data.get("Emotional Triggers", ""), help="What specific events or observations triggered shifts in your emotions?", key="emotional_triggers_input")
                    discipline_score = st.slider("Discipline Score (1=Low, 5=High)", 1, 5, value=int(initial_data.get("Discipline Score 1-5", 3)), help="Rate your adherence to your plan and discipline.", key="discipline_score_input")

            st.markdown("---")

            # --- Section: Post-Trade Analysis & Learning ---
            with st.expander("💡 Post-Trade Analysis & Learning", expanded=True):
                st.markdown("Critical review and growth points after the trade.")
                cols_post_analysis, cols_lessons_learned = st.columns(2)
                with cols_post_analysis:
                    st.markdown("##### Post-Trade Analysis (What happened?)")
                    initial_pta_json = initial_data.get("Post-Trade Analysis", json.dumps({'text': '', 'style': {'font_size': '14px', 'font_color': '#FFFFFF', 'bold': False, 'italic': False, 'underline': False}}))
                    initial_pta_dict = json.loads(initial_pta_json)
                    post_trade_analysis_text = initial_pta_dict.get('text', '')
                    pta_style = initial_pta_dict.get('style', {})

                    cols_style_pta_text_1, cols_style_pta_text_2 = st.columns([1,1])
                    with cols_style_pta_text_1:
                        pta_font_size = st.selectbox("Font Size  ", ["12px", "14px", "16px", "18px"], index=["12px", "14px", "16px", "18px"].index(pta_style.get('font_size', '14px')), key="pta_font_size_sel")
                        pta_font_color = st.color_picker("Font Color  ", pta_style.get('font_color', '#FFFFFF'), key="pta_font_color_cp")
                    with cols_style_pta_text_2:
                        pta_bold = st.checkbox("Bold Text  ", value=pta_style.get('bold', False), key="pta_bold_cb")
                        pta_italic = st.checkbox("Italic Text  ", value=pta_style.get('italic', False), key="pta_italic_cb")
                        pta_underline = st.checkbox("Underline Text  ", value=pta_style.get('underline', False), key="pta_underline_cb")

                    post_trade_analysis = st.text_area(
                        "Objective review of price action, decision-making, divergences from plan, etc.",
                        value=post_trade_analysis_text, height=200, help="Analyze objectively what transpired and why.", key="post_trade_analysis_input"
                    )
                with cols_lessons_learned:
                    st.markdown("##### Lessons Learned (What did I learn?)")
                    initial_ll_json = initial_data.get("Lessons Learned", json.dumps({'text': '', 'style': {'font_size': '14px', 'font_color': '#FFFFFF', 'bold': False, 'italic': False, 'underline': False}}))
                    initial_ll_dict = json.loads(initial_ll_json)
                    lessons_learned_text = initial_ll_dict.get('text', '')
                    ll_style = initial_ll_dict.get('style', {})

                    cols_style_ll_text_1, cols_style_ll_text_2 = st.columns([1,1])
                    with cols_style_ll_text_1:
                        ll_font_size = st.selectbox("Font Size   ", ["12px", "14px", "16px", "18px"], index=["12px", "14px", "16px", "18px"].index(ll_style.get('font_size', '14px')), key="ll_font_size_sel")
                        ll_font_color = st.color_picker("Font Color   ", ll_style.get('font_color', '#FFFFFF'), key="ll_font_color_cp")
                    with cols_style_ll_text_2:
                        ll_bold = st.checkbox("Bold Text   ", value=ll_style.get('bold', False), key="ll_bold_cb")
                        ll_italic = st.checkbox("Italic Text   ", value=ll_style.get('italic', False), key="ll_italic_cb")
                        ll_underline = st.checkbox("Underline Text   ", value=ll_style.get('underline', False), key="ll_underline_cb")

                    lessons_learned = st.text_area(
                        "Key Takeaways for Future Improvement & What could have been done better.",
                        value=lessons_learned_text, height=150, help="Synthesize actionable insights for your next trades.", key="lessons_learned_input"
                    )
                    adjustments = st.text_area("Specific Adjustments to Strategy/Trading Plan", value=initial_data.get("Adjustments", ""), help="Concrete changes you plan to implement based on this trade.", key="adjustments_input")
            
            st.markdown("---")

            # --- Section: General Journal Notes & Tags ---
            with st.expander("📚 General Notes & Categorization", expanded=False):
                st.markdown("Capture any remaining thoughts and categorize your trade.")
                cols_notes_tags_1, cols_notes_tags_2 = st.columns(2)
                with cols_notes_tags_1:
                    st.markdown("##### Additional General Notes")
                    initial_general_notes_json = initial_data.get("Journal Notes", json.dumps({'text': '', 'style': {'font_size': '14px', 'font_color': '#FFFFFF', 'bold': False, 'italic': False, 'underline': False}}))
                    initial_general_notes_dict = json.loads(initial_general_notes_json)
                    journal_notes_general_text = initial_general_notes_dict.get('text', '')
                    general_notes_style = initial_general_notes_dict.get('style', {})

                    cols_style_notes_text_1, cols_style_notes_text_2 = st.columns([1,1])
                    with cols_style_notes_text_1:
                        journal_notes_font_size = st.selectbox("Notes Font Size", ["12px", "14px", "16px", "18px"], index=["12px", "14px", "16px", "18px"].index(general_notes_style.get('font_size', '14px')), key="journal_notes_font_size_sel")
                        journal_notes_font_color = st.color_picker("Notes Font Color", general_notes_style.get('font_color', '#FFFFFF'), key="journal_notes_font_color_cp")
                    with cols_style_notes_text_2:
                        journal_notes_bold = st.checkbox("Notes Bold", value=general_notes_style.get('bold', False), key="journal_notes_bold_cb")
                        journal_notes_italic = st.checkbox("Notes Italic", value=general_notes_style.get('italic', False), key="journal_notes_italic_cb")
                        journal_notes_underline = st.checkbox("Notes Underline", value=general_notes_style.get('underline', False), key="journal_notes_underline_cb")

                    journal_notes_general = st.text_area(
                        "Any additional general thoughts or information (Markdown supported).",
                        value=journal_notes_general_text, height=150, key="journal_notes_general_input"
                    )
                with cols_notes_tags_2:
                    st.markdown("##### Categorize Your Trade")
                    current_tags_in_journal = sorted(list(set(st.session_state.tools_trade_journal['Tags'].str.split(',').explode().dropna().astype(str).str.strip())))
                    suggested_tags = ["Setup: Reversal", "Setup: Trend-Continuation", "Mistake: Overtrading", "Mistake: No SL", "Emotion: FOMO", "Emotion: Revenge", "Session: London", "Session: NY", "Chart Pattern: Flag", "RSI Divergence", "News Play", "Range Breakout"]
                    all_tag_options = sorted(list(set(current_tags_in_journal + suggested_tags)))
                    initial_tags = initial_data.get("Tags", "").split(',') if initial_data.get("Tags") else []
                    tags = st.multiselect("Trade Tags", all_tag_options, 
                                          default=[t for t in initial_tags if t in all_tag_options], help="Categorize your trade for easier analysis later.", key="tags_input")

            # Clear edit state after form is rendered with initial_data
            if is_editing:
                # Remove this to prevent constant pre-filling when editing and not submitting form in same rerun
                # del st.session_state['edit_trade_data']
                pass # The deletion will happen after successful form submission

            st.markdown("---")
            submit_button = st.form_submit_button(
                f"{'Update Trade' if is_editing else 'Save New Trade'}", 
                type="primary", 
                help=f"{'Click to update this trade log.' if is_editing else 'Click to save your new trade log.'}"
            )
        
            if submit_button:
                # --- Calculations for derived fields ---
                pip_multiplier = 0 # default, will be set below
                approx_pip_value_usd_per_lot = 0.0 # default, will be set below
                pip_scale = 0.0 # default, will be set below

                if "JPY" in symbol:
                    pip_multiplier = 100
                    approx_pip_value_usd_per_lot = 8.5 # Rough average for JPY pairs ($/std lot/pip)
                    pip_scale = 0.01 
                else: # Most other pairs like EUR/USD, GBP/USD
                    pip_multiplier = 10000 
                    approx_pip_value_usd_per_lot = 10.0 # Rough average for non-JPY pairs ($/std lot/pip)
                    pip_scale = 0.0001 
                
                trade_pips_gain = 0.0
                pnL_dollars = 0.0
                win_loss_status = "No-Trade (Study)" # Default for clarity if calculations fail or type is 'Study'

                if trade_type in ["Long", "Short"]:
                    if entry_price > 0.0 and final_exit_price > 0.0:
                        if trade_type == "Long":
                            trade_pips_gain = (final_exit_price - entry_price) / pip_scale
                        elif trade_type == "Short":
                            trade_pips_gain = (entry_price - final_exit_price) / pip_scale
                        
                        pnL_dollars = trade_pips_gain * lots * (approx_pip_value_usd_per_lot / pip_multiplier * 10000) # (Pips * Lots * PipValueFactorPerPip)

                        if pnL_dollars > 0.0:
                            win_loss_status = "Win"
                        elif pnL_dollars < 0.0:
                            win_loss_status = "Loss"
                        else:
                            win_loss_status = "Breakeven"
                    else: # If pricing is incomplete for actual trades (no final exit price, etc)
                        win_loss_status = "Pending / Invalid Prices"
                        trade_pips_gain = 0.0
                        pnL_dollars = 0.0
                elif trade_type == "Breakeven":
                    win_loss_status = "Breakeven"
                    pnL_dollars = 0.0
                    trade_pips_gain = 0.0
                # else: trade_type is "No-Trade (Study)" - values remain 0/No-Trade (default above handles this)


                initial_r_calc = 0.0
                if entry_price > 0.0 and stop_loss_price > 0.0 and take_profit_price > 0.0 and trade_type in ["Long", "Short"]:
                    risk_per_unit = abs(entry_price - stop_loss_price)
                    reward_per_unit = abs(take_profit_price - entry_price)
                    if risk_per_unit > 0.0:
                        initial_r_calc = reward_per_unit / risk_per_unit
                
                realized_r_calc = 0.0
                if entry_price > 0.0 and stop_loss_price > 0.0 and final_exit_price > 0.0 and trade_type in ["Long", "Short"]:
                    risk_per_unit_realized = abs(entry_price - stop_loss_price)
                    realized_pnl_raw = final_exit_price - entry_price if trade_type == "Long" else entry_price - final_exit_price
                    if risk_per_unit_realized > 0.0:
                        realized_r_calc = realized_pnl_raw / risk_per_unit_realized # Can be negative

                # Prepare rich text fields for JSON storage
                reasons_entry_formatted_json = json.dumps({
                    'text': reasons_for_entry,
                    'style': {'font_size': reasons_entry_font_size, 'font_color': reasons_entry_font_color,
                              'bold': reasons_entry_bold, 'italic': reasons_entry_italic, 'underline': reasons_entry_underline}
                })
                reasons_exit_formatted_json = json.dumps({
                    'text': reasons_for_exit,
                    'style': {'font_size': reasons_exit_font_size, 'font_color': reasons_exit_font_color,
                              'bold': reasons_exit_bold, 'italic': reasons_exit_italic, 'underline': reasons_exit_underline}
                })
                post_trade_analysis_formatted_json = json.dumps({
                    'text': post_trade_analysis,
                    'style': {'font_size': pta_font_size, 'font_color': pta_font_color,
                              'bold': pta_bold, 'italic': pta_italic, 'underline': pta_underline}
                })
                lessons_learned_formatted_json = json.dumps({
                    'text': lessons_learned,
                    'style': {'font_size': ll_font_size, 'font_color': ll_font_color,
                              'bold': ll_bold, 'italic': ll_italic, 'underline': ll_underline}
                })
                journal_notes_general_formatted_json = json.dumps({
                    'text': journal_notes_general,
                    'style': {'font_size': journal_notes_font_size, 'font_color': journal_notes_font_color,
                              'bold': journal_notes_bold, 'italic': journal_notes_italic, 'underline': journal_notes_underline}
                })

                # Create the new trade entry or update existing
                new_trade_data = {
                    "Trade ID": trade_id_input, # Should always be set
                    "Date": pd.to_datetime(trade_date),
                    "Entry Time": pd.to_datetime(f"{trade_date} {entry_time}"),
                    "Exit Time": pd.to_datetime(f"{trade_date} {exit_time}"),
                    "Symbol": symbol,
                    "Trade Type": trade_type,
                    "Strategy Used": selected_strategy if selected_strategy != "(Select One)" else "",
                    "Lots": lots,
                    "Entry Price": entry_price,
                    "Stop Loss Price": stop_loss_price,
                    "Take Profit Price": take_profit_price,
                    "Final Exit Price": final_exit_price,
                    "Initial R": initial_r_calc,
                    "Realized R": realized_r_calc,
                    "Win/Loss": win_loss_status,
                    "PnL ($)": pnL_dollars,
                    "Pips": trade_pips_gain,
                    "Weekly Bias": weekly_bias,
                    "Daily Bias": daily_bias,
                    "4H Structure": h4_structure,
                    "1H Structure": h1_structure,
                    "Primary Correlation": primary_corr,
                    "Secondary Correlation": secondary_corr,
                    "News Event Impact": ','.join(news_impact),
                    "Market State (HTF)": market_state,
                    "Setup Name": setup_name,
                    "Indicators Used": ','.join(indicators_used),
                    "Entry Trigger": entry_trigger,
                    "Reasons for Entry": reasons_entry_formatted_json,
                    "Reasons for Exit": reasons_exit_formatted_json,
                    "Pre-Trade Mindset": pre_trade_mindset,
                    "In-Trade Emotions": ','.join(in_trade_emotions),
                    "Emotional Triggers": emotional_triggers,
                    "Discipline Score 1-5": float(discipline_score),
                    "Order Type": order_type,
                    "Partial Exits": partial_exits,
                    "Entry Screenshot": entry_screenshot_storage, # Use the path from storage
                    "Exit Screenshot": exit_screenshot_storage,   # Use the path from storage
                    "Journal Notes": journal_notes_general_formatted_json,
                    "Post-Trade Analysis": post_trade_analysis_formatted_json,
                    "Lessons Learned": lessons_learned_formatted_json,
                    "Adjustments": adjustments,
                    "Tags": ','.join(tags)
                }

                new_trade_df_row = pd.DataFrame([new_trade_data])
                # Ensure correct dtypes for the new row for concatenation
                for col, dtype in journal_dtypes.items():
                    if col in new_trade_df_row.columns:
                        if dtype == str:
                            new_trade_df_row[col] = new_trade_df_row[col].fillna('').astype(str)
                        elif 'datetime' in str(dtype):
                            new_trade_df_row[col] = pd.to_datetime(new_trade_df_row[col], errors='coerce')
                        elif dtype == float:
                            new_trade_df_row[col] = pd.to_numeric(new_trade_df_row[col], errors='coerce').fillna(0.0).astype(float)
                        elif dtype == bool:
                            new_trade_df_row[col] = new_trade_df_row[col].fillna(False).astype(bool)


                if is_editing:
                    # Update existing row
                    trade_to_update_idx = st.session_state.tools_trade_journal[st.session_state.tools_trade_journal['Trade ID'] == trade_id_input].index
                    if not trade_to_update_idx.empty:
                        for col in journal_cols: # Update all columns according to new_trade_df_row
                            st.session_state.tools_trade_journal.loc[trade_to_update_idx, col] = new_trade_df_row[col].iloc[0]
                        st.success(f"Trade {trade_id_input} updated successfully!")
                    else:
                        st.error(f"Error: Trade with ID {trade_id_input} not found for update.")
                else:
                    # Append new trade to journal
                    st.session_state.tools_trade_journal = pd.concat(
                        [st.session_state.tools_trade_journal, new_trade_df_row],
                        ignore_index=True
                    ).astype(journal_dtypes, errors='ignore') # Re-assert all dtypes
                    st.success("New trade saved successfully!")
                
                # Save to database if user is logged in
                if 'logged_in_user' in st.session_state:
                    username = st.session_state.logged_in_user
                    if _ta_save_journal(username, st.session_state.tools_trade_journal):
                        ta_update_xp(10) # 10 XP per trade log
                        ta_update_streak()
                        logging.info(f"Trade {'updated' if is_editing else 'logged'} and saved to database for user {username} with ID {trade_id_input}")
                    else:
                        st.error("Failed to save trade to account. Saved locally only.")
                else:
                    st.warning("Trade saved locally (not synced to account, please log in to save).")
                    logging.info("Trade logged for anonymous user")
                
                # Clear pre-fill state if it was an edit, and rerun to show updated data
                if 'edit_trade_data' in st.session_state:
                    del st.session_state['edit_trade_data']
                st.rerun()

        st.subheader("Recent Trades Overview")
        # Define a simplified column config for the overview table, for better readability
        overview_column_config = {
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Trade ID": st.column_config.TextColumn("Trade ID", width="small"),
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Trade Type": st.column_config.TextColumn("Type"),
            "Lots": st.column_config.NumberColumn("Lots", format="%.2f"),
            "PnL ($)": st.column_config.NumberColumn("P&L ($)", format="$%.2f", help="Profit and Loss in account currency"),
            "Realized R": st.column_config.NumberColumn("Realized R", format="%.2f", help="Risk-Reward multiple realized"),
            "Win/Loss": st.column_config.TextColumn("Outcome"),
            "Strategy Used": st.column_config.TextColumn("Strategy"),
            "Tags": st.column_config.TextColumn("Tags", width="small", help="Categorization tags")
        }
        
        # Display the journal, possibly selecting a subset of columns for the initial view
        # Ensure only columns that exist in the dataframe and are desired for overview are shown.
        cols_to_display = [col for col in overview_column_config.keys() if col in st.session_state.tools_trade_journal.columns]
        
        st.dataframe(st.session_state.tools_trade_journal[cols_to_display],
                     column_config=overview_column_config,
                     hide_index=True,
                     use_container_width=True)
        
        # Export options
        st.subheader("Export Journal")
        col_export1, col_export2 = st.columns(2)
    
        with col_export1:
            csv = st.session_state.tools_trade_journal.to_csv(index=False)
            st.download_button("Download CSV", csv, "trade_journal.csv", "text/csv")
    
        with col_export2:
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF..."):
                    try:
                        # Reusing your existing latex generation - enhance to include more fields.
                        # This PDF generation is simplistic. For richer PDF, consider reportlab/fpdf or templating.
                        latex_content = """
                        \\documentclass{article}
                        \\usepackage{booktabs}
                        \\usepackage{geometry}
                        \\geometry{a4paper, margin=1in}
                        \\usepackage{pdflscape}
                        \\begin{document}
                        \\section*{Trade Journal Report}
                        \\begin{landscape}
                        \\begin{tabular}{llrrlll}
                        \\toprule
                        Date & Symbol & Entry Price & Exit Price & P&L ($) & Realized R & Tags \\\\
                        \\midrule
                        """
                        # Ensure these columns exist in the DataFrame for export
                        export_cols_for_pdf = ["Date", "Symbol", "Entry Price", "Final Exit Price", "PnL ($)", "Realized R", "Tags"]
                        temp_df = st.session_state.tools_trade_journal[[col for col in export_cols_for_pdf if col in st.session_state.tools_trade_journal.columns]].copy()
                        temp_df = temp_df.fillna({'Entry Price':0.0, 'Final Exit Price':0.0, 'PnL ($)':0.0, 'Realized R':0.0, 'Tags':''}).round({'Entry Price':5, 'Final Exit Price':5, 'PnL ($)':2, 'Realized R':2})
                        
                        for _, row in temp_df.iterrows():
                            date_str = row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else ''
                            # Sanitize LaTeX input: replace _ and & if they are in Tags/Symbol
                            sanitized_symbol = row['Symbol'].replace('_', '\\_')
                            sanitized_tags = row['Tags'].replace('_', '\\_').replace('&', '\\&')
                            latex_content += f"{date_str} & {sanitized_symbol} & {row['Entry Price']:.5f} & {row['Final Exit Price']:.5f} & {row['PnL ($)']:.2f} & {row['Realized R']:.2f} & {sanitized_tags} \\\\\n"
                    
                        latex_content += """
                        \\bottomrule
                        \\end{tabular}
                        \\end{landscape}
                        \\end{document}
                        """
                    
                        with open("trade_journal_report.tex", "w") as f:
                            f.write(latex_content)
                    
                        import subprocess
                        # Using shell=True for latexmk as it often relies on environment variables (like PATH)
                        # but in production, consider absolute paths to latexmk and pdflatex for security.
                        process = subprocess.run(["latexmk", "-pdf", "trade_journal_report.tex"], check=True, capture_output=True, text=True, shell=True) 
                        if process.returncode != 0:
                            st.error(f"LaTeX compilation failed: {process.stderr}")
                            logging.error(f"LaTeX compilation failed: {process.stderr}", exc_info=True)
                            raise RuntimeError("LaTeX compilation failed")

                        with open("trade_journal_report.pdf", "rb") as f:
                            st.download_button("Download PDF Report", f, "trade_journal_report.pdf", "application/pdf")
                    except FileNotFoundError:
                        st.error("LaTeX compilation tools not found. Please install a TeX distribution (e.g., MiKTeX on Windows, TeX Live on Linux/macOS) to generate PDFs.")
                        logging.error("LaTeX tools not found for PDF generation.")
                    except subprocess.CalledProcessError as e:
                        st.error(f"PDF generation failed: LaTeX command returned an error. See logs for details.")
                        logging.error(f"LaTeX compilation failed: {e.stderr}", exc_info=True)
                    except Exception as e:
                        st.error(f"PDF generation failed unexpectedly: {str(e)}")
                        logging.error(f"PDF generation error: {str(e)}", exc_info=True)


    # =========================================================
    # ANALYTICS TAB
    # =========================================================
    with tab_analytics:
        st.subheader("📊 Trade Analytics & Insights")
        st.markdown("Dive deeper into your performance with these comprehensive analytics.")
        
        if not st.session_state.tools_trade_journal.empty:
            df_analytics = st.session_state.tools_trade_journal.copy()
            df_analytics = df_analytics[df_analytics["Win/Loss"].isin(["Win", "Loss", "Breakeven", "Pending / Invalid Prices"])].copy() # Include more statuses if useful

            if df_analytics.empty:
                st.info("No completed trades to analyze. Log some trades first.")
            else:
                df_analytics["Date"] = pd.to_datetime(df_analytics["Date"])
                df_analytics['Strategy Used'] = df_analytics['Strategy Used'].replace('', 'N/A')
                
                # Filters
                st.markdown("---")
                st.markdown("#### Filter Your Analytics")
                col_filter_a, col_filter_b, col_filter_c, col_filter_d = st.columns(4)
                with col_filter_a:
                    analytics_symbol_filter = st.multiselect(
                        "Filter by Symbol",
                        options=df_analytics['Symbol'].unique(),
                        default=df_analytics['Symbol'].unique(), key="analytics_symbol_filter"
                    )
                with col_filter_b:
                    analytics_trade_type_filter = st.multiselect(
                        "Filter by Trade Type",
                        options=df_analytics['Trade Type'].unique(),
                        default=df_analytics['Trade Type'].unique(), key="analytics_trade_type_filter"
                    )
                with col_filter_c:
                    tag_options_analytics = sorted(list(set(df_analytics['Tags'].str.split(',').explode().dropna().astype(str).str.strip())))
                    analytics_tag_filter = st.multiselect("Filter by Tags", options=tag_options_analytics, key="analytics_tag_filter")
                with col_filter_d:
                    strategy_options_analytics = sorted(df_analytics['Strategy Used'].unique())
                    analytics_strategy_filter = st.multiselect("Filter by Strategy", options=strategy_options_analytics, default=strategy_options_analytics, key="analytics_strategy_filter")
                
                if analytics_symbol_filter:
                    df_analytics = df_analytics[df_analytics['Symbol'].isin(analytics_symbol_filter)]
                if analytics_trade_type_filter:
                    df_analytics = df_analytics[df_analytics['Trade Type'].isin(analytics_trade_type_filter)]
                if analytics_tag_filter:
                    df_analytics = df_analytics[df_analytics['Tags'].apply(lambda x: any(tag in x.split(',') for tag in analytics_tag_filter) if isinstance(x, str) and x else False)]
                if analytics_strategy_filter:
                    df_analytics = df_analytics[df_analytics['Strategy Used'].isin(analytics_strategy_filter)]

                if df_analytics.empty:
                    st.warning("No trades match the current filter criteria.")
                else:
                    # Metrics Section
                    st.markdown("---")
                    st.markdown("#### Key Performance Indicators")
                    total_trades_ana = len(df_analytics)
                    winning_trades_ana = df_analytics[df_analytics["Win/Loss"] == "Win"]
                    losing_trades_ana = df_analytics[df_analytics["Win/Loss"] == "Loss"]
                    # breakeven_trades_ana = df_analytics[df_analytics["Win/Loss"] == "Breakeven"] # Not used here, keep for consistency
                    
                    # Ensure no division by zero for these
                    net_profit_ana = df_analytics["PnL ($)"].sum()
                    gross_profit_ana = winning_trades_ana["PnL ($)"].sum()
                    gross_loss_ana = losing_trades_ana["PnL ($)"].sum() # already negative

                    win_rate_ana = (len(winning_trades_ana) / total_trades_ana * 100) if total_trades_ana > 0 else 0
                    
                    avg_win_ana = winning_trades_ana["PnL ($)"].mean() if not winning_trades_ana.empty else 0.0
                    avg_loss_ana = losing_trades_ana["PnL ($)"].mean() if not losing_trades_ana.empty else 0.0

                    expectancy_ana = ((len(winning_trades_ana) / total_trades_ana) * avg_win_ana) + ((len(losing_trades_ana) / total_trades_ana) * avg_loss_ana) if total_trades_ana > 0 else 0.0
                    
                    profit_factor_val = gross_profit_ana / abs(gross_loss_ana) if gross_loss_ana != 0 else (np.inf if gross_profit_ana > 0 else 0.0)

                    col_metrics_ana1, col_metrics_ana2, col_metrics_ana3, col_metrics_ana4, col_metrics_ana5 = st.columns(5)
                    col_metrics_ana1.metric("Net P&L", f"${net_profit_ana:,.2f}")
                    col_metrics_ana2.metric("Total Trades", total_trades_ana)
                    col_metrics_ana3.metric("Win Rate", f"{win_rate_ana:,.2f}%")
                    col_metrics_ana4.metric("Avg Win", f"${avg_win_ana:,.2f}")
                    col_metrics_ana5.metric("Avg Loss", f"${abs(avg_loss_ana):,.2f}") # Display as positive amount

                    col_metrics_ana6, col_metrics_ana7, col_metrics_ana8, col_metrics_ana9 = st.columns(4)
                    col_metrics_ana6.metric("Profit Factor", f"{profit_factor_val:,.2f}" if profit_factor_val != np.inf else "∞")
                    col_metrics_ana7.metric("Expectancy", f"${expectancy_ana:,.2f} per trade")
                    # Calculate longest streak dynamically from 'Win/Loss' column
                    longest_win_streak = 0
                    current_win_streak = 0
                    longest_loss_streak = 0
                    current_loss_streak = 0

                    for outcome in df_analytics['Win/Loss']:
                        if outcome == "Win":
                            current_win_streak += 1
                            longest_win_streak = max(longest_win_streak, current_win_streak)
                            current_loss_streak = 0 # Reset loss streak
                        elif outcome == "Loss":
                            current_loss_streak += 1
                            longest_loss_streak = max(longest_loss_streak, current_loss_streak)
                            current_win_streak = 0 # Reset win streak
                        else: # Breakeven or other statuses reset both streaks
                            current_win_streak = 0
                            current_loss_streak = 0

                    col_metrics_ana8.metric("Longest Win Streak", longest_win_streak)
                    col_metrics_ana9.metric("Longest Loss Streak", longest_loss_streak)
                    
                    st.markdown("---")
                    st.markdown("#### Performance Visualizations")

                    # Equity Curve
                    st.subheader("Equity Curve")
                    df_analytics['Cumulative PnL'] = df_analytics["PnL ($)"].cumsum()
                    fig_equity = px.line(df_analytics, x=df_analytics.index, y="Cumulative PnL",
                                         title="Equity Curve", labels={"index": "Trade Number"},
                                         color_discrete_sequence=['#58b3b1'])
                    fig_equity.update_layout(hovermode="x unified", template="plotly_dark",
                                            xaxis_title="Trade Number", yaxis_title="Cumulative P&L ($)")
                    st.plotly_chart(fig_equity, use_container_width=True)

                    # Performance by Symbol
                    st.subheader("Performance by Symbol")
                    df_sym_perf = df_analytics.groupby('Symbol').agg(
                        Total_PnL=('PnL ($)', 'sum'),
                        Trades=('Trade ID', 'count'),
                        Wins=('Win/Loss', lambda x: (x == "Win").sum()),
                        Losses=('Win/Loss', lambda x: (x == "Loss").sum())
                    ).reset_index()
                    df_sym_perf['Win Rate (%)'] = (df_sym_perf['Wins'] / df_sym_perf['Trades']) * 100 if df_sym_perf['Trades'].sum() > 0 else 0.0
                    df_sym_perf.sort_values("Total_PnL", ascending=False, inplace=True)

                    fig_symbol = px.bar(df_sym_perf, x='Symbol', y='Total_PnL', color='Win Rate (%)',
                                        title='Total P&L by Symbol (Colored by Win Rate)',
                                        labels={'Total_PnL': 'Total P&L ($)', 'Win Rate (%)': 'Win Rate (%)'},
                                        color_continuous_scale=px.colors.sequential.Viridis)
                    fig_symbol.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_symbol, use_container_width=True)

                    # Performance by Strategy
                    st.subheader("Performance by Strategy")
                    df_strat_perf = df_analytics.groupby('Strategy Used').agg(
                        Total_PnL=('PnL ($)', 'sum'),
                        Trades=('Trade ID', 'count'),
                        Wins=('Win/Loss', lambda x: (x == "Win").sum()),
                        Losses=('Win/Loss', lambda x: (x == "Loss").sum())
                    ).reset_index()
                    df_strat_perf['Win Rate (%)'] = (df_strat_perf['Wins'] / df_strat_perf['Trades']) * 100 if df_strat_perf['Trades'].sum() > 0 else 0.0
                    df_strat_perf.sort_values("Total_PnL", ascending=False, inplace=True)
                    fig_strategy = px.bar(df_strat_perf, x='Strategy Used', y='Total_PnL', color='Win Rate (%)',
                                        title='Total P&L by Strategy Used (Colored by Win Rate)',
                                        labels={'Total_PnL': 'Total P&L ($)', 'Win Rate (%)': 'Win Rate (%)'},
                                        color_continuous_scale=px.colors.sequential.Plasma)
                    fig_strategy.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_strategy, use_container_width=True)


                    # R-Multiples Distribution
                    st.subheader("Realized R-Multiples Distribution")
                    df_analytics_r = df_analytics[df_analytics['Realized R'].notna() & (df_analytics['Realized R'] != 0.0)].copy() # Filter out zero R-Multiples
                    if not df_analytics_r.empty:
                        fig_r_dist = px.histogram(df_analytics_r, x="Realized R", nbins=20,
                                                  title="Distribution of Realized R-Multiples",
                                                  labels={'Realized R': 'Realized R-Multiple'},
                                                  color_discrete_sequence=['#4d7171'])
                        fig_r_dist.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_r_dist, use_container_width=True)
                    else:
                        st.info("No trades with valid Realized R-multiples to display.")

                    # Emotional Analysis
                    st.subheader("Emotional Impact on Performance")
                    # First, ensure 'In-Trade Emotions' column exists and contains non-empty values
                    if "In-Trade Emotions" in df_analytics.columns and not df_analytics["In-Trade Emotions"].str.strip().eq("").all():
                        df_emo_perf = df_analytics[df_analytics["In-Trade Emotions"] != ""].copy()
                        df_emo_perf["Emotion"] = df_emo_perf["In-Trade Emotions"].str.split(',').explode().str.strip()
                        # Drop any empty string emotions from the explode operation
                        df_emo_perf = df_emo_perf[df_emo_perf["Emotion"] != ""].copy()
                        
                        if not df_emo_perf.empty:
                            df_emo_grouped = df_emo_perf.groupby("Emotion").agg(
                                Avg_PnL=('PnL ($)', 'mean'),
                                Trades=('Trade ID', 'count'),
                                Wins=('Win/Loss', lambda x: (x == "Win").sum()),
                                Losses=('Win/Loss', lambda x: (x == "Loss").sum())
                            ).reset_index()
                            df_emo_grouped['Win Rate (%)'] = (df_emo_grouped['Wins'] / df_emo_grouped['Trades']) * 100
                            df_emo_grouped.sort_values("Avg_PnL", ascending=False, inplace=True)
                            
                            fig_emotion = px.bar(df_emo_grouped, x='Emotion', y='Avg_PnL', color='Win Rate (%)',
                                                title='Average P&L by In-Trade Emotion',
                                                labels={'Avg_PnL': 'Average P&L ($)', 'Win Rate (%)': 'Win Rate (%)'},
                                                color_continuous_scale=px.colors.sequential.Magenta)
                            fig_emotion.update_layout(template="plotly_dark")
                            st.plotly_chart(fig_emotion, use_container_width=True)
                        else:
                            st.info("No valid emotional data found after filtering. Log more specific emotions.")
                    else:
                        st.info("No emotional data logged to analyze. Fill 'In-Trade Emotions' in the journal.")


                    # Discipline Score correlation
                    st.subheader("Discipline Score vs. P&L")
                    df_discipline = df_analytics[df_analytics['Discipline Score 1-5'].notna() & (df_analytics['Discipline Score 1-5'] > 0)].copy()
                    if not df_discipline.empty:
                        fig_discipline = px.scatter(df_discipline, x="Discipline Score 1-5", y="PnL ($)",
                                                    trendline="ols", title="P&L vs. Discipline Score",
                                                    labels={'Discipline Score 1-5': 'Discipline Score', 'PnL ($)': 'Profit/Loss ($)'},
                                                    color_discrete_sequence=['#58b3b1'])
                        fig_discipline.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_discipline, use_container_width=True)
                    else:
                        st.info("No discipline score data available to plot. Fill 'Discipline Score' in the journal.")

        else:
            st.info("No trades logged yet. Add trades in the 'Log Trade' tab to view analytics.")
            
    # =========================================================
    # TRADE HISTORY (REVIEW/REPLAY) TAB
    # =========================================================
    with tab_history:
        st.subheader("📜 Detailed Trade History & Review")
        st.markdown("Select a trade to review its detailed parameters, performance, and your comprehensive notes.")
    
        if not st.session_state.tools_trade_journal.empty:
            df_history = st.session_state.tools_trade_journal.sort_values(by="Date", ascending=False).reset_index(drop=True)
            
            # Create a more descriptive option string for the selectbox
            trade_to_review_options = [
                f"{row['Date'].strftime('%Y-%m-%d %H:%M')} - {row['Symbol']} ({row['Trade ID']}) ({row['Win/Loss']}: ${row['PnL ($)']:.2f})"
                for _, row in df_history.iterrows()
            ]
            
            selected_trade_display = st.selectbox(
                "Select a Trade to Review/Edit",
                options=trade_to_review_options, 
                key="select_trade_to_review",
                help="Choose a logged trade from your journal to view or edit its details."
            )

            # Defensive fix for IndexError:
            # Extract Trade ID from the display string
            try:
                # Assuming format "DATE - SYMBOL (TRADE_ID) (WIN/LOSS: $PNL)"
                selected_trade_id_parts = selected_trade_display.split('(')
                # The Trade ID is the part right before the last closing parenthesis (e.g., "(TRADE_ID) (WIN/LOSS...")
                # So we take the second-to-last part and strip its opening parenthesis and trailing whitespace
                selected_trade_id = selected_trade_id_parts[-2].strip().strip(')')

            except (IndexError, AttributeError):
                # Fallback if the format is not as expected or the string is malformed.
                # In such cases, it's safer to consider no trade selected or warn the user.
                st.error("Could not parse Trade ID from the selected entry. Please select another trade.")
                selected_trade_id = None

            selected_trade_row = None
            if selected_trade_id:
                matching_rows = df_history[df_history['Trade ID'] == selected_trade_id]
                if not matching_rows.empty:
                    selected_trade_row = matching_rows.iloc[0]
                    trade_idx_in_df = matching_rows.index[0] # Original index for update/delete
                else:
                    st.warning(f"Trade with ID `{selected_trade_id}` not found in journal. It might have been deleted.")

            
            if selected_trade_row is not None:
                st.markdown(f"### Reviewing Trade: `{selected_trade_row['Trade ID']}`")
                st.markdown("---")

                # --- Section: Summary Overview ---
                with st.expander("📊 Trade Summary", expanded=True):
                    cols_summary_r1, cols_summary_r2, cols_summary_r3 = st.columns(3)
                    cols_summary_r1.metric("Date", selected_trade_row['Date'].strftime('%Y-%m-%d'))
                    cols_summary_r2.metric("Symbol", selected_trade_row['Symbol'])
                    cols_summary_r3.metric("Strategy Used", selected_trade_row['Strategy Used'] if selected_trade_row['Strategy Used'] else "N/A")

                    cols_summary_r4, cols_summary_r5, cols_summary_r6 = st.columns(3)
                    cols_summary_r4.metric("Trade Type", selected_trade_row['Trade Type'])
                    cols_summary_r5.metric("Outcome", selected_trade_row['Win/Loss'])
                    # Provide delta only if PnL is non-zero
                    delta_pnl = f"{selected_trade_row['Pips']:.2f} pips" if selected_trade_row['Pips'] != 0 else None
                    delta_color_pnl = "normal" if selected_trade_row['PnL ($)'] != 0 else "off"

                    cols_summary_r6.metric("P&L ($)", f"${selected_trade_row['PnL ($)']:.2f}",
                                            delta=delta_pnl,
                                            delta_color=delta_color_pnl)
                    
                    cols_summary_r7, cols_summary_r8 = st.columns(2)
                    cols_summary_r7.metric("Initial R", f"{selected_trade_row['Initial R']:.2f}")
                    cols_summary_r8.metric("Realized R", f"{selected_trade_row['Realized R']:.2f}")


                # --- Section: Detailed Execution ---
                with st.expander("📈 Execution Details"):
                    cols_exec_time, cols_pricing_ex, cols_lots_type = st.columns(3)
                    cols_exec_time.write(f"**Entry Time:** {selected_trade_row['Entry Time'].strftime('%H:%M:%S') if pd.notna(selected_trade_row['Entry Time']) else 'N/A'}")
                    cols_exec_time.write(f"**Exit Time:** {selected_trade_row['Exit Time'].strftime('%H:%M:%S') if pd.notna(selected_trade_row['Exit Time']) else 'N/A'}")

                    cols_pricing_ex.write(f"**Entry Price:** {selected_trade_row['Entry Price']:.5f}")
                    cols_pricing_ex.write(f"**Stop Loss Price:** {selected_trade_row['Stop Loss Price']:.5f}")
                    cols_pricing_ex.write(f"**Take Profit Price:** {selected_trade_row['Take Profit Price']:.5f}")
                    cols_pricing_ex.write(f"**Final Exit Price:** {selected_trade_row['Final Exit Price']:.5f}")
                    
                    cols_lots_type.write(f"**Lots:** {selected_trade_row['Lots']:.2f}")
                    cols_lots_type.write(f"**Order Type:** {selected_trade_row['Order Type']}")
                    cols_lots_type.write(f"**Partial Exits:** {'Yes' if selected_trade_row['Partial Exits'] else 'No'}")


                # --- Section: Market & Bias Analysis ---
                with st.expander("🌍 Market Context & Bias"):
                    cols_mb_1, cols_mb_2, cols_mb_3 = st.columns(3)
                    with cols_mb_1:
                        st.write(f"**Weekly Bias:** {selected_trade_row['Weekly Bias']}")
                        st.write(f"**Daily Bias:** {selected_trade_row['Daily Bias']}")
                    with cols_mb_2:
                        st.write(f"**4H Structure:** {selected_trade_row['4H Structure']}")
                        st.write(f"**1H Structure:** {selected_trade_row['1H Structure']}")
                    with cols_mb_3:
                        st.write(f"**Market State (HTF):** {selected_trade_row['Market State (HTF)']}")
                        st.write(f"**Primary Correlation:** {selected_trade_row['Primary Correlation']}")
                        st.write(f"**Secondary Correlation:** {selected_trade_row['Secondary Correlation']}")
                        st.write(f"**News Event Impact:** {selected_trade_row['News Event Impact'].replace(',', ', ')}")

                # --- Section: Setup & Entry Details ---
                with st.expander("🎯 Setup & Entry Plan"):
                    st.write(f"**Setup Name:** {selected_trade_row['Setup Name']}")
                    st.write(f"**Indicators Used:** {selected_trade_row['Indicators Used'].replace(',', ', ')}")
                    st.write(f"**Entry Trigger:** {selected_trade_row['Entry Trigger']}")
                    
                    st.markdown("##### Reasons for Entry:")
                    try:
                        entry_reason_data = json.loads(selected_trade_row["Reasons for Entry"])
                        st.markdown(apply_text_styles(
                            entry_reason_data['text'], **entry_reason_data['style']),
                            unsafe_allow_html=True
                        )
                    except (json.JSONDecodeError, KeyError): # Fallback for old/malformed data
                        st.markdown(f'<div class="styled-text-area-display" style="font-size: 14px; color: #FFFFFF;">{selected_trade_row["Reasons for Entry"]}</div>', unsafe_allow_html=True)


                # --- Section: Exit Details ---
                with st.expander("🚪 Exit Details"):
                    st.markdown("##### Reasons for Exit:")
                    try:
                        exit_reason_data = json.loads(selected_trade_row["Reasons for Exit"])
                        st.markdown(apply_text_styles(
                            exit_reason_data['text'], **exit_reason_data['style']),
                            unsafe_allow_html=True
                        )
                    except (json.JSONDecodeError, KeyError): # Fallback for old/malformed data
                        st.markdown(f'<div class="styled-text-area-display" style="font-size: 14px; color: #FFFFFF;">{selected_trade_row["Reasons for Exit"]}</div>', unsafe_allow_html=True)

                    entry_screenshot_link = selected_trade_row['Entry Screenshot'] if pd.notna(selected_trade_row['Entry Screenshot']) else ""
                    exit_screenshot_link = selected_trade_row['Exit Screenshot'] if pd.notna(selected_trade_row['Exit Screenshot']) else ""
                    
                    if entry_screenshot_link or exit_screenshot_link:
                        st.markdown("---")
                        st.markdown("#### Screenshots")
                        col_screens_1, col_screens_2 = st.columns(2)
                        if entry_screenshot_link:
                            with col_screens_1:
                                if os.path.exists(entry_screenshot_link):
                                    st.image(entry_screenshot_link, caption="Entry Screenshot", use_column_width=True)
                                else:
                                    st.info(f"Entry screenshot referenced, but file not found: `{os.path.basename(entry_screenshot_link)}`")
                        if exit_screenshot_link:
                            with col_screens_2:
                                if os.path.exists(exit_screenshot_link):
                                    st.image(exit_screenshot_link, caption="Exit Screenshot", use_column_width=True)
                                else:
                                    st.info(f"Exit screenshot referenced, but file not found: `{os.path.basename(exit_screenshot_link)}`")
                    else:
                        st.info("No screenshots linked to this trade.")
                
                # --- Section: Psychological Factors ---
                with st.expander("🧠 Psychological Factors"):
                    st.write(f"**Pre-Trade Mindset / Plan:** {selected_trade_row['Pre-Trade Mindset']}")
                    st.write(f"**In-Trade Emotions:** {selected_trade_row['In-Trade Emotions'].replace(',', ', ')}")
                    st.write(f"**Emotional Triggers:** {selected_trade_row['Emotional Triggers']}")
                    st.write(f"**Discipline Score:** {selected_trade_row['Discipline Score 1-5']:.0f}/5")

                # --- Section: Reflections & Learning ---
                with st.expander("💡 Reflections & Learning"):
                    st.markdown("##### Post-Trade Analysis:")
                    try:
                        pta_data = json.loads(selected_trade_row["Post-Trade Analysis"])
                        st.markdown(apply_text_styles(
                            pta_data['text'], **pta_data['style']),
                            unsafe_allow_html=True
                        )
                    except (json.JSONDecodeError, KeyError): # Fallback for old/malformed data
                         st.markdown(f'<div class="styled-text-area-display" style="font-size: 14px; color: #FFFFFF;">{selected_trade_row["Post-Trade Analysis"]}</div>', unsafe_allow_html=True)

                    st.markdown("##### Lessons Learned:")
                    try:
                        ll_data = json.loads(selected_trade_row["Lessons Learned"])
                        st.markdown(apply_text_styles(
                            ll_data['text'], **ll_data['style']),
                            unsafe_allow_html=True
                        )
                    except (json.JSONDecodeError, KeyError): # Fallback for old/malformed data
                        st.markdown(f'<div class="styled-text-area-display" style="font-size: 14px; color: #FFFFFF;">{selected_trade_row["Lessons Learned"]}</div>', unsafe_allow_html=True)

                    st.write(f"**Adjustments for Future Trades:** {selected_trade_row['Adjustments']}")
                
                # --- Section: General Journal Notes & Tags ---
                with st.expander("📚 General Notes & Tags"):
                    st.markdown("##### General Journal Notes:")
                    try:
                        notes_data = json.loads(selected_trade_row["Journal Notes"])
                        st.markdown(apply_text_styles(
                            notes_data['text'], **notes_data['style']),
                            unsafe_allow_html=True
                        )
                    except (json.JSONDecodeError, KeyError): # Fallback for old/malformed data
                        st.markdown(f'<div class="styled-text-area-display" style="font-size: 14px; color: #FFFFFF;">{selected_trade_row["Journal Notes"]}</div>', unsafe_allow_html=True)

                    st.markdown(f"**Tags:** {selected_trade_row['Tags'].replace(',', ', ')}")

                st.markdown("---")
                col_review_buttons_final = st.columns(2)
                with col_review_buttons_final[0]:
                    if st.button("Edit This Trade", key=f"edit_trade_history_{selected_trade_row['Trade ID']}", type="primary"):
                        # Prepare data for pre-filling, convert datetime to Python objects expected by date_input/time_input
                        trade_data_to_edit = selected_trade_row.to_dict()
                        if pd.notna(trade_data_to_edit.get("Date")): trade_data_to_edit["Date"] = pd.to_datetime(trade_data_to_edit["Date"]).to_pydatetime().date()
                        if pd.notna(trade_data_to_edit.get("Entry Time")): trade_data_to_edit["Entry Time"] = pd.to_datetime(trade_data_to_edit["Entry Time"]).to_pydatetime().time()
                        if pd.notna(trade_data_to_edit.get("Exit Time")): trade_data_to_edit["Exit Time"] = pd.to_datetime(trade_data_to_edit["Exit Time"]).to_pydatetime().time()
                        
                        st.session_state.edit_trade_data = trade_data_to_edit

                        # Redirect and notify
                        st.session_state.current_page = 'backtesting'
                        # A rerunning causes a reload, making it visually seamless to prefill the form
                        st.info(f"Form pre-filled for Trade ID `{selected_trade_row['Trade ID']}`. Please go to the **Log Trade** tab to modify.")
                        st.rerun() 

                with col_review_buttons_final[1]:
                    if st.button("Delete This Trade", key=f"delete_trade_history_{selected_trade_row['Trade ID']}"):
                        st.session_state.tools_trade_journal = st.session_state.tools_trade_journal.drop(index=trade_idx_in_df).reset_index(drop=True)
                        if 'logged_in_user' in st.session_state:
                            _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal)
                        st.success("Trade deleted successfully!")
                        st.rerun()
            # This 'else' clause handles the case where selected_trade_row is None after all checks.
            else:
                st.info("Select a trade from the dropdown above to view its details.")

        else:
            st.info("No trades logged yet. Add trades in the 'Log Trade' tab.")
                
    # Challenge Mode (remains unchanged)
    st.markdown("---")
    st.subheader("🏅 Challenge Mode")
    st.write("30-Day Journaling Discipline Challenge - Gain 300 XP for completing, XP can be exchanged for gift cards!")
    streak = st.session_state.get('streak', 0)
    progress = min(streak / 30.0, 1.0)
    st.progress(progress)
    if progress >= 1.0 and st.session_state.get('challenge_30day_completed', False) == False:
        st.success("Challenge completed! Great job on your consistency.")
        if 'logged_in_user' in st.session_state:
            ta_update_xp(300) # Bonus XP for completion
        st.session_state.challenge_30day_completed = True # Ensure XP is only given once
    elif progress >= 1.0:
        st.info("You've already completed this challenge!")

    # Leaderboard / Self-Competition (remains largely unchanged)
    st.subheader("🏆 Leaderboard - Consistency")
    users_from_db = c.execute("SELECT username, data FROM users").fetchall()
    leader_data = []
    for u, d in users_from_db:
        user_d = json.loads(d) if d else {}
        journal_entries = user_d.get("tools_trade_journal", [])
        if isinstance(journal_entries, list): # Check if it's a list of dicts (actual rows)
            trade_count = sum(1 for entry in journal_entries if entry.get("Win/Loss") in ["Win", "Loss", "Breakeven", "Pending / Invalid Prices"])
        else: # Fallback for any unexpected/older formats
            trade_count = 0
            logging.warning(f"Journal data for user {u} is not in expected list format. Entries will not be counted.")


        leader_data.append({"Username": u, "Journaled Trades": trade_count})
    if leader_data:
        leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
        leader_df["Rank"] = leader_df.index + 1
        st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]], hide_index=True)
    else:
        st.info("No leaderboard data yet. Log some trades!")
# =========================================================
# PERFORMANCE DASHBOARD PAGE (MT5)
# =========================================================
elif st.session_state.current_page == 'mt5':
    st.title("📊 Performance Dashboard")
    st.caption("Analyze your MT5 trading history with advanced metrics and visualizations.")
    st.markdown('---')

    # Custom CSS for theme consistency and new progress bars, with fixed box sizes, and CALENDAR styles
    st.markdown(
        """
        <style>
        /* General Metric Box Styling */
        .metric-box {
            background-color: #2d4646;
            padding: 10px 15px; /* Reduced padding slightly */
            border-radius: 8px;
            text-align: center;
            border: 1px solid #58b3b1;
            color: #ffffff;
            transition: all 0.3s ease-in-out;
            margin: 5px 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between; /* Distribute space between title, value, and bar */
            height: 100px; /* Fixed height for all metric boxes */
            min-width: 150px; /* Ensure a minimum width */
            box-sizing: border-box; /* Include padding and border in the height */
            font-size: 0.9em; /* Overall text size adjustment for uniformity */
        }
        .metric-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(88, 179, 177, 0.3);
        }
        .metric-box strong { /* Style for metric titles */
            font-size: 1em; /* Keep title size consistent */
            margin-bottom: 3px;
            display: block;
        }
        .metric-box .metric-value { /* Style for the main metric value */
            font-size: 1.2em; /* Slightly larger for the main number */
            font-weight: bold;
            display: block;
            line-height: 1.3; /* Ensure spacing for single-line values */
            flex-grow: 1; /* Allows value to take up more space in non-bar boxes */
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .metric-box .sub-value { /* For values like the parenthetical total loss */
            font-size: 0.8em;
            color: #ccc;
            line-height: 1; /* Tighter line height for sub-values */
            padding-bottom: 5px; /* Ensure space from the bottom */
        }
        .metric-box .day-info { /* For best/worst performing day text, which can be longer */
            font-size: 0.85em; /* Adjusted for longer text to fit */
            line-height: 1.2;
            flex-grow: 1; /* Allow this text to take up available space */
            display: flex;
            align-items: center; /* Vertically center the text if shorter */
            justify-content: center; /* Horizontally center the text */
            padding-top: 5px; /* Little padding above the info text */
            padding-bottom: 5px; /* Little padding below the info text */
        }


        /* Tab Styling */
        .stTabs [data-baseweb="tab"] {
            color: #ffffff !important;
            background-color: #2d4646 !important;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            margin-right: 5px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #58b3b1 !important;
            color: #ffffff !important;
            font-weight: 600;
            border-bottom: 2px solid #4d7171 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #4d7171 !important;
            color: #ffffff !important;
        }

        /* Progress Bar Styling */
        .progress-container {
            width: 100%;
            background-color: #333; /* Dark background for the bar container */
            border-radius: 5px;
            overflow: hidden;
            height: 8px; /* Slightly reduced height for the bar */
            margin-top: auto; /* Pushes the bar to the bottom */
            flex-shrink: 0; /* Prevents bar from shrinking */
        }

        .progress-bar {
            height: 100%;
            border-radius: 5px;
            text-align: right;
            line-height: 8px;
            color: white;
            font-size: 7px; /* Smaller text within the bar if any */
            box-sizing: border-box;
            white-space: nowrap;
        }
        .progress-bar.green { background-color: #5cb85c; } /* Green for positive */
        .progress-bar.red { background-color: #d9534f; }   /* Red for negative */
        .progress-bar.neutral { background-color: #5bc0de; } /* Blue for neutral, if needed */

        /* Specific styles for the combined win/loss bar */
        .win-loss-bar-container {
            display: flex;
            width: 100%;
            background-color: #d9534f; /* Default to red for full bar if no wins */
            border-radius: 5px;
            overflow: hidden;
            height: 8px;
            margin-top: auto; /* Pushes the bar to the bottom */
            flex-shrink: 0;
        }
        .win-bar {
            height: 100%;
            background-color: #5cb85c; /* Green for wins */
            border-radius: 5px 0 0 5px; /* Rounded on left */
            flex-shrink: 0;
        }
        .loss-bar {
            height: 100%;
            background-color: #d9534f; /* Red for losses */
            border-radius: 0 5px 5px 0; /* Rounded on right */
            flex-shrink: 0;
        }

        /* Trading Score Bar */
        .trading-score-bar-container {
            width: 100%;
            background-color: #d9534f; /* Red background for the whole bar representing max possible score */
            border-radius: 5px;
            overflow: hidden;
            height: 8px;
            margin-top: auto; /* Pushes the bar to the bottom */
            flex-shrink: 0;
            position: relative;
        }
        .trading-score-bar {
            height: 100%;
            background-color: #5cb85c; /* Green part for the actual score */
            border-radius: 5px;
        }

        /* CALENDAR STYLES */
        .calendar-container {
            background-color: #262730; /* Darker background for the calendar itself */
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            color: #ffffff;
            font-family: Arial, sans-serif;
            border: 1px solid #3d3d4b;
        }
        .calendar-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            font-size: 1.4em;
            font-weight: bold;
        }
        /* Style for Streamlit's selectbox within the calendar, if needed */
        div[data-baseweb="select"] {
            margin: 0; /* Remove default margin from Streamlit elements */
        }
        .calendar-nav {
            display: flex;
            justify-content: center; /* Center the selectbox and any nav arrows */
            align-items: center;
        }
        .calendar-nav .nav-arrow {
            background: none;
            border: none;
            color: #58b3b1;
            font-size: 1.8em;
            cursor: pointer;
            padding: 0 5px;
            margin: 0 5px;
            text-decoration: none; /* Remove underline for anchor tags if used */
        }
        .calendar-weekdays {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 5px;
            margin-bottom: 10px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .calendar-weekday-item {
            text-align: center;
            padding: 5px;
            color: #ccc;
        }
        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 5px;
        }
        .calendar-day-box {
            background-color: #2d2e37; /* Default day box background */
            padding: 8px;
            border-radius: 6px;
            height: 70px; /* Fixed height for day boxes */
            display: flex;
            flex-direction: column;
            justify-content: flex-start; /* Align content to the top */
            position: relative;
            border: 1px solid #3d3d4b; /* Subtle border for all cells */
            overflow: hidden; /* Hide overflowing text */
            box-sizing: border-box; /* Include padding and border in the height */
        }
        .calendar-day-box.empty-month-day { /* For days that are not in the current month */
            background-color: #2d2e37; /* Match no-trade day color */
            border: 1px solid #3d3d4b;
            visibility: hidden; /* Hide empty boxes for preceding/succeeding month days */
        }
        .calendar-day-box .day-number {
            font-size: 0.8em;
            color: #bbbbbb;
            text-align: left;
            margin-bottom: 5px;
            line-height: 1; /* Tighter line height for day number */
        }
        .calendar-day-box .profit-amount {
            font-size: 0.9em;
            font-weight: bold;
            text-align: center;
            line-height: 1.1; /* Adjust line height for profit amount */
            flex-grow: 1; /* Allow profit amount to take up space */
            display: flex;
            align-items: center; /* Vertically center the amount */
            justify-content: center; /* Horizontally center */
            white-space: nowrap; /* Prevent profit amount from wrapping */
            text-overflow: ellipsis; /* Add ellipsis for overflowing text */
            overflow: hidden; /* Hide overflow */
        }
        .calendar-day-box.profitable {
            background-color: #0f2b0f; /* Green for profit */
            border-color: #5cb85c; /* Green border */
        }
        .calendar-day-box.losing {
            background-color: #2b0f0f; /* Red for loss */
            border-color: #d9534f; /* Red border */
        }
        .calendar-day-box.current-day {
            border: 2px solid #ff7f50; /* Orange border for current day */
        }
        .calendar-day-box .dot-indicator {
            position: absolute;
            top: 5px;
            right: 5px;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #ffcc00; /* Yellow dot for other indicators */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --------------------------
    # Helper functions (MT5 page specific)
    # --------------------------
    def _ta_human_pct_mt5(value):
        try:
            if value is None or pd.isna(value):
                return "N/A"
            return f"{float(value) * 100:.2f}%"
        except Exception:
            return "N/A"

    def _ta_human_num_mt5(value):
        """
        Formats a numerical value to a comma-separated string with two decimal places.
        Returns 'N/A' for None, NaN, or non-numeric input.
        """
        try:
            if value is None or pd.isna(value):
                return "N/A"
            float_val = float(value)
            return f"{float_val:,.2f}"
        except (ValueError, TypeError):
            return "N/A"

    def _ta_compute_sharpe(df, risk_free_rate=0.02):
        if "Profit" not in df.columns or df.empty:
            return np.nan
        
        # Ensure 'Close Time' is datetime and then set as DatetimeIndex for resampling
        df_for_sharpe = df.copy()
        df_for_sharpe["Close Time"] = pd.to_datetime(df_for_sharpe["Close Time"], errors='coerce')
        df_for_sharpe = df_for_sharpe.dropna(subset=["Close Time"]) # Remove NaT
        
        if df_for_sharpe.empty:
            return np.nan

        daily_pnl_series = df_for_sharpe.set_index("Close Time")["Profit"].resample('D').sum().fillna(0.0)

        if daily_pnl_series.empty or len(daily_pnl_series) < 2:
            return np.nan

        # Calculate returns, then drop NaNs from returns (e.g. first value will be NaN)
        returns = daily_pnl_series.pct_change().dropna()
        
        # Ensure we have enough actual return data points
        if len(returns) < 2:
            return np.nan
        
        mean_return = returns.mean() * 252  # Annualized (assuming trading 252 days/year)
        std_return = returns.std() * np.sqrt(252)  # Annualized
        return (mean_return - risk_free_rate) / std_return if std_return != 0 else np.nan

    def _ta_daily_pnl_mt5(df):
        """
        Returns a dictionary mapping datetime.date to total profit for that day.
        Includes all days that had at least one trade in the CSV, even if net profit is zero.
        """
        if "Close Time" in df.columns and "Profit" in df.columns and not df.empty and not df["Profit"].isnull().all():
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["Close Time"]).dt.date
            # Group by date and sum profits; this naturally creates entries for all trade days.
            return df_copy.groupby("date")["Profit"].sum().to_dict()
        return {}

    def _ta_profit_factor_mt5(df):
        wins_sum = df[df["Profit"] > 0]["Profit"].sum()
        losses_sum = abs(df[df["Profit"] < 0]["Profit"].sum()) # Ensure losses sum is positive for the ratio
        return wins_sum / losses_sum if losses_sum != 0.0 else np.nan

    def _ta_show_badges_mt5(df):
        st.subheader("🎖️ Your Trading Badges")
        
        total_profit_val = df["Profit"].sum()
        total_trades_val = len(df)
        
        col_badge1, col_badge2, col_badge3 = st.columns(3)
        with col_badge1:
            if total_profit_val > 10000:
                st.markdown("<div class='metric-box profitable'><strong>🎖️ Profit Pioneer</strong><br><span class='metric-value'>Achieved over $10,000 profit!</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-box'><strong>🎖️ Profit Pioneer</strong><br><span class='metric-value'>Goal: $10,000 profit</span></div>", unsafe_allow_html=True)

        with col_badge2:
            if total_trades_val >= 30: 
                st.markdown("<div class='metric-box profitable'><strong>🎖️ Active Trader</strong><br><span class='metric-value'>Completed over 30 trades!</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-box'><strong>🎖️ Active Trader</strong><br><span class='metric-value'>Goal: 30 trades ({max(0, 30 - total_trades_val)} left)</span></div>", unsafe_allow_html=True)
        
        avg_win_for_badge = df[df["Profit"] > 0]["Profit"].mean()
        avg_loss_for_badge = df[df["Profit"] < 0]["Profit"].mean() 
        
        with col_badge3:
            if pd.notna(avg_win_for_badge) and pd.notna(avg_loss_for_badge) and avg_loss_for_badge < 0.0:
                if avg_win_for_badge > abs(avg_loss_for_badge):
                    st.markdown("<div class='metric-box profitable'><strong>🎖️ Smart Scaler</strong><br><span class='metric-value'>Avg Win > Avg Loss!</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='metric-box'><strong>🎖️ Smart Scaler</strong><br><span class='metric-value'>Improve R:R ratio</span></div>", unsafe_allow_html=True)
            else:
                 st.markdown("<div class='metric-box'><strong>Smart Scaler</strong><br><span class='metric-value'>Trade more to assess!</span></div>", unsafe_allow_html=True)


    # --------------------------
    # File Uploader (MT5 page)
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
                    st.error(f"Missing required columns: {', '.join(missing_cols)}. Please ensure your CSV has all necessary columns.")
                    st.session_state.mt5_df = pd.DataFrame()
                    if "selected_calendar_month" in st.session_state: del st.session_state.selected_calendar_month
                    st.stop()

                # Coerce 'Open Time' and 'Close Time' to datetime, handling errors by setting to NaT
                df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
                df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")
                
                # Coerce 'Profit' to numeric, handling errors by setting to NaN, then fill NaNs with 0.0
                df["Profit"] = pd.to_numeric(df["Profit"], errors='coerce').fillna(0.0)
                
                # Filter out rows where 'Close Time' or 'Open Time' became NaT after coercion
                df = df.dropna(subset=["Open Time", "Close Time"])

                # Check if the filtered DataFrame is empty
                if df.empty:
                    st.error("Uploaded CSV resulted in no valid trading data after processing timestamps or profits.")
                    st.session_state.mt5_df = pd.DataFrame()
                    if "selected_calendar_month" in st.session_state: del st.session_state.selected_calendar_month
                    st.stop()


                df["Trade Duration"] = (df["Close Time"] - df["Open Time"]).dt.total_seconds() / 3600

                # Calculate daily_pnl_map once for both metrics and calendar.
                # This map contains entries for all days with at least one trade, including zero-net-profit days.
                daily_pnl_map = _ta_daily_pnl_mt5(df)
                
                # For aggregate stats (like drawdown, sharpe) that may require a continuous time series.
                # Fill in zeros for days without trades between min/max trading days.
                daily_pnl_df_for_stats = pd.DataFrame(columns=["date", "Profit"]) # Initialize empty

                if daily_pnl_map:
                    min_data_date = min(daily_pnl_map.keys())
                    max_data_date = max(daily_pnl_map.keys())
                    all_dates_in_data_range = pd.date_range(start=min_data_date, end=max_data_date).date
                    daily_pnl_df_for_stats = pd.DataFrame([
                        {"date": d, "Profit": daily_pnl_map.get(d, 0.0)} # Use 0.0 for days not in map but in range
                        for d in all_dates_in_data_range
                    ])
                elif not df.empty and pd.notna(df['Close Time'].min()) and pd.notna(df['Close Time'].max()):
                    # Even if daily_pnl_map is empty (e.g., all trades resulted in 0.0 profit),
                    # create a range of dates for stats calculation if Close Time is valid
                    min_date_raw = df['Close Time'].min().date()
                    max_date_raw = df['Close Time'].max().date()
                    all_dates_raw_range = pd.date_range(start=min_date_raw, end=max_date_raw).date
                    daily_pnl_df_for_stats = pd.DataFrame([
                        {"date": d, "Profit": 0.0} for d in all_dates_raw_range
                    ])


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
                    wins_df = df[df["Profit"] > 0]
                    losses_df = df[df["Profit"] < 0] # Filter for strictly losing trades here for correct avg_loss

                    win_rate = len(wins_df) / total_trades if total_trades else 0.0
                    net_profit = df["Profit"].sum()
                    profit_factor = _ta_profit_factor_mt5(df)
                    avg_win = wins_df["Profit"].mean() if not wins_df.empty else 0.0
                    avg_loss = losses_df["Profit"].mean() if not losses_df.empty else 0.0 # avg_loss will be negative or 0.0
                    
                    max_drawdown = (daily_pnl_df_for_stats["Profit"].cumsum() - daily_pnl_df_for_stats["Profit"].cumsum().cummax()).min() if not daily_pnl_df_for_stats.empty else 0.0
                    sharpe_ratio = _ta_compute_sharpe(df)
                    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) if total_trades else 0.0
                    longest_win_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] > 0) if k), default=0)
                    longest_loss_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] < 0) if k), default=0)

                    # Calculate additional metrics for the top row
                    # Avg R:R - assuming this is (Avg Win / Avg Loss magnitude)
                    avg_r_r = avg_win / abs(avg_loss) if avg_loss != 0.0 else np.nan # Calculate the numerical ratio
                    

                    # Trading Score (example: could be a composite of factors, here just a placeholder for visual)
                    trading_score_value = 90.98 # Example value, replace with your actual calculation
                    max_trading_score = 100
                    trading_score_percentage = (trading_score_value / max_trading_score) * 100

                    # Hit Rate - often synonymous with Win Rate, but sometimes calculated differently (e.g., successful trades vs total attempts)
                    hit_rate = win_rate # Assuming Hit Rate is the same as Win Rate for this example

                    # Most Profitable Asset
                    most_profitable_asset_calc = "N/A"
                    if not df.empty and "Symbol" in df.columns and not df["Profit"].isnull().all():
                        profitable_assets = df.groupby("Symbol")["Profit"].sum()
                        if not profitable_assets.empty and profitable_assets.max() > 0.0: # Ensure there is actual positive profit
                            most_profitable_asset_calc = profitable_assets.idxmax()
                        elif not profitable_assets.empty and profitable_assets.max() <= 0.0 and profitable_assets.min() <= 0.0:
                             most_profitable_asset_calc = "None Profitable" # All symbols had losses or zero profit
                        # If profitable_assets is empty, it remains "N/A"
                    
                    # Best and Worst Performing Day
                    best_day_profit = 0.0
                    best_performing_day_name = "N/A"
                    worst_day_loss = 0.0 
                    worst_performing_day_name = "N/A"

                    if not daily_pnl_df_for_stats.empty and not daily_pnl_df_for_stats["Profit"].empty:
                        # Filter for days with actual non-zero P&L for "best/worst day"
                        days_with_pnl_actual_trades = daily_pnl_df_for_stats[daily_pnl_df_for_stats["Profit"] != 0.0] 
                        
                        if not days_with_pnl_actual_trades.empty:
                            # Best Performing Day
                            best_day_profit = days_with_pnl_actual_trades["Profit"].max()
                            if pd.notna(best_day_profit) and best_day_profit > 0.0:
                                best_performing_day_date = days_with_pnl_actual_trades.loc[days_with_pnl_actual_trades["Profit"].idxmax(), "date"]
                                best_performing_day_name = pd.to_datetime(str(best_performing_day_date)).strftime('%A')
                            else:
                                best_performing_day_name = "No Profitable Days"

                            # Worst Performing Day
                            worst_day_loss = days_with_pnl_actual_trades["Profit"].min()
                            if pd.notna(worst_day_loss) and worst_day_loss < 0.0:
                                worst_performing_day_date = days_with_pnl_actual_trades.loc[days_with_pnl_actual_trades["Profit"].idxmin(), "date"]
                                worst_performing_day_name = pd.to_datetime(str(worst_performing_day_date)).strftime('%A')
                            else:
                                worst_performing_day_name = "No Losing Days"
                        else: # No non-zero profit days found in stats df
                            best_performing_day_name = "No Trades With Non-Zero P&L"
                            worst_performing_day_name = "No Trades With Non-Zero P&L"
                    else: # daily_pnl_df_for_stats is empty
                        best_performing_day_name = "No P&L Data"
                        worst_performing_day_name = "No P&L Data"


                    # Row 1: Metrics with bars (Avg R:R, Win Rate, Trading Score, Hit Rate, Total Trades)
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        r_r_bar_width = min(100, (avg_r_r / 2) * 100 if pd.notna(avg_r_r) and avg_r_r > 0 else 0)
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Avg R:R</strong>
                                <span class='metric-value'>{_ta_human_num_mt5(avg_r_r)}</span>
                                <div class="progress-container">
                                    <div class="progress-bar green" style="width: {r_r_bar_width:.2f}%;"></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        win_rate_percent = win_rate * 100
                        loss_rate_percent = 100 - win_rate_percent
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Win Rate</strong>
                                <span class='metric-value'>{_ta_human_pct_mt5(win_rate)}</span>
                                <div class="win-loss-bar-container">
                                    <div class="win-bar" style="width: {win_rate_percent:.2f}%;"></div>
                                    <div class="loss-bar" style="width: {loss_rate_percent:.2f}%;"></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Trading score</strong>
                                <span class='metric-value'>{_ta_human_num_mt5(trading_score_value)}</span>
                                <div class="trading-score-bar-container">
                                    <div class="trading-score-bar" style="width: {trading_score_percentage:.2f}%;"></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        hit_rate_percent = hit_rate * 100
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Hit Rate</strong>
                                <span class='metric-value'>{_ta_human_pct_mt5(hit_rate)}</span>
                                <div class="win-loss-bar-container">
                                    <div class="win-bar" style="width: {hit_rate_percent:.2f}%;"></div>
                                    <div class="loss-bar" style="width: {100-hit_rate_percent:.2f}%;"></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col5:
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Total Trades</strong>
                                <span class='metric-value'>{_ta_human_num_mt5(total_trades)}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---") # Separator between rows of metrics

                    # Row 2: Avg Profit, Best Performing Day, Total Profit, Worst Performing Day, Most Profitable Asset
                    col6, col7, col8, col9, col10 = st.columns(5)

                    with col6:
                        avg_win_formatted = _ta_human_num_mt5(avg_win)
                        avg_win_display = f"<span style='color: #5cb85c;'>${avg_win_formatted}</span>" if avg_win > 0.0 and avg_win_formatted != "N/A" else f"${avg_win_formatted}"
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Avg Profit</strong>
                                <span class='metric-value'>{avg_win_display}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    with col7:
                        best_day_profit_val = best_day_profit 
                        best_day_profit_formatted = _ta_human_num_mt5(best_day_profit_val)

                        if best_day_profit_val > 0.0 and best_day_profit_formatted != "N/A":
                            best_day_profit_display_html = f"<span style='color: #5cb85c;'>${best_day_profit_formatted}</span>"
                            day_info_text = f"{best_performing_day_name} with an average profit of {best_day_profit_display_html}."
                        elif best_performing_day_name in ["No Profitable Days", "No P&L Data", "No Trades With Non-Zero P&L"]:
                            day_info_text = best_performing_day_name
                        else:
                            day_info_text = "N/A" # Fallback if specific status isn't caught
                        
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Best Performing Day</strong>
                                <span class='day-info'>{day_info_text}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    with col8:
                        net_profit_val = net_profit 
                        net_profit_formatted = _ta_human_num_mt5(abs(net_profit_val))

                        if net_profit_val >= 0.0 and net_profit_formatted != "N/A":
                            net_profit_value_display_html = f"<span style='color: #5cb85c;'>${net_profit_formatted}</span>"
                        elif net_profit_formatted != "N/A":
                            net_profit_value_display_html = f"<span style='color: #d9534f;'>-${net_profit_formatted}</span>"
                        else:
                            net_profit_value_display_html = "N/A"
                        
                        # Total losses magnitude for the parentheses display. Image shows ($267,157.00) in red.
                        total_losses_magnitude = abs(losses_df['Profit'].sum()) if not losses_df.empty else 0.0
                        formatted_total_loss_in_parentheses_val = _ta_human_num_mt5(total_losses_magnitude)

                        if formatted_total_loss_in_parentheses_val != "N/A":
                            formatted_total_loss_in_parentheses_html = f"<span style='color: #d9534f;'>($-{formatted_total_loss_in_parentheses_val})</span>"
                        else:
                            formatted_total_loss_in_parentheses_html = f"<span style='color: #d9534f;'>(N/A)</span>"

                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Total Profit</strong>
                                <span class='metric-value'>{net_profit_value_display_html}</span>
                                <span class='sub-value'>{formatted_total_loss_in_parentheses_html}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    with col9:
                        worst_day_loss_val = worst_day_loss 
                        worst_day_loss_formatted = _ta_human_num_mt5(abs(worst_day_loss_val))

                        if worst_day_loss_val < 0.0 and worst_day_loss_formatted != "N/A":
                            worst_day_loss_display_html = f"<span style='color: #d9534f;'>-${worst_day_loss_formatted}</span>"
                            day_info_text = f"{worst_performing_day_name} with an average loss of {worst_day_loss_display_html}."
                        elif worst_performing_day_name in ["No Losing Days", "No P&L Data", "No Trades With Non-Zero P&L"]:
                            day_info_text = worst_performing_day_name
                        else:
                            day_info_text = "N/A" # Fallback
                        
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Worst Performing Day</strong>
                                <span class='day-info'>{day_info_text}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    with col10:
                        st.markdown(f"""
                            <div class='metric-box'>
                                <strong>Most Profitable Asset</strong>
                                <span class='metric-value'>{most_profitable_asset_calc}</span>
                            </div>
                        """, unsafe_allow_html=True)
                # ---------- End of Summary Metrics Tab ----------


                # ---------- Visualizations Tab ----------
                with tab_charts:
                    st.subheader("Visualizations")
                    st.write("Visualizations will be displayed here.")
                    # Add your charting code here

                # ---------- Edge Finder Tab ----------
                with tab_edge:
                    st.subheader("Edge Finder")
                    st.write("Analyze your trading edge here.")
                    # Add your edge analysis code here

                # ---------- Export Reports Tab ----------
                with tab_export:
                    st.subheader("Export Reports")
                    st.write("Export your trading data and reports.")
                    # Add your export functionality here


            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}. Please check your CSV format and ensure it contains the required columns with valid data.")
                logging.error(f"Error processing CSV: {str(e)}", exc_info=True)
                st.session_state.mt5_df = pd.DataFrame() # Clear session state if error
                if "selected_calendar_month" in st.session_state: del st.session_state.selected_calendar_month
    else:
        st.info("👆 Upload your MT5 trading history CSV to explore advanced performance metrics.")
        # Ensure that if file is not uploaded, there is no lingering dataframe
        if "mt5_df" in st.session_state:
            del st.session_state.mt5_df
        # Remove selected calendar month when no file is uploaded
        if "selected_calendar_month" in st.session_state:
            del st.session_state.selected_calendar_month
            

    # Gamification Badges (Now placed correctly below the tabs section)
    if "mt5_df" in st.session_state and not st.session_state.mt5_df.empty:
        try:
            st.markdown("---") # Separator before badges
            _ta_show_badges_mt5(st.session_state.mt5_df)
        except Exception as e:
            logging.error(f"Error displaying badges: {str(e)}")
    else:
        pass # No trades uploaded, no badges to show yet.
    
    # ----- Daily Performance Calendar -----
    if "mt5_df" in st.session_state and not st.session_state.mt5_df.empty:
        st.markdown("---") # Separator before calendar
        st.subheader("🗓️ Daily Performance Calendar")

        df_for_calendar = st.session_state.mt5_df 
        
        # --- Calendar month selection logic ---
        selected_month_date = date(datetime.now().year, datetime.now().month, 1) # Default to current month start

        if not df_for_calendar.empty and not df_for_calendar["Close Time"].isnull().all():
            min_date_data = pd.to_datetime(df_for_calendar["Close Time"]).min().date()
            max_date_data = pd.to_datetime(df_for_calendar["Close Time"]).max().date()
            
            # Ensure proper handling of 'MS' frequency when creating periods for month selection
            all_months_in_data = pd.date_range(start=min_date_data.replace(day=1), 
                                                end=max_date_data.replace(day=1), freq='MS').to_period('M')
            available_months_periods = sorted(list(all_months_in_data), reverse=True)

            if available_months_periods:
                display_options = [f"{p.strftime('%B %Y')}" for p in available_months_periods]
                
                # Default to the month of the latest trade
                latest_data_month_str = available_months_periods[0].strftime('%B %Y') 
                
                if 'selected_calendar_month' not in st.session_state:
                     st.session_state.selected_calendar_month = latest_data_month_str 
                
                selected_month_year_str = st.selectbox(
                    "Select Month",
                    options=display_options,
                    index=display_options.index(st.session_state.selected_calendar_month) if st.session_state.selected_calendar_month in display_options else 0,
                    key="calendar_month_selector"
                )
                st.session_state.selected_calendar_month = selected_month_year_str

                selected_period = next((p for p in available_months_periods if p.strftime('%B %Y') == selected_month_year_str), None)
                selected_month_date = selected_period.start_time.date() if selected_period else date(datetime.now().year, datetime.now().month, 1)
            else:
                 st.warning("No trade data with valid 'Close Time' to display in the calendar.")
                 selected_month_date = date(datetime.now().year, datetime.now().month, 1) 
        else: 
            st.warning("No trade data with valid 'Close Time' to display in the calendar.")
            selected_month_date = date(datetime.now().year, datetime.now().month, 1)

        # --- Daily P&L for calendar display ---
        daily_pnl_map_for_calendar = _ta_daily_pnl_mt5(df_for_calendar)
        
        # --- Generate calendar grid HTML ---
        cal = calendar.Calendar(firstweekday=calendar.SUNDAY) 
        month_days = cal.monthdatescalendar(selected_month_date.year, selected_month_date.month)

        calendar_html = f"""
        <div class="calendar-container">
            <div class="calendar-header">
                {calendar.month_name[selected_month_date.month]} {selected_month_date.year}
            </div>
            <div class="calendar-weekdays">
                <div class="calendar-weekday-item">Sun</div>
                <div class="calendar-weekday-item">Mon</div>
                <div class="calendar-weekday-item">Tue</div>
                <div class="calendar-weekday-item">Wed</div>
                <div class="calendar-weekday-item">Thu</div>
                <div class="calendar-weekday-item">Fri</div>
                <div class="calendar-weekday-item">Sat</div>
            </div>
            <div class="calendar-grid">
        """

        today = datetime.now().date()
        for week in month_days:
            for day_date in week:
                day_class = ""
                profit_amount_html = ""
                dot_indicator_html = "" # Placeholder for future dots, currently unused
                
                if day_date.month == selected_month_date.month:
                    profit = daily_pnl_map_for_calendar.get(day_date)
                    if profit is not None: # This means there was *at least one trade* on this specific day in the CSV
                        if profit > 0.0:
                            day_class += " profitable"
                            profit_amount_html = f"<span style='color:#5cb85c;'>${_ta_human_num_mt5(profit)}</span>"
                        elif profit < 0.0:
                            day_class += " losing"
                            profit_amount_html = f"<span style='color:#d9534f;'>-${_ta_human_num_mt5(abs(profit))}</span>"
                        else: # profit == 0.0 for the day, meaning trades occurred but summed to zero.
                            profit_amount_html = f"<span style='color:#cccccc;'>$0.00</span>" 
                    else: # No trades recorded at all for this day in the map (no data for this day in CSV for the current month)
                         profit_amount_html = "<span style='color:#cccccc;'>$0.00</span>"
                else: # Days from the previous/next month displayed in the grid
                    day_class += " empty-month-day" # Hides these days via CSS visibility: hidden
                    profit_amount_html = "" # Ensure no profit is shown for hidden days
                
                if day_date == today:
                    day_class += " current-day"
                        
                calendar_html += f"""
                    <div class="calendar-day-box {day_class}">
                        <span class="day-number">{day_date.day if day_date.month == selected_month_date.month else ''}</span>
                        <div class="profit-amount">{profit_amount_html}</div>
                        {dot_indicator_html}
                    </div>
                """
        
        calendar_html += """
            </div>
        </div>
        """
        
        # --- DEBUGGING MARKERS ---
        st.write("--- BEGINNING OF CALENDAR OUTPUT ---")
        st.markdown(f"Is calendar_html a string? {isinstance(calendar_html, str)}")
        st.markdown(f"Length of calendar_html: {len(calendar_html)}")
        st.markdown(calendar_html, unsafe_allow_html=True)
        st.write("--- END OF CALENDAR OUTPUT ---")


    # Report Export & Sharing
    if 'df' in locals() and not df.empty:
        st.markdown("---") 
        if st.button("📄 Generate Performance Report"):
            total_trades = len(df)
            wins_df = df[df["Profit"] > 0]
            losses_df = df[df["Profit"] < 0] 
            win_rate = len(wins_df) / total_trades if total_trades else 0.0
            net_profit = df["Profit"].sum()
            profit_factor = _ta_profit_factor_mt5(df)
            longest_win_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] > 0) if k), default=0)
            longest_loss_streak = max((len(list(g)) for k, g in df.groupby(df["Profit"] < 0) if k), default=0)

            report_html = f"""
            <html>
            <head>
                <title>Performance Report</title>
                <style>
                    body {{ font-family: sans-serif; margin: 20px; background-color: #1a1a1a; color: #f0f0f0; }}
                    h2 {{ color: #58b3b1; }}
                    p {{ margin-bottom: 5px; }}
                    .positive {{ color: #5cb85c; }}
                    .negative {{ color: #d9534f; }}
                </style>
            </head>
            <body>
            <h2>Performance Report</h2>
            <p><strong>Total Trades:</strong> {_ta_human_num_mt5(total_trades)}</p>
            <p><strong>Win Rate:</strong> {_ta_human_pct_mt5(win_rate)}</p>
            <p><strong>Net Profit:</strong> <span class='{'positive' if net_profit >= 0 else 'negative'}'>
            {'$' if net_profit >= 0 else '-$'}{_ta_human_num_mt5(abs(net_profit))}</span></p>
            <p><strong>Profit Factor:</strong> {_ta_human_num_mt5(profit_factor)}</p>
            <p><strong>Biggest Win:</strong> <span class='positive'>${_ta_human_num_mt5(wins_df["Profit"].max() if not wins_df.empty else 0.0)}</span></p>
            <p><strong>Biggest Loss:</strong> <span class='negative'>-${_ta_human_num_mt5(abs(losses_df["Profit"].min()) if not losses_df.empty else 0.0)}</span></p>
            <p><strong>Longest Win Streak:</strong> {_ta_human_num_mt5(longest_win_streak)}</p>
            <p><strong>Longest Loss Streak:</strong> {_ta_human_num_mt5(longest_loss_streak)}</p>
            <p><strong>Avg Trade Duration:</strong> {_ta_human_num_mt5(df['Trade Duration'].mean())}h</p>
            <p><strong>Total Volume:</strong> {_ta_human_num_mt5(df['Volume'].sum())}</p>
            <p><strong>Avg Volume:</strong> {_ta_human_num_mt5(df['Volume'].mean())}</p>
            <p><strong>Profit / Trade:</strong> <span class='{'positive' if (net_profit/total_trades if total_trades else 0.0) >= 0 else 'negative'}'>
            {'$' if (net_profit/total_trades if total_trades else 0.0) >= 0 else '-$'}{_ta_human_num_mt5(abs(net_profit/total_trades if total_trades else 0.0))}</span></p>
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

# =========================================================
# MANAGE MY STRATEGY PAGE
# =========================================================
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

# =========================================================
# ACCOUNT PAGE
# =========================================================
elif st.session_state.current_page == 'account':
    # >>> START REPLACEMENT HERE <<<
    if "logged_in_user" not in st.session_state:
        st.title("👤 My Account") # This title will now only appear when not logged in
        st.markdown(
            """
            Manage your account, save your data, and sync your trading journal and drawings. Signing in lets you:
            - Keep your trading journal and strategies backed up.
            - Track your progress and gamification stats.
            - Sync across devices.
            - Import/export your account data easily.
            """
        )
    # >>> END REPLACEMENT HERE <<<
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
                                st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
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
    else:
        # --------------------------
        # LOGGED-IN USER VIEW
        # --------------------------
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

        st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")
        st.markdown("This is your personal dashboard. Track your progress and manage your account.")
        st.markdown("---")
        

        # --- Main Dashboard Layout using Columns ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Progress Snapshot")
            
            # --- Custom CSS for the KPI cards ---
            st.markdown("""
            <style>
            .kpi-card {
                background-color: rgba(45, 70, 70, 0.5);
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                border: 1px solid #58b3b1;
                margin-bottom: 10px;
            }
            .kpi-icon {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .kpi-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #FFFFFF;
            }
            .kpi-label {
                font-size: 0.9em;
                color: #A0A0A0;
            }
            .insights-card {
                background-color: rgba(45, 70, 70, 0.3);
                border-left: 5px solid #58b3b1;
                padding: 15px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)

            # --- Row 1: KPI Cards ---
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                level = st.session_state.get('level', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">🧙‍♂️</div>
                    <div class="kpi-value">Level {level}</div>
                    <div class="kpi-label">Trader's Rank</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col2:
                streak = st.session_state.get('streak', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">🔥</div>
                    <div class="kpi-value">{streak} Days</div>
                    <div class="kpi-label">Journaling Streak</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col3:
                total_xp = st.session_state.get('xp', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">⭐</div>
                    <div class="kpi-value">{total_xp:,}</div>
                    <div class="kpi-label">Total Experience</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")

            # --- Row 2: Progress Chart and Insights ---
            # Use the 'gap' parameter to create significant horizontal space between the columns
            chart_col, spacer_col, insights_col = st.columns([10, 1, 10])

            with chart_col:
                st.markdown("<h5 style='text-align: center;'>Progress to Next Level</h5>", unsafe_allow_html=True)
                total_xp = st.session_state.get('xp', 0) # Ensure total_xp is defined here
                xp_in_level = total_xp % 100
                xp_needed = 100 - xp_in_level

                fig = go.Figure(go.Pie(
                    values=[xp_in_level, xp_needed],
                    labels=['XP Gained', 'XP Needed'],
                    hole=0.6,
                    marker_colors=['#58b3b1', '#2d4646'],
                    textinfo='none',
                    hoverinfo='label+value',
                    direction='clockwise',
                    sort=False
                ))
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(text=f'<b>{xp_in_level}<span style="font-size:0.6em">/100</span></b>', x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")],
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # The spacer column is left empty to create the space.
            with spacer_col:
                st.empty()

            with insights_col:
                st.markdown("<h5 style='text-align: center;'>Personalized Insights</h5>", unsafe_allow_html=True)
                streak = st.session_state.get('streak', 0) # Ensure streak is defined here
                
                insight_message = ""
                if streak > 21:
                    insight_message = "Your journaling consistency is elite! This level of discipline is a key trait of professional traders."
                elif streak > 7:
                    insight_message = "Over a week of consistent journaling! You're building a powerful habit. Keep the momentum going."
                else:
                    insight_message = "Every trade journaled is a step forward. Stay consistent to build a strong foundation for your trading career."
                
                st.markdown(f"<div class='insights-card'><p>💡 {insight_message}</p></div>", unsafe_allow_html=True)

                num_trades = len(st.session_state.tools_trade_journal)
                next_milestone = ""
                if num_trades < 10:
                    next_milestone = f"Log **{10 - num_trades} more trades** to earn the 'Ten Trades' badge!"
                elif num_trades < 50:
                    next_milestone = f"You're **{50 - num_trades} trades** away from the '50 Club' badge. Keep it up!"
                else:
                     next_milestone = "The next streak badge is at 30 days. You've got this!"

                st.markdown(f"<div class='insights-card' style='margin-top: 10px;'><p>🎯 **Next Up:** {next_milestone}</p></div>", unsafe_allow_html=True)

        # --- Row 3: XP Journey Chart (This part goes right after the `with col1:` and `with col2:` blocks) ---
        st.markdown("<hr style='border-color: #4d7171;'>", unsafe_allow_html=True)
        st.subheader("🚀 Your XP Journey")
        journal_df = st.session_state.tools_trade_journal
        if not journal_df.empty and 'Date' in journal_df.columns:
            journal_df['Date'] = pd.to_datetime(journal_df['Date'])
            xp_data = journal_df.sort_values(by='Date').copy()
            xp_data['xp_gained'] = 10 
            xp_data['cumulative_xp'] = xp_data['xp_gained'].cumsum()
            
            fig_line = px.area(xp_data, x='Date', y='cumulative_xp', 
                                title="XP Growth Over Time (Based on Journal Entries)",
                                labels={'Date': 'Journal Entry Date', 'cumulative_xp': 'Cumulative XP'})
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(45, 70, 70, 0.3)',
                xaxis=dict(gridcolor='#4d7171'),
                yaxis=dict(gridcolor='#4d7171'),
                font_color="white"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Log your first trade in the 'Backtesting' tab to start your XP Journey!")
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

# =========================================================
# COMMUNITY TRADE IDEAS PAGE
# =========================================================
elif st.session_state.current_page == 'community':
    st.title("🌐 Community Trade Ideas")
    st.markdown(""" Share and explore trade ideas with the community. Upload your chart screenshots and discuss strategies with other traders. """)
    st.write('---')
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
                if "ImagePath" in idea and pd.notna(idea['ImagePath']) and os.path.exists(idea['ImagePath']):
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
# =========================================================
# TOOLS PAGE
# =========================================================
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
    # --------------------------
    # PROFIT / LOSS CALCULATOR
    # --------------------------
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
    # --------------------------
    # PRICE ALERTS
    # --------------------------
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
    # --------------------------
    # CURRENCY CORRELATION HEATMAP
    # --------------------------
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
    # --------------------------
    # RISK MANAGEMENT CALCULATOR
    # --------------------------
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
    # --------------------------
    # TRADING SESSION TRACKER
    # --------------------------
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
    # --------------------------
    # DRAWDOWN RECOVERY PLANNER
    # --------------------------
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
    # --------------------------
    # PRE-TRADE CHECKLIST
    # --------------------------
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
    # --------------------------
    # PRE-MARKET CHECKLIST
    # --------------------------
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

# =========================================================
# ZENVO ACADEMY PAGE
# =========================================================
elif st.session_state.current_page == "Zenvo Academy":
    st.title("📚 Zenvo Academy")
    st.caption("Your journey to trading mastery starts here. Explore interactive courses, track your progress, and unlock your potential.")
    st.markdown('---')

    # --- Academy Tabs ---
    tab1, tab2, tab3 = st.tabs(["🎓 Learning Path", "📈 My Progress", "🛠️ Resources"])

    with tab1:
        st.markdown("### 🗺️ Your Learning Path")
        st.write("Our Academy provides a clear learning path for traders of all levels. Start from the beginning or jump into a topic that interests you.")

        # --- Course: Forex Fundamentals ---
        with st.expander("Forex Fundamentals - Level 1 (100 XP)", expanded=True):
            st.markdown("**Description:** This course is your first step into the world of Forex trading. You'll learn the essential concepts that every trader needs to know.")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                    - **Lesson 1:** What is Forex?
                    - **Lesson 2:** How to Read a Currency Pair
                    - **Lesson 3:** Understanding Pips and Lots
                    - **Lesson 4:** Introduction to Charting
                """)
            with col2:
                st.button("Start Learning", key="start_forex_fundamentals")
                st.progress(st.session_state.get('forex_fundamentals_progress', 0))


        # --- Course: Technical Analysis 101 ---
        with st.expander("Technical Analysis 101 - Level 2 (150 XP)"):
            st.markdown("**Description:** Learn how to analyze price charts to identify trading opportunities. This course covers the foundational tools of technical analysis.")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                    - **Lesson 1:** Candlestick Patterns
                    - **Lesson 2:** Support and Resistance
                    - **Lesson 3:** Trendlines and Channels
                    - **Lesson 4:** Moving Averages
                """)
            with col2:
                st.button("Start Course", key="start_technical_analysis", disabled=st.session_state.level < 1)


    with tab2:
        st.markdown("### 🚀 Your Progress")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Level", st.session_state.get('level', 0))
        with col2:
            st.metric("Experience Points (XP)", f"{st.session_state.get('xp', 0)} / {(st.session_state.get('level', 0) + 1) * 100}")
        with col3:
            st.metric("Badges Earned", len(st.session_state.get('badges', [])))

        st.markdown("---")
        st.markdown("#### 📜 Completed Courses")
        if 'completed_courses' in st.session_state and st.session_state.completed_courses:
            for course in st.session_state.completed_courses:
                st.success(f"**{course}** - Completed!")
        else:
            st.info("You haven't completed any courses yet. Get started on the Learning Path!")

        st.markdown("#### 🎖️ Your Badges")
        if 'badges' in st.session_state and st.session_state.badges:
            for badge in st.session_state.badges:
                st.markdown(f"- 🏅 {badge}")
        else:
            st.info("No badges earned yet. Complete courses to unlock them!")


    with tab3:
        st.markdown("### 🧰 Trading Resources")
        st.info("This section is under development. Soon you will find helpful tools, articles, and more to aid your trading journey!")


    if st.button("Log Out", key="logout_academy_page"):
        if 'logged_in_user' in st.session_state:
            del st.session_state.logged_in_user
        st.session_state.drawings = {}
        # Ensure correct journal_cols and journal_dtypes are used here, referencing the main app's definitions
        st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
        st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"]) # Match structure defined in main app
        st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
        st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
        st.session_state.xp = 0
        st.session_state.level = 0
        st.session_state.badges = []
        st.session_state.streak = 0
        st.session_state.completed_courses = []
        st.success("Logged out successfully!")
        logging.info("User logged out")
        st.session_state.current_page = "account" # Changed to 'account' as 'login' page state doesn't directly exist
        st.rerun()
