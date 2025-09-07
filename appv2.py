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
import calendar
from datetime import datetime, date, timedelta

# =========================================================
# GLOBAL CSS & GRIDLINE SETTINGS
# =========================================================
st.markdown(
    """
    <style>
    /* --- Main App Styling (from Code A, adapted to use Code B's background) --- */
    .stApp {
        /* Retain Code B's background styling, adding only text color */
        background-color: #000000; /* black background */
        color: #c9d1d9; /* Text color from Code A */
        /* Code B gridline background */
        background-image:
        linear-gradient(rgba(88, 179, 177, 0.16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(88, 179, 177, 0.16) 1px, transparent 1px);
        background-size: 40px 40px;
        background-attachment: fixed;
    }

    .block-container {
        padding: 1.5rem 2.5rem 2rem 2.5rem !important; /* From Code A */
    }

    h1, h2, h3, h4 {
        color: #c9d1d9 !important; /* From Code A */
    }

    /* --- Global Horizontal Line Style --- (From Code B, then refined with Code A's) */
    hr {
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        border-top: 1px solid #30363d !important; /* Adjusted from Code A */
        border-bottom: none !important;
        background-color: transparent !important;
        height: 1px !important;
    }

    /* Hide Streamlit Branding (From Code A & B merged) */
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }

    /* Optional: remove extra padding/margin from main page (from Code B, adapted) */
    .css-18e3th9, .css-1d391kg {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    /* --- Metric Card Styling (from Code A) --- */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        border-color: #58a6ff;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #8b949e;
    }

    /* --- Tab Styling (from Code A, adapted) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border: 1px solid #30363d; /* Adjusted border from Code A */
        border-radius: 8px;
        padding: 0 24px;
        transition: all 0.2s ease-in-out;
        color: #c9d1d9; /* Default text color for tabs */
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #161b22;
        color: #58a6ff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #161b22;
        border-color: #58a6ff;
        color: #c9d1d9; /* Active tab text color from Code A */
    }

    /* --- Styling for Markdown in Trade Playbook (from Code A) --- */
    .trade-notes-display {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }
    .trade-notes-display p { font-size: 15px; color: #c9d1d9; line-height: 1.6; }
    .trade-notes-display h1, h2, h3, h4 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# LOGGING & DATABASE SETUP (Merged Code A & B)
# =========================================================
# Ensure Code A's logging config is used (INFO level is sufficient for general ops)
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Using Code B's DB_FILE "users.db" and structure for overall app consistency.
# Code A's logic will be adapted to store journal in 'data' field.
DB_FILE = "users.db"

# Custom JSON encoder for handling datetime objects and NaNs (from Code A, as it's provided in prompt)
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)): return obj.isoformat()
        # pandas NA and numpy NaN handling added for robustness from Code A
        if pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)): return None
        return super().default(obj)

try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS community_data (key TEXT PRIMARY KEY, data TEXT)''') # From Code B
    conn.commit()
    logging.info("SQLite database initialized successfully")
except Exception as e:
    st.error("Fatal Error: Could not connect to the database.")
    logging.critical(f"Failed to initialize SQLite database: {str(e)}", exc_info=True)
    st.stop()


# =========================================================
# HELPER & GAMIFICATION FUNCTIONS (Adapted from Code A and B)
# =========================================================
# Code B helper for logging info in case no-ops (kept, generally useful)
def ta_safe_lower(s):
    return str(s).strip().lower().replace(" ", "")

# Code A formatting functions (more specific) for general use
def ta_human_pct(x, nd=1):
    if pd.isna(x):
        return "—"
    return f"{x*100:.{nd}f}%"

def _ta_human_num(x, nd=2): # Code A's _ta_human_num (non-MT5 specific)
    if pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"

def _ta_hash(): # from Code B
    return uuid.uuid4().hex[:12]

def _ta_percent_gain_to_recover(drawdown_pct): # from Code B
    if drawdown_pct <= 0:
        return 0.0
    if drawdown_pct >= 0.99:
        return float("inf")
    return drawdown_pct / (1 - drawdown_pct)

# User Data Management (Adapted from Code A to fit Code B's DB structure)
def get_user_data(username):
    c.execute("SELECT data FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    return json.loads(result[0]) if result and result[0] else {}

def save_user_data(username, data):
    try:
        json_data = json.dumps(data, cls=CustomJSONEncoder)
        c.execute("UPDATE users SET data = ? WHERE username = ?", (json_data, username))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Failed to save data for {username}: {e}", exc_info=True)
        return False

# Community Data Management (from Code B)
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

# Trade Journal Save (Adapted from Code A to use get/save_user_data)
def _ta_save_journal(username, journal_df):
    user_data = get_user_data(username)
    # The journal data will be stored under the key 'trade_journal'
    user_data["trade_journal"] = journal_df.to_dict('records')
    if save_user_data(username, user_data):
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    st.error("Failed to save journal.")
    return False

# XP notification system (from Code B)
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

# XP Update (Adapted from Code A to integrate with Code B's XP notification)
def ta_update_xp(username, amount): # from Code A
    user_data = get_user_data(username)
    user_data['xp'] = user_data.get('xp', 0) + amount
    level = user_data['xp'] // 100 # From Code B's XP update logic for levelling
    if level > user_data.get('level', 0):
        user_data['level'] = level
        user_data['badges'] = user_data.get('badges', []) + [f"Level {level}"]
        st.balloons()
        st.success(f"Level up! You are now level {level}.")
    save_user_data(username, user_data)
    # The session state needs to be updated for other parts of the app
    st.session_state.xp = user_data['xp']
    st.session_state.level = user_data['level']
    st.session_state.badges = user_data['badges']
    show_xp_notification(amount) # Integrates Code B's notification system


# Streak Update (Adapted from Code A, ensuring correct session state update)
def ta_update_streak(username): # from Code A
    user_data = get_user_data(username)
    today = dt.date.today()
    last_date_str = user_data.get('last_journal_date')
    streak = user_data.get('streak', 0)

    if last_date_str:
        last_date = dt.date.fromisoformat(last_date_str)
        if last_date == today: return # Already journaled today
        if last_date == today - dt.timedelta(days=1): streak += 1
        else: streak = 1 # Streak broken or new streak begins
    else: streak = 1 # First time journaling or no previous streak
        
    user_data.update({'streak': streak, 'last_journal_date': today.isoformat()})
    
    # Badge for streak achievement (from Code B, adapted)
    if streak % 7 == 0:
        badge = f"{streak}-Day Streak"
        if badge not in user_data.get('badges', []):
            user_data['badges'] = user_data.get('badges', []) + [badge]
            st.balloons()
            st.success(f"Unlocked: {badge} Discipline Badge!")

    save_user_data(username, user_data)
    st.session_state.streak = streak # Update session state directly
    st.session_state.badges = user_data['badges'] # Update session state badges


# MOCK AUTHENTICATION & SESSION STATE SETUP (Code A, adapted slightly)
# This mock is for initial setup, the full login in the account page will take precedence.
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = "pro_trader"
    c.execute("SELECT username FROM users WHERE username = ?", (st.session_state.logged_in_user,))
    if not c.fetchone():
        hashed_password = hashlib.sha256("password".encode()).hexdigest()
        # Initialize with Code A's trade_journal key and empty structures
        initial_data = json.dumps({
            'xp': 0, 'streak': 0, 'trade_journal': [], 'level': 0, 'badges': [],
            'drawings': {}, 'strategies': [], 'emotion_log': [], 'reflection_log': []
        })
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)",
                  (st.session_state.logged_in_user, hashed_password, initial_data))
        conn.commit()

# =========================================================
# JOURNAL SCHEMA & ROBUST DATA MIGRATION (from Code A)
# CLEANED UP SCHEMA with safe names for columns
journal_cols = [
    "TradeID", "Date", "Symbol", "Direction", "Outcome", "PnL", "RR",
    "Strategy", "Tags", "EntryPrice", "StopLoss", "FinalExit", "Lots",
    "EntryRationale", "TradeJournalNotes", "EntryScreenshot", "ExitScreenshot"
]
journal_dtypes = {
    "TradeID": str, "Date": "datetime64[ns]", "Symbol": str, "Direction": str,
    "Outcome": str, "PnL": float, "RR": float, "Strategy": str,
    "Tags": str, "EntryPrice": float, "StopLoss": float, "FinalExit": float, "Lots": float,
    "EntryRationale": str, "TradeJournalNotes": str,
    "EntryScreenshot": str, "ExitScreenshot": str
}

if 'trade_journal' not in st.session_state:
    user_data = get_user_data(st.session_state.logged_in_user)
    journal_data = user_data.get("trade_journal", []) # Consistent with 'trade_journal' key
    df = pd.DataFrame(journal_data)

    # Safely migrate data to the new, safer schema
    # Extended legacy_col_map to try and cover potential old Code B journal columns
    legacy_col_map = {
        "Trade ID": "TradeID", "Entry Price": "EntryPrice", "Stop Loss": "StopLoss",
        "Final Exit": "FinalExit", "PnL ($)": "PnL", "R:R": "RR",
        "Entry Rationale": "EntryRationale", "Trade Journal Notes": "TradeJournalNotes",
        "Entry Screenshot": "EntryScreenshot", "Exit Screenshot": "ExitScreenshot",
        # --- Potential mappings from Code B's older, different journal structure ---
        "Date": "Date", # Date is already fine
        "Weekly Bias": "Strategy",
        "Daily Bias": "Strategy",
        "4H Structure": "Strategy",
        "1H Structure": "Strategy",
        "Positive Correlated Pair & Bias": "EntryRationale", # Can merge with EntryRationale
        "Potential Entry Points": "EntryRationale", # Can merge
        "5min/15min Setup?": "EntryRationale", # Can merge
        "Entry Conditions": "EntryRationale", # Can merge
        "Planned R:R": "RR", # Can map to RR directly if numeric or extract
        "News Filter": "EntryRationale",
        "Alerts": "TradeJournalNotes", # Move to notes
        "Concerns": "TradeJournalNotes", # Move to notes
        "Emotions": "TradeJournalNotes", # Move to notes
        "Confluence Score 1-7": "Strategy", # Could be part of strategy/rationale
        "Outcome / R:R Realised": "Outcome", # Map Outcome and then parse RR if needed
        "Notes/Journal": "TradeJournalNotes",
        "Entry Price": "EntryPrice", # These are direct matches now but good to be explicit
        "Stop Loss Price": "StopLoss",
        "Take Profit Price": "FinalExit", # Map to FinalExit for A's schema
        "Lots": "Lots",
        "Tags": "Tags",
        "Direction": "Direction", # Ensure this is also mapped
        "Symbol": "Symbol" # Ensure this is also mapped
    }

    df.rename(columns=legacy_col_map, inplace=True)

    # Clean up and ensure data types and consistent values
    if 'Outcome' in df.columns:
        def standardize_outcome(x):
            if pd.isna(x):
                return 'No Trade/Study'
            x_str = str(x).lower().strip()
            if 'win' in x_str:
                return 'Win'
            elif 'loss' in x_str:
                return 'Loss'
            elif 'breakeven' in x_str:
                return 'Breakeven'
            elif 'no trade' in x_str or x_str in ('0.0', 'nan', '', '0.0r'):
                return 'No Trade/Study'
            return x # return original if unable to parse (might be raw RR string like '1:2.5')

        df['Outcome'] = df['Outcome'].apply(standardize_outcome)

    if 'RR' in df.columns:
        # Convert any '1:X.XX' strings to float, handling existing floats as-is
        def parse_rr_to_float(x):
            if isinstance(x, str) and ':' in x:
                try:
                    return float(x.split(':')[1])
                except (ValueError, IndexError):
                    return 0.0 # Default if parse fails
            elif pd.isna(x):
                return 0.0
            return float(x) # Convert existing numbers (or NaN) to float

        df['RR'] = df['RR'].apply(parse_rr_to_float)

    # Convert PnL to float explicitly after potential merges and if it was incorrectly string formatted
    if 'PnL' in df.columns:
        df['PnL'] = pd.to_numeric(df['PnL'], errors='coerce').fillna(0.0)

    # Ensure all new columns (from journal_cols) exist with appropriate default values
    for col, dtype in journal_dtypes.items():
        if col not in df.columns:
            if dtype == str:
                df[col] = ''
            elif 'datetime' in str(dtype):
                df[col] = pd.NaT
            elif dtype == float:
                df[col] = 0.0
            else:
                df[col] = None
        else:
            # Attempt to convert existing column to correct dtype
            try:
                df[col] = df[col].astype(dtype, errors='coerce')
            except Exception as e:
                logging.warning(f"Failed to cast column {col} to {dtype}: {e}")
                if 'datetime' in str(dtype): df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == float: df[col] = pd.to_numeric(df[col], errors='coerce')


    st.session_state.trade_journal = df[journal_cols].astype(journal_dtypes, errors='ignore')
    st.session_state.trade_journal['Date'] = pd.to_datetime(st.session_state.trade_journal['Date'], errors='coerce')
    # Fill any remaining NaT in 'Date' with a default or drop for consistency
    st.session_state.trade_journal['Date'] = st.session_state.trade_journal['Date'].fillna(pd.Timestamp.now().floor('D'))


# =========================================================
# PAGE CONFIGURATION (Code B global)
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide") # From Code B


# =========================================================
# CUSTOM SIDEBAR CSS (Code B global)
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
# NEWS & ECONOMIC CALENDAR DATA / HELPERS (Code B global)
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
        date_str = published[:10] if published else ""
        currency = detect_currency(title)
        polarity = TextBlob(title).sentiment.polarity
        impact = rate_impact(polarity)
        summary = getattr(entry, "summary", "")
        rows.append({
            "Date": date_str, # Keep as string for now
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
# SESSION STATE INITIALIZATION (from Code B, adapted for unified journal name)
# =========================================================
# Init other session states if not already done by auth or journal init
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'fundamentals'

if 'current_subpage' not in st.session_state:
    st.session_state.current_subpage = None

if 'show_tools_submenu' not in st.session_state:
    st.session_state.show_tools_submenu = False

# This was 'tools_trade_journal' in Code B, but Code A uses 'trade_journal'
# It's now handled by the journal_schema setup directly, so no need for 'if' here.
# But keeping any other journal-like states initialized empty if not used elsewhere explicitly.
if "temp_journal" not in st.session_state:
    st.session_state.temp_journal = None


# =========================================================
# SIDEBAR NAVIGATION (Code B global, adapted page name)
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
logo = Image.open("logo22.png") # Assuming 'logo22.png' exists
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

# Navigation items (UPDATED: 'backtesting' to 'trading_journal')
nav_items = [
    ('fundamentals', 'Forex Fundamentals'),
    ('trading_journal', 'Trading Journal'), # RENAMED HERE
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
# FUNDAMENTALS PAGE (Code B)
# =========================================================
if st.session_state.current_page == 'fundamentals':
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📅 Forex Fundamentals")
        st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
        st.markdown('---')
    with col2:
        st.info("See the Trading Journal tab for live charts + detailed news.") # Updated page reference
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
# TRADING JOURNAL PAGE (Replaces old 'Backtesting' from Code B with Code A's Journal)
# =========================================================
elif st.session_state.current_page == 'trading_journal': # RENAMED HERE
    # PAGE LAYOUT (FROM CODE A, adapted)
    st.title("📊 Trading Journal") # Renamed from 'Pro Journal & Backtesting Environment'
    st.caption(f"A streamlined interface for professional trade analysis. | Logged in as: **{st.session_state.logged_in_user}**")
    st.markdown("---")

    # --- CHARTING AREA (REMOVED - per request: "remove tradingview widget and save, load buttons") ---
    # The Code A's 'pairs_map' and all the 'tv_html' and `st.components.v1.html(tv_html, height=560)` are REMOVED.
    # Also, any save/load buttons for drawings specific to the chart are REMOVED from this section.

    # =========================================================
    # TRADING JOURNAL TABS (FROM CODE A)
    # =========================================================
    tab_entry, tab_playbook, tab_analytics = st.tabs(["**📝 Log New Trade**", "**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])

    # --- TAB 1: LOG NEW TRADE ---
    with tab_entry:
        st.header("Log a New Trade")
        st.caption("Focus on a quick, essential entry. You can add detailed notes and screenshots later in the Playbook.")

        with st.form("trade_entry_form", clear_on_submit=True):
            st.markdown("##### ⚡ Trade Entry Details")
            col1, col2, col3 = st.columns(3) # Adjusted to 3 columns

            # Defined map, now just for Symbol selection and removed direct TV symbol use
            pairs_map_for_selection = {
                "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD", "USD/CHF": "OANDA:USDCHF",
                "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD", "USD/CAD": "FX:USDCAD"
            } # Keeping this map as a list for symbol dropdown only

            with col1:
                date_val = st.date_input("Date", dt.date.today())
                # Symbol selection will just use keys of the map, without direct TV integration here
                symbol_options = list(pairs_map_for_selection.keys()) + ["Other"]
                symbol = st.selectbox("Symbol", symbol_options, index=0) # Default to EUR/USD or first
                if symbol == "Other": symbol = st.text_input("Custom Symbol")
            with col2:
                direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
                lots = st.number_input("Size (Lots)", min_value=0.01, max_value=1000.0, value=0.10, step=0.01, format="%.2f")
            with col3:
                entry_price = st.number_input("Entry Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
                stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.00001, format="%.5f")

            st.markdown("---")
            st.markdown("##### Trade Results & Metrics")
            res_col1, res_col2, res_col3 = st.columns(3) # New 3-column row for results

            with res_col1:
                final_exit = st.number_input("Final Exit Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
                outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven", "No Trade/Study"])

            with res_col2:
                manual_pnl_input = st.number_input("Manual PnL ($)", value=0.0, format="%.2f", help="Enter the profit/loss amount manually.")

            with res_col3:
                manual_rr_input = st.number_input("Manual Risk:Reward (R)", value=0.0, format="%.2f", help="Enter the risk-to-reward ratio manually.")

            calculate_pnl_rr = st.checkbox("Calculate PnL/RR from Entry/Stop/Exit Prices", value=False,
                                           help="Check this to automatically calculate PnL and R:R based on prices entered above, overriding manual inputs.")
            st.markdown("---")
            st.markdown("##### Rationale & Tags")
            entry_rationale = st.text_area("Why did you enter this trade?", height=100)

            all_tags_list = []
            if 'Tags' in st.session_state.trade_journal.columns:
                 all_tags_list = st.session_state.trade_journal['Tags'].str.split(',').explode().dropna().str.strip().tolist()

            all_tags = sorted(list(set(all_tags_list)))

            suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged"]
            tags_selection = st.multiselect("Select Existing Tags", options=sorted(list(set(all_tags + suggested_tags))))

            new_tags_input = st.text_input("Add New Tags (comma-separated)", placeholder="e.g., strong momentum, poor entry, ...")

            submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
            if submitted:
                final_pnl, final_rr = 0.0, 0.0 # Initialize final values

                if calculate_pnl_rr:
                    # Logic adapted from Code A, robustified.
                    if stop_loss == 0.0 or entry_price == 0.0: # Cannot calculate RR or reliable PnL without proper prices
                        st.error("Entry Price and Stop Loss must be greater than 0 to calculate PnL/RR automatically.")
                        st.stop() # Prevent form submission

                    # Define multipliers for lot size (1 lot = 100,000 units usually) and pip value.
                    # This is a generic estimation; actual values depend on the pair and account currency.
                    # Assuming a base account currency (e.g., USD) for PnL
                    pip_size_for_pair = 0.0001 # Default for most pairs
                    if "JPY" in symbol.upper():
                        pip_size_for_pair = 0.01

                    # Crude Pip Value calculation (highly dependent on quote currency, can be improved)
                    # For simplicity: assuming 1 lot value is roughly 10 USD per pip for non-JPY, 1000 JPY per pip for JPY
                    # For accuracy, you'd need live exchange rates for conversion.
                    if "JPY" in symbol.upper(): # E.g., USD/JPY - if JPY is quote, pip value is USD amount per pip
                        pip_value_per_lot = 1000 # Example: 1000 JPY/pip, then converted to USD later for PnL. Or more simply directly use dollar per pip = $10 / (USD/JPY rate)
                        # To be more realistic for PnL in USD, for JPY pairs (e.g., USDJPY, GBPJPY):
                        # Pips = (exit - entry) / pip_size
                        # For USDJPY, PnL = pips * (1 USD / (current USDJPY rate)) * Lot_size_units (100000)
                        # We need an "implied" USD equivalent for PnL and RR in USD terms.
                        # Simplification for PnL, for USD denominated account:
                        # (change in points * (Lot Size / value of point in quote) ) * quote to USD rate
                        if symbol.upper() == 'USD/JPY': # Assuming PnL in USD
                            # Simplified, rough USD/JPY pips calc, value of one standard lot (100k) 1000 JPY/pip.
                            # Converted to USD: 1000 JPY / JPY/USD rate (which is 1 / USD/JPY rate) => 1000 * (USD/JPY rate). Approx $10/pip.
                            # Change in value in JPY terms
                            # Example for rough estimate
                            final_pnl = ((final_exit - entry_price) * lots * 100) if direction == "Long" else ((entry_price - final_exit) * lots * 100) # Times 100 for crude $ PnL per pip on 0.01 pips, if 1 lot=$10
                            final_pnl *= (final_exit + entry_price)/2 / (100 * pip_size_for_pair) # Correction attempt using price average. Very simplified.

                        else: # JPY as counter like EUR/JPY
                             final_pnl = ((final_exit - entry_price) * lots * 100) if direction == "Long" else ((entry_price - final_exit) * lots * 100)
                             final_pnl *= 0.007 # A very crude conversion from JPY to USD
                    else: # Non-JPY pairs like EUR/USD
                        pip_value_per_lot = 10 # Assuming roughly $10/pip for standard lot in USD terms
                        final_pnl = ((final_exit - entry_price) * pip_multiplier * lots * pip_value_per_lot / 10000) if direction == "Long" else ((entry_price - final_exit) * pip_multiplier * lots * pip_value_per_lot / 10000)


                    # For RR calculation, it's about price difference, so relative units are fine.
                    if risk_per_unit > 0.0:
                        reward_amount = abs(final_exit - entry_price)
                        final_rr = reward_amount / risk_per_unit

                        # Adjust RR sign based on actual outcome relative to direction
                        is_win = (direction == "Long" and final_exit > entry_price) or (direction == "Short" and final_exit < entry_price)
                        if not is_win and final_exit != entry_price: # Not breakeven
                             final_rr *= -1 # This implies a loss-making trade, even if a small one

                    else:
                        final_rr = 0.0


                else: # Manual PnL and RR inputs are used
                    final_pnl = manual_pnl_input
                    final_rr = manual_rr_input

                # Combine tags from multiselect and new text input
                newly_added_tags = [tag.strip() for tag in new_tags_input.split(',') if tag.strip()]
                final_tags_list = sorted(list(set(tags_selection + newly_added_tags)))

                new_trade_data = {
                    "TradeID": f"TRD-{uuid.uuid4().hex[:6].upper()}", "Date": pd.to_datetime(date_val),
                    "Symbol": symbol, "Direction": direction, "Outcome": outcome,
                    "Lots": lots, "EntryPrice": entry_price, "StopLoss": stop_loss, "FinalExit": final_exit,
                    "PnL": final_pnl, "RR": final_rr,
                    "Tags": ','.join(final_tags_list), "EntryRationale": entry_rationale,
                    "Strategy": '', "TradeJournalNotes": '', "EntryScreenshot": '', "ExitScreenshot": ''
                }
                new_df = pd.DataFrame([new_trade_data])
                st.session_state.trade_journal = pd.concat([st.session_state.trade_journal, new_df], ignore_index=True)

                if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                    ta_update_xp(st.session_state.logged_in_user, 10)
                    ta_update_streak(st.session_state.logged_in_user)
                    st.success(f"Trade {new_trade_data['TradeID']} logged successfully!")
                st.rerun()

    # --- TAB 2: TRADE PLAYBOOK ---
    with tab_playbook:
        st.header("Your Trade Playbook")
        df_playbook = st.session_state.trade_journal
        if df_playbook.empty:
            st.info("Your logged trades will appear here as playbook cards. Log your first trade to get started!")
        else:
            st.caption("Filter and review your past trades to refine your strategy and identify patterns.")

            filter_cols = st.columns([1, 1, 1, 2])
            outcome_filter = filter_cols[0].multiselect("Filter Outcome", df_playbook['Outcome'].unique(), default=df_playbook['Outcome'].unique())
            symbol_filter = filter_cols[1].multiselect("Filter Symbol", df_playbook['Symbol'].unique(), default=df_playbook['Symbol'].unique())
            direction_filter = filter_cols[2].multiselect("Filter Direction", df_playbook['Direction'].unique(), default=df_playbook['Direction'].unique())

            # Safely handle 'Tags' column for filter options
            tag_options_raw = df_playbook['Tags'].astype(str).str.split(',').explode().dropna().str.strip()
            if not tag_options_raw.empty:
                tag_options = sorted(list(set(tag_options_raw)))
            else:
                tag_options = []

            tag_filter = filter_cols[3].multiselect("Filter Tag", options=tag_options)

            filtered_df = df_playbook[
                (df_playbook['Outcome'].isin(outcome_filter)) &
                (df_playbook['Symbol'].isin(symbol_filter)) &
                (df_playbook['Direction'].isin(direction_filter))
            ]
            if tag_filter:
                # Filter where *any* of the selected tags are in the trade's tags string
                filtered_df = filtered_df[filtered_df['Tags'].astype(str).apply(lambda x: any(tag in x.split(',') for tag in tag_filter))]


            for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
                outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e", "No Trade/Study": "#58a6ff"}.get(row['Outcome'], "#30363d") # Added "No Trade/Study"
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #30363d; border-left: 8px solid {outcome_color}; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem;">
                        <h4>{row['Symbol']} <span style="font-weight: 500; color: {outcome_color};">{row['Direction']} / {row['Outcome']}</span></h4>
                        <span style="color: #8b949e; font-size: 0.9em;">{row['Date'].strftime('%A, %d %B %Y')} | {row['TradeID']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Net PnL", f"${row['PnL']:.2f}")
                    metric_cols[1].metric("R-Multiple", f"{row['RR']:.2f}R")
                    metric_cols[2].metric("Position Size", f"{row['Lots']:.2f} lots")

                    if row['EntryRationale']:
                        st.markdown(f"**Entry Rationale:** *{row['EntryRationale']}*")
                    if row['Tags']:
                        tags_list = [f"`{tag.strip()}`" for tag in str(row['Tags']).split(',') if tag.strip()]
                        if tags_list: # Ensure tags_list is not empty after filtering out empty strings
                            st.markdown(f"**Tags:** {', '.join(tags_list)}")


                    with st.expander("Journal Notes & Actions"):
                        # Using unique keys based on TradeID is critical inside a loop
                        notes = st.text_area(
                            "Trade Journal Notes",
                            value=row['TradeJournalNotes'],
                            key=f"notes_{row['TradeID']}",
                            height=150
                        )

                        action_cols = st.columns([1, 1, 4])

                        if action_cols[0].button("Save Notes", key=f"save_{row['TradeID']}", type="primary"):
                            st.session_state.trade_journal.loc[st.session_state.trade_journal['TradeID'] == row['TradeID'], 'TradeJournalNotes'] = notes
                            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                                st.toast(f"Notes for {row['TradeID']} saved!", icon="✅")
                                st.rerun() # Rerun to reflect saved notes in the textarea if the user re-opens
                            else:
                                st.error("Failed to save notes.")

                        if action_cols[1].button("Delete Trade", key=f"delete_{row['TradeID']}"):
                            index_to_drop = st.session_state.trade_journal[st.session_state.trade_journal['TradeID'] == row['TradeID']].index
                            st.session_state.trade_journal.drop(index_to_drop, inplace=True)
                            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                                st.toast(f"Trade {row['TradeID']} deleted.", icon="🗑️")
                                st.rerun()
                            else:
                                st.error("Failed to delete trade.")

                    st.markdown("---")


    # --- TAB 3: ANALYTICS DASHBOARD ---
    with tab_analytics:
        st.header("Your Performance Dashboard")
        df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()

        if df_analytics.empty:
            st.info("Complete at least one winning or losing trade to view your performance analytics.")
        else:
            # High-Level KPIs
            total_pnl = df_analytics['PnL'].sum()
            total_trades = len(df_analytics)
            wins = df_analytics[df_analytics['Outcome'] == 'Win']
            losses = df_analytics[df_analytics['Outcome'] == 'Loss']

            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            avg_win = wins['PnL'].mean() if not wins.empty else 0
            avg_loss = losses['PnL'].mean() if not losses.empty else 0
            # Ensure profit factor calculation handles zero losses gracefully (from Code A's approach)
            profit_factor = wins['PnL'].sum() / abs(losses['PnL'].sum()) if not losses.empty and losses['PnL'].sum() != 0 else (float('inf') if wins['PnL'].sum() > 0 else 0)


            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Net PnL ($)", f"${total_pnl:,.2f}", delta=f"{total_pnl:+.2f}")
            kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
            kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
            kpi_cols[3].metric("Avg. Win / Loss ($)", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}")

            st.markdown("---")

            # Visualizations
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("Cumulative PnL")
                df_analytics_sorted = df_analytics.sort_values(by='Date').copy() # Ensure sorted by date for cumulative sum
                df_analytics_sorted['CumulativePnL'] = df_analytics_sorted['PnL'].cumsum()
                fig_equity = px.area(df_analytics_sorted, x='Date', y='CumulativePnL', title="Your Equity Curve", template="plotly_dark")
                fig_equity.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                st.plotly_chart(fig_equity, use_container_width=True)

            with chart_cols[1]:
                st.subheader("Performance by Symbol")
                pnl_by_symbol = df_analytics.groupby('Symbol')['PnL'].sum().sort_values(ascending=False)
                fig_pnl_symbol = px.bar(pnl_by_symbol, title="Net PnL by Symbol", template="plotly_dark")
                fig_pnl_symbol.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", showlegend=False)
                st.plotly_chart(fig_pnl_symbol, use_container_width=True)


# =========================================================
# PERFORMANCE DASHBOARD PAGE (MT5) (Code B, adapted session state name for journal)
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
    # Helper functions (MT5 page specific, from Code B)
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
        return wins_sum / losses_sum if losses_sum != 0.0 else (np.inf if wins_sum > 0 else np.nan)

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
                    
                    # For win/loss streaks on daily data
                    def _ta_compute_streaks(df):
                        d = df.sort_values(by="date") # Ensure sorted
                        if d.empty:
                            return {"current_win": 0, "best_win": 0, "current_loss": 0, "best_loss": 0}

                        current_win_streak = 0
                        best_win_streak = 0
                        current_loss_streak = 0
                        best_loss_streak = 0

                        for pnl in d["Profit"]:
                            if pnl > 0:
                                current_win_streak += 1
                                best_win_streak = max(best_win_streak, current_win_streak)
                                current_loss_streak = 0 # Reset loss streak
                            elif pnl < 0:
                                current_loss_streak += 1
                                best_loss_streak = max(best_loss_streak, current_loss_streak)
                                current_win_streak = 0 # Reset win streak
                            else: # Break even days, typically don't count towards streak
                                current_win_streak = 0
                                current_loss_streak = 0

                        return {"current_win": current_win_streak, "best_win": best_win_streak,
                                "current_loss": current_loss_streak, "best_loss": best_loss_streak}

                    streaks = _ta_compute_streaks(daily_pnl_df_for_stats) # Use daily PnL for streaks
                    longest_win_streak = streaks["best_win"]
                    longest_loss_streak = streaks["best_loss"]


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
                        elif not profitable_assets.empty and profitable_assets.min() <= 0.0 and profitable_assets.max() <= 0.0: # All symbols were zero or loss
                             most_profitable_asset_calc = "None Profitable"
                    
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
                            day_info_text = f"{best_performing_day_name} with profit of {best_day_profit_display_html}."
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
                            day_info_text = f"{worst_performing_day_name} with loss of {worst_day_loss_display_html}."
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
                    if not daily_pnl_df_for_stats.empty:
                        df_for_chart = daily_pnl_df_for_stats.copy()
                        df_for_chart["Cumulative Profit"] = df_for_chart["Profit"].cumsum()
                        fig_equity = px.line(df_for_chart, x="date", y="Cumulative Profit", title="Equity Curve")
                        st.plotly_chart(fig_equity, use_container_width=True)

                        profit_by_symbol = df.groupby("Symbol")["Profit"].sum().reset_index()
                        fig_symbol = px.bar(profit_by_symbol, x="Symbol", y="Profit", title="Profit by Symbol")
                        st.plotly_chart(fig_symbol, use_container_width=True)

                        trade_types = df["Type"].value_counts().reset_index(name="Count")
                        trade_types.columns = ['Type', 'Count']
                        fig_type = px.pie(trade_types, names="Type", values="Count", title="Trades by Type")
                        st.plotly_chart(fig_type, use_container_width=True)
                    else:
                        st.info("Upload your MT5 data to see visualizations.")


                # ---------- Edge Finder Tab ----------
                with tab_edge:
                    st.subheader("Edge Finder")
                    st.write("Analyze your trading edge here by breaking down performance by various factors.")

                    if not df.empty:
                        analysis_options = ['Symbol', 'Type', 'Trade Duration']
                        analysis_by = st.selectbox("Analyze by:", analysis_options)

                        if analysis_by == 'Trade Duration':
                            df_for_edge = df.copy()
                            df_for_edge['Duration Bin'] = pd.cut(df_for_edge['Trade Duration'], bins=5, labels=False)
                            grouped_data = df_for_edge.groupby('Duration Bin')['Profit'].agg(['sum', 'count', 'mean']).reset_index()
                            grouped_data['Duration Bin'] = grouped_data['Duration Bin'].apply(lambda x: f"Bin {x}") # For better display
                            fig_edge = px.bar(grouped_data, x='Duration Bin', y='sum', title=f"Profit by {analysis_by}")
                            st.plotly_chart(fig_edge, use_container_width=True)
                        else:
                            grouped_data = df.groupby(analysis_by)['Profit'].agg(['sum', 'count', 'mean']).reset_index()
                            fig_edge = px.bar(grouped_data, x=analysis_by, y='sum', title=f"Profit by {analysis_by}")
                            st.plotly_chart(fig_edge, use_container_width=True)

                    else:
                        st.info("Upload your MT5 data to use the Edge Finder.")

                # ---------- Export Reports Tab ----------
                with tab_export:
                    st.subheader("Export Reports")
                    st.write("Export your trading data and reports.")

                    # Download processed dataframe
                    csv_processed = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Processed CSV",
                        data=csv_processed,
                        file_name="processed_mt5_history.csv",
                        mime="text/csv",
                    )
                    
                    # Placeholder for more complex PDF reports or interactive dashboards (e.g., as HTML)
                    st.info("Further reporting options (e.g., custom PDF reports) could be integrated here.")


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

        # --- Display Calendar HTML ---
        st.markdown(calendar_html, unsafe_allow_html=True)


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
            
            # Using daily PnL dataframe to get proper streaks
            if not daily_pnl_df_for_stats.empty:
                streaks = _ta_compute_streaks(daily_pnl_df_for_stats)
                longest_win_streak = streaks['best_win']
                longest_loss_streak = streaks['best_loss']
            else:
                longest_win_streak = 0
                longest_loss_streak = 0


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
# MANAGE MY STRATEGY PAGE (Code B, adapted journal reference)
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
                    user_data = get_user_data(username) # Use global helper
                    user_data["strategies"] = st.session_state.strategies.to_dict(orient="records")
                    save_user_data(username, user_data) # Use global helper
                    st.success("Strategy saved to your account!")
                    logging.info(f"Strategy saved for {username}: {strategy_name}")
                except Exception as e:
                    st.error(f"Failed to save strategy: {str(e)}")
                    logging.error(f"Error saving strategy for {username}: {str(e)}")
            st.success(f"Strategy '{strategy_name}' added successfully!")

    # Display Strategies
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
                            user_data = get_user_data(username) # Use global helper
                            user_data["strategies"] = st.session_state.strategies.to_dict(orient="records")
                            save_user_data(username, user_data) # Use global helper
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
    journal_df = st.session_state.trade_journal # Now using 'trade_journal'
    mt5_df = st.session_state.get('mt5_df', pd.DataFrame()) # Code B's MT5 df

    # We need to unify the columns before combining. Code A's journal is primary.
    # Align MT5 df columns if possible, otherwise use Code A's journal only.
    combined_df = journal_df.copy() # Start with Code A's journal
    
    # Example mapping: PnL and RR are common
    if not mt5_df.empty and 'Profit' in mt5_df.columns:
        mt5_temp = pd.DataFrame()
        mt5_temp['Date'] = pd.to_datetime(mt5_df['Close Time'])
        mt5_temp['PnL'] = mt5_df['Profit']
        mt5_temp['Symbol'] = mt5_df['Symbol']
        mt5_temp['Outcome'] = mt5_df['Profit'].apply(lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Breakeven'))
        # For 'RR', MT5 raw data usually doesn't have it. Will need to calculate or default.
        mt5_temp['RR'] = mt5_df.apply(lambda row: row['Profit'] / abs(row['StopLoss']) if 'StopLoss' in mt5_df.columns and row['StopLoss'] != 0 else 0, axis=1) # Simplified RR calculation

        # Pad with empty columns to match journal_cols
        for col in journal_cols:
            if col not in mt5_temp.columns:
                mt5_temp[col] = pd.Series(dtype=journal_dtypes.get(col))

        combined_df = pd.concat([combined_df, mt5_temp[journal_cols]], ignore_index=True)


    if "RR" in combined_df.columns:
        combined_df['r'] = pd.to_numeric(combined_df['RR'], errors='coerce') # Ensure RR is numeric for expectancy calculation
    
    group_cols = ["Symbol"] if "Symbol" in combined_df.columns else []

    if group_cols and 'r' in combined_df.columns and not combined_df['r'].isnull().all(): # Only proceed if 'r' column has data
        # Calculate expectancy, ensuring not to pass NaN to agg
        g = combined_df.dropna(subset=["r"]).groupby(group_cols)

        # Simplified _ta_expectancy_by_group to work with Code A's definitions where possible
        # This part requires robust functions from original Code B, which were not in provided snippet, re-creating minimal necessary
        res_data = []
        for name, group in g:
            wins_r = group[group['r'] > 0]['r']
            losses_r = group[group['r'] < 0]['r']

            winrate_calc = len(wins_r) / len(group) if len(group) > 0 else 0.0
            avg_win_r = wins_r.mean() if not wins_r.empty else 0.0
            avg_loss_r = abs(losses_r.mean()) if not losses_r.empty else 0.0 # Absolute value for loss

            expectancy_calc = (winrate_calc * avg_win_r) - ((1 - winrate_calc) * avg_loss_r)
            
            res_data.append({
                "Symbol": name if isinstance(name, str) else name[0], # Handle tuple for groupby
                "trades": len(group),
                "winrate": winrate_calc,
                "avg_win_R": avg_win_r,
                "avg_loss_R": avg_loss_r,
                "expectancy_R": expectancy_calc
            })
        
        agg = pd.DataFrame(res_data).sort_values("expectancy_R", ascending=False)
        st.write("Your refined edge profile based on logged trades:")
        st.dataframe(agg)
    else:
        st.info("Log more trades with symbols and outcomes/RR to evolve your playbook. Ensure 'RR' column has numerical data.")


import streamlit as st
import hashlib
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import logging

# --- Configuration (assumed from context, needs to be defined if not globally available) ---
# Assuming 'conn', 'c', 'journal_cols', 'journal_dtypes' are defined elsewhere in the main script.
# For the purpose of this replacement, I'll mock them or provide minimal definitions.

# Mock definitions for execution: In a real app, these would be loaded or globally defined.
try:
    # This block attempts to use Streamlit's connection for SQLITE
    conn = st.connection("sqlite_db", type="sql")
    c = conn.cursor()
except AttributeError:
    # Fallback for local execution or if st.connection is not desired/available
    # In a real app, ensure 'users.db' is handled correctly (e.g., in a data folder)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    data TEXT
)
""")
conn.commit()

journal_cols = ["Date", "Symbol", "Direction", "Entry Price", "Exit Price", "Profit/Loss", "Notes"]
journal_dtypes = {
    "Date": 'datetime64[ns]',
    "Symbol": str,
    "Direction": str,
    "Entry Price": float,
    "Exit Price": float,
    "Profit/Loss": float,
    "Notes": str
}

# Ensure basic session state vars are initialized to avoid KeyError on first run
# This needs to be robust for all session states used later, including new ones like 'redeemed_rxp'
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'account'
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None # Or 'pro_trader' for initial mock
if 'xp' not in st.session_state:
    st.session_state.xp = 0
if 'level' not in st.session_state:
    st.session_state.level = 0
if 'badges' not in st.session_state:
    st.session_state.badges = []
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'redeemed_rxp' not in st.session_state:
    st.session_state.redeemed_rxp = 0
# Initialize trade_journal if not present, especially crucial for calculations before login
if 'trade_journal' not in st.session_state:
    st.session_state.trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)


# =========================================================
# ACCOUNT PAGE
# =========================================================
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

        # --- MODIFICATION START: HELPER FUNCTION TO SAVE USER DATA ---
        def save_user_data(username):
            """
            Saves the current session state data for the logged-in user to the database.
            """
            # Create a dictionary with the user's data from the session state
            user_data = {
                "drawings": st.session_state.get("drawings", {}),
                "tools_trade_journal": st.session_state.get("tools_trade_journal", pd.DataFrame()).to_dict('records'),
                "strategies": st.session_state.get("strategies", pd.DataFrame()).to_dict('records'),
                "emotion_log": st.session_state.get("emotion_log", pd.DataFrame()).to_dict('records'),
                "reflection_log": st.session_state.get("reflection_log", pd.DataFrame()).to_dict('records'),
                "xp": st.session_state.get("xp", 0),
                "level": st.session_state.get("level", 0),
                "badges": st.session_state.get("badges", []),
                "streak": st.session_state.get("streak", 0),
                "last_journal_date": st.session_state.get("last_journal_date", None)
            }
            # Convert dictionary to a JSON string
            user_data_json = json.dumps(user_data, default=str) # Using default=str to handle any non-serializable types
            try:
                # Update the database
                c.execute("UPDATE users SET data = ? WHERE username = ?", (user_data_json, username))
                conn.commit()
                logging.info(f"Successfully saved data for user {username}")
                return True
            except Exception as e:
                logging.error(f"Failed to save data for user {username}: {e}")
                st.error("Could not save your progress. Please contact support.")
                return False
        # --- MODIFICATION END ---
        
        def handle_logout():
            """
            Clears all user-specific data from the session state upon logout.
            This modular function makes the main code cleaner and the logic reusable.
            """
            # Save final data before logging out
            if 'logged_in_user' in st.session_state:
                save_user_data(st.session_state.logged_in_user)

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
        

        # --- Main Dashboard Layout ---
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
            margin-bottom: 10px; /* MODIFICATION: Added margin-bottom */
        }
        .redeem-card {
            background-color: rgba(45, 70, 70, 0.5);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #58b3b1;
            text-align: center;
            height: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- MODIFICATION START: Row 1: KPI Cards (Now with 4 columns for RXP) ---
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
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
                <div class="kpi-label">Total Experience (XP)</div>
            </div>
            """, unsafe_allow_html=True)
        # --- NEW RXP KPI CARD ---
        with kpi_col4:
            total_xp = st.session_state.get('xp', 0)
            redeemable_xp = int(total_xp / 2) # Every 10 XP = 5 RXP
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">💎</div>
                <div class="kpi-value">{redeemable_xp:,}</div>
                <div class="kpi-label">Redeemable XP (RXP)</div>
            </div>
            """, unsafe_allow_html=True)
        # --- MODIFICATION END ---
        
        st.markdown("---")

        # --- Row 2: Progress Chart, Insights, and Badges ---
        # --- MODIFICATION: Adjusted column ratios to make chart smaller ---
        chart_col, insights_col = st.columns([1, 2])

        with chart_col:
            # --- MODIFICATION: The container is now smaller ---
            st.markdown("<h5 style='text-align: center;'>Progress to Next Level</h5>", unsafe_allow_html=True)
            total_xp = st.session_state.get('xp', 0)
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
                # Made font slightly smaller to fit better in the smaller chart
                annotations=[dict(text=f'<b>{xp_in_level}<span style="font-size:0.6em">/100</span></b>', x=0.5, y=0.5, font_size=18, showarrow=False, font_color="white")],
                margin=dict(t=20, b=20, l=20, r=20) # Added some margin
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # --- MODIFICATION: Badges moved here, into insights_col ---
        with insights_col:
            st.markdown("<h5 style='text-align: center;'>Personalized Insights & Badges</h5>", unsafe_allow_html=True)
            
            # Insights sub-column
            insight_sub_col, badge_sub_col = st.columns(2)
            
            with insight_sub_col:
                st.markdown("<h6>💡 Insights</h6>", unsafe_allow_html=True)
                streak = st.session_state.get('streak', 0)
                
                insight_message = ""
                if streak > 21:
                    insight_message = "Your journaling consistency is elite! This is a key trait of professional traders."
                elif streak > 7:
                    insight_message = "Over a week of consistent journaling! You're building a powerful habit."
                else:
                    insight_message = "Every trade journaled is a step forward. Stay consistent to build a strong foundation."
                
                st.markdown(f"<div class='insights-card'><p>{insight_message}</p></div>", unsafe_allow_html=True)

                num_trades = len(st.session_state.tools_trade_journal)
                next_milestone = ""
                if num_trades < 10:
                    next_milestone = f"Log **{10 - num_trades} more trades** to earn the 'Ten Trades' badge!"
                elif num_trades < 50:
                    next_milestone = f"You're **{50 - num_trades} trades** away from the '50 Club' badge. Keep it up!"
                else:
                    next_milestone = "The next streak badge is at 30 days. You've got this!"

                st.markdown(f"<div class='insights-card'><p>🎯 **Next Up:** {next_milestone}</p></div>", unsafe_allow_html=True)

            # Badges sub-column
            with badge_sub_col:
                st.markdown("<h6>🏆 Badges Earned</h6>", unsafe_allow_html=True)
                badges = st.session_state.get('badges', [])
                if badges:
                    for badge in badges:
                        st.markdown(f"- 🏅 {badge}")
                else:
                    st.info("No badges earned yet. Keep trading to unlock them!")
        
        # --- XP Journey Chart ---
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

        st.markdown("---")
        
        # --- MODIFICATION START: NEW SECTION "REDEEM RXP" ---
        st.subheader("💎 Redeem Your RXP")
        
        current_rxp = int(st.session_state.get('xp', 0) / 2)
        st.info(f"You have **{current_rxp:,} RXP** available to spend.")
        
        # Define redeemable items
        items = {
            "1_month_access": {"name": "1 Month Free Access", "cost": 1000, "icon": "🗓️"},
            "consultation": {"name": "30-Min Pro Consultation", "cost": 2500, "icon": "🧑‍🏫"},
            "advanced_course": {"name": "Advanced Indicators Course", "cost": 5000, "icon": "📚"}
        }

        redeem_cols = st.columns(len(items))

        for i, (item_key, item_details) in enumerate(items.items()):
            with redeem_cols[i]:
                st.markdown(
                    f"""
                    <div class="redeem-card">
                        <h3>{item_details['icon']}</h3>
                        <h5>{item_details['name']}</h5>
                        <p>Cost: <strong>{item_details['cost']:,} RXP</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"Redeem {item_details['name']}", key=f"redeem_{item_key}", use_container_width=True):
                    # Check if user has enough RXP
                    if current_rxp >= item_details['cost']:
                        # Calculate the cost in XP (RXP * 2)
                        xp_cost = item_details['cost'] * 2
                        
                        # Update session state
                        st.session_state.xp -= xp_cost
                        
                        # Save the updated data to the database
                        if save_user_data(st.session_state.logged_in_user):
                            st.success(f"Successfully redeemed '{item_details['name']}'!")
                            time.sleep(1) # Brief pause to let user see message
                            st.rerun()
                        # If saving fails, the helper function will show an error.
                            
                    else:
                        st.warning("You do not have enough RXP for this item.")
        # --- MODIFICATION END ---

        st.markdown("---")

        # --- Account Details and Actions using an Expander ---
        with st.expander("⚙️ Manage Account"):
            st.write(f"**Username**: `{st.session_state.logged_in_user}`")
            st.write("**Email**: `trader.pro@email.com` (example)")
            if st.button("Log Out", key="logout_account_page", type="primary"):
                handle_logout()
# =========================================================
# COMMUNITY TRADE IDEAS PAGE (Code B, adapted journal reference)
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
                user_data_dir = os.path.join("user_data", username) # Generic user dir. Adjust if _ta_user_dir exists and is robust
                os.makedirs(os.path.join(user_data_dir, "community_images"), exist_ok=True) # Ensure images folder exists

                idea_id = _ta_hash()
                idea_data = {
                    "Username": username,
                    "Pair": trade_pair,
                    "Direction": trade_direction,
                    "Description": trade_description,
                    "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "IdeaID": idea_id,
                    "ImagePath": None # Initialize, will update if image uploaded
                }
                if uploaded_image:
                    image_path = os.path.join(user_data_dir, "community_images", f"{idea_id}.png")
                    try:
                        with open(image_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())
                        idea_data["ImagePath"] = image_path
                        st.session_state.trade_ideas = pd.concat([st.session_state.trade_ideas, pd.DataFrame([idea_data])], ignore_index=True)
                        _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                        st.success("Trade idea shared successfully!")
                        logging.info(f"Trade idea shared by {username}: {idea_id}")
                    except Exception as e:
                        st.error(f"Failed to save image: {e}. Trade idea not saved.")
                        logging.error(f"Error saving image for trade idea: {e}")
                else:
                    st.session_state.trade_ideas = pd.concat([st.session_state.trade_ideas, pd.DataFrame([idea_data])], ignore_index=True)
                    _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                    st.success("Trade idea shared successfully!")
                    logging.info(f"Trade idea shared by {username}: {idea_id} (no image)")
                st.rerun() # Refresh to show new idea
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
        trades = len(user_d.get("trade_journal", [])) # Changed from "tools_trade_journal"
        leader_data.append({"Username": u, "Journaled Trades": trades})
    if leader_data:
        leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
        leader_df["Rank"] = leader_df.index + 1
        st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]])
    else:
        st.info("No leaderboard data yet.")
# =========================================================
# TOOLS PAGE (Code B, adapted journal references)
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
            close_price = st.number_input("Close Price", value=1.1050, step=0.0001, format="%.5f", key="pl_close_price")
        with col_calc2:
            account_currency = st.selectbox("Account Currency", ["USD", "EUR", "GBP", "JPY"], index=0, key="pl_account_currency")
            open_price = st.number_input("Open Price", value=1.1000, step=0.0001, format="%.5f", key="pl_open_price")
            trade_direction = st.radio("Trade Direction", ["Long", "Short"], key="pl_trade_direction")
        pip_multiplier = 100 if "JPY" in currency_pair else 10000
        pip_movement = abs(close_price - open_price) * pip_multiplier
        # The exchange rate and pip value calculation are placeholders and would need actual real-time data
        # For simplicity and given the example, assuming a flat $10 per pip for non-JPY for a standard lot and approx $1 for JPY.
        # This part requires a proper FX rate API for real accuracy for account_currency conversion
        if "JPY" in currency_pair:
            # Very crude approximation, assuming a USD account
            pip_value_basis = 0.01 * (100000 / ( (open_price + close_price)/2) ) # Pip value for JPY in terms of base currency * 100k units
            # Converting to USD: If USDJPY is 150, 1 pip (0.01 JPY) is 0.01/150 USD approx 0.000067 USD.
            # 1 lot (100,000 units) * 0.000067 = 6.7 USD/pip for USDJPY (varies by current rate)
            # Placeholder simplified conversion: assume for a lot, value is '10' for non-JPY. For JPY, assume average of $7 per pip per standard lot.
            pip_dollar_value_per_lot = 7.0
            # For general P/L cal, using price difference as 'points' and converting based on lot size and fixed multiplier
            profit_loss = ( (close_price - open_price) if trade_direction == "Long" else (open_price - close_price) ) * (1 / pip_size_for_pair) * position_size * pip_dollar_value_per_lot

        else: # Non-JPY pair
            # e.g., EUR/USD, if current EURUSD is 1.1, 1 pip (0.0001) means for 1 lot ($100,000 of base currency)
            # You get 10 USD per pip * lot size
            pip_dollar_value_per_lot = 10.0 # Standard approx. $10 per pip per lot
            profit_loss = ( (close_price - open_price) if trade_direction == "Long" else (open_price - close_price) ) * pip_multiplier * position_size * (pip_dollar_value_per_lot / 10) # Div 10 because pip_multiplier for non-JPY is 10000
        
        # Apply conversion if account currency is not USD, very rough, uses flat exchange rate as placeholder
        # Actual impl. would need `api.exchangerate.host` to get current rates.
        if account_currency == "EUR": profit_loss /= 1.08 # Example rate
        elif account_currency == "GBP": profit_loss /= 1.25 # Example rate
        elif account_currency == "JPY": profit_loss /= 0.0067 # Example rate (for converting USD PnL to JPY PnL)


        st.write(f"Pip Movement: {pip_movement:.2f} pips")
        st.write(f"Estimated Value Per Pip: {pip_dollar_value_per_lot * position_size:.2f} {account_currency}") # Display lot adjusted pip value
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
                pair = st.selectbox("Currency Pair", forex_pairs, key="alert_pair")
            with col2:
                price = st.number_input("Target Price", min_value=0.0, format="%.5f", key="alert_price")
            submitted = st.form_submit_button("➕ Add Alert")
            if submitted:
                new_alert = {"Pair": pair, "Target Price": price, "Triggered": False}
                st.session_state.price_alerts = pd.concat([st.session_state.price_alerts, pd.DataFrame([new_alert])], ignore_index=True)
                st.success(f"Alert added: {pair} at {price}")
                logging.info(f"Alert added: {pair} at {price}")
        st.subheader("Your Alerts")
        st.dataframe(st.session_state.price_alerts, use_container_width=True, height=220)
        # Conditional autorefresh only if alerts are being tracked (moved logic below)

        active_pairs = st.session_state.price_alerts["Pair"].unique().tolist()
        live_prices = {}
        for p in active_pairs:
            if not p:
                continue
            base, quote = p[:3], p[3:]
            try:
                # Using a generic endpoint. This might be rate-limited or require an API key in a real app.
                # 'api.exchangerate.host' provides real-time and historical currency exchange rates via a simple JSON API.
                # However, for a Streamlit app embedded within a client (not always accessible directly from Python back-end or may incur cost)
                # it's just for demo purpose. For a real trading app, it should use a reliable FX data provider.
                r = requests.get(f"https://api.exchangerate.host/latest?base={base}&symbols={quote}", timeout=6)
                data = r.json()
                price_val = data.get("rates", {}).get(quote)
                live_prices[p] = float(price_val) if price_val is not None else None
                logging.info(f"Fetched price for {p}: {live_prices[p]}")
            except Exception as e:
                live_prices[p] = None
                logging.error(f"Error fetching price for {p}: {str(e)}")

        triggered_alerts = []
        if not st.session_state.price_alerts.empty:
            for idx, row in st.session_state.price_alerts.iterrows():
                pair = row["Pair"]
                target = row["Target Price"]
                current_price = live_prices.get(pair)
                if isinstance(current_price, (int, float)):
                    # Adjusting trigger tolerance for JPY pairs (fewer decimals)
                    tolerance = 0.0005 # for non-JPY
                    if "JPY" in pair:
                        tolerance = 0.01 # for JPY

                    if not row["Triggered"] and abs(current_price - target) <= tolerance: # Use <= for including exact matches
                        st.session_state.price_alerts.at[idx, "Triggered"] = True
                        triggered_alerts.append((idx, f"{pair} reached {target:.5f} (Current: {current_price:.5f})"))
                        logging.info(f"Alert triggered: {pair} at {target}")

            if triggered_alerts:
                for idx, alert_msg in triggered_alerts:
                    st.balloons()
                    st.success(f"⚡ {alert_msg}")

        # Auto-refresh only if there are active, non-triggered alerts
        if not st.session_state.price_alerts.empty and any(not row["Triggered"] for _, row in st.session_state.price_alerts.iterrows()):
             st_autorefresh(interval=5000, key="price_alert_autorefresh") # Refresh every 5 seconds

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
                        Current: {current_price_display} &nbsp;&nbsp;&nbsp; Target: {target:.5f}
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
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=10000.0, key="rm_balance")
        with col2:
            risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, key="rm_risk_percent")
        with col3:
            stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=20.0, key="rm_stop_loss_pips")
        with col4:
            pip_value_currency = st.selectbox("Pip Value (per Lot)", ["$10 (Major USD Pairs)", "$7 (Major JPY Pairs)"], key="rm_pip_value_select")
        
        calculated_pip_value_per_lot = 10.0 # Default to $10
        if "JPY" in pip_value_currency:
            calculated_pip_value_per_lot = 7.0 # Approximated $7 per pip per lot for JPY pairs
            
        if st.button("Calculate Lot Size"):
            if stop_loss_pips <= 0 or calculated_pip_value_per_lot <= 0:
                st.error("Stop Loss (pips) and Pip Value must be positive numbers.")
            else:
                risk_amount = balance * (risk_percent / 100)
                lot_size = risk_amount / (stop_loss_pips * calculated_pip_value_per_lot)
                st.success(f"✅ Recommended Lot Size: {lot_size:.2f} lots")
                logging.info(f"Calculated lot size: {lot_size}")
        # 🔄 What-If Analyzer
        st.subheader('🔄 What-If Analyzer')
        base_equity = st.number_input('Starting Equity', value=10000.0, min_value=0.0, step=100.0, key='whatif_equity')
        risk_pct = st.slider('Risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key='whatif_risk') / 100.0
        winrate = st.slider('Win rate (%)', 10.0, 90.0, 50.0, 1.0, key='whatif_wr') / 100.0
        avg_r = st.slider('Average R multiple', 0.5, 5.0, 1.5, 0.1, key='whatif_avg_r')
        trades = st.slider('Number of trades', 10, 500, 100, 10, key='whatif_trades')
        
        # Calculate Expectancy per R-unit, assuming Loss is -1R (standard)
        E_R = (winrate * avg_r) - ((1 - winrate) * 1.0)
        
        # Avoid zero or negative base in exponentiation which could lead to NaN or inf
        if (1 + risk_pct * E_R) <= 0:
            exp_growth = 0.0 # or handle as specific warning if mathematically possible
            st.warning("Calculated growth factor is non-positive. This indicates very high risk/low expectancy, resulting in probable account wipeout.")
        else:
            exp_growth = (1 + risk_pct * E_R) ** trades
        
        st.metric('Expected Growth Multiplier', f"{exp_growth:.2f}x")
        
        alt_risk = st.slider('What if risk per trade was (%)', 0.1, 5.0, 0.5, 0.1, key='whatif_alt') / 100.0
        
        if (1 + alt_risk * E_R) <= 0:
            alt_growth = 0.0
        else:
            alt_growth = (1 + alt_risk * E_R) ** trades
            
        st.metric('Alt Growth Multiplier', f"{alt_growth:.2f}x")
        
        # 📈 Equity Projection
        # Ensure only positive or safe values are used for log scaling or direct charting to prevent errors
        if (1 + risk_pct * E_R) <= 0:
            sim_equity_base = [base_equity * (1 + risk_pct * E_R)] * (trades + 1) if (1 + risk_pct * E_R) < 0 else [base_equity] * (trades + 1)
        else:
            sim_equity_base = base_equity * (1 + risk_pct * E_R) ** np.arange(trades + 1)
        
        if (1 + alt_risk * E_R) <= 0:
            sim_equity_alt = [base_equity * (1 + alt_risk * E_R)] * (trades + 1) if (1 + alt_risk * E_R) < 0 else [base_equity] * (trades + 1)
        else:
            sim_equity_alt = base_equity * (1 + alt_risk * E_R) ** np.arange(trades + 1)

        sim = pd.DataFrame({
            'trade': list(range(trades + 1)),
            'equity_base': sim_equity_base,
            'equity_alt': sim_equity_alt,
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
        
        # Attempt to unify journal and mt5 data for session tracking, preferring Code A's fields
        current_journal_df = st.session_state.trade_journal.copy()

        # Rename relevant columns from Code A's journal to match potential 'session' concept from Code B or derive.
        # This part requires some inference of "session" for Code A's journal. We don't have direct session in Code A
        # For this demo, let's create a dummy session or assume for a simplified integration from 'Date'.
        # For simplicity, assign based on time of day in a simplified way, this isn't true "session" but can simulate
        def assign_session(timestamp):
            if pd.isna(timestamp):
                return 'Unknown'
            hour = timestamp.hour
            if 0 <= hour < 9: return 'Tokyo'
            if 8 <= hour < 17: return 'London' # Overlaps slightly with Tokyo in the start
            if 13 <= hour < 22: return 'New York' # Overlaps with London
            return 'Sydney' # Roughly from 22-7 UTC for previous day's end/new day's start

        if not current_journal_df.empty:
            current_journal_df['datetime'] = pd.to_datetime(current_journal_df['Date'])
            current_journal_df['r'] = pd.to_numeric(current_journal_df['RR'], errors='coerce')
            current_journal_df['session'] = current_journal_df['datetime'].apply(assign_session) # Assign a session based on 'Date' field from Code A

        # If MT5 data exists, convert its columns and combine
        if not mt5_df.empty:
            mt5_for_sessions = mt5_df.copy()
            mt5_for_sessions['datetime'] = pd.to_datetime(mt5_for_sessions['Close Time'])
            mt5_for_sessions['r'] = mt5_for_sessions['Profit'] # For expectancy calculation
            mt5_for_sessions['session'] = mt5_for_sessions['datetime'].apply(assign_session) # Apply session logic
            
            # Select common columns and concatenate
            cols_to_combine = ['datetime', 'r', 'session']
            if 'Symbol' in current_journal_df.columns:
                mt5_for_sessions['Symbol'] = mt5_for_sessions['Symbol'] # Use MT5 Symbol
                cols_to_combine.append('Symbol')
            
            # Only use columns that exist in both for concatenation without issues, then add others if needed
            filtered_journal = current_journal_df[[col for col in cols_to_combine if col in current_journal_df.columns]].dropna(subset=['r', 'datetime'])
            filtered_mt5 = mt5_for_sessions[[col for col in cols_to_combine if col in mt5_for_sessions.columns]].dropna(subset=['r', 'datetime'])

            df_sessions_combined = pd.concat([filtered_journal, filtered_mt5], ignore_index=True)
        else:
            df_sessions_combined = current_journal_df[['datetime', 'r', 'session']].dropna(subset=['r', 'datetime'])


        if not df_sessions_combined.empty and 'session' in df_sessions_combined.columns and not df_sessions_combined['r'].isnull().all():
            
            def _ta_expectancy_by_group_session(df_input, group_cols):
                g = df_input.dropna(subset=["r"]).groupby(group_cols)
                res = g["r"].agg(
                    trades="count",
                    winrate=lambda s: (s>0).mean(),
                    avg_win=lambda s: s[s>0].mean() if (s>0).any() else 0.0,
                    avg_loss=lambda s: -s[s<0].mean() if (s<0).any() else 0.0, # Make loss positive for formula
                    expectancy=lambda s: (s>0).mean()*(s[s>0].mean() if (s>0).any() else 0.0) - (1-(s>0).mean())*(-s[s<0].mean() if (s<0).any() else 0.0)
                ).reset_index()
                return res
            
            by_sess = _ta_expectancy_by_group_session(df_sessions_combined, ['session'])
            
            by_sess.rename(columns={'winrate': 'Win Rate', 'expectancy': 'Expectancy (R)'}, inplace=True)
            st.dataframe(by_sess, use_container_width=True)
            
            fig = px.bar(by_sess, x='session', y='Win Rate', title='Win Rate by Session', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Log trades in the 'Trading Journal' or upload MT5 trades to analyze performance by trading session. Ensure there are enough valid trade data points with a calculated 'R' value.")
        
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
            
            time_until_label = ""
            time_diff_hours = 0.0
            if not is_open:
                # Calculate time until open
                if local_hour < start:
                    time_diff_hours = start - local_hour
                else: # local_hour > end
                    time_diff_hours = (24 - local_hour) + start
                time_until_label = "Opens in"
            else:
                # Calculate time until close
                if local_hour < end:
                    time_diff_hours = end - local_hour
                else: # Should not happen if is_open is true
                    pass
                time_until_label = "Closes in"

            session_status.append({
                "Session": session["name"],
                "Status": "Open" if is_open else "Closed",
                "Local Time": local_time.strftime("%H:%M"),
                "Time Until": f"{time_diff_hours:.1f}" if is_open or time_diff_hours > 0 else "0.0"
            })
        session_df = pd.DataFrame(session_status)
        st.dataframe(session_df, use_container_width=True)
        for session in session_status:
            color = "#2ecc71" if session["Status"] == "Open" else "#e74c3c"
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                {session['Session']} Session: {session['Status']} (Local: {session['Local Time']}, {'Closes in' if session['Status'] == 'Open' else 'Opens in'} {session['Time Until']} hours)
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
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 50.0, 10.0, key="dd_pct") / 100
        recovery_pct_val = _ta_percent_gain_to_recover(drawdown_pct) # Using helper
        st.metric("Required Gain to Recover", f"{recovery_pct_val*100:.2f}%")
        st.subheader("📈 Recovery Simulation")
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_equity = st.number_input("Initial Equity ($)", min_value=100.0, value=10000.0, key="dd_initial_equity")
        with col2:
            win_rate = st.slider("Expected Win Rate (%)", 10, 90, 50, key="dd_win_rate") / 100
        with col3:
            avg_rr = st.slider("Average R:R", 0.5, 5.0, 1.5, 0.1, key="dd_avg_rr")
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, key="dd_risk_per_trade") / 100

        # Ensure calculation logic handles potential division by zero or log of non-positive numbers.
        expectancy_sim = (win_rate * avg_rr) - ((1 - win_rate) * 1.0)
        
        trades_needed = 0
        if (1 + risk_per_trade * expectancy_sim) <= 0:
            st.error("Expected growth factor is non-positive, recovery not possible under these conditions. Adjust risk or expectancy.")
            trades_needed = float('inf') # Impossible recovery
        elif drawdown_pct >= 1.0: # 100% drawdown or more, technically impossible to recover to original state purely on percentage basis
             trades_needed = float('inf')
        elif 1 / (1 - drawdown_pct) <= 0 : # Defensive check
            trades_needed = float('inf')
        elif (1 + risk_per_trade * expectancy_sim) == 1.0: # No growth if expectancy is 0, or 0 risk, etc.
             trades_needed = float('inf')
        elif expectancy_sim <= 0: # If expectation is negative, infinite trades
            trades_needed = float('inf')
        else:
            try:
                # More accurate log calculation for trades needed
                numerator = math.log(initial_equity / (initial_equity * (1 - drawdown_pct))) # Log of the recovery multiplier
                denominator = math.log(1 + risk_per_trade * expectancy_sim)
                trades_needed = math.ceil(numerator / denominator) if denominator != 0 else float('inf')

            except (ValueError, ZeroDivisionError):
                trades_needed = float('inf')

        st.write(f"Estimated Trades to Recover: {trades_needed if trades_needed != float('inf') else 'Infinite (Impossible)'}")

        sim_equity = [initial_equity * (1 - drawdown_pct)]
        if trades_needed != float('inf') :
            for _ in range(min(trades_needed + 10, 100)): # Limit max trades to prevent excessively long simulations
                if (1 + risk_per_trade * expectancy_sim) > 0:
                    sim_equity.append(sim_equity[-1] * (1 + risk_per_trade * expectancy_sim))
                else:
                    # If factor drops to zero or negative, equity becomes zero
                    sim_equity.append(0.0) # Or previous low if only positive factor allowed
                    break # Stop further growth


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
                        user_data = get_user_data(username) # Use global helper
                        user_data["reflection_log"] = st.session_state.reflection_log.to_dict(orient="records")
                        save_user_data(username, user_data) # Use global helper
                    except Exception as e:
                        logging.error(f"Error saving reflection: {str(e)}")
                st.success("Reflection logged!")
        if "reflection_log" in st.session_state and not st.session_state.reflection_log.empty:
            st.dataframe(st.session_state.reflection_log)

# =========================================================
# ZENVO ACADEMY PAGE (Code B, adapted journal and state names)
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
                # Placeholder for start learning action
                if st.button("Start Learning", key="start_forex_fundamentals"):
                    # Trigger some action or navigate to lesson page (not implemented in this code)
                    st.info("Starting 'Forex Fundamentals' module!")
                    # Award initial XP upon first start, track completion in user_data
                    if 'logged_in_user' in st.session_state:
                        username = st.session_state.logged_in_user
                        user_data = get_user_data(username)
                        completed_courses = user_data.get('completed_courses', [])
                        if "Forex Fundamentals" not in completed_courses:
                            completed_courses.append("Forex Fundamentals")
                            user_data['completed_courses'] = completed_courses
                            save_user_data(username, user_data)
                            ta_update_xp(username, 100) # Award XP
                
                # Assume a 'forex_fundamentals_progress' in session state is maintained externally for visual
                if 'forex_fundamentals_progress' not in st.session_state:
                    # Initialize progress, possibly loading from user data later
                    st.session_state.forex_fundamentals_progress = 0

                # Example of linking progress to XP, very simplified.
                st.progress(st.session_state.forex_fundamentals_progress)


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
                # Only enable if user is at least Level 1 (XP based from user_data/session_state.level)
                can_start_ta = st.session_state.get('level', 0) >= 1
                if st.button("Start Course", key="start_technical_analysis", disabled=not can_start_ta):
                    if not can_start_ta:
                        st.warning("You need to reach Level 1 to start this course!")
                    else:
                        st.info("Starting 'Technical Analysis 101' module!")
                        if 'logged_in_user' in st.session_state:
                            username = st.session_state.logged_in_user
                            user_data = get_user_data(username)
                            completed_courses = user_data.get('completed_courses', [])
                            if "Technical Analysis 101" not in completed_courses:
                                completed_courses.append("Technical Analysis 101")
                                user_data['completed_courses'] = completed_courses
                                save_user_data(username, user_data)
                                ta_update_xp(username, 150) # Award XP


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
        # Load from user_data, then update session_state for consistency
        completed_courses = []
        if 'logged_in_user' in st.session_state:
             user_data_acad = get_user_data(st.session_state.logged_in_user)
             completed_courses = user_data_acad.get('completed_courses', [])

        if completed_courses:
            for course in completed_courses:
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
        
        # Clear/reset all session states relevant to a logged-in user
        user_session_keys_to_reset = [
            'drawings', 'trade_journal', 'strategies',
            'emotion_log', 'reflection_log', 'xp', 'level', 'badges', 'streak',
            'last_journal_date', 'selected_calendar_month', 'forex_fundamentals_progress',
            'current_subpage', 'show_tools_submenu'
        ]
        for key in user_session_keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        # Re-initialize to default empty structures
        st.session_state.drawings = {}
        st.session_state.trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
        st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"])
        st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
        st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
        st.session_state.xp = 0
        st.session_state.level = 0
        st.session_state.badges = []
        st.session_state.streak = 0
        
        # The academy page doesn't have a concept of 'completed_courses' in session_state, but in DB
        # If it was in session_state for faster access, clear it too.
        if 'completed_courses' in st.session_state:
            del st.session_state['completed_courses']
        

        st.success("Logged out successfully!")
        logging.info("User logged out from Academy page")
        st.session_state.current_page = "account" # Changed to 'account' as 'login' page state doesn't directly exist
        st.rerun()
