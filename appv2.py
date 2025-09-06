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
#import scipy.stats # Uncomment if scipy is actually used
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
        border-top: 1px solid #4d7171 !important; /* Code 2 default */
        border-bottom: none !important;
        background-color: transparent !important;
        height: 1px !important;
    }

    /* Hide Streamlit branding (merged from Code 1 for comprehensive hiding) */
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; height: 0 !important; overflow: hidden !important; }

    /* Remove top padding and margins for main content */
    .css-18e3th9, .css-1d391kg {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
    }
    /* Optional: reduce padding inside Streamlit containers */
    .block-container {
        /* Code 2's general padding, adjusted slightly to be less aggressive for overall layout */
        padding-top: 1rem !important; 
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* --- Main App Styling (merged from Code 1, preserving Code 2's background) --- */
    .stApp {
        /* Code 2's background: #000000 with grid - kept */
        color: #c9d1d9; /* Code 1's text color for better readability in journal */
    }
    h1, h2, h3, h4 { /* Updated to include h4 for journal consistency */
        color: #c9d1d9 !important; /* Code 1's header color */
    }
    
    /* --- Metric Card Styling from Code 1 (for the new journal analytics) --- */
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

    /* --- Tab Styling from Code 1 (for the new journal tabs, higher specificity) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    /* Default tab styling */
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0 24px;
        transition: all 0.2s ease-in-out;
        color: #c9d1d9 !important; /* Ensure text is visible for default state */
    }
    /* Hover state for tabs */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #161b22;
        color: #58a6ff !important; /* Highlight color on hover */
    }
    /* Selected tab state */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #161b22;
        border-color: #58a6ff; /* Border color for selected tab */
        color: #c9d1d9 !important; /* Ensure text is visible for selected tab */
    }

    /* --- Styling for Markdown in Trade Playbook from Code 1 --- */
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
    # Ensure 'r' column exists before trying to access it
    if 'r' not in df.columns:
        return pd.DataFrame() # Return empty if 'r' column is missing

    g = df.dropna(subset=["r"]).groupby(group_cols)
    if g.empty: # Check if groupby resulted in an empty object
        return pd.DataFrame()

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

# Code 1's get_user_data function, added to Code 2's helpers
def get_user_data(username):
    c.execute("SELECT data FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    return json.loads(result[0]) if result and result[0] else {}

# Original _ta_daily_pnl (for MT5, expects 'datetime' and 'pnl')
def _ta_daily_pnl(df):
    if "datetime" in df.columns and "pnl" in df.columns:
        tmp = df.dropna(subset=["datetime"]).copy()
        tmp["date"] = pd.to_datetime(tmp["datetime"]).dt.date
        return tmp.groupby("date", as_index=False)["pnl"].sum()
    return pd.DataFrame(columns=["date","pnl"])

# New _ta_daily_pnl_journal for the updated journal schema (from code 1's logic, adapted for code 2's naming)
def _ta_daily_pnl_journal(df):
    if "Date" in df.columns and "PnL" in df.columns:
        tmp = df.dropna(subset=["Date"]).copy()
        tmp["date_only"] = pd.to_datetime(tmp["Date"]).dt.date
        tmp["PnL"] = pd.to_numeric(tmp["PnL"], errors='coerce').fillna(0.0) # Ensure PnL is numeric
        return tmp.groupby("date_only", as_index=False)["PnL"].sum()
    return pd.DataFrame(columns=["date_only","PnL"]) # Return a dataframe matching the structure if empty

def _ta_compute_streaks(df):
    # This function is used in Code 2's gamification for 'streak'
    # It relies on _ta_daily_pnl, which processes MT5-like data with 'datetime' and 'pnl'.
    # For user journaling streak, `ta_update_streak` handles it directly.
    # If this function was intended for the new journal, it would need modification or a separate call.
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
    """
    Show a visually appealing XP notification using Streamlit's native st.toast.
    This will appear in the top-right corner.
    """
    # st.toast automatically handles top-right positioning and a default duration (~3 seconds).
    st.toast(f"⭐ +{xp_gained} XP Earned!", icon="🎉")

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
        # Handle datetime and date objects by returning ISO format.
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        # Handle Pandas NA values by returning None, ensuring JSON compatibility.
        if pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return super().default(obj)
# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide", initial_sidebar_state="collapsed") # Updated from Code 1

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
# JOURNAL & DRAWING INITIALIZATION (UPDATED FROM CODE 1)
# =========================================================
# Initialize drawings in session_state
if "drawings" not in st.session_state:
    st.session_state.drawings = {}
    logging.info("Initialized st.session_state.drawings")

# Define journal columns and dtypes (UPDATED FROM CODE 1)
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

# Initialize trading journal with proper dtypes and robust data migration (UPDATED FROM CODE 1)
if 'tools_trade_journal' not in st.session_state:
    # Ensure logged_in_user is set for fetching data.
    # If not logged in, attempt a mock login for initial setup (consistent with Code 1's mock behavior)
    if 'logged_in_user' not in st.session_state:
        st.session_state.logged_in_user = "pro_trader" 
        try:
            c.execute("SELECT username FROM users WHERE username = ?", (st.session_state.logged_in_user,))
            if not c.fetchone():
                hashed_password = hashlib.sha256("password".encode()).hexdigest()
                initial_data_mock = json.dumps({'xp': 0, 'streak': 0, 'tools_trade_journal': []}) # Using tools_trade_journal key
                c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", 
                          (st.session_state.logged_in_user, hashed_password, initial_data_mock))
                conn.commit()
                logging.info("Mock user 'pro_trader' created for initial journal setup.")
        except Exception as e:
            logging.error(f"Error initializing mock user for journal: {e}")
            # st.warning("Could not initialize mock user for journal. Some features may not work.") # Commented to reduce noise

    user_data = get_user_data(st.session_state.logged_in_user)
    # Use 'tools_trade_journal' as the key to store journal data in user_data
    journal_data = user_data.get("tools_trade_journal", []) 
    df = pd.DataFrame(journal_data)
    
    # Safely migrate data to the new, safer schema from code 1
    # This mapping covers common old columns from code 2's journal_cols
    # and maps them to the new schema's generic columns.
    legacy_col_map = {
        "Trade ID": "TradeID", # From old code 1
        "Entry Price": "EntryPrice", "Stop Loss Price": "StopLoss", # Old Code 2's column name
        "Take Profit Price": "FinalExit", # Old Code 2's column name mapped to new FinalExit
        "PnL ($)": "PnL", # From old code 1
        "R:R": "RR", # From old code 1
        "Entry Rationale": "EntryRationale", # From old code 1
        "Trade Journal Notes": "TradeJournalNotes", # From old code 1
        "Entry Screenshot": "EntryScreenshot", # From old code 1
        "Exit Screenshot": "ExitScreenshot", # From old code 1
        
        # Mapping old code 2 specific columns to new generic ones for consolidation
        "Weekly Bias": "Strategy", 
        "Daily Bias": "Strategy", 
        "4H Structure": "TradeJournalNotes_temp", # Use temp to avoid direct overwrite
        "1H Structure": "TradeJournalNotes_temp",
        "Positive Correlated Pair & Bias": "TradeJournalNotes_temp", 
        "Potential Entry Points": "EntryRationale_temp", 
        "5min/15min Setup?": "TradeJournalNotes_temp", 
        "Entry Conditions": "EntryRationale_temp", 
        "Planned R:R": "RR_temp", # Use temp for RR that will be calculated
        "News Filter": "TradeJournalNotes_temp", 
        "Alerts": "TradeJournalNotes_temp", 
        "Concerns": "TradeJournalNotes_temp", 
        "Emotions": "TradeJournalNotes_temp", 
        "Confluence Score 1-7": "TradeJournalNotes_temp", 
        "Outcome / R:R Realised": "Outcome_temp" # Use temp for Outcome that needs parsing
    }
    df.rename(columns=legacy_col_map, inplace=True)

    # Process and consolidate 'Outcome / R:R Realised' -> 'Outcome' and 'RR'
    if "Outcome_temp" in df.columns:
        df["Outcome_original"] = df["Outcome_temp"].copy() # Keep original for debugging/migration
        def parse_outcome_rr(value):
            if isinstance(value, str) and ':' in value:
                parts = value.split(' ', 1) # Split only on first space
                outcome = parts[0] if parts[0] in ["Win", "Loss", "Breakeven", "No Trade/Study"] else "Undefined"
                rr_str = parts[1] if len(parts) > 1 else "0.0"
                try:
                    rr = float(rr_str.split(':')[1])
                except (ValueError, IndexError):
                    rr = 0.0
                return outcome, rr
            return "Undefined", 0.0 # Default if format is not as expected

        outcome_parsed, rr_parsed = zip(*df["Outcome_temp"].apply(parse_outcome_rr))
        df["Outcome"] = pd.Series(outcome_parsed)
        df["RR"] = pd.Series(rr_parsed)
        df.drop(columns=["Outcome_temp", "Outcome_original"], errors='ignore', inplace=True) # Drop temporary columns
    else:
        df["Outcome"] = "Undefined" # Default if no old outcome column
        df["RR"] = 0.0

    # Consolidate temporary rationale and notes into their final columns
    if "EntryRationale_temp" in df.columns:
        if "EntryRationale" not in df.columns:
            df["EntryRationale"] = ''
        df["EntryRationale"] = df["EntryRationale"].fillna('') + " " + df["EntryRationale_temp"].fillna('')
        df.drop(columns=["EntryRationale_temp"], errors='ignore', inplace=True)

    if "TradeJournalNotes_temp" in df.columns:
        if "TradeJournalNotes" not in df.columns:
            df["TradeJournalNotes"] = ''
        # Filter out empty strings before joining
        df["TradeJournalNotes"] = df.apply(lambda row: 
            " ".join(filter(None, [row["TradeJournalNotes"].strip() if pd.notna(row["TradeJournalNotes"]) else '', row["TradeJournalNotes_temp"].strip() if pd.notna(row["TradeJournalNotes_temp"]) else ''])),
            axis=1
        ).str.strip()
        df.drop(columns=["TradeJournalNotes_temp"], errors='ignore', inplace=True)

    # Convert any old 'Planned R:R' or similar to RR if 'RR' is still 0.0 or None
    if "RR_temp" in df.columns:
        df["RR"] = df.apply(lambda row: float(str(row["RR_temp"]).split(':')[1]) if isinstance(row["RR_temp"], str) and ':' in str(row["RR_temp"]) else row["RR"], axis=1)
        df.drop(columns=["RR_temp"], errors='ignore', inplace=True)

    # Fill any missing columns from the new schema
    for col, dtype in journal_dtypes.items():
        if col not in df.columns:
            if dtype == str: df[col] = ''
            elif 'datetime' in str(dtype): df[col] = pd.NaT
            elif dtype == float: df[col] = 0.0
            else: df[col] = None
    
    # Ensure 'Tags' column exists and is string type
    if 'Tags' not in df.columns:
        df['Tags'] = ''
    df['Tags'] = df['Tags'].astype(str).fillna('')

    st.session_state.tools_trade_journal = df[journal_cols].astype(journal_dtypes, errors='ignore')
    st.session_state.tools_trade_journal['Date'] = pd.to_datetime(st.session_state.tools_trade_journal['Date'], errors='coerce')


# Initialize temporary journal for form
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
            
            # Show XP notification - This line is correctly placed here.
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
                badge = f"Discipline Badge ({streak} Days)" # Added streak amount to badge name
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
        daily_pnl = _ta_daily_pnl(mt5_df) # This assumes MT5 df has 'datetime' and 'pnl' columns
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
                "AUD/USD": "↑ AUD weakens vs USD",
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
    
    # Using TradingView widget from Code 1, which has `height=550` or `height=560`, using 800 from Code 2
    tv_html = f"""
    <div id="tradingview_widget"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({{
        "container_id": "tradingview_widget",
        "width": "100%",
        "height": 800, /* Retaining Code 2's larger height for backtesting */
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6", /* Code 2's toolbar bg, consistent with its look */
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
                parent.window.postMessage({{action: 'save_drawings', pair: '{pair}'}}, '*'); /* Changed targetOrigin to '*' for broader compatibility, ensure security */
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
                            parent.window.postMessage({{action: 'load_drawings', pair: '{pair}', content: {json.dumps(content)}}}, '*'); /* Changed targetOrigin to '*' for broader compatibility, ensure security */
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
                        st.session_state.tools_trade_journal = pd.DataFrame(user_data.get("tools_trade_journal", [])).astype(journal_dtypes, errors='ignore')
                        st.session_state.tools_trade_journal['Date'] = pd.to_datetime(st.session_state.tools_trade_journal['Date'], errors='coerce')
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
                    # Clean up session state flags
                    if drawings_key in st.session_state:
                        del st.session_state[drawings_key]
                    if f"bt_save_trigger_{pair}" in st.session_state:
                        del st.session_state[f"bt_save_trigger_{pair}"]
            else:
                st.warning("No valid drawing content received. Ensure you have drawn on the chart.")
                logging.warning(f"No valid drawing content received for {pair}: {content}")
                # Still clean up flags if no valid content
                if drawings_key in st.session_state:
                    del st.session_state[drawings_key]
                if f"bt_save_trigger_{pair}" in st.session_state:
                    del st.session_state[f"bt_save_trigger_{pair}"]
    else:
        st.info("Sign in via the My Account tab to save/load drawings and trading journal.")
        logging.info("User not logged in, save/load drawings disabled")


    # Trading Journal Tabs (UPDATED FROM CODE 1)
    st.markdown("### 📝 Trading Journal")
    st.markdown("A streamlined interface for professional trade analysis.") # Adjusted caption

    tab_entry, tab_playbook, tab_analytics = st.tabs(["**📝 Log New Trade**", "**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])

    # --- TAB 1: LOG NEW TRADE ---
    with tab_entry:
        st.header("Log a New Trade")
        st.caption("Focus on a quick, essential entry. You can add detailed notes and screenshots later in the Playbook.")

        with st.form("trade_entry_form", clear_on_submit=True):
            st.markdown("##### ⚡ Trade Entry Details")
            col1, col2, col3 = st.columns(3)

            with col1:
                date_val = st.date_input("Date", dt.date.today())
                symbol_options = list(pairs_map.keys()) + ["Other"]
                # Use the 'pair' selected for TradingView as default
                default_symbol_index = symbol_options.index(pair) if pair in symbol_options else 0
                symbol = st.selectbox("Symbol", symbol_options, index=default_symbol_index)
                if symbol == "Other": symbol = st.text_input("Custom Symbol", value="") # Ensure custom symbol has default empty string
            with col2:
                direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
                lots = st.number_input("Size (Lots)", min_value=0.01, max_value=1000.0, value=0.10, step=0.01, format="%.2f")
            with col3:
                entry_price = st.number_input("Entry Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
                stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
            
            st.markdown("---")
            st.markdown("##### Trade Results & Metrics")
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                final_exit = st.number_input("Final Exit Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
                outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven", "No Trade/Study"])
            
            with res_col2:
                manual_pnl_input = st.number_input("Manual PnL ($)", value=0.0, format="%.2f", help="Enter the profit/loss amount manually.")
            
            with res_col3:
                manual_rr_input = st.number_input("Manual Risk:Reward (R)", value=0.0, format="%.2f", help="Enter the risk-to-reward ratio manually.")
            
            calculate_pnl_rr = st.checkbox("Calculate PnL/RR from Entry/Stop/Exit Prices", value=False, 
                                           help="Check this to automatically calculate PnL and R:R based on prices entered above, overriding manual inputs.")

            with st.expander("Add Quick Rationale & Tags (Optional)"):
                entry_rationale = st.text_area("Why did you enter this trade?", height=100)
                # Ensure tags are pulled from the correct session state variable
                # Handle cases where 'Tags' might be empty or contain non-string values
                all_tags_raw = st.session_state.tools_trade_journal['Tags'].str.split(',').explode().dropna().str.strip()
                all_tags = sorted(list(set(all_tags_raw[all_tags_raw != '']))) # Filter out empty strings
                
                suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged"]
                tags = st.multiselect("Trade Tags", options=sorted(list(set(all_tags + suggested_tags))))

            submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
            if submitted:
                final_pnl, final_rr = 0.0, 0.0

                if calculate_pnl_rr:
                    # Determine pip value based on symbol for more accurate PnL
                    # Assuming standard lot (100,000 units) and typical pip values
                    if entry_price > 0 and stop_loss > 0:
                        risk_pips_amount = abs(entry_price - stop_loss)
                    else:
                        risk_pips_amount = 0.0

                    if "JPY" in symbol:
                        pip_scale = 100 # JPY pairs move 2 decimal places, 1 pip = 0.01
                        pip_value_per_lot = 1000 # 100,000 units * 0.01 JPY/unit = 1000 JPY/pip/lot
                    else:
                        pip_scale = 10000 # Most other pairs move 4 decimal places, 1 pip = 0.0001
                        pip_value_per_lot = 1000 # 100,000 units * 0.0001 USD/unit = 10 USD/pip/lot (simplified)
                    
                    # PnL calculation based on price difference * lot size * pip value per lot
                    if outcome in ["Win", "Loss"] and final_exit > 0 and entry_price > 0:
                        price_diff_raw = (final_exit - entry_price) if direction == "Long" else (entry_price - final_exit)
                        pnl_calculated = (price_diff_raw * pip_scale) * (lots * (pip_value_per_lot / pip_scale)) # Simplified example
                    else:
                        pnl_calculated = 0.0
                    
                    if risk_pips_amount > 0:
                        profit_pips = abs(final_exit - entry_price) * pip_scale
                        risk_pips = risk_pips_amount * pip_scale
                        final_rr = (profit_pips / risk_pips) if pnl_calculated >= 0 else -(profit_pips / risk_pips)
                    else:
                        final_rr = 0.0
                    
                    final_pnl = pnl_calculated
                else:
                    final_pnl = manual_pnl_input
                    final_rr = manual_rr_input

                new_trade_data = {
                    "TradeID": f"TRD-{uuid.uuid4().hex[:6].upper()}", "Date": pd.to_datetime(date_val),
                    "Symbol": symbol, "Direction": direction, "Outcome": outcome,
                    "Lots": lots, "EntryPrice": entry_price, "StopLoss": stop_loss, "FinalExit": final_exit,
                    "PnL": final_pnl, "RR": final_rr, 
                    "Tags": ','.join(tags), "EntryRationale": entry_rationale,
                    "Strategy": '', "TradeJournalNotes": '', "EntryScreenshot": '', "ExitScreenshot": ''
                }
                new_df = pd.DataFrame([new_trade_data])
                # Ensure using the correct session state variable name: tools_trade_journal
                st.session_state.tools_trade_journal = pd.concat([st.session_state.tools_trade_journal, new_df], ignore_index=True)
                
                # Use code 2's existing save and gamification functions
                if _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal):
                    ta_update_xp(10) # Call code 2's ta_update_xp
                    ta_update_streak() # Call code 2's ta_update_streak
                    st.success(f"Trade {new_trade_data['TradeID']} logged successfully!")
                st.rerun()

    # --- TAB 2: TRADE PLAYBOOK (Replaces old Trade History from Code 2) ---
    with tab_playbook:
        st.header("Your Trade Playbook")
        # Ensure using the correct session state variable name: tools_trade_journal
        df_playbook = st.session_state.tools_trade_journal
        if df_playbook.empty:
            st.info("Your logged trades will appear here as playbook cards. Log your first trade to get started!")
        else:
            st.caption("Filter and review your past trades to refine your strategy and identify patterns.")
            
            filter_cols = st.columns([1, 1, 1, 2])
            outcome_filter = filter_cols[0].multiselect("Filter Outcome", df_playbook['Outcome'].unique(), default=df_playbook['Outcome'].unique())
            symbol_filter = filter_cols[1].multiselect("Filter Symbol", df_playbook['Symbol'].unique(), default=df_playbook['Symbol'].unique())
            direction_filter = filter_cols[2].multiselect("Filter Direction", df_playbook['Direction'].unique(), default=df_playbook['Direction'].unique())
            
            all_tags_in_df = df_playbook['Tags'].str.split(',').explode().dropna().str.strip()
            tag_options = sorted(list(set(all_tags_in_df[all_tags_raw != '']))) # Filter out empty strings
            tag_filter = filter_cols[3].multiselect("Filter Tag", options=tag_options)
            
            filtered_df = df_playbook[
                (df_playbook['Outcome'].isin(outcome_filter)) &
                (df_playbook['Symbol'].isin(symbol_filter)) &
                (df_playbook['Direction'].isin(direction_filter))
            ]
            if tag_filter:
                # Filter by checking if ANY selected tag is present in the 'Tags' string
                filtered_df = filtered_df[
                    filtered_df['Tags'].astype(str).apply(lambda x: 
                        any(tag.lower() in t.lower() for t in x.split(',') for tag in tag_filter) 
                    )
                ]

            for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
                outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e", "No Trade/Study": "#30363d"}.get(row['Outcome'], "#30363d")
                with st.container(border=True): # Using border=True for a subtle card effect
                    st.markdown(f"""
                    <div style="border-left: 8px solid {outcome_color}; border-radius: 8px 0 0 8px; padding: 0.5rem 1rem; margin-bottom: 0.5rem;">
                        <h4 style="margin:0; padding:0;">{row['Symbol']} <span style="font-weight: 500; color: {outcome_color};">{row['Direction']} / {row['Outcome']}</span></h4>
                        <span style="color: #8b949e; font-size: 0.85em;">{row['Date'].strftime('%A, %d %B %Y')}</span>
                        {/* REMOVED: st.markdown("---") was here, now removed as per requirement */}
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
                        st.markdown(f"**Tags:** {', '.join(tags_list)}")
                    
                    # --- START ADDITION: Editable Notes Section ---
                    st.subheader(f"Notes for Trade {row['TradeID']}")
                    
                    # Ensure 'TradeJournalNotes' is a string for text_area
                    current_notes = str(row.get('TradeJournalNotes', '')) if pd.notna(row.get('TradeJournalNotes')) else ''
                    
                    edited_notes = st.text_area("Detailed Notes", value=current_notes, height=150, key=f"notes_editor_{row['TradeID']}")
                    
                    save_delete_cols = st.columns(2)
                    with save_delete_cols[0]:
                        if st.button("💾 Save Notes", key=f"save_notes_{row['TradeID']}", use_container_width=True):
                            # Update the DataFrame
                            st.session_state.tools_trade_journal.loc[index, 'TradeJournalNotes'] = edited_notes
                            # Save to database
                            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal):
                                st.success(f"Notes for Trade {row['TradeID']} saved successfully!")
                            else:
                                st.error("Failed to save notes.")
                            st.rerun() # Rerun to reflect saved notes

                    with save_delete_cols[1]:
                        if st.button("🗑️ Delete Trade", key=f"delete_trade_{row['TradeID']}", type="secondary", use_container_width=True):
                            if st.session_state.logged_in_user:
                                # Remove the trade from the DataFrame
                                st.session_state.tools_trade_journal = st.session_state.tools_trade_journal.drop(index).reset_index(drop=True)
                                # Save the updated DataFrame to the database
                                if _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal):
                                    ta_update_xp(-10) # Deduct 10 XP
                                    st.success(f"Trade {row['TradeID']} deleted successfully! -10 XP.")
                                else:
                                    st.error("Failed to delete trade.")
                                st.rerun() # Rerun to update the trade list and XP
                            else:
                                st.error("Please log in to delete trades.")
                    # --- END ADDITION ---

                    # --- Original screenshot display ---
                    screenshot_cols = st.columns(2)
                    if row['EntryScreenshot']:
                        screenshot_cols[0].image(row['EntryScreenshot'], caption="Entry Screenshot", use_column_width=True)
                    if row['ExitScreenshot']:
                        screenshot_cols[1].image(row['ExitScreenshot'], caption="Exit Screenshot", use_column_width=True)
                    
                    st.markdown("---") # Keep a separator between individual trade cards
                                       # Moved from inside the initial header markdown
                                       # to here, after all trade details.


    # --- TAB 3: ANALYTICS DASHBOARD (Replaces old Analytics tab from Code 2) ---
    with tab_analytics:
        st.header("Your Performance Dashboard")
        # Ensure using the correct session state variable name: tools_trade_journal
        # Only consider trades with 'Win' or 'Loss' for performance metrics
        df_analytics = st.session_state.tools_trade_journal[st.session_state.tools_trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()
        
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
            # Ensure no division by zero for profit factor
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
                df_analytics_sorted = df_analytics.sort_values(by='Date')
                df_analytics_sorted['CumulativePnL'] = df_analytics_sorted['PnL'].cumsum()
                fig_equity = px.area(df_analytics_sorted, x='Date', y='CumulativePnL', title="Your Equity Curve", template="plotly_dark")
                fig_equity.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                st.plotly_chart(fig_equity, use_container_width=True)
                
            with chart_cols[1]:
                st.subheader("Performance by Symbol")
                pnl_by_symbol = df_analytics.groupby('Symbol')['PnL'].sum().sort_values(ascending=False).reset_index()
                fig_pnl_symbol = px.bar(pnl_by_symbol, x='Symbol', y='PnL', title="Net PnL by Symbol", template="plotly_dark")
                fig_pnl_symbol.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", showlegend=False, font_color="#c9d1d9")
                st.plotly_chart(fig_pnl_symbol, use_container_width=True)

    # --- Gamification Features (kept from Code 2, references updated) ---
    # Challenge Mode
    st.subheader("🏅 Challenge Mode")
    st.write("30-Day Journaling Discipline Challenge - Gain 300 XP for completing, XP can be exchanged for gift cards!")
    streak = st.session_state.get('streak', 0)
    progress = min(streak / 30.0, 1.0)
    st.progress(progress)
    if progress >= 1.0:
        if "challenge_30_days_completed" not in st.session_state or not st.session_state.challenge_30_days_completed:
            ta_update_xp(300) # Bonus XP for completion
            st.session_state.challenge_30_days_completed = True # Prevent multiple XP awards
        st.success("Challenge completed! Great job on your consistency.")

    # Leaderboard / Self-Competition
    st.subheader("🏆 Leaderboard - Consistency")
    users = c.execute("SELECT username, data FROM users").fetchall()
    leader_data = []
    for u, d in users:
        user_d = json.loads(d) if d else {}
        # Ensure to use 'tools_trade_journal'
        trades = len(user_d.get("tools_trade_journal", []))
        leader_data.append({"Username": u, "Journaled Trades": trades})
    if leader_data:
        leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
        leader_df["Rank"] = leader_df.index + 1
        st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]])
    else:
        st.info("No leaderboard data yet.")

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
        # Assuming percentage change in portfolio value for Sharpe, not PnL directly
        # For PnL, one might use PnL / initial_capital or similar, but percentage change is more standard for returns.
        # This is a simplification; a full Sharpe calculation needs proper equity curve or return series.
        daily_equity = daily_pnl_series.cumsum()
        returns = daily_equity.pct_change().dropna()
        
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
                    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss if total_trades else 0.0 # avg_loss is already negative
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
                            day_info_text = f"{best_performing_day_name}"
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
                            formatted_total_loss_in_parentheses_html = f"<span style='color: #d9534f;'>(-${formatted_total_loss_in_parentheses_val})</span>"
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
                            day_info_text = f"{worst_performing_day_name}"
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
                        df_for_charting = daily_pnl_df_for_stats.copy()
                        df_for_charting['Cumulative Profit'] = df_for_charting['Profit'].cumsum()

                        st.write("#### Equity Curve")
                        fig_equity_mt5 = px.line(df_for_charting, x='date', y='Cumulative Profit', title='Cumulative Profit (Equity Curve)', template='plotly_dark')
                        fig_equity_mt5.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                        st.plotly_chart(fig_equity_mt5, use_container_width=True)

                        st.write("#### Daily Profit Distribution")
                        fig_dist = px.histogram(df_for_charting[df_for_charting['Profit'] != 0], x='Profit', nbins=50, title='Daily Profit Distribution', template='plotly_dark')
                        fig_dist.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                        st.plotly_chart(fig_dist, use_container_width=True)

                        st.write("#### Profit by Symbol")
                        pnl_by_symbol_mt5 = df.groupby('Symbol')['Profit'].sum().reset_index()
                        fig_symbol_pnl_mt5 = px.bar(pnl_by_symbol_mt5, x='Symbol', y='Profit', title='Profit by Symbol', template='plotly_dark')
                        fig_symbol_pnl_mt5.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                        st.plotly_chart(fig_symbol_pnl_mt5, use_container_width=True)
                    else:
                        st.info("Upload trade data to see visualizations.")

                # ---------- Edge Finder Tab ----------
                with tab_edge:
                    st.subheader("Edge Finder")
                    st.markdown("Analyze your trading edge by various dimensions. Understand where your profits truly come from.")

                    if not df.empty:
                        st.write("#### Profitability by Trade Type")
                        pnl_by_type = df.groupby('Type')['Profit'].sum().reset_index()
                        fig_type = px.bar(pnl_by_type, x='Type', y='Profit', title='Profit by Trade Type (Buy/Sell)', template='plotly_dark')
                        fig_type.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                        st.plotly_chart(fig_type, use_container_width=True)

                        st.write("#### Profitability by Trade Duration (Hours)")
                        # Define duration bins
                        bins = [0, 1, 4, 8, 24, 72, 168, 720, np.inf] # 0-1h, 1-4h, 4-8h, 8-24h, 1-3d, 3-7d, 1-4w, >1m
                        labels = ['<1h', '1-4h', '4-8h', '8-24h', '1-3d', '3-7d', '1-4w', '>1m']
                        df['Duration_Bins'] = pd.cut(df['Trade Duration'], bins=bins, labels=labels, right=False)
                        pnl_by_duration = df.groupby('Duration_Bins')['Profit'].sum().reset_index()
                        
                        if not pnl_by_duration.empty:
                            fig_duration = px.bar(pnl_by_duration, x='Duration_Bins', y='Profit', title='Profit by Trade Duration', template='plotly_dark')
                            fig_duration.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                            st.plotly_chart(fig_duration, use_container_width=True)
                        else:
                            st.info("Not enough data to analyze profit by trade duration.")

                    else:
                        st.info("Upload trade data to find your trading edge.")

                # ---------- Export Reports Tab ----------
                with tab_export:
                    st.subheader("Export Reports")
                    st.write("Export your trading data and reports.")
                    
                    csv_export = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Raw Data CSV",
                        data=csv_export,
                        file_name="mt5_history_processed.csv",
                        mime="text/csv",
                    )
                    st.info("Download your processed MT5 history as a CSV file.")

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
        st.markdown(calendar_html, unsafe_allow_html=True)


    # Report Export & Sharing
    # Ensure `df` is available here, it should be from the uploader
    if "mt5_df" in st.session_state and not st.session_state.mt5_df.empty:
        df = st.session_state.mt5_df
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
                    c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
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
                            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
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
    journal_df = st.session_state.tools_trade_journal # Use the new journal
    mt5_df = st.session_state.get('mt5_df', pd.DataFrame())

    # Create a unified PnL column for expectancy calculation
    # Assume MT5 'Profit' maps to 'PnL' in the journal
    journal_for_exp = journal_df.copy()
    journal_for_exp['pnl'] = journal_for_exp['PnL'] # Use PnL from the new journal
    journal_for_exp['r'] = journal_for_exp['RR'] # Use RR from the new journal
    journal_for_exp['Symbol'] = journal_for_exp['Symbol'] # Keep Symbol for grouping

    mt5_for_exp = mt5_df.copy()
    if not mt5_for_exp.empty:
        mt5_for_exp['pnl'] = mt5_for_exp['Profit']
        mt5_for_exp['r'] = mt5_for_exp.apply(lambda row: row['Profit'] / (abs(row['StopLoss']) if 'StopLoss' in row and row['StopLoss'] != 0 else 1) if row['Profit'] >=0 else row['Profit'] / (abs(row['StopLoss']) if 'StopLoss' in row and row['StopLoss'] != 0 else 1) , axis=1) # Simplified R for MT5 if no direct R:R
        mt5_for_exp['Symbol'] = mt5_for_exp['Symbol']

    combined_df = pd.concat([journal_for_exp[['Symbol', 'pnl', 'r']], mt5_for_exp[['Symbol', 'pnl', 'r']]], ignore_index=True) if not mt5_for_exp.empty else journal_for_exp[['Symbol', 'pnl', 'r']]
    
    group_cols = ["Symbol"] if "Symbol" in combined_df.columns else []
    
    if group_cols and 'r' in combined_df.columns and not combined_df['r'].isnull().all():
        agg = _ta_expectancy_by_group(combined_df, group_cols).sort_values("expectancy", ascending=False)
        st.write("Your refined edge profile based on logged trades:")
        st.dataframe(agg)
    else:
        st.info("Log more trades with symbols and outcomes to evolve your playbook.")

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
    if "logged_in_user" not in st.session_state or st.session_state.logged_in_user == "pro_trader": # Also check for the mock user
        # Tabs for Sign In and Sign Up
        tab_signin, tab_signup, tab_debug = st.tabs(["🔑 Sign In", "📝 Sign Up", "🛠 Debug"])
        # --------------------------
        # SIGN IN TAB
        # --------------------------
        with tab_signin:
            st.subheader("Welcome back! Please sign in to access your account.")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
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
                            # Ensure all journal_cols are present and dtypes are correct for loaded data
                            for col in journal_cols:
                                if col not in loaded_df.columns:
                                    loaded_df[col] = pd.Series(dtype=journal_dtypes[col])
                            st.session_state.tools_trade_journal = loaded_df[journal_cols].astype(journal_dtypes, errors='ignore')
                            st.session_state.tools_trade_journal['Date'] = pd.to_datetime(st.session_state.tools_trade_journal['Date'], errors='coerce')
                        else:
                             st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)


                        if "strategies" in user_data:
                            st.session_state.strategies = pd.DataFrame(user_data["strategies"])
                        else:
                            st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"])

                        if "emotion_log" in user_data:
                            st.session_state.emotion_log = pd.DataFrame(user_data["emotion_log"])
                        else:
                            st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])

                        if "reflection_log" in user_data:
                            st.session_state.reflection_log = pd.DataFrame(user_data["reflection_log"])
                        else:
                            st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
                            
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
                new_username = st.text_input("New Username", key="reg_username")
                new_password = st.text_input("New Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
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
                'emotion_log', 'reflection_log', 'xp', 'level', 'badges', 'streak',
                'last_journal_date', 'forex_fundamentals_progress', 'challenge_30_days_completed' # Clear relevant flags too
            ]
            for key in user_session_keys:
                if key in st.session_state:
                    del st.session_state[key]

            # Re-initialize core data structures to their empty state using the NEW schemas
            st.session_state.drawings = {}
            st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
            st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"])
            st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
            st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
            st.session_state.xp = 0
            st.session_state.level = 0
            st.session_state.badges = []
            st.session_state.streak = 0
            st.session_state.last_journal_date = None

            logging.info("User logged out")
            st.session_state.current_page = "account" # Ensure redirection to the same page
            st.rerun()

        st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")
        st.markdown("This is your personal dashboard. Track your progress and manage your account.")
        st.markdown("---")
        

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

        st.subheader("📈 Progress Snapshot")

        # --- KPI Cards (Now all 4 in a single row) ---
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
                <div class="kpi-label">Total Experience</div>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi_col4:
            total_xp = st.session_state.get('xp', 0)
            # UPDATED RXP CALCULATION: Every 10 XP is 5 RXP
            redeemable_xp = (total_xp // 10) * 5 
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">💎</div>
                <div class="kpi-value">{redeemable_xp:,}</div>
                <div class="kpi-label">Redeemable XP (RXP)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")

        # --- Row 2: Progress Chart, Insights, and Badges (rearranged) ---
        chart_col, insights_col, badges_col = st.columns([0.8, 1, 0.7]) # Adjusted ratios for better visual balance

        with chart_col:
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
                annotations=[dict(text=f'<b>{xp_in_level}<span style="font-size:0.6em">/100</span></b>', x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")],
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with insights_col:
            st.markdown("<h5 style='text-align: center;'>Personalized Insights</h5>", unsafe_allow_html=True)
            streak = st.session_state.get('streak', 0)
            
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

        with badges_col: # Badges moved here
            st.markdown("<h5 style='text-align: center;'>🏆 Badges</h5>", unsafe_allow_html=True)
            badges = st.session_state.get('badges', [])
            if badges:
                for badge in badges:
                    st.markdown(f"- 🏅 {badge}")
            else:
                st.info("No badges earned yet. Keep up the great work to unlock them!")

        # --- XP Journey Chart (This part goes right after the `chart_col`, `insights_col`, `badges_col` blocks) ---
        st.markdown("<hr style='border-color: #4d7171;'>", unsafe_allow_html=True)
        st.subheader("🚀 Your XP Journey")
        journal_df = st.session_state.tools_trade_journal
        if not journal_df.empty and 'Date' in journal_df.columns:
            journal_df['Date'] = pd.to_datetime(journal_df['Date'])
            # Ensure PnL is numeric before creating xp_gained
            journal_df['PnL'] = pd.to_numeric(journal_df['PnL'], errors='coerce').fillna(0)
            
            xp_data = journal_df.sort_values(by='Date').copy()
            xp_data['xp_gained'] = 10 # Each trade logged earns 10 XP as per ta_update_xp logic in logging a trade
            xp_data['cumulative_xp'] = xp_data['xp_gained'].cumsum()
            
            fig_line = px.area(xp_data, x='Date', y='cumulative_xp', 
                                title="XP Growth Over Time (Based on Journal Entries)",
                                labels={'Date': 'Journal Entry Date', 'cumulative_xp': 'Cumulative XP'})
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(45, 70, 70, 0.3)',
                xaxis=dict(gridcolor='#4d7171', showgrid=True),
                yaxis=dict(gridcolor='#4d7171', showgrid=True),
                font_color="white"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Log your first trade in the 'Backtesting' tab to start your XP Journey!")

        st.markdown("---")
        
        # --- NEW SECTION: Redeemable Items ---
        st.subheader("🎁 Redeem Your RXP")
        st.markdown(f"You have **{redeemable_xp:,} RXP** available to redeem!")
        st.markdown("Here are some rewards you can unlock:")

        rewards = [
            {"item": "1 Month Premium Subscription", "cost": 100},
            {"item": "Exclusive Trading E-Book", "cost": 50},
            {"item": "Personalized Strategy Review Session (30 min)", "cost": 250},
            {"item": "Zenvo Academy Advanced Course Access", "cost": 150},
        ]

        for reward in rewards:
            col_reward_item, col_reward_btn = st.columns([3, 1])
            with col_reward_item:
                can_redeem = redeemable_xp >= reward["cost"]
                status_text = "✨ Available" if can_redeem else f"🚫 Needs {reward['cost'] - redeemable_xp} more RXP"
                st.markdown(f"**{reward['item']}** ({reward['cost']} RXP) - *{status_text}*")
            with col_reward_btn:
                if st.button("Redeem", key=f"redeem_{reward['item'].replace(' ', '_').lower()}", disabled=not can_redeem):
                    st.success(f"Successfully redeemed '{reward['item']}'! (RXP deduction not implemented yet)")
                    # In a real app, you would deduct RXP from user_data and save it to the DB
                    # To deduct XP (since RXP is derived from XP), you'd convert RXP cost back to XP:
                    # xp_cost = (reward['cost'] * 10) / 5
                    # user_data['xp'] = user_data.get('xp', 0) - xp_cost
                    # _ta_save_user_data(st.session_state.logged_in_user, user_data) # You'd need a save_user_data function
                    st.rerun() # Rerun to update RXP balance visually


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
                    "IdeaID": idea_id,
                    "ImagePath": None # Initialize ImagePath to None
                }
                if uploaded_image:
                    image_dir = os.path.join(user_dir, "community_images")
                    os.makedirs(image_dir, exist_ok=True) # Ensure directory exists
                    image_path = os.path.join(image_dir, f"{idea_id}.png")
                    try:
                        with open(image_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())
                        idea_data["ImagePath"] = image_path
                    except Exception as e:
                        st.warning(f"Failed to save image: {e}")
                        logging.error(f"Failed to save uploaded image for idea {idea_id}: {e}")

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
        # Sort ideas by timestamp for most recent first
        st.session_state.trade_ideas['Timestamp'] = pd.to_datetime(st.session_state.trade_ideas['Timestamp'])
        sorted_ideas = st.session_state.trade_ideas.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)

        for idx, idea in sorted_ideas.iterrows():
            with st.expander(f"{idea['Pair']} - {idea['Direction']} by {idea['Username']} ({idea['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')})"):
                st.markdown(f"Description: {idea['Description']}")
                if "ImagePath" in idea and pd.notna(idea['ImagePath']) and os.path.exists(idea['ImagePath']):
                    try:
                        st.image(idea['ImagePath'], caption="Chart Screenshot", use_column_width=True)
                    except Exception as e:
                        st.warning(f"Could not load image at {idea['ImagePath']}: {e}")
                        logging.error(f"Error loading image: {e}")
                
                # Check if the logged-in user is the author of the idea
                is_author = "logged_in_user" in st.session_state and st.session_state.logged_in_user == idea["Username"]
                
                if is_author and st.button("Delete Idea", key=f"delete_idea_{idea['IdeaID']}"):
                    # Remove the image file if it exists
                    if "ImagePath" in idea and pd.notna(idea['ImagePath']) and os.path.exists(idea['ImagePath']):
                        try:
                            os.remove(idea['ImagePath'])
                            logging.info(f"Deleted image file: {idea['ImagePath']}")
                        except Exception as e:
                            logging.error(f"Error deleting image file {idea['ImagePath']}: {e}")

                    st.session_state.trade_ideas = st.session_state.trade_ideas.drop(st.session_state.trade_ideas[st.session_state.trade_ideas['IdeaID'] == idea['IdeaID']].index).reset_index(drop=True)
                    _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                    st.success("Trade idea deleted successfully!")
                    logging.info(f"Trade idea {idea['IdeaID']} deleted by {st.session_state.logged_in_user}")
                    st.rerun()
                elif not is_author and "logged_in_user" in st.session_state: # If user is logged in but not author
                    st.caption("You can only delete your own trade ideas.")
                elif "logged_in_user" not in st.session_state:
                    st.caption("Log in to manage your trade ideas.")

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
        # Sort templates by timestamp for most recent first
        st.session_state.community_templates['Timestamp'] = pd.to_datetime(st.session_state.community_templates['Timestamp'])
        sorted_templates = st.session_state.community_templates.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)

        for idx, template in sorted_templates.iterrows():
            with st.expander(f"{template['Type']} - {template['Name']} by {template['Username']} ({template['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')})"):
                st.markdown(template['Content'])
                
                # Check if the logged-in user is the author of the template
                is_author = "logged_in_user" in st.session_state and st.session_state.logged_in_user == template["Username"]

                if is_author and st.button("Delete Template", key=f"delete_template_{template['ID']}"):
                    st.session_state.community_templates = st.session_state.community_templates.drop(st.session_state.community_templates[st.session_state.community_templates['ID'] == template['ID']].index).reset_index(drop=True)
                    _ta_save_community('templates', st.session_state.community_templates.to_dict('records'))
                    st.success("Template deleted successfully!")
                    logging.info(f"Template {template['ID']} deleted by {st.session_state.logged_in_user}")
                    st.rerun()
                elif not is_author and "logged_in_user" in st.session_state: # If user is logged in but not author
                    st.caption("You can only delete your own templates.")
                elif "logged_in_user" not in st.session_state:
                    st.caption("Log in to manage your templates.")
    else:
        st.info("No templates shared yet. Share one above!")
    # Leaderboard / Self-Competition
    st.subheader("🏆 Leaderboard - Consistency")
    users = c.execute("SELECT username, data FROM users").fetchall()
    leader_data = []
    for u, d in users:
        user_d = json.loads(d) if d else {}
        # Ensure to use 'tools_trade_journal'
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
    /* Specific tab styling for the Tools page to override generic .stTabs styling for visual differentiation */
    .stTabs [data-baseweb="tablist"] {
        background-color: transparent !important; /* Make tablist background transparent */
        border-bottom: 1px solid #4d7171; /* Add a subtle line under the tabs */
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d4646 !important; /* Specific background for inactive tabs */
        border: 1px solid #4d7171 !important;
        color: #ffffff !important;
        border-radius: 8px 8px 0 0 !important; /* Rounded top corners */
        margin-right: 5px !important;
        padding: 10px 15px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4d7171 !important; /* Darker on hover */
        color: #f0f0f0 !important;
        border-color: #58b3b1 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #58b3b1 !important; /* Highlight color for active tab */
        color: #ffffff !important;
        border-color: #58b3b1 !important;
        font-weight: bold !important;
        border-bottom: none !important; /* No bottom border for active tab */
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
            trade_direction = st.radio("Trade Direction", ["Long", "Short"], key="pl_trade_direction", horizontal=True) # Made horizontal

        # Standard Forex pip values for a standard lot (100,000 units)
        # These values are illustrative and might vary slightly by broker or exact pair.
        pip_value_mapping = {
            "EUR/USD": 10.0, "GBP/USD": 10.0, "AUD/USD": 10.0, "NZD/USD": 10.0,
            "USD/JPY": 9.09, # Approx for 110.00 USDJPY (10/110)
            "USD/CHF": 10.0, "USD/CAD": 10.0,
            "EUR/GBP": 10.0, "EUR/JPY": 9.09, # Approx
            "GBP/JPY": 9.09 # Approx
        }
        base_pip_value = pip_value_mapping.get(currency_pair, 10.0) # Default to 10 if not found

        # Calculate pips moved
        if "JPY" in currency_pair:
            pip_divisor = 0.01
        else:
            pip_divisor = 0.0001

        pips_moved = abs(close_price - open_price) / pip_divisor

        # Calculate profit/loss
        if trade_direction == "Long":
            price_change = close_price - open_price
        else: # Short
            price_change = open_price - close_price
        
        # Profit/Loss = (Price Change in Pips) * (Pip Value per Lot) * (Position Size in Lots)
        calculated_pnl = (price_change / pip_divisor) * (base_pip_value * position_size)

        st.write(f"Pip Movement: {pips_moved:.2f} pips")
        st.write(f"Estimated Pip Value per Lot ({currency_pair} in {account_currency}): ${base_pip_value:.2f}") # Display estimated pip value
        st.write(f"Potential Profit/Loss: {calculated_pnl:.2f} {account_currency}")

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
        # Displaying the alerts in a nicer format
        if not st.session_state.price_alerts.empty:
            st.dataframe(st.session_state.price_alerts.assign(
                Action=[st.button("❌ Cancel", key=f"cancel_{idx}") for idx in st.session_state.price_alerts.index]
            ), use_container_width=True, hide_index=True)

            cancel_indices = [idx for idx, row in st.session_state.price_alerts.iterrows() if st.session_state[f"cancel_{idx}"]]
            if cancel_indices:
                st.session_state.price_alerts = st.session_state.price_alerts.drop(index=cancel_indices).reset_index(drop=True)
                st.success("Selected alerts cancelled.")
                st.rerun()
        else:
            st.info("No price alerts set. Add one above to start monitoring prices.")
        
        # Auto-refresh and live price checking
        st.markdown("---")
        st.markdown("### 📊 Active Price Monitoring")
        # Auto-refresh logic (can be made optional by user)
        # st_autorefresh(interval=15000, key="price_alert_autorefresh") # Refresh every 15 seconds

        active_pairs = st.session_state.price_alerts["Pair"].unique().tolist()
        live_prices = {}
        if active_pairs:
            st.write("Fetching live prices...")
            with st.spinner("Updating prices..."):
                for p in active_pairs:
                    if not p: continue
                    try:
                        # Using an alternative API that is more robust or free
                        # forex_api_url = f"https://api.exchangerate.host/latest?base={p[:3]}&symbols={p[3:]}"
                        # Using a mock for this example as free real-time APIs are limited
                        mock_price = round(np.random.uniform(1.05000, 1.15000), 5) if "USD" in p else round(np.random.uniform(140.00, 150.00), 2)
                        live_prices[p] = mock_price
                        # For actual API, uncomment below (and find a good API key)
                        # r = requests.get(forex_api_url, timeout=6)
                        # data = r.json()
                        # price_val = data.get("rates", {}).get(p[3:])
                        # live_prices[p] = float(price_val) if price_val is not None else None
                        logging.info(f"Fetched price for {p}: {live_prices[p]}")
                    except Exception as e:
                        live_prices[p] = None
                        logging.error(f"Error fetching price for {p}: {str(e)}")
            
            triggered_alerts = []
            for idx, row in st.session_state.price_alerts.iterrows():
                pair = row["Pair"]
                target = row["Target Price"]
                current_price = live_prices.get(pair)
                
                # Check for JPY pairs (2 decimal places) vs others (4/5 decimal places)
                tolerance = 0.0005 if "JPY" not in pair else 0.01

                if isinstance(current_price, (int, float)):
                    if not row["Triggered"] and abs(current_price - target) <= tolerance: # Adjusted to be <= tolerance
                        st.session_state.price_alerts.at[idx, "Triggered"] = True
                        triggered_alerts.append((idx, f"{pair} reached {target} (Current: {current_price:.5f})"))
                        logging.info(f"Alert triggered: {pair} at {target}")
            
            if triggered_alerts:
                for idx, alert_msg in triggered_alerts:
                    st.balloons()
                    st.success(f"⚡ {alert_msg}")

            st.write("#### Live Prices & Alert Status")
            current_alerts_data = []
            for idx, row in st.session_state.price_alerts.iterrows():
                pair = row["Pair"]
                target = row["Target Price"]
                triggered = row["Triggered"]
                current_price = live_prices.get(pair)
                current_price_display = f"{current_price:.5f}" if isinstance(current_price, (int, float)) else "N/A"
                status_text = "TRIGGERED!" if triggered else "Pending"
                current_alerts_data.append([pair, f"{target:.5f}", current_price_display, status_text])
            
            if current_alerts_data:
                current_alerts_df = pd.DataFrame(current_alerts_data, columns=["Pair", "Target Price", "Current Price", "Status"])
                st.dataframe(current_alerts_df, use_container_width=True, hide_index=True)
            else:
                st.info("No active alerts to display live prices for.")

        else:
            st.info("No alerts set. Live price monitoring will start once you add an alert.")

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
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9") # Added dark theme support
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
            pip_value = st.number_input("Pip Value per Standard Lot ($)", min_value=0.01, value=10.0, key="rm_pip_value")
        if st.button("Calculate Lot Size", key="calc_lot_size"):
            if stop_loss_pips > 0 and pip_value > 0:
                risk_amount = balance * (risk_percent / 100)
                lot_size = risk_amount / (stop_loss_pips * pip_value)
                st.success(f"✅ Recommended Lot Size: {lot_size:.2f} lots")
                logging.info(f"Calculated lot size: {lot_size}")
            else:
                st.error("Stop Loss (pips) and Pip Value per Standard Lot must be greater than zero.")
        # 🔄 What-If Analyzer
        st.subheader('🔄 What-If Analyzer')
        base_equity = st.number_input('Starting Equity', value=10000.0, min_value=0.0, step=100.0, key='whatif_equity')
        risk_pct = st.slider('Risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key='whatif_risk') / 100.0
        winrate = st.slider('Win rate (%)', 10.0, 90.0, 50.0, 1.0, key='whatif_wr') / 100.0
        avg_r = st.slider('Average R multiple', 0.5, 5.0, 1.5, 0.1, key='whatif_avg_r')
        trades = st.slider('Number of trades', 10, 500, 100, 10, key='whatif_trades')

        if (winrate * avg_r - (1 - winrate)) <= -1: # Prevent division by zero or infinite growth
            st.error("The calculated expectancy (Win Rate * Avg R - (1 - Win Rate)) is too low for a positive growth projection. Please adjust your parameters.")
        else:
            E_R = winrate * avg_r - (1 - winrate) * 1.0
            if E_R <= -1: # Ensure the growth factor (1 + risk_pct * E_R) is not zero or negative when exponentiating
                st.error("Negative expectancy combined with current risk settings will lead to rapid account depletion. Please increase Win Rate or Avg R, or decrease Risk per trade.")
            else:
                exp_growth = (1 + risk_pct * E_R) ** trades
                st.metric('Expected Growth Multiplier', f"{exp_growth:.2f}x")
                alt_risk = st.slider('What if risk per trade was (%)', 0.1, 5.0, 0.5, 0.1, key='whatif_alt') / 100.0
                if (1 + alt_risk * E_R) <= 0:
                    st.error("Alternate risk settings lead to an unrealistic growth scenario. Adjust alternative risk %.")
                else:
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
                    fig.update_layout(title='Equity Projection – Base vs What-If', xaxis_title='Trade #', yaxis_title='Equity ($)', template='plotly_dark')
                    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                    st.plotly_chart(fig, use_container_width=True)
    # --------------------------
    # TRADING SESSION TRACKER
    # --------------------------
    with tabs[4]:
        st.header("🕒 Forex Market Sessions")
        st.markdown(""" Stay aware of active trading sessions to trade when volatility is highest. Each session has unique characteristics: Sydney/Tokyo for Asia-Pacific news, London for Europe, New York for US data. Overlaps like London/New York offer highest liquidity and volatility, ideal for major pairs. Track your performance per session to identify your edge. """)
        st.markdown('---')
        st.subheader('📊 Session Statistics')
        # This section expects a 'session' column, which the new journal does not natively have.
        # This part will mostly display dummy data or require manual tagging of sessions in the journal
        # or MT5 data to be useful. For now, it will show a warning if 'session' column is missing.
        mt5_df = st.session_state.get('mt5_df', pd.DataFrame())
        journal_df_for_sessions = st.session_state.tools_trade_journal.copy()
        
        # We need a 'session' column for this to work meaningfully. Let's create a dummy one if it doesn't exist
        # Or, we can guide the user to add it if they want to use this feature.
        if 'session' not in journal_df_for_sessions.columns:
            journal_df_for_sessions['session'] = 'Unknown' # Default value
        
        # Merge MT5 and Journal data for combined session analysis if both exist
        combined_session_df = pd.DataFrame()
        if not mt5_df.empty:
            mt5_df_temp = mt5_df.copy()
            mt5_df_temp['date_time'] = pd.to_datetime(mt5_df_temp['Close Time'])
            # Infer session from datetime for MT5 data. This is a simplification.
            def get_session(timestamp):
                # Basic heuristic for session (UTC times)
                if 0 <= timestamp.hour < 9: return 'Tokyo'
                if 8 <= timestamp.hour < 17: return 'London'
                if 13 <= timestamp.hour < 22: return 'New York'
                return 'Sydney' # Or Asian session overlap
            mt5_df_temp['session'] = mt5_df_temp['date_time'].apply(get_session)
            mt5_df_temp['r'] = mt5_df_temp['Profit'] # Use Profit as 'r' for simple aggregation
            
            combined_session_df = pd.concat([combined_session_df, mt5_df_temp[['session', 'r']]], ignore_index=True)

        if not journal_df_for_sessions.empty:
            journal_df_for_sessions['date_time'] = pd.to_datetime(journal_df_for_sessions['Date'])
            def get_session_journal(timestamp):
                if 0 <= timestamp.hour < 9: return 'Tokyo'
                if 8 <= timestamp.hour < 17: return 'London'
                if 13 <= timestamp.hour < 22: return 'New York'
                return 'Sydney'
            journal_df_for_sessions['session'] = journal_df_for_sessions['date_time'].apply(get_session_journal)
            journal_df_for_sessions['r'] = journal_df_for_sessions['RR'] # Use RR from new journal
            
            combined_session_df = pd.concat([combined_session_df, journal_df_for_sessions[['session', 'r']]], ignore_index=True)


        if not combined_session_df.empty and 'r' in combined_session_df.columns:
            # Drop NaN values in 'r' before grouping for correct stats
            combined_session_df_cleaned = combined_session_df.dropna(subset=['r'])
            
            if not combined_session_df_cleaned.empty:
                by_sess = combined_session_df_cleaned.groupby(['session']).agg(
                    trades=('r', 'count'),
                    winrate=('r', lambda s: (s > 0).mean()),
                    avg_r=('r', 'mean')
                ).reset_index()
                st.dataframe(by_sess, use_container_width=True)
                
                fig = px.bar(by_sess, x='session', y='winrate', title='Win Rate by Session', template='plotly_dark')
                fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid trades with R:R values to analyze performance by trading session.")
        else:
            st.info("Upload trades (MT5 or Logged Journal) to analyze performance by trading session.")

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
            
            # Calculate time until next state change
            if is_open:
                time_diff_hours = (end - local_hour) % 24
                status_msg = f"Closes in {time_diff_hours:.1f} hours"
            else:
                if local_hour < start:
                    time_diff_hours = start - local_hour
                else: # local_hour >= end (and wrapped around past midnight)
                    time_diff_hours = (24 - local_hour) + start
                status_msg = f"Opens in {time_diff_hours:.1f} hours"


            session_status.append({
                "Session": session["name"],
                "Status": "Open" if is_open else "Closed",
                "Local Time": local_time.strftime("%H:%M"),
                "Time Until": status_msg
            })
        session_df = pd.DataFrame(session_status)
        st.dataframe(session_df, use_container_width=True, hide_index=True)
        for session in session_status:
            color = "#2ecc71" if session["Status"] == "Open" else "#e74c3c"
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                {session['Session']} Session: {session['Status']} (Local: {session['Local Time']}, {session['Time Until']})
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
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 50.0, 10.0, key="dd_drawdown_pct") / 100
        recovery_pct = _ta_percent_gain_to_recover(drawdown_pct)
        st.metric("Required Gain to Recover", f"{recovery_pct*100:.2f}%")
        st.subheader("📈 Recovery Simulation")
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_equity = st.number_input("Initial Equity ($)", min_value=100.0, value=10000.0, key="dd_initial_equity")
        with col2:
            win_rate = st.slider("Expected Win Rate (%)", 10, 90, 50, key="dd_win_rate") / 100
        with col3:
            avg_rr = st.slider("Average R:R", 0.5, 5.0, 1.5, 0.1, key="dd_avg_rr")
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, key="dd_risk_per_trade") / 100
        
        expected_r_per_trade = (win_rate * avg_rr) - ((1 - win_rate) * 1.0) # Assuming 1R loss per losing trade

        if expected_r_per_trade <= 0:
            st.error("Expected R per trade is non-positive. Recovery is not possible with these parameters. Adjust Win Rate, Average R:R, or Risk per Trade.")
            trades_needed = float('inf')
        else:
            try:
                # Formula: Final_Equity = Initial_Equity * (1 + Risk_per_trade * Expected_R)^Trades
                # We want Final_Equity >= Initial_Equity (meaning current_equity * (1 + recovery_pct) )
                # current_equity * (1 + recovery_pct) = current_equity * (1 + risk_per_trade * Expected_R)^Trades
                # (1 + recovery_pct) = (1 + risk_per_trade * Expected_R)^Trades
                # log(1 + recovery_pct) = Trades * log(1 + risk_per_trade * Expected_R)
                # Trades = log(1 + recovery_pct) / log(1 + risk_per_trade * Expected_R)
                
                # The recovery is from the *current* equity in drawdown, not the initial.
                # Let current_equity = initial_equity * (1 - drawdown_pct)
                # We need to reach initial_equity from current_equity.
                # Factor to recover = initial_equity / current_equity = 1 / (1 - drawdown_pct)
                # So we need (1 + risk_per_trade * expected_r)^trades = 1 / (1 - drawdown_pct)
                # trades_needed = log(1 / (1 - drawdown_pct)) / log(1 + risk_per_trade * expected_r)

                if (1 + risk_per_trade * expected_r_per_trade) <= 0:
                     st.error("The growth factor (1 + risk per trade * expected R) is zero or negative. Cannot simulate recovery with these parameters.")
                     trades_needed = float('inf')
                else:
                    trades_needed = math.ceil(math.log(1 / (1 - drawdown_pct)) / math.log(1 + risk_per_trade * expected_r_per_trade))
            except (ValueError, ZeroDivisionError):
                st.error("Cannot calculate trades needed. Ensure expected R per trade and drawdown percentage are valid.")
                trades_needed = float('inf')

        if trades_needed == float('inf'):
            st.write("Estimated Trades to Recover: Cannot be achieved with current parameters.")
        else:
            st.write(f"Estimated Trades to Recover: {trades_needed}")

        sim_equity = [initial_equity * (1 - drawdown_pct)]
        # Simulate a reasonable number of trades, capping to avoid excessive calculations
        max_sim_trades = min(trades_needed + 50, 500) if trades_needed != float('inf') else 100
        
        for _ in range(max_sim_trades):
            if (1 + risk_per_trade * expected_r_per_trade) > 0: # Prevent negative equity growth
                next_equity = sim_equity[-1] * (1 + risk_per_trade * expected_r_per_trade)
                sim_equity.append(next_equity)
            else:
                sim_equity.append(sim_equity[-1]) # Stop growth if factor is bad

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(sim_equity))), y=sim_equity, mode='lines', name='Equity'))
        fig.add_hline(y=initial_equity, line_dash="dash", line_color="green", annotation_text="Initial Equity")
        fig.update_layout(title='Drawdown Recovery Simulation', xaxis_title='Trade #', yaxis_title='Equity ($)', template='plotly_dark')
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9")
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
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
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
