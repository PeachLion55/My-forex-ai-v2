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

# --- Gridline background settings ---
grid_color = "#58b3b1"  # gridline color
grid_opacity = 0.18      # 0.0 (transparent) to 1.0 (solid)
grid_size = 40          # distance between gridlines in px

# Convert HEX to RGB
r = int(grid_color[1:3], 16)
g = int(grid_color[3:5], 16)
b = int(grid_color[5:7], 16)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #000000;  /* black background */
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
logo = logo.resize((60, 50))  # adjust width/height as needed

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

# Display content based on current page
if st.session_state.current_page == 'tools':
    st.title("Tools")
    st.markdown('---')
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
    selected_tool = st.selectbox("Select a Tool", tools_options, key="tool_select")
    st.write(f"You selected: {selected_tool}")
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
        {"Currency": "USD", "Current": "4.50%", "Previous": "4.75%", "Changed": "12-18-2024", "Next Meeting": "2025-09-18"},
        {"Currency": "GBP", "Current": "4.00%", "Previous": "4.25%", "Changed": "08-07-2025", "Next Meeting": "2025-09-19"},
        {"Currency": "EUR", "Current": "2.15%", "Previous": "2.40%", "Changed": "06-05-2025", "Next Meeting": "2025-09-12"},
        {"Currency": "JPY", "Current": "0.50%", "Previous": "0.25%", "Changed": "01-24-2025", "Next Meeting": "2025-09-20"},
        {"Currency": "AUD", "Current": "3.60%", "Previous": "3.85%", "Changed": "08-12-2025", "Next Meeting": "2025-09-24"},
        {"Currency": "CAD", "Current": "2.75%", "Previous": "3.00%", "Changed": "03-12-2025", "Next Meeting": "2025-09-04"},
        {"Currency": "NZD", "Current": "3.25%", "Previous": "3.50%", "Changed": "05-28-2025", "Next Meeting": "2025-10-09"},
        {"Currency": "CHF", "Current": "0.00%", "Previous": "0.25%", "Changed": "06-19-2025", "Next Meeting": "2025-09-26"},
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

# Display events in side-by-side boxes
for ev in forex_high_impact_events:
    positive_html = "<br>".join([f"{k}: {v}" for k, v in ev["impact_positive"].items()])
    negative_html = "<br>".join([f"{k}: {v}" for k, v in ev["impact_negative"].items()])

    st.markdown(
        f"""
        <div style="display: flex; gap: 20px; margin-bottom: 15px;">
            <div style="background-color: #000000; color: #ffffff; padding: 15px; border-radius: 8px; flex: 1;">
                <strong>{ev['event']}</strong><br>
                <em>What it is:</em> {ev['description']}<br>
                <em>Why it matters:</em> {ev['why_it_matters']}<br><br>
                <strong>Positive Impact →</strong><br>
                {positive_html}
            </div>
            <div style="background-color: #000000; color: #ffffff; padding: 15px; border-radius: 8px; flex: 1;">
                <strong>Negative Impact →</strong><br>
                {negative_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Backtesting page
if st.session_state.current_page == 'backtesting':
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
    st.markdown("""
    <style>
    .metric-box {{
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }}
    .metric-box.positive {{
        background-color: #d4edda;
        color: #155724;
    }}
    .metric-box.negative {{
        background-color: #f8d7da;
        color: #721c24;
    }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    📊 Performance Dashboard
    Upload your MT5 trading history CSV to analyze your trading performance
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
    st.markdown(""" Trading psychology is critical to success. This section helps you track your emotions, reflect on your mindset, and maintain discipline through structured journaling and analysis. """)
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
        ta_update_xp(100)  # Bonus XP for completion

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
        tab_signin, tab_signup = st.tabs(["🔑 Sign In", "📝 Sign Up"])

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
                            initial_data = json.dumps({"xp": 0, "level": 0, "badges": [], "streak": 0})
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
    else:
        # --------------------------
        # LOGGED-IN USER VIEW
        # --------------------------
        st.subheader(f"Welcome, {st.session_state.logged_in_user}!")

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

elif st.session_state.current_page == 'tools':
    st.title("🛠 Tools")
    if st.session_state.current_subpage is None:
        st.write("Please select a tool from the sidebar.")
    elif st.session_state.current_subpage == 'profit_loss':
        st.header("💰 Profit / Loss Calculator")
        st.markdown("Calculate your potential profit or loss for a trade.")
        st.write('---')
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            currency_pair =st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY"], key="pl_currency_pair")
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
    elif st.session_state.current_subpage == 'alerts':
        st.header("⏰ Price Alerts")
        st.markdown("Set price alerts for your favourite forex pairs and get notified when the price hits your target.")
        st.write('---')
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
                        Current: {current_price_display}    Target: {target}
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
    elif st.session_state.current_subpage == 'correlation':
        st.header("📊 Currency Correlation Heatmap")
        st.markdown("Understand how forex pairs move relative to each other.")
        st.write('---')
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
    elif st.session_state.current_subpage == 'risk_mgmt':
        st.header("🛡️ Risk Management Calculator")
        st.markdown(""" Proper position sizing keeps your account safe. Risk management is crucial to long-term trading success. It helps prevent large losses, preserves capital, and allows you to stay in the game during drawdowns. Always risk no more than 1-2% per trade, use stop losses, and calculate position sizes based on your account size and risk tolerance. """)
        st.write('---')
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
    elif st.session_state.current_subpage == 'sessions':
        st.header("🕒 Forex Market Sessions")
        st.markdown(""" Stay aware of active trading sessions to trade when volatility is highest. Each session has unique characteristics: Sydney/Tokyo for Asia-Pacific news, London for Europe, New York for US data. Overlaps like London/New York offer highest liquidity and volatility, ideal for major pairs. Track your performance per session to identify your edge. """)
        st.write('---')
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
    elif st.session_state.current_subpage == 'drawdown':
        st.header("📉 Drawdown Recovery Planner")
        st.markdown(""" Plan your recovery from a drawdown. Understand the percentage gain required to recover losses and simulate recovery based on your trading parameters. """)
        st.write('---')
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
    elif st.session_state.current_subpage == 'checklist':
        st.header("✅ Pre-Trade Checklist")
        st.markdown(""" Ensure discipline by running through this checklist before every trade. A structured approach reduces impulsive decisions and aligns trades with your strategy. """)
        st.write('---')
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
    elif st.session_state.current_subpage == 'premarket':
        st.header("📅 Pre-Market Checklist")
        st.markdown(""" Build consistent habits with pre-market checklists and end-of-day reflections. These rituals help maintain discipline and continuous improvement. """)
        st.write('---')
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

elif st.session_state.current_page == 'settings':
    st.title("Settings")
    st.markdown('---')
    # Add settings content if needed

# Close database connection
conn.close()
