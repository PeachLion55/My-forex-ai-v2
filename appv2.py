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
st.set_page_config(page_title="Forex Dashboard", layout="wide")

# ------------------ Custom CSS ------------------
st.markdown("""
    <style>
    /* Centered vertical navigation bar */
    div[data-baseweb="tab-list"] {
        position: fixed !important;
        left: 24px;
        top: 50%;
        transform: translateY(-50%);
        width: 220px;
        max-height: calc(100vh - 48px);
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 16px;
        background: linear-gradient(180deg, #121212 0%, #0a0a0a 100%);
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        z-index: 1000;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #444 #1a1a1a;
    }
    div[data-baseweb="tab-list"]::-webkit-scrollbar {
        width: 8px;
    }
    div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 4px;
    }
    div[data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    div[data-baseweb="tab-list"] button {
        width: 100% !important;
        text-align: left;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        font-size: 14px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        justify-content: flex-start;
        color: #fff !important;
        background: linear-gradient(90deg, #c76b12, #ff9a3d) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border: none !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(90deg, #ffd37a, #ff8a00) !important;
        color: #000 !important;
        font-weight: 600;
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(0,0,0,0.5);
    }
    div[data-baseLectbox select:focus, .stDataFrame .stNumberInput input:focus, .stDataFrame .stDateInput input:focus {
        background-color: #2a2a2a;
        border-color: #FFD700;
        box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
        outline: none;
    }
    </style>
""", unsafe_allow_html=True)

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

# Initialize selected main tab
if "selected_main_tab" not in st.session_state:
    st.session_state.selected_main_tab = "Dashboard"

# =========================================================
# NAVIGATION
# =========================================================
main_tabs = [
    "Dashboard", "Markets", "Calendar",
    "Analytics", "Calculator", "Mentai",
    "Backtest", "Trades", "Add Trade"
]
tools_subtabs = [
    "Profit/Loss Calculator", "Price Alerts", "Currency Correlation Heatmap",
    "Risk Management Calculator", "Trading Session Tracker", "Drawdown Recovery Planner",
    "Pre-Trade Checklist", "Pre-Market Checklist"
]

# Determine which tabs to show in the navigation bar
nav_tabs = main_tabs
if st.session_state.selected_main_tab == "Calculator":
    nav_tabs = main_tabs + tools_subtabs

# Create the navigation tabs
tab_instances = st.tabs(nav_tabs)

# Map tab names to their instances
tab_map = {name: tab_instances[i] for i, name in enumerate(nav_tabs)}

# Update selected main tab when a main tab is clicked
for tab_name in main_tabs:
    if tab_name in tab_map:
        with tab_map[tab_name]:
            if st.session_state.selected_main_tab != tab_name:
                st.session_state.selected_main_tab = tab_name
                st.rerun()

# =========================================================
# TAB 1: Dashboard
# =========================================================
with tab_map.get("Dashboard", st.container()):
    st.title("Dashboard")
    st.write("Overview and key metrics")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
    with col2:
        st.info("See the **Backtest** tab for live charts + detailed news.")
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
    st.markdown("""
    Interest rates are a key driver in forex markets. Higher rates attract foreign capital, strengthening the currency. 
    Lower rates can weaken it. Monitor changes and forward guidance from central banks for trading opportunities.
    Below are current rates, with details on recent changes, next meeting dates, and market expectations.
    """)
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
                    <div class="card">
                        <div style="
                            background-color:{color};
                            border-radius:10px;
                            padding:15px;
                            text-align:center;
                            color:white;
                        ">
                            <h3 style="margin: 0 0 6px 0;">{rate['Currency']}</h3>
                            <p style="margin: 2px 0;"><b>Current:</b> {rate['Current']}</p>
                            <p style="margin: 2px 0;"><b>Previous:</b> {rate['Previous']}</p>
                            <p style="margin: 2px 0;"><b>Changed On:</b> {rate['Changed']}</p>
                            <p style="margin: 2px 0;"><b>Next Meeting:</b> {rate['Next Meeting']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
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
# TAB 2: Markets
# =========================================================
with tab_map.get("Markets", st.container()):
    st.title("Markets")
    st.write("Live market data")
    st.caption("Live TradingView chart for backtesting and trading journal for the selected pair.")
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
    <div class="tradingview-widget-container" style="height:780px; width:100%">
      <div id="tradingview_chart_{tv_symbol.replace(':','_')}" style="height:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      console.log("Initializing TradingView widget for {tv_symbol}");
      try {{
        const widget = new TradingView.widget({{
          "autosize": true,
          "symbol": "{tv_symbol}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "hide_top_toolbar": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "save_image": true,
          "container_id": "tradingview_chart_{tv_symbol.replace(':','_')}"
        }});
        widget.onChartReady(() => {{
          console.log("Chart ready for {tv_symbol}");
          const chart = widget.activeChart();
          window.chart = chart;
          const initialContent = {initial_content};
          if (Object.keys(initialContent).length > 0) {{
            console.log("Loading initial content:", initialContent);
            chart.setContent(initialContent);
          }} else {{
            console.log("No initial content to load for {tv_symbol}");
          }}
        }});
      }} catch (error) {{
        console.error("Error initializing TradingView widget:", error);
      }}
      </script>
    </div>
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
                try {{
                  console.log("Attempting to save drawings for {pair}");
                  window.parent.chart.getContent((content) => {{
                    console.log("Drawing content received:", content);
                    window.parent.postMessage({{
                      type: 'streamlit:setComponentValue',
                      value: content,
                      dataType: 'json',
                      key: 'bt_drawings_key_{pair}'
                    }}, '*');
                  }});
                }} catch (error) {{
                  console.error("Error saving drawings:", error);
                }}
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
                            console.log("Loading drawings for {pair}:", {json.dumps(content)});
                            window.parent.chart.setContent({json.dumps(content)});
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

# =========================================================
# TAB 3: Calendar
# =========================================================
with tab_map.get("Calendar", st.container()):
    st.title("Calendar")
    st.write("Economic events and schedules")
    st.markdown("### 🗓️ Upcoming Economic Events")
    if 'selected_currency_1' not in st.session_state:
        st.session_state.selected_currency_1 = None
    if 'selected_currency_2' not in st.session_state:
        st.session_state.selected_currency_2 = None
    uniq_ccy = sorted(set(list(econ_df["Currency"].unique()) + list(df_news["Currency"].unique())))
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        currency_filter_1 = st.selectbox("Primary currency to highlight", options=["None"] + uniq_ccy, key="cal_curr_3")
        st.session_state.selected_currency_1 = None if currency_filter_1 == "None" else currency_filter_1
    with col_filter2:
        currency_filter_2 = st.selectbox("Secondary currency to highlight", options=["None"] + uniq_ccy, key="cal_curr_4")
        st.session_state.selected_currency_2 = None if currency_filter_2 == "None" else currency_filter_2
    def highlight_currency(row):
        styles = [''] * len(row)
        if st.session_state.selected_currency_1 and row['Currency'] == st.session_state.selected_currency_1:
            styles = ['background-color: #171447; color: white' if col == 'Currency' else 'background-color: #171447' for col in row.index]
        if st.session_state.selected_currency_2 and row['Currency'] == st.session_state.selected_currency_2:
            styles = ['background-color: #471414; color: white' if col == 'Currency' else 'background-color: #471414' for col in row.index]
        return styles
    st.dataframe(econ_df.style.apply(highlight_currency, axis=1), use_container_width=True, height=360)

# =========================================================
# TAB 4: Analytics
# =========================================================
with tab_map.get("Analytics", st.container()):
    st.title("Analytics")
    st.write("Performance and trade analytics")
    st.markdown("""
        <style>
        .title-container {
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 20px;
        }
        .title {
            font-size: 24px;
            font-weight: bold;
            color: #FFFFFF;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 16px;
            color: #FFFFFF;
        }
        .metrics-row {
            display: flex;
            flex-wrap: nowrap;
            gap: 20px;
            padding: 15px 0;
            justify-content: space-between;
        }
        .metric-card {
            flex: 1;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            background: linear-gradient(180deg, #2a2a2a, #1a1a1a);
            border: 1px solid #3a3a3a;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            min-height: 120px;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .metric-title {
            font-size: 16px;
            font-weight: 600;
            color: #FFD700;
            margin-bottom: 12px;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
        }
        .positive .metric-value {
            color: #00ff00;
        }
        .negative .metric-value {
            color: #ff0000;
        }
        .neutral .metric-value {
            color: white;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            color: white;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .upload-container {
            background: #1a1a1a !important;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            border: 1px solid #3a3a3a;
        }
        .stFileUploader > div > div > div {
            background-color: transparent !important;
            border-radius: 8px;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="title-container">
            <div class="title">📊 MT5 Performance Dashboard</div>
            <div class="subtitle">Upload your MT5 trading history CSV to analyze your trading performance</div>
        </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose your MT5 History CSV file",
            type=["csv"],
            help="Upload a CSV file exported from MetaTrader 5 containing your trading history."
        )
        st.markdown('</div>', unsafe_allow_html=True)
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
                                    <div class="metric-card {style}">
                                        <div class="metric-title">{title}</div>
                                        <div class="metric-value">{value}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    # Visualizations
                    st.markdown('<div class="section-title">📊 Profit by Instrument</div>', unsafe_allow_html=True)
                    profit_symbol = df.groupby("Symbol")["Profit"].sum().reset_index()
                    fig_symbol = px.bar(
                        profit_symbol, x="Symbol", y="Profit", color="Profit",
                        title="Profit by Instrument", template="plotly_white",
                        color_continuous_scale=px.colors.diverging.Tealrose
                    )
                    fig_symbol.update_layout(
                        title_font_size=18, title_x=0.5,
                        font_color="#333333"
                    )
                    st.plotly_chart(fig_symbol, use_container_width=True)
                    st.markdown('<div class="section-title">🔎 Trade Distribution</div>', unsafe_allow_html=True)
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
                    st.success("✅ MT5 Performance Dashboard Loaded Successfully!")
                    _ta_update_xp(50)
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
        if "timeframe" in df.columns: group_cols.append("timeframe")
        if "symbol" in df.columns: group_cols.append("symbol")
        if "setup" in df.columns: group_cols.append("setup")
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
            "Total Trades", "Win Rate", "Avg R", "Profit Factor",
            "Max Drawdown (PnL)", "Best Symbol", "Worst Symbol",
            "Best Timeframe", "Worst Timeframe"
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
                "Win Rate": _ta_human_pct((df["r"]>0).mean()) if "r" in df.columns else "—",
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
        if "timeframe" in df.columns: group_cols.append("timeframe")
        if "symbol" in df.columns: group_cols.append("symbol")
        if "setup" in df.columns: group_cols.append("setup")
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
            <h1>MT5 Performance Report</h1>
            <p><b>Total Trades:</b> {total_trades}</p>
            <p><b>Win Rate:</b> {win_rate:.2f}%</p>
            <p><b>Net Profit:</b> ${net_profit:,.2f}</p>
            <p><b>Profit Factor:</b> {profit_factor}</p>
            <p><b>Biggest Win:</b> ${biggest_win:,.2f}</p>
            <p><b>Biggest Loss:</b> ${biggest_loss:,.2f}</p>
            <p><b>Longest Win Streak:</b> {longest_win_streak}</p>
            <p><b>Longest Loss Streak:</b> {longest_loss_streak}</p>
            <p><b>Avg Trade Duration:</b> {avg_trade_duration:.2f}h</p>
            <p><b>Total Volume:</b> {total_volume:,.2f}</p>
            <p><b>Avg Volume:</b> {avg_volume:.2f}</p>
            <p><b>Profit / Trade:</b> ${profit_per_trade:.2f}</p>
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
# TAB 5: Calculator
# =========================================================
with tab_map.get("Calculator", st.container()):
    st.title("Calculator")
    st.write("Position size and risk calculations")
    if st.session_state.selected_main_tab == "Calculator":
        tools_subtab_instances = st.tabs(tools_subtabs)
        tools_subtab_map = {name: tools_subtab_instances[i] for i, name in enumerate(tools_subtabs)}
    else:
        tools_subtab_map = {name: st.container() for name in tools_subtabs}

    with tools_subtab_map.get("Profit/Loss Calculator", st.container()):
        st.header("💰 Profit / Loss Calculator")
        st.markdown("Calculate your potential profit or loss for a trade.")
        st.write('---')
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
            (0.0001 / exchange_rate) * position_size * 100000
            if "JPY" not in currency_pair
            else (0.01 / exchange_rate) * position_size * 100000
        )
        profit_loss = pip_movement * pip_value
        st.write(f"**Pip Movement**: {pip_movement:.2f} pips")
        st.write(f"**Pip Value**: {pip_value:.2f} {account_currency}")
        st.write(f"**Potential Profit/Loss**: {profit_loss:.2f} {account_currency}")

    with tools_subtab_map.get("Price Alerts", st.container()):
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
                        <div style="
                            border-radius:10px;
                            padding:12px;
                            margin-bottom:10px;
                            background: linear-gradient(180deg,#0f1720, #0b0f14);
                            border:1px solid rgba(255,255,255,0.03);
                        ">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="font-size:1.05rem;">{pair}</strong>
                                <span style="color:{color}; font-weight:700;">{status}</span>
                            </div>
                            <div style="margin-top:8px;">
                                <small class="small-muted">Current:</small> <b>{current_price_display}</b> &nbsp;&nbsp;
                                <small class="small-muted">Target:</small> <b>{target}</b>
                            </div>
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

    with tools_subtab_map.get("Currency Correlation Heatmap", st.container()):
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
        fig = px.imshow(corr_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        title="Forex Pair Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with tools_subtab_map.get("Risk Management Calculator", st.container()):
        st.header("🛡️ Risk Management Calculator")
        st.markdown("""
        Proper position sizing keeps your account safe. Risk management is crucial to long-term trading success. 
        It helps prevent large losses, preserves capital, and allows you to stay in the game during drawdowns. 
        Always risk no more than 1-2% per trade, use stop losses, and calculate position sizes based on your account size and risk tolerance.
        """)
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
            st.success(f"✅ Recommended Lot Size: **{lot_size:.2f} lots**")
            logging.info(f"Calculated lot size: {lot_size}")
        # 🔄 What-If Analyzer
        st.subheader('🔄 What-If Analyzer')
        base_equity = st.number_input('Starting Equity', value=10000.0, min_value=0.0, step=100.0, key='whatif_equity')
        risk_pct = st.slider('Risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key='whatif_risk') / 100.0
        winrate = st.slider('Win rate (%)', 10.0, 90.0, 50.0, 1.0, key='whatif_wr') / 100.0
        avg_r = st.slider('Average R multiple', 0.5, 5.0, 1.5, 0.1, key='whatif_avg_r')
        trades = st.slider('Number of trades', 10, 500, 100, 10, key='whatif_trades')
        if st.button("Run What-If Analysis"):
            risk_per_trade = base_equity * risk_pct
            wins = int(trades * winrate)
            losses = trades - wins
            avg_win = risk_per_trade * avg_r
            avg_loss = risk_per_trade
            expected_profit = (wins * avg_win) - (losses * avg_loss)
            final_equity = base_equity + expected_profit
            growth_pct = (final_equity - base_equity) / base_equity * 100
            st.markdown("### Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Profit", f"${expected_profit:,.2f}")
            with col2:
                st.metric("Final Equity", f"${final_equity:,.2f}")
            with col3:
                st.metric("Growth", f"{growth_pct:.2f}%")
            logging.info(f"What-If Analysis: base=${base_equity}, risk={risk_pct*100}%, winrate={winrate*100}%, avg_r={avg_r}, trades={trades}, profit=${expected_profit}")

    with tools_subtab_map.get("Trading Session Tracker", st.container()):
        st.header("⏰ Trading Session Tracker")
        st.markdown("Track active trading sessions based on current time.")
        st.write('---')
        now = dt.datetime.now(pytz.UTC)
        sessions = [
            {"name": "Sydney", "start": 22, "end": 7, "tz": "Australia/Sydney"},
            {"name": "Tokyo", "start": 0, "end": 9, "tz": "Asia/Tokyo"},
            {"name": "London", "start": 8, "end": 17, "tz": "Europe/London"},
            {"name": "New York", "start": 13, "end": 22, "tz": "America/New_York"}
        ]
        active_sessions = []
        for session in sessions:
            tz = pytz.timezone(session["tz"])
            local_time = now.astimezone(tz).hour
            start = session["start"]
            end = session["end"]
            if end < start:
                is_active = local_time >= start or local_time < end
            else:
                is_active = start <= local_time < end
            if is_active:
                active_sessions.append(session["name"])
            st.markdown(f"**{session['name']} Session**: {'🟢 Active' if is_active else '🔴 Inactive'}")
        if active_sessions:
            st.success(f"Active sessions: {', '.join(active_sessions)}")
        else:
            st.info("No trading sessions are currently active.")

    with tools_subtab_map.get("Drawdown Recovery Planner", st.container()):
        st.header("📉 Drawdown Recovery Planner")
        st.markdown("Plan your recovery from a drawdown.")
        st.write('---')
        drawdown_pct = st.number_input("Current Drawdown (%)", min_value=0.0, max_value=100.0, value=10.0)
        recovery_r = st.number_input("Average R per Trade", min_value=0.1, value=1.5)
        recovery_trades = st.number
