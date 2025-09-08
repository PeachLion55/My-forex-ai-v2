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
    /* This styling for st.metric is kept for non-editable metrics elsewhere */
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
    
    /* --- Custom Playbook Metric Display (new for editable section) --- */
    .playbook-metric-display {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 10px; /* Space between rows of metrics/actions */
    }
    .playbook-metric-display .label {
        font-size: 0.9em;
        color: #8b949e;
        margin-bottom: 5px;
    }
    .playbook-metric-display .value {
        font-size: 1.1em;
        font-weight: bold;
        color: #c9d1d9;
    }
    .playbook-metric-display.profit-positive {
        border-color: #2da44e; /* Green border for profit */
        background-color: #0b1f15; /* Darker green background */
    }
    .playbook-metric-display.profit-negative {
        border-color: #cf222e; /* Red border for loss */
        background-color: #260d0d; /* Darker red background */
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# LOGGING & DATABASE SETUP
# =========================================================
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_FILE = "users.db"

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)): return obj.isoformat()
        if pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)): return None
        return super().default(obj)

try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS community_data (key TEXT PRIMARY KEY, data TEXT)''')
    conn.commit()
    logging.info("SQLite database initialized successfully")
except Exception as e:
    st.error("Fatal Error: Could not connect to the database.")
    logging.critical(f"Failed to initialize SQLite database: {str(e)}", exc_info=True)
    st.stop()


# =========================================================
# JOURNAL SCHEMA & ROBUST DATA MIGRATION (GLOBAL)
# =========================================================
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
    "EntryScreenshot": str, # Added for screenshots
    "ExitScreenshot": str   # Added for screenshots
}

# =========================================================
# CORE HELPER FUNCTIONS (GLOBAL)
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

def _ta_hash():
    return uuid.uuid4().hex[:12]

def _ta_percent_gain_to_recover(drawdown_pct):
    if drawdown_pct <= 0:
        return 0.0
    if drawdown_pct >= 0.99:
        return float("inf")
    return drawdown_pct / (1 - drawdown_pct)

def get_user_data(username):
    c.execute("SELECT data FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    return json.loads(result[0]) if result and result[0] else {}

def save_user_data(username):
    """
    Saves the current session state data for the logged-in user to the database.
    This function should be called when any user-specific data in st.session_state changes.
    """
    user_data = {
        "drawings": st.session_state.get("drawings", {}),
        "trade_journal": st.session_state.get("trade_journal", pd.DataFrame(columns=journal_cols).astype(journal_dtypes, errors='ignore')).to_dict('records'),
        "strategies": st.session_state.get("strategies", pd.DataFrame()).to_dict('records'),
        "emotion_log": st.session_state.get("emotion_log", pd.DataFrame()).to_dict('records'),
        "reflection_log": st.session_state.get("reflection_log", pd.DataFrame()).to_dict('records'),
        "xp": st.session_state.get("xp", 0),
        "level": st.session_state.get("level", 0),
        "badges": st.session_state.get("badges", []),
        "streak": st.session_state.get("streak", 0),
        "last_journal_date": st.session_state.get("last_journal_date", None), # for journaling streak
        "last_login_xp_date": st.session_state.get("last_login_xp_date", None), # For daily login XP
        "gamification_flags": st.session_state.get("gamification_flags", {}), # To track various awarded flags
        'xp_log': st.session_state.get('xp_log', []), # XP transactions
        'chatroom_rules_accepted': st.session_state.get('chatroom_rules_accepted', False), # Added for chat
        'chatroom_nickname': st.session_state.get('user_nickname', None), # Added for chat (renamed for DB)
    }
    user_data_json = json.dumps(user_data, default=str)
    try:
        c.execute("UPDATE users SET data = ? WHERE username = ?", (user_data_json, username))
        conn.commit()
        logging.info(f"Successfully saved data for user {username}")
        return True
    except Exception as e:
        logging.error(f"Failed to save data for user {username}: {e}")
        st.error("Could not save your progress. Please contact support.")
        return False

def _ta_load_community(key, default=None):
    if default is None:
        default = [] # Ensure default is a mutable type if intended to be appended to
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

def _ta_save_journal(username, journal_df):
    st.session_state.trade_journal = journal_df
    return save_user_data(username)

def show_xp_notification(xp_gained):
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

def ta_update_xp(username, amount, description="XP activity"):
    """
    Updates user XP, checks for level up, logs the transaction, and persists data.
    """
    if username is None: return

    # Ensure xp_log is a list in session state
    if 'xp_log' not in st.session_state:
        st.session_state.xp_log = []

    st.session_state.xp = st.session_state.get('xp', 0) + amount
    
    new_level = st.session_state.xp // 100
    if new_level > st.session_state.get('level', 0):
        st.session_state.level = new_level
        badge_name = f"Level {new_level}"
        if badge_name not in st.session_state.badges:
            st.session_state.badges.append(badge_name)
        st.balloons()
        st.success(f"Level up! You are now level {new_level}!")

    # Log XP transaction
    st.session_state.xp_log.append({
        "Date": dt.date.today().isoformat(), # Use ISO format for consistent storage
        "Amount": amount,
        "Description": description
    })

    save_user_data(username) # Persist all session state XP/Level/Badges/xp_log changes
    if amount != 0: # Only show notification for non-zero XP changes
        show_xp_notification(amount) # Show notification

def ta_award_badge(username, badge_name):
    """Awards a badge to the user and triggers a toast notification."""
    if username is None: return

    if badge_name not in st.session_state.get('badges', []): # Only award if not already present
        st.session_state.badges.append(badge_name)
        save_user_data(username)
        # Using st.toast here for chatroom, but st.success for bigger app-level achievements is fine too
        st.toast(f"🏅 New Badge Earned: **{badge_name}**!", icon="🎖️") # Changed from success to toast
        #st.balloons() # Can also trigger balloons for badges if desired
        logging.info(f"User {username} awarded badge: {badge_name}")

def ta_update_streak(username):
    """
    Updates journaling streak, awards streak badges and XP, and persists data.
    """
    if username is None: return
    today = dt.date.today()
    last_journal_date_str = st.session_state.get('last_journal_date')
    current_streak = st.session_state.get('streak', 0)

    last_journal_date = dt.date.fromisoformat(last_journal_date_str) if last_journal_date_str else None
    
    if last_journal_date is None: # First journal entry ever
        current_streak = 1
    elif last_journal_date == today - dt.timedelta(days=1): # Consecutive day
        current_streak += 1
    elif last_journal_date == today: # Already journaled today, no change to streak or XP
        return
    else: # Streak broken
        current_streak = 1
        
    st.session_state.streak = current_streak
    st.session_state.last_journal_date = today.isoformat() # Update last journaling date

    # Award streak badges and XP
    if current_streak > 0 and current_streak % 7 == 0:
        badge_name = f"{current_streak}-Day Streak"
        # We need to make sure ta_award_badge does not re-add to session.badges if already there, 
        # but ta_award_badge already does this.
        ta_award_badge(username, badge_name) # Uses the general award badge, which includes persistence and checks.
        ta_update_xp(username, 15, f"Unlocked: {badge_name} Discipline Badge") # Bonus XP for streak badge
        st.balloons() # Removed the local success as ta_award_badge gives a toast
    
    save_user_data(username) # Persist updated streak, date, and possibly badge/XP

# =========================================================
# NEW GAMIFICATION FEATURES - HELPER FUNCTIONS
# (Defined here so they are available globally)
# =========================================================

def award_xp_for_notes_added_if_changed(username, trade_id, current_notes):
    """
    Awards XP if notes are added or significantly changed for a trade.
    Uses a content hash in gamification_flags to prevent repeat awards for the same notes content.
    """
    if username is None: return
    
    gamification_flags = st.session_state.get('gamification_flags', {})
    notes_award_key = f"xp_notes_for_trade_{trade_id}_content_hash" # Specific key for this XP event based on trade ID

    current_notes_hash = hashlib.md5(current_notes.strip().encode()).hexdigest() if current_notes.strip() else ""
    last_awarded_notes_hash = gamification_flags.get(notes_award_key)

    # Award XP if non-empty notes provided AND content hash has changed (or it's new notes for this trade_id)
    if current_notes.strip() and current_notes_hash != last_awarded_notes_hash:
        gamification_flags[notes_award_key] = current_notes_hash # Update the stored hash for this trade's notes XP
        st.session_state.gamification_flags = gamification_flags # Persist this flag change to session_state
        
        ta_update_xp(username, 5, f"Added/updated notes for trade {trade_id}") # Calls global XP updater and logger
        return True
    return False

def check_and_award_trade_milestones(username):
    """
    Checks if trade count milestones have been hit and awards XP/badges.
    Prevents re-awarding by checking gamification_flags.
    """
    if username is None: return

    current_total_trades = len(st.session_state.trade_journal)
    
    trade_milestones = {
        10: {"xp": 20, "badge_name": "Ten Trades Novice"},
        50: {"xp": 50, "badge_name": "Fifty Trades Apprentice"},
        100: {"xp": 100, "badge_name": "Centurion Trader"}
    }

    gamification_flags = st.session_state.get('gamification_flags', {})

    for milestone_count, details in trade_milestones.items():
        badge_name = details["badge_name"]
        milestone_flag_key = f"trade_count_{milestone_count}_awarded" # Unique flag key for this milestone
        
        if current_total_trades >= milestone_count and not gamification_flags.get(milestone_flag_key):
            gamification_flags[milestone_flag_key] = True # Mark this milestone as awarded
            st.session_state.gamification_flags = gamification_flags # Persist flag change to session_state
            
            ta_award_badge(username, badge_name) # Add badge using global function
            ta_update_xp(username, details["xp"], f"Achieved '{badge_name}' trade milestone") # Award bonus XP
            logging.info(f"User {username} hit trade milestone {badge_name}. Awarded {details['xp']} XP.")
    save_user_data(username) # Always persist gamification flags changes

def check_and_award_performance_milestones(username):
    """
    Checks and awards XP for performance milestones (Profit Factor, Avg R:R, Win Rate).
    Prevents re-awarding by checking gamification_flags.
    """
    if username is None: return

    df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()

    if df_analytics.empty or len(df_analytics) < 5: # Require minimum trades for meaningful stats
        return

    df_analytics['PnL'] = pd.to_numeric(df_analytics['PnL'], errors='coerce').fillna(0.0)
    df_analytics['RR'] = pd.to_numeric(df_analytics['RR'], errors='coerce').fillna(0.0)

    total_wins_sum = df_analytics[df_analytics['Outcome'] == 'Win']['PnL'].sum()
    total_losses_sum_abs = abs(df_analytics[df_analytics['Outcome'] == 'Loss']['PnL'].sum())

    profit_factor_val = total_wins_sum / total_losses_sum_abs if total_losses_sum_abs > 0 else (float('inf') if total_wins_sum > 0 else 0)
    avg_rr = df_analytics['RR'].mean()
    
    if len(df_analytics) > 0:
        win_rate = (len(df_analytics[df_analytics['Outcome'] == 'Win']) / len(df_analytics)) * 100
    else:
        win_rate = 0.0

    gamification_flags = st.session_state.get('gamification_flags', {})

    # Milestone 1: High Profit Factor
    profit_factor_threshold = 2.0
    pf_milestone_key = 'milestone_high_profit_factor_2.0_awarded' # Unique flag key
    if profit_factor_val >= profit_factor_threshold and not gamification_flags.get(pf_milestone_key):
        gamification_flags[pf_milestone_key] = True
        st.session_state.gamification_flags = gamification_flags # Persist
        ta_update_xp(username, 30, f"Achieved Profit Factor of {profit_factor_val:.2f} milestone")
        st.success(f"Performance Milestone: Achieved Profit Factor of {profit_factor_val:.2f}! Bonus 30 XP!")

    # Milestone 2: High Average R:R
    avg_rr_threshold = 1.5
    avg_rr_milestone_key = 'milestone_high_avg_rr_1.5_awarded' # Unique flag key
    if avg_rr >= avg_rr_threshold and not gamification_flags.get(avg_rr_milestone_key):
        gamification_flags[avg_rr_milestone_key] = True
        st.session_state.gamification_flags = gamification_flags # Persist
        ta_update_xp(username, 25, f"Achieved average R:R of {avg_rr:.2f} milestone")
        st.success(f"Performance Milestone: Achieved average R:R of {avg_rr:.2f}! Bonus 25 XP!")

    # Milestone 3: High Win Rate
    win_rate_threshold = 60 # 60% win rate
    wr_milestone_key = 'milestone_high_win_rate_60_percent_awarded' # Unique flag key
    if win_rate >= win_rate_threshold and not gamification_flags.get(wr_milestone_key):
        gamification_flags[wr_milestone_key] = True
        st.session_state.gamification_flags = gamification_flags # Persist
        ta_update_xp(username, 20, f"Achieved {win_rate:.1f}% Win Rate milestone")
        st.success(f"Performance Milestone: Achieved {win_rate:.1f}% Win Rate! Bonus 20 XP!")
    
    save_user_data(username) # Always persist gamification flags changes

def render_xp_leaderboard():
    """
    Renders the global XP leaderboard on the Account page.
    """
    st.subheader("🏆 Global XP Leaderboard")

    try:
        c.execute("SELECT username, data FROM users")
        all_users_raw = c.fetchall()

        leaderboard_data = []
        for uname, udata_json in all_users_raw:
            user_d = json.loads(udata_json) if udata_json else {}
            user_xp = user_d.get('xp', 0)
            leaderboard_data.append({"Username": uname, "XP Earned": user_xp})
        
        if leaderboard_data:
            leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by="XP Earned", ascending=False).reset_index(drop=True)
            leaderboard_df["Rank"] = leaderboard_df.index + 1

            def highlight_current_user_row(row):
                if st.session_state.logged_in_user is not None and row['Username'] == st.session_state.logged_in_user:
                    return ['background-color: #000000; color: white;'] * len(row)
                return [''] * len(row)
            
            st.dataframe(leaderboard_df[['Rank', 'Username', 'XP Earned']].style.apply(highlight_current_user_row, axis=1), use_container_width=True)
            
        else:
            st.info("No users registered yet. Be the first to earn XP!")

    except Exception as e:
        st.error(f"Error loading leaderboard: {e}")
        logging.error(f"Leaderboard load error: {e}", exc_info=True)


# =========================================================
# SESSION STATE INITIALIZATION (GLOBAL)
# This is now centralized and ensures consistent state.
# =========================================================

DEFAULT_APP_STATE = {
    'logged_in_user': None,
    'current_page': 'account', # Default to account page for login/signup
    'current_subpage': None,
    'show_tools_submenu': False,
    'temp_journal': None,
    'drawings': {},
    'xp': 0,
    'level': 0,
    'badges': [],
    'streak': 0,
    'last_journal_date': None,
    'last_login_xp_date': None,
    'gamification_flags': {},
    'xp_log': [], # Stores XP transactions
    'trade_journal': pd.DataFrame(columns=journal_cols).astype(journal_dtypes, errors='ignore'),
    'strategies': pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"]),
    'emotion_log': pd.DataFrame(columns=["Date", "Emotion", "Notes"]),
    'reflection_log': pd.DataFrame(columns=["Date", "Reflection"]),
    'mt5_df': pd.DataFrame(),
    'price_alerts': pd.DataFrame(columns=["Pair", "Target Price", "Triggered"]),
    'selected_calendar_month': datetime.now().strftime('%B %Y'),
    'trade_ideas': pd.DataFrame(columns=["Username", "Pair", "Direction", "Description", "Timestamp", "IdeaID", "ImagePath"]),
    'community_templates': pd.DataFrame(columns=["Username", "Type", "Name", "Content", "Timestamp", "ID"]),
    # Chatroom specific states
    'chatroom_rules_accepted': False,
    'user_nickname': None, # This will store the nickname for the current session/user if set
    # Global chat messages for community. Loaded from _ta_load_community for each channel.
    'chat_messages': {
        "General Discussion": [],
        "Trading Psychology": [],
        "Trade Reviews": [],
        "Market News": []
    }
}

def initialize_and_load_session_state():
    """
    Initializes all expected session state variables to their defaults if not set,
    then attempts to load persisted user data if a user is logged in.
    Also handles daily login XP award.
    """
    # 1. Initialize all session state variables to their defaults if they don't exist
    for key, default_value in DEFAULT_APP_STATE.items():
        if key not in st.session_state:
            # Handle mutable defaults by making a copy
            if isinstance(default_value, (pd.DataFrame, dict, list)):
                st.session_state[key] = default_value.copy()
            else:
                st.session_state[key] = default_value
    
    # 2. If a user IS logged in, attempt to load their specific data and apply gamification checks
    if st.session_state.logged_in_user is not None:
        username = st.session_state.logged_in_user
        user_data_from_db = get_user_data(username)

        # Load scalar values and dicts
        st.session_state.xp = user_data_from_db.get('xp', DEFAULT_APP_STATE['xp'])
        st.session_state.level = user_data_from_db.get('level', DEFAULT_APP_STATE['level'])
        st.session_state.badges = user_data_from_db.get('badges', DEFAULT_APP_STATE['badges'].copy())
        st.session_state.streak = user_data_from_db.get('streak', DEFAULT_APP_STATE['streak'])
        st.session_state.last_journal_date = user_data_from_db.get('last_journal_date', DEFAULT_APP_STATE['last_journal_date'])
        st.session_state.drawings = user_data_from_db.get("drawings", DEFAULT_APP_STATE['drawings'].copy())
        
        st.session_state.gamification_flags = user_data_from_db.get('gamification_flags', DEFAULT_APP_STATE['gamification_flags'].copy())
        st.session_state.last_login_xp_date = user_data_from_db.get('last_login_xp_date', DEFAULT_APP_STATE['last_login_xp_date'])
        st.session_state.xp_log = user_data_from_db.get('xp_log', DEFAULT_APP_STATE['xp_log'].copy())

        # Load chatroom specific user data
        st.session_state.chatroom_rules_accepted = user_data_from_db.get('chatroom_rules_accepted', DEFAULT_APP_STATE['chatroom_rules_accepted'])
        # Store nickname in a consistent user_data field, and then load into session state
        st.session_state.user_nickname = user_data_from_db.get('chatroom_nickname', DEFAULT_APP_STATE['user_nickname'])

        # Load DataFrames, ensuring proper column structure and types
        def load_dataframe_robustly(data_list, columns, dtypes, default_df):
            if data_list:
                df = pd.DataFrame(data_list)
                for col in columns:
                    if col not in df.columns:
                        df[col] = pd.Series(dtype=dtypes.get(col))
                return df[columns].astype(dtypes, errors='ignore')
            return default_df.copy()

        st.session_state.trade_journal = load_dataframe_robustly(user_data_from_db.get("trade_journal", []), journal_cols, journal_dtypes, DEFAULT_APP_STATE['trade_journal'])
        
        st.session_state.strategies = load_dataframe_robustly(user_data_from_db.get("strategies", []), 
                                                           DEFAULT_APP_STATE['strategies'].columns.tolist(),
                                                           {col: str for col in DEFAULT_APP_STATE['strategies'].columns},
                                                           DEFAULT_APP_STATE['strategies'])

        st.session_state.emotion_log = load_dataframe_robustly(user_data_from_db.get("emotion_log", []),
                                                           DEFAULT_APP_STATE['emotion_log'].columns.tolist(),
                                                           {'Date': 'datetime64[ns]', 'Emotion': str, 'Notes': str},
                                                           DEFAULT_APP_STATE['emotion_log'])
        if not st.session_state.emotion_log.empty:
            st.session_state.emotion_log['Date'] = pd.to_datetime(st.session_state.emotion_log['Date'], errors='coerce')


        st.session_state.reflection_log = load_dataframe_robustly(user_data_from_db.get("reflection_log", []),
                                                               DEFAULT_APP_STATE['reflection_log'].columns.tolist(),
                                                               {'Date': 'datetime64[ns]', 'Reflection': str},
                                                               DEFAULT_APP_STATE['reflection_log'])
        if not st.session_state.reflection_log.empty:
            st.session_state.reflection_log['Date'] = pd.to_datetime(st.session_state.reflection_log['Date'], errors='coerce')
        

        # --- AWARD DAILY LOGIN XP ---
        today = dt.date.today()
        last_login_xp_date_obj = dt.date.fromisoformat(st.session_state.last_login_xp_date) if st.session_state.last_login_xp_date else None

        if last_login_xp_date_obj is None or last_login_xp_date_obj < today:
            # We explicitly update the last_login_xp_date BEFORE calling ta_update_xp and save_user_data
            # to ensure the flag is set immediately. ta_update_xp will call save_user_data again,
            # so the updated last_login_xp_date will be saved correctly.
            st.session_state.last_login_xp_date = today.isoformat()
            ta_update_xp(username, 10, "Daily Login Bonus") # ta_update_xp contains st.rerun
            logging.info(f"Daily login XP (10) awarded to {username}")
        # --- END DAILY LOGIN XP ---


    st.session_state.trade_ideas = pd.DataFrame(_ta_load_community('trade_ideas', []), columns=DEFAULT_APP_STATE['trade_ideas'].columns).copy()
    st.session_state.community_templates = pd.DataFrame(_ta_load_community('templates', []), columns=DEFAULT_APP_STATE['community_templates'].columns).copy()

    # Load GLOBAL community chat messages (always load all channels, even if no user logged in, to show for public access logic if applicable,
    # but specifically needed for chatroom regardless if logged in is used by actual display)
    # The actual chat_messages are part of the app state, loaded *from* community_data
    for channel_key in DEFAULT_APP_STATE['chat_messages'].keys():
        db_key = f'chat_channel_{channel_key.lower().replace(" ", "_")}'
        st.session_state.chat_messages[channel_key] = _ta_load_community(db_key, default=[]) # Load into a list

initialize_and_load_session_state()

# =========================================================
# PAGE CONFIGURATION (Streamlit requires this at top level)
# =========================================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide")


# =========================================================
# CUSTOM SIDEBAR CSS
# =========================================================
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        overflow: hidden !important;
        max-height: 100vh !important;
    }
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
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: linear-gradient(to right, rgba(88, 179, 177, 1.0), rgba(0, 0, 0, 1.0)) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
        color: #f0f0f0 !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[data-active="true"] {
        background: rgba(88, 179, 177, 0.7) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
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
        date_str = published[:10] if published else ""
        currency = detect_currency(title)
        polarity = TextBlob(title).sentiment.polarity
        impact = rate_impact(polarity)
        summary = getattr(entry, "summary", "")
        rows.append({
            "Date": date_str,
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
# SIDEBAR NAVIGATION
# =========================================================

st.markdown(
    """
    <style>
    .sidebar-content {
        padding-top: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

nav_items = [
    ('fundamentals', 'Forex Fundamentals'),
    ('trading_journal', 'Trading Journal'),
    ('mt5', 'Performance Dashboard'),
    ('tools', 'Tools'),
    ('strategy', 'Manage My Strategy'),
    ('community', 'Community Trade Ideas'),
    ('Community Chatroom', 'Community Chatroom'), # ADDED CHATROOM NAV ITEM
    ('Zenvo Academy', 'Zenvo Academy'),
    ('account', 'My Account')
]

for page_key, page_name in nav_items:
    is_active = (st.session_state.current_page == page_key)
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
        st.info("See the Trading Journal tab for live charts + detailed news.")
    
    st.markdown("### 🗓️ Upcoming Economic Events")
    
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
        selected_1 = st.session_state.get('selected_currency_1')
        selected_2 = st.session_state.get('selected_currency_2')

        if selected_1 and row['Currency'] == selected_1:
            styles = ['background-color: #4c7170; color: white' if col == 'Currency' else 'background-color: #4c7170' for col in row.index]
        if selected_2 and row['Currency'] == selected_2:
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
                background-color:#00000;
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
# TRADING JOURNAL PAGE
# =========================================================
elif st.session_state.current_page == 'trading_journal':
    if st.session_state.logged_in_user is None:
        st.warning("Please log in to access your Trading Journal.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.title("📊 Trading Journal")
    st.caption(f"A streamlined interface for professional trade analysis. | Logged in as: **{st.session_state.logged_in_user}**")
    st.markdown("---")

    tab_entry, tab_playbook, tab_analytics = st.tabs(["**📝 Log New Trade**", "**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])

    # --- TAB 1: LOG NEW TRADE ---
    with tab_entry:
        st.header("Log a New Trade")
        st.caption("Focus on a quick, essential entry. You can add detailed notes and screenshots later in the Playbook.")

        with st.form("trade_entry_form", clear_on_submit=True):
            st.markdown("##### ⚡ Trade Entry Details")
            col1, col2, col3 = st.columns(3)

            pairs_map_for_selection = {
                "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD", "USD/CHF": "OANDA:USDCHF",
                "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD", "USD/CAD": "FX:USDCAD"
            }

            with col1:
                date_val = st.date_input("Date", dt.date.today())
                symbol_options = list(pairs_map_for_selection.keys()) + ["Other"]
                symbol = st.selectbox("Symbol", symbol_options, index=0)
                if symbol == "Other": symbol = st.text_input("Custom Symbol")
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
            st.markdown("---")
            
            st.markdown("##### Rationale & Tags")
            entry_rationale = st.text_area("Why did you enter this trade?", height=100)

            all_tags = sorted(list(set(st.session_state.trade_journal['Tags'].str.split(',').explode().dropna().str.strip().tolist()))) if not st.session_state.trade_journal.empty and 'Tags' in st.session_state.trade_journal.columns else []

            suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged"]
            tags_selection = st.multiselect("Select Existing Tags", options=sorted(list(set(all_tags + suggested_tags))))

            new_tags_input = st.text_input("Add New Tags (comma-separated)", placeholder="e.g., strong momentum, poor entry, ...")

            submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
            if submitted:
                final_pnl, final_rr = 0.0, 0.0

                if calculate_pnl_rr:
                    if stop_loss == 0.0 or entry_price == 0.0:
                        st.error("Entry Price and Stop Loss must be greater than 0 to calculate PnL/RR automatically.")
                        st.stop()
                    
                    risk_per_unit = abs(entry_price - stop_loss)
                    pip_size_for_pair_calc = 0.0001
                    if "JPY" in symbol.upper():
                        pip_size_for_pair_calc = 0.01
                    
                    usd_per_pip_per_standard_lot = 10.0

                    price_change_raw = final_exit - entry_price
                    pips_moved = price_change_raw / pip_size_for_pair_calc
                    
                    if direction == "Long":
                        final_pnl = pips_moved * (lots * usd_per_pip_per_standard_lot) / 10 
                    else: # Short
                        final_pnl = -pips_moved * (lots * usd_per_pip_per_standard_lot) / 10
                    
                    if risk_per_unit > 0.0:
                        reward_per_unit = abs(final_exit - entry_price)
                        final_rr = reward_per_unit / risk_per_unit
                        
                        if (direction == "Long" and final_exit < entry_price) or (direction == "Short" and final_exit > entry_price):
                            final_rr *= -1

                        if final_exit == entry_price:
                            final_rr = 0.0
                    else:
                        final_rr = 0.0

                else: # Manual PnL and RR inputs are used
                    final_pnl = manual_pnl_input
                    final_rr = manual_rr_input

                newly_added_tags = [tag.strip() for tag in new_tags_input.split(',') if tag.strip()]
                final_tags_list = sorted(list(set(tags_selection + newly_added_tags)))

                trade_id_new = f"TRD-{uuid.uuid4().hex[:6].upper()}"

                entry_screenshot_path_saved = None
                exit_screenshot_path_saved = None

                new_trade_data = {
                    "TradeID": trade_id_new, "Date": pd.to_datetime(date_val),
                    "Symbol": symbol, "Direction": direction, "Outcome": outcome,
                    "Lots": lots, "EntryPrice": entry_price, "StopLoss": stop_loss, "FinalExit": final_exit,
                    "PnL": final_pnl, "RR": final_rr,
                    "Tags": ','.join(final_tags_list), "EntryRationale": entry_rationale,
                    "Strategy": '', "TradeJournalNotes": '', 
                    "EntryScreenshot": entry_screenshot_path_saved,
                    "ExitScreenshot": exit_screenshot_path_saved
                }
                new_df = pd.DataFrame([new_trade_data])
                st.session_state.trade_journal = pd.concat([st.session_state.trade_journal, new_df], ignore_index=True)

                if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                    ta_update_xp(st.session_state.logged_in_user, 10, "Logged a new trade")
                    ta_update_streak(st.session_state.logged_in_user)
                    st.success(f"Trade {new_trade_data['TradeID']} logged successfully!")
                    
                    check_and_award_trade_milestones(st.session_state.logged_in_user)
                    check_and_award_performance_milestones(st.session_state.logged_in_user)
                    
                    st.rerun()
                else:
                    st.error("Failed to save trade.")

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
                filtered_df = filtered_df[filtered_df['Tags'].astype(str).apply(lambda x: any(tag in x.split(',') for tag in tag_filter))]

            if 'edit_state' not in st.session_state: st.session_state.edit_state = {}


            for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
                trade_id_key = row['TradeID']
                outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e", "No Trade/Study": "#58a6ff"}.get(row['Outcome'], "#30363d")

                with st.container(border=True):
                    # Trade Header
                    st.markdown(f"""
                    <h4 style='margin-bottom:0px;'>{row['Symbol']} <span style="font-weight: 500; color: {outcome_color};">{row['Direction']} / {row['Outcome']}</span></h4>
                    <span style="color: #8b949e; font-size: 0.9em;">{row['Date'].strftime('%A, %d %B %Y')} | {trade_id_key}</span>
                    """, unsafe_allow_html=True)
                    st.markdown("---")

                    # Metrics Section
                    metric_cols = st.columns(3)
                    
                    pnl_val = float(pd.to_numeric(row.get('PnL', 0.0), errors='coerce') or 0.0)
                    rr_val = float(pd.to_numeric(row.get('RR', 0.0), errors='coerce') or 0.0)
                    lots_val = float(pd.to_numeric(row.get('Lots', 0.01), errors='coerce') or 0.01)

                    # Helper function to render a metric display or its editing form
                    def render_metric_cell_or_form(col_obj, metric_label, db_column, current_value, key_suffix, format_str, is_pnl_metric=False):
                        is_editing = st.session_state.edit_state.get(f"{key_suffix}_{trade_id_key}", False)
                        with col_obj:
                            if is_editing:
                                with st.form(f"form_{key_suffix}_{trade_id_key}", clear_on_submit=False):
                                    st.markdown(f"**Edit {metric_label}**")
                                    new_value = st.number_input("", value=current_value, format=format_str, key=f"input_{key_suffix}_{trade_id_key}")
                                    s_col, c_col = st.columns(2)
                                    if s_col.form_submit_button("✓ Save", type="primary", use_container_width=True):
                                        st.session_state.trade_journal.loc[index, db_column] = new_value
                                        _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal)
                                        st.session_state.edit_state[f"{key_suffix}_{trade_id_key}"] = False
                                        st.rerun()
                                    if c_col.form_submit_button("✗ Cancel", use_container_width=True):
                                        st.session_state.edit_state[f"{key_suffix}_{trade_id_key}"] = False
                                        st.rerun()
                            else:
                                display_val_str = ""
                                border_style = ""
                                if is_pnl_metric:
                                    border_color = "#2da44e" if current_value > 0 else ("#cf222e" if current_value < 0 else "#30363d")
                                    val_color = "#50fa7b" if current_value > 0 else ("#ff5555" if current_value < 0 else "#c9d1d9")
                                    border_style = f"border: 1px solid {border_color};"
                                    display_val_str = f"<div class='value' style='color:{val_color};'>${current_value:.2f}</div>"
                                elif metric_label == "R-Multiple":
                                    display_val_str = f"<div class='value'>{current_value:.2f}R</div>"
                                else: # Position Size
                                    display_val_str = f"<div class='value'>{current_value:.2f} lots</div>"
                                
                                # Use Streamlit's native components for reliability,
                                # wrapping content for styling
                                st.markdown(
                                    f"""
                                    <div class='playbook-metric-display' style='{border_style} position:relative;'>
                                        <div class='label' style='margin-right:25px;'>{metric_label}</div>
                                        {display_val_str}
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                                # Button absolutely positioned within the parent Streamlit column/context for simple access
                                if st.button("✏️", key=f"edit_btn_{key_suffix}_{trade_id_key}", help=f"Edit {metric_label}"):
                                    st.session_state.edit_state[f"{key_suffix}_{trade_id_key}"] = True
                                    st.rerun()
                                st.markdown(
                                    """
                                    <style>
                                        /* Basic override to visually pull button into top-right of metric display */
                                        button[key*="edit_btn_"] {
                                            position: absolute;
                                            top: 5px;
                                            right: 5px;
                                            z-index: 10;
                                            background-color: transparent;
                                            border: none;
                                            color: #c9d1d9; /* Visible pencil */
                                            font-size: 0.2em;
                                            padding: 0;
                                            height: 5px; /* Small clickable area */
                                            width: 5px; /* Small clickable area */
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                        }
                                        /* This ensures the containing elements do not stretch excessively when the button is positioned absolutely */
                                        div[data-testid="stColumn"] > div > div:nth-child(2) > div:nth-child(2) > div > button[key*="edit_btn_"] {
                                            margin-top: 0px !important;
                                        }
                                        .playbook-metric-display {
                                            padding-right: 30px; /* Ensure space for the button */
                                        }
                                    </style>
                                    """, unsafe_allow_html=True
                                )

                    render_metric_cell_or_form(metric_cols[0], "Net PnL", "PnL", pnl_val, "pnl", "%.2f", is_pnl_metric=True)
                    render_metric_cell_or_form(metric_cols[1], "R-Multiple", "RR", rr_val, "rr", "%.2f")
                    render_metric_cell_or_form(metric_cols[2], "Position Size", "Lots", lots_val, "lots", "%.2f")
                    
                    st.markdown("---")

                    if row['EntryRationale']: st.markdown(f"**Entry Rationale:** *{row['EntryRationale']}*")
                    if row['Tags']:
                        tags_list = [f"`{tag.strip()}`" for tag in str(row['Tags']).split(',') if tag.strip()]
                        if tags_list: st.markdown(f"**Tags:** {', '.join(tags_list)}")
                    
                    # Journal Notes & Screenshots (EXPANDER - DEFAULT CLOSED)
                    with st.expander("Journal Notes & Screenshots", expanded=False):
                        notes = st.text_area(
                            "Trade Journal Notes",
                            value=row['TradeJournalNotes'],
                            key=f"notes_{trade_id_key}",
                            height=150
                        )

                        action_cols_notes_delete = st.columns([1, 1, 4]) 

                        if action_cols_notes_delete[0].button("Save Notes", key=f"save_notes_{trade_id_key}", type="primary"):
                            original_notes_from_df = st.session_state.trade_journal.loc[st.session_state.trade_journal['TradeID'] == trade_id_key, 'TradeJournalNotes'].iloc[0]
                            st.session_state.trade_journal.loc[index, 'TradeJournalNotes'] = notes # Use index for robustness
                            
                            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                                current_notes_hash = hashlib.md5(notes.strip().encode()).hexdigest() if notes.strip() else ""
                                gamification_flags = st.session_state.get('gamification_flags', {})
                                notes_award_key = f"xp_notes_for_trade_{trade_id_key}_content_hash"
                                last_awarded_notes_hash = gamification_flags.get(notes_award_key)

                                if notes.strip() and current_notes_hash != last_awarded_notes_hash:
                                    award_xp_for_notes_added_if_changed(st.session_state.logged_in_user, trade_id_key, notes)
                                else:
                                    st.toast(f"Notes for {row['TradeID']} updated (no new XP for same content).", icon="✅")
                                save_user_data(st.session_state.logged_in_user)
                                st.rerun()
                            else:
                                st.error("Failed to save notes.")

                        if action_cols_notes_delete[1].button("Delete Trade", key=f"delete_trade_{trade_id_key}"):
                            username = st.session_state.logged_in_user
                            xp_deduction_amount = 0
                            
                            xp_deduction_amount += 10 # Base XP per logged trade

                            gamification_flags = st.session_state.get('gamification_flags', {})
                            notes_award_key_for_deleted = f"xp_notes_for_trade_{trade_id_key}_content_hash"
                            if notes_award_key_for_deleted in gamification_flags:
                                xp_deduction_amount += 5
                                del gamification_flags[notes_award_key_for_deleted]
                            
                            if trade_id_key in st.session_state.edit_state:
                                # Clean up edit states related to this deleted trade
                                for key in list(st.session_state.edit_state.keys()):
                                    if trade_id_key in key:
                                        del st.session_state.edit_state[key]
                            
                            st.session_state.gamification_flags = gamification_flags
                            
                            if xp_deduction_amount > 0:
                                ta_update_xp(username, -xp_deduction_amount, f"Deleted trade {row['TradeID']}")
                                st.toast(f"Trade {row['TradeID']} deleted. {xp_deduction_amount} XP deducted.", icon="🗑️")
                            else:
                                st.toast(f"Trade {row['TradeID']} deleted.", icon="🗑️")

                            # Delete trade from DataFrame using its actual index
                            st.session_state.trade_journal.drop(index, inplace=True)
                            st.session_state.trade_journal.reset_index(drop=True, inplace=True) # Reset index after drop
                            
                            if row['EntryScreenshot'] and os.path.exists(row['EntryScreenshot']):
                                try: os.remove(row['EntryScreenshot'])
                                except OSError as e: logging.error(f"Error deleting entry screenshot {row['EntryScreenshot']}: {e}")
                            if row['ExitScreenshot'] and os.path.exists(row['ExitScreenshot']):
                                try: os.remove(row['ExitScreenshot'])
                                except OSError as e: logging.error(f"Error deleting exit screenshot {row['ExitScreenshot']}: {e}")

                            if _ta_save_journal(username, st.session_state.trade_journal):
                                check_and_award_trade_milestones(username)
                                check_and_award_performance_milestones(username)
                                st.rerun()
                            else:
                                st.error("Failed to delete trade.")
                        
                        # --- Upload Before/After Screenshots AFTER TRADE LOGGING ---
                        st.markdown("---")
                        st.subheader("Update Screenshots")
                        
                        image_base_path = os.path.join("user_data", st.session_state.logged_in_user, "journal_images")
                        os.makedirs(image_base_path, exist_ok=True)

                        upload_cols = st.columns(2)
                        
                        with upload_cols[0]:
                            new_entry_screenshot_file = st.file_uploader(
                                f"Upload/Update Entry Screenshot", 
                                type=["png", "jpg", "jpeg"], 
                                key=f"update_entry_ss_uploader_{trade_id_key}"
                            )
                            if new_entry_screenshot_file:
                                if st.button("Save Entry Image", key=f"save_new_entry_ss_btn_{trade_id_key}", type="secondary", use_container_width=True):
                                    if row['EntryScreenshot'] and os.path.exists(row['EntryScreenshot']):
                                        try: os.remove(row['EntryScreenshot'])
                                        except OSError as e: logging.error(f"Error deleting old entry screenshot {row['EntryScreenshot']}: {e}")

                                    entry_screenshot_filename = f"{trade_id_key}_entry_{uuid.uuid4().hex[:4]}_{new_entry_screenshot_file.name}"
                                    entry_screenshot_full_path = os.path.join(image_base_path, entry_screenshot_filename)
                                    with open(entry_screenshot_full_path, "wb") as f:
                                        f.write(new_entry_screenshot_file.getbuffer())
                                    
                                    st.session_state.trade_journal.loc[index, 'EntryScreenshot'] = entry_screenshot_full_path
                                    _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal)
                                    st.toast("Entry screenshot updated!", icon="📸")
                                    st.rerun()

                        with upload_cols[1]:
                            new_exit_screenshot_file = st.file_uploader(
                                f"Upload/Update Exit Screenshot", 
                                type=["png", "jpg", "jpeg"], 
                                key=f"update_exit_ss_uploader_{trade_id_key}"
                            )
                            if new_exit_screenshot_file:
                                if st.button("Save Exit Image", key=f"save_new_exit_ss_btn_{trade_id_key}", type="secondary", use_container_width=True):
                                    if row['ExitScreenshot'] and os.path.exists(row['ExitScreenshot']):
                                        try: os.remove(row['ExitScreenshot'])
                                        except OSError as e: logging.error(f"Error deleting old exit screenshot {row['ExitScreenshot']}: {e}")

                                    exit_screenshot_filename = f"{trade_id_key}_exit_{uuid.uuid4().hex[:4]}_{new_exit_screenshot_file.name}"
                                    exit_screenshot_full_path = os.path.join(image_base_path, exit_screenshot_filename)
                                    with open(exit_screenshot_full_path, "wb") as f:
                                        f.write(new_exit_screenshot_file.getbuffer())

                                    st.session_state.trade_journal.loc[index, 'ExitScreenshot'] = exit_screenshot_full_path
                                    _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal)
                                    st.toast("Exit screenshot updated!", icon="📸")
                                    st.rerun()
                                    
                        st.markdown("---")
                        # Current Visuals Display
                        st.subheader("Current Visuals")
                        visual_cols = st.columns(2)
                        if row['EntryScreenshot'] and os.path.exists(row['EntryScreenshot']):
                            visual_cols[0].image(row['EntryScreenshot'], caption="Entry", width=250)
                        else:
                            visual_cols[0].info("No Entry Screenshot available.")
                        
                        if row['ExitScreenshot'] and os.path.exists(row['ExitScreenshot']):
                            visual_cols[1].image(row['ExitScreenshot'], caption="Exit", width=250)
                        else:
                            visual_cols[1].info("No Exit Screenshot available.")
                            
                    st.markdown("---") # End of a single trade container


    # --- TAB 3: ANALYTICS DASHBOARD ---
    with tab_analytics:
        st.header("Your Performance Dashboard")
        df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()

        if df_analytics.empty:
            st.info("Complete at least one winning or losing trade to view your performance analytics.")
        else:
            total_pnl = pd.to_numeric(df_analytics['PnL'], errors='coerce').fillna(0.0).sum()
            total_trades = len(df_analytics)
            wins = df_analytics[df_analytics['Outcome'] == 'Win']
            losses = df_analytics[df_analytics['Outcome'] == 'Loss']

            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            avg_win = pd.to_numeric(wins['PnL'], errors='coerce').mean() if not wins.empty else 0.0
            avg_loss = pd.to_numeric(losses['PnL'], errors='coerce').mean() if not losses.empty else 0.0
            profit_factor = pd.to_numeric(wins['PnL'], errors='coerce').sum() / abs(pd.to_numeric(losses['PnL'], errors='coerce').sum()) if not losses.empty and pd.to_numeric(losses['PnL'], errors='coerce').sum() != 0 else (float('inf') if pd.to_numeric(wins['PnL'], errors='coerce').sum() > 0 else 0)


            kpi_cols = st.columns(4)

            pnl_metric_color = "#2da44e" if total_pnl >= 0 else "#cf222e" # Custom border color
            pnl_value_color_inner = "#50fa7b" if total_pnl >= 0 else "#ff5555" # Inner text color
            pnl_delta_icon = "⬆️" if total_pnl >= 0 else "⬇️"
            pnl_delta_display = f'<span style="font-size: 0.875rem; color: {pnl_value_color_inner};">{pnl_delta_icon} {abs(total_pnl):,.2f}</span>'


            kpi_cols[0].markdown(
                f"""
                <div class="stMetric" style="background-color: #161b22; border: 1px solid {pnl_metric_color}; border-radius: 8px; padding: 1.2rem; transition: all 0.2s ease-in-out;">
                    <div data-testid="stMetricLabel" style="font-weight: 500; color: #8b949e;">Net PnL ($)</div>
                    <div data-testid="stMetricValue" style="font-size: 2.25rem; line-height: 1.2; font-weight: 600; color: {pnl_value_color_inner};">${total_pnl:,.2f}</div>
                    {pnl_delta_display}
                </div>
                """, unsafe_allow_html=True
            )

            kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
            kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
            kpi_cols[3].metric("Avg. Win / Loss ($)", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}")

            st.markdown("---")

            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("Cumulative PnL")
                df_analytics_sorted = df_analytics.sort_values(by='Date').copy()
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
# PERFORMANCE DASHBOARD PAGE (MT5)
# =========================================================
elif st.session_state.current_page == 'mt5':
    if st.session_state.logged_in_user is None:
        st.warning("Please log in to access the Performance Dashboard.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.title("📊 Performance Dashboard")
    st.caption("Analyze your MT5 trading history with advanced metrics and visualizations.")
    st.markdown('---')
    st.markdown(
        """
        <style>
        /* General Metric Box Styling */
        .metric-box {
            background-color: #2d4646;
            padding: 10px 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #58b3b1;
            color: #ffffff;
            transition: all 0.3s ease-in-out;
            margin: 5px 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100px;
            min-width: 150px;
            box-sizing: border-box;
            font-size: 0.9em;
        }
        .metric-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(88, 179, 177, 0.3);
        }
        .metric-box strong {
            font-size: 1em;
            margin-bottom: 3px;
            display: block;
        }
        .metric-box .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            display: block;
            line-height: 1.3;
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .metric-box .sub-value {
            font-size: 0.8em;
            color: #ccc;
            line-height: 1;
            padding-bottom: 5px;
        }
        .metric-box .day-info {
            font-size: 0.85em;
            line-height: 1.2;
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding-top: 5px;
            padding-bottom: 5px;
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
            background-color: #333;
            border-radius: 5px;
            overflow: hidden;
            height: 8px;
            margin-top: auto;
            flex-shrink: 0;
        }
        .progress-bar {
            height: 100%;
            border-radius: 5px;
            text-align: right;
            line-height: 8px;
            color: white;
            font-size: 7px;
            box-sizing: border-box;
            white-space: nowrap;
        }
        .progress-bar.green { background-color: #5cb85c; }
        .progress-bar.red { background-color: #d9534f; }
        .progress-bar.neutral { background-color: #5bc0de; }
        /* Specific styles for the combined win/loss bar */
        .win-loss-bar-container {
            display: flex;
            width: 100%;
            background-color: #d9534f;
            border-radius: 5px;
            overflow: hidden;
            height: 8px;
            margin-top: auto;
            flex-shrink: 0;
        }
        .win-bar {
            height: 100%;
            background-color: #5cb85c;
            border-radius: 5px 0 0 5px;
            flex-shrink: 0;
        }
        .loss-bar {
            height: 100%;
            background-color: #d9534f;
            border-radius: 0 5px 5px 0;
            flex-shrink: 0;
        }
        /* Trading Score Bar */
        .trading-score-bar-container {
            width: 100%;
            background-color: #d9534f;
            border-radius: 5px;
            overflow: hidden;
            height: 8px;
            margin-top: auto;
            flex-shrink: 0;
            position: relative;
        }
        .trading-score-bar {
            height: 100%;
            background-color: #5cb85c;
            border-radius: 5px;
        }
        /* CALENDAR STYLES */
        .calendar-container {
            background-color: #262730;
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
        div[data-baseweb="select"] {
            margin: 0;
        }
        .calendar-nav {
            display: flex;
            justify-content: center;
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
            text-decoration: none;
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
            background-color: #2d2e37;
            padding: 8px;
            border-radius: 6px;
            height: 70px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            position: relative;
            border: 1px solid #3d3d4b;
            overflow: hidden;
            box-sizing: border-box;
        }
        .calendar-day-box.empty-month-day {
            background-color: #2d2e37;
            border: 1px solid #3d3d4b;
            visibility: hidden;
        }
        .calendar-day-box .day-number {
            font-size: 0.8em;
            color: #bbbbbb;
            text-align: left;
            margin-bottom: 5px;
            line-height: 1;
        }
        .calendar-day-box .profit-amount {
            font-size: 0.9em;
            font-weight: bold;
            text-align: center;
            line-height: 1.1;
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
        }
        .calendar-day-box.profitable {
            background-color: #0f2b0f;
            border-color: #5cb85c;
        }
        .calendar-day-box.losing {
            background-color: #2b0f0f;
            border-color: #d9534f;
        }
        .calendar-day-box.current-day {
            border: 2px solid #ff7f50;
        }
        .calendar-day-box .dot-indicator {
            position: absolute;
            top: 5px;
            right: 5px;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #ffcc00;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # The Python code for this MT5 page's Streamlit components would start below this point,
    # following the `st.markdown('---')` that it just rendered.
    # For example: `uploaded_file = st.file_uploader(...)` would appear here.
    
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

    def _ta_compute_sharpe(df_trades, risk_free_rate=0.02):
        if "Profit" not in df_trades.columns or df_trades.empty:
            return np.nan

        df_for_sharpe = df_trades.copy()
        df_for_sharpe["Close Time"] = pd.to_datetime(df_for_sharpe["Close Time"], errors='coerce')
        df_for_sharpe = df_for_sharpe.dropna(subset=["Close Time"])

        if df_for_sharpe.empty:
            return np.nan

        daily_pnl_series = df_for_sharpe.set_index("Close Time")["Profit"].resample('D').sum().fillna(0.0)

        if daily_pnl_series.empty or len(daily_pnl_series) < 2:
            return np.nan

        returns = daily_pnl_series.pct_change().dropna()

        if len(returns) < 2:
            return np.nan

        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        return (mean_return - risk_free_rate) / std_return if std_return != 0 else np.nan

    def _ta_daily_pnl_mt5(df_trades):
        """
        Returns a dictionary mapping datetime.date to total profit for that day.
        Includes all days that had at least one trade in the CSV.
        """
        if "Close Time" in df_trades.columns and "Profit" in df_trades.columns and not df_trades.empty and not df_trades["Profit"].isnull().all():
            df_copy = df_trades.copy()
            df_copy["date"] = pd.to_datetime(df_copy["Close Time"]).dt.date
            return df_copy.groupby("date")["Profit"].sum().to_dict()
        return {}

    def _ta_profit_factor_mt5(df_trades):
        wins_sum = df_trades[df_trades["Profit"] > 0]["Profit"].sum()
        losses_sum = abs(df_trades[df_trades["Profit"] < 0]["Profit"].sum())
        return wins_sum / losses_sum if losses_sum != 0.0 else (np.inf if wins_sum > 0 else np.nan)

    def _ta_show_badges_mt5(df_trades):
        st.subheader("🎖️ Your Trading Badges")

        total_profit_val = df_trades["Profit"].sum()
        total_trades_val = len(df_trades)

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

        avg_win_for_badge = df_trades[df_trades["Profit"] > 0]["Profit"].mean()
        avg_loss_for_badge = df_trades[df_trades["Profit"] < 0]["Profit"].mean()

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
                    st.session_state.mt5_df = DEFAULT_APP_STATE['mt5_df'].copy()
                    st.session_state.selected_calendar_month = DEFAULT_APP_STATE['selected_calendar_month']
                    st.stop()

                df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
                df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")
                df["Profit"] = pd.to_numeric(df["Profit"], errors='coerce').fillna(0.0)
                df = df.dropna(subset=["Open Time", "Close Time"])

                if df.empty:
                    st.error("Uploaded CSV resulted in no valid trading data after processing timestamps or profits.")
                    st.session_state.mt5_df = DEFAULT_APP_STATE['mt5_df'].copy()
                    st.session_state.selected_calendar_month = DEFAULT_APP_STATE['selected_calendar_month']
                    st.stop()


                df["Trade Duration"] = (df["Close Time"] - df["Open Time"]).dt.total_seconds() / 3600

                daily_pnl_map = _ta_daily_pnl_mt5(df)

                daily_pnl_df_for_stats = pd.DataFrame(columns=["date", "Profit"])

                if daily_pnl_map:
                    min_data_date = min(daily_pnl_map.keys())
                    max_data_date = max(daily_pnl_map.keys())
                    all_dates_in_data_range = pd.date_range(start=min_data_date, end=max_data_date).date
                    daily_pnl_df_for_stats = pd.DataFrame([
                        {"date": d, "Profit": daily_pnl_map.get(d, 0.0)}
                        for d in all_dates_in_data_range
                    ])
                elif not df.empty and pd.notna(df['Close Time'].min()) and pd.notna(df['Close Time'].max()):
                    min_date_raw = df['Close Time'].min().date()
                    max_date_raw = df['Close Time'].max().date()
                    all_dates_raw_range = pd.date_range(start=min_date_raw, end=max_date_raw).date
                    daily_pnl_df_for_stats = pd.DataFrame([
                        {"date": d, "Profit": 0.0} for d in all_dates_raw_range
                    ])

                tab_summary, tab_charts, tab_edge, tab_export = st.tabs([
                    "📈 Summary Metrics",
                    "📊 Visualizations",
                    "🔍 Edge Finder",
                    "📤 Export Reports"
                ])

                with tab_summary:
                    st.subheader("Key Performance Metrics")
                    total_trades = len(df)
                    wins_df = df[df["Profit"] > 0]
                    losses_df = df[df["Profit"] < 0]

                    win_rate = len(wins_df) / total_trades if total_trades else 0.0
                    net_profit = df["Profit"].sum()
                    profit_factor = _ta_profit_factor_mt5(df)
                    avg_win = wins_df["Profit"].mean() if not wins_df.empty else 0.0
                    avg_loss = losses_df["Profit"].mean() if not losses_df.empty else 0.0

                    max_drawdown = (daily_pnl_df_for_stats["Profit"].cumsum() - daily_pnl_df_for_stats["Profit"].cumsum().cummax()).min() if not daily_pnl_df_for_stats.empty else 0.0
                    sharpe_ratio = _ta_compute_sharpe(df)
                    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) if total_trades else 0.0
                    
                    def _ta_compute_streaks(df_pnl_daily):
                        d = df_pnl_daily.sort_values(by="date")
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
                                current_loss_streak = 0
                            elif pnl < 0:
                                current_loss_streak += 1
                                best_loss_streak = max(best_loss_streak, current_loss_streak)
                                current_win_streak = 0
                            else:
                                current_win_streak = 0
                                current_loss_streak = 0

                        return {"current_win": current_win_streak, "best_win": best_win_streak,
                                "current_loss": current_loss_streak, "best_loss": best_loss_streak}

                    streaks = _ta_compute_streaks(daily_pnl_df_for_stats)
                    longest_win_streak = streaks["best_win"]
                    longest_loss_streak = streaks["best_loss"]

                    avg_r_r = avg_win / abs(avg_loss) if avg_loss != 0.0 else np.nan

                    trading_score_value = 90.98
                    max_trading_score = 100
                    trading_score_percentage = (trading_score_value / max_trading_score) * 100

                    hit_rate = win_rate

                    most_profitable_asset_calc = "N/A"
                    if not df.empty and "Symbol" in df.columns and not df["Profit"].isnull().all():
                        profitable_assets = df.groupby("Symbol")["Profit"].sum()
                        if not profitable_assets.empty and profitable_assets.max() > 0.0:
                            most_profitable_asset_calc = profitable_assets.idxmax()
                        elif not profitable_assets.empty and profitable_assets.min() <= 0.0 and profitable_assets.max() <= 0.0:
                             most_profitable_asset_calc = "None Profitable"
                    
                    best_day_profit = 0.0
                    best_performing_day_name = "N/A"
                    worst_day_loss = 0.0
                    worst_performing_day_name = "N/A"

                    if not daily_pnl_df_for_stats.empty and not daily_pnl_df_for_stats["Profit"].empty:
                        days_with_pnl_actual_trades = daily_pnl_df_for_stats[daily_pnl_df_for_stats["Profit"] != 0.0]

                        if not days_with_pnl_actual_trades.empty:
                            best_day_profit = days_with_pnl_actual_trades["Profit"].max()
                            if pd.notna(best_day_profit) and best_day_profit > 0.0:
                                best_performing_day_date = days_with_pnl_actual_trades.loc[days_with_pnl_actual_trades["Profit"].idxmax(), "date"]
                                best_performing_day_name = pd.to_datetime(str(best_performing_day_date)).strftime('%A')
                            else:
                                best_performing_day_name = "No Profitable Days"

                            worst_day_loss = days_with_pnl_actual_trades["Profit"].min()
                            if pd.notna(worst_day_loss) and worst_day_loss < 0.0:
                                worst_performing_day_date = days_with_pnl_actual_trades.loc[days_with_pnl_actual_trades["Profit"].idxmin(), "date"]
                                worst_performing_day_name = pd.to_datetime(str(worst_performing_day_date)).strftime('%A')
                            else:
                                worst_performing_day_name = "No Losing Days"
                        else:
                            best_performing_day_name = "No Trades With Non-Zero P&L"
                            worst_performing_day_name = "No Trades With Non-Zero P&L"
                    else:
                        best_performing_day_name = "No P&L Data"
                        worst_performing_day_name = "No P&L Data"


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

                    st.markdown("---")

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
                            day_info_text = "N/A"

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
                            day_info_text = "N/A"

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
                            grouped_data['Duration Bin'] = grouped_data['Duration Bin'].apply(lambda x: f"Bin {x}")
                            fig_edge = px.bar(grouped_data, x='Duration Bin', y='sum', title=f"Profit by {analysis_by}")
                            st.plotly_chart(fig_edge, use_container_width=True)
                        else:
                            grouped_data = df.groupby(analysis_by)['Profit'].agg(['sum', 'count', 'mean']).reset_index()
                            fig_edge = px.bar(grouped_data, x=analysis_by, y='sum', title=f"Profit by {analysis_by}")
                            st.plotly_chart(fig_edge, use_container_width=True)

                    else:
                        st.info("Upload your MT5 data to use the Edge Finder.")

                with tab_export:
                    st.subheader("Export Reports")
                    st.write("Export your trading data and reports.")

                    csv_processed = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Processed CSV",
                        data=csv_processed,
                        file_name="processed_mt5_history.csv",
                        mime="text/csv",
                    )
                    
                    st.info("Further reporting options (e.g., custom PDF reports) could be integrated here.")


            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}. Please check your CSV format and ensure it contains the required columns with valid data.")
                logging.error(f"Error processing CSV: {str(e)}", exc_info=True)
                st.session_state.mt5_df = DEFAULT_APP_STATE['mt5_df'].copy()
                st.session_state.selected_calendar_month = DEFAULT_APP_STATE['selected_calendar_month']
    else:
        st.info("👆 Upload your MT5 trading history CSV to explore advanced performance metrics.")
        st.session_state.mt5_df = DEFAULT_APP_STATE['mt5_df'].copy()
        st.session_state.selected_calendar_month = DEFAULT_APP_STATE['selected_calendar_month']

    if st.session_state.mt5_df is not None and not st.session_state.mt5_df.empty:
        try:
            st.markdown("---")
            _ta_show_badges_mt5(st.session_state.mt5_df)
        except Exception as e:
            logging.error(f"Error displaying badges: {str(e)}")
    
    if st.session_state.mt5_df is not None and not st.session_state.mt5_df.empty:
        st.markdown("---")
        st.subheader("🗓️ Daily Performance Calendar")

        df_for_calendar = st.session_state.mt5_df

        selected_month_date = date(datetime.now().year, datetime.now().month, 1)

        if not df_for_calendar.empty and not df_for_calendar["Close Time"].isnull().all():
            min_date_data = pd.to_datetime(df_for_calendar["Close Time"]).min().date()
            max_date_data = pd.to_datetime(df_for_calendar["Close Time"]).max().date()

            all_months_in_data = pd.date_range(start=min_date_data.replace(day=1),
                                                end=max_date_data.replace(day=1), freq='MS').to_period('M')
            available_months_periods = sorted(list(all_months_in_data), reverse=True)

            if available_months_periods:
                display_options = [f"{p.strftime('%B %Y')}" for p in available_months_periods]

                latest_data_month_str = available_months_periods[0].strftime('%B %Y')

                if 'selected_calendar_month' not in st.session_state or st.session_state.selected_calendar_month not in display_options:
                     st.session_state.selected_calendar_month = latest_data_month_str

                selected_month_year_str = st.selectbox(
                    "Select Month",
                    options=display_options,
                    index=display_options.index(st.session_state.selected_calendar_month),
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

        daily_pnl_map_for_calendar = _ta_daily_pnl_mt5(df_for_calendar)

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
                dot_indicator_html = ""

                if day_date.month == selected_month_date.month:
                    profit = daily_pnl_map_for_calendar.get(day_date)
                    if profit is not None:
                        if profit > 0.0:
                            day_class += " profitable"
                            profit_amount_html = f"<span style='color:#5cb85c;'>${_ta_human_num_mt5(profit)}</span>"
                        elif profit < 0.0:
                            day_class += " losing"
                            profit_amount_html = f"<span style='color:#d9534f;'>-${_ta_human_num_mt5(abs(profit))}</span>"
                        else:
                            profit_amount_html = f"<span style='color:#cccccc;'>$0.00</span>"
                    else:
                         profit_amount_html = "<span style='color:#cccccc;'>$0.00</span>"

                else:
                    day_class += " empty-month-day"
                    profit_amount_html = ""

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


    if 'df' in locals() and not df.empty:
        st.markdown("---")
        if st.button("📄 Generate Performance Report"):
            total_trades = len(df)
            wins_df = df[df["Profit"] > 0]
            losses_df = df[df["Profit"] < 0]
            win_rate = len(wins_df) / total_trades if total_trades else 0.0
            net_profit = df["Profit"].sum()
            profit_factor = _ta_profit_factor_mt5(df)
            
            if st.session_state.get('mt5_df', pd.DataFrame()) is not None and not st.session_state.mt5_df.empty:
                daily_pnl_for_streaks = _ta_daily_pnl_mt5(st.session_state.mt5_df)
                streaks = _ta_compute_streaks(pd.DataFrame(list(daily_pnl_for_streaks.items()), columns=['date', 'Profit']))
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
# MANAGE MY STRATEGY PAGE
# =========================================================
elif st.session_state.current_page == 'strategy':
    if st.session_state.logged_in_user is None:
        st.warning("Please log in to manage your strategies.")
        st.session_state.current_page = 'account'
        st.rerun()

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
            st.session_state.strategies = pd.concat([st.session_state.strategies, pd.DataFrame([strategy_data])], ignore_index=True)
            if st.session_state.logged_in_user is not None:
                username = st.session_state.logged_in_user
                try:
                    save_user_data(username)
                    st.success("Strategy saved to your account!")
                    logging.info(f"Strategy saved for {username}: {strategy_name}")
                except Exception as e:
                    st.error(f"Failed to save strategy: {str(e)}")
                    logging.error(f"Error saving strategy for {username}: {str(e)}")
            st.success(f"Strategy '{strategy_name}' added successfully!")

    if not st.session_state.strategies.empty:
        st.subheader("Your Strategies")
        for idx, row in st.session_state.strategies.iterrows():
            with st.expander(f"Strategy: {row['Name']} (Added: {row['Date Added']})"):
                st.markdown(f"Description: {row['Description']}")
                st.markdown(f"Entry Rules: {row['Entry Rules']}")
                st.markdown(f"Exit Rules: {row['Exit Rules']}")
                st.markdown(f"Risk Management: {row['Risk Management']}")
                if st.button("Delete Strategy", key=f"delete_strategy_{idx}"):
                    st.session_state.strategies = st.session_state.strategies.drop(idx).reset_index(drop=True)
                    if st.session_state.logged_in_user is not None:
                        username = st.session_state.logged_in_user
                        try:
                            save_user_data(username)
                            st.success("Strategy deleted and account updated!")
                            logging.info(f"Strategy deleted for {username}")
                        except Exception as e:
                            st.error(f"Failed to delete strategy: {str(e)}")
                            logging.error(f"Error deleting strategy for {username}: {str(e)}")
                    st.rerun()
    else:
        st.info("No strategies defined yet. Add one above.")
    
    st.subheader("📖 Evolving Playbook")
    journal_df = st.session_state.trade_journal
    mt5_df = st.session_state.mt5_df

    combined_df = journal_df.copy()
    
    if not mt5_df.empty and 'Profit' in mt5_df.columns:
        mt5_temp = pd.DataFrame()
        mt5_temp['Date'] = pd.to_datetime(mt5_df['Close Time'], errors='coerce')
        mt5_temp['PnL'] = pd.to_numeric(mt5_df['Profit'], errors='coerce').fillna(0.0)
        mt5_temp['Symbol'] = mt5_df['Symbol']
        mt5_temp['Outcome'] = mt5_df['Profit'].apply(lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Breakeven'))
        if 'Open Price' in mt5_df.columns and 'StopLoss' in mt5_df.columns and 'Close Time' in mt5_df.columns: # Fixed to use mt5_df
             mt5_temp['RR'] = mt5_df.apply(lambda row: (row['Profit'] / abs(row['Open Price'] - row['StopLoss'])) if abs(row['Open Price'] - row['StopLoss']) > 0 else 0, axis=1)
        else:
            mt5_temp['RR'] = 0.0
        
        for col in journal_cols:
            if col not in mt5_temp.columns:
                mt5_temp[col] = pd.Series(dtype=journal_dtypes.get(col))

        combined_df = pd.concat([combined_df, mt5_temp[journal_cols]], ignore_index=True)

    if "RR" in combined_df.columns:
        combined_df['r'] = pd.to_numeric(combined_df['RR'], errors='coerce')
    
    group_cols = ["Symbol"] if "Symbol" in combined_df.columns else []

    if group_cols and 'r' in combined_df.columns and not combined_df['r'].isnull().all():
        g = combined_df.dropna(subset=["r"]).groupby(group_cols)

        res_data = []
        for name, group in g:
            wins_r = group[group['r'] > 0]['r']
            losses_r = group[group['r'] < 0]['r']

            winrate_calc = len(wins_r) / len(group) if len(group) > 0 else 0.0
            avg_win_r = wins_r.mean() if not wins_r.empty else 0.0
            avg_loss_r = abs(losses_r.mean()) if not losses_r.empty else 0.0

            expectancy_calc = (winrate_calc * avg_win_r) - ((1 - winrate_calc) * avg_loss_r)
            
            res_data.append({
                "Symbol": name if isinstance(name, str) else name[0],
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

# =========================================================
# ACCOUNT PAGE
# =========================================================
elif st.session_state.current_page == 'account':
    # This introductory section should ONLY show when the user is NOT logged in.
    if st.session_state.logged_in_user is None:
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
    
        # Tabs for Sign In and Sign Up (only visible when logged_in_user is None)
        tab_signin, tab_signup, tab_debug = st.tabs(["🔑 Sign In", "📝 Sign Up", "🛠 Debug"])
        # --------------------------
        # SIGN IN TAB
        # --------------------------
        with tab_signin:
            st.subheader("Welcome back! Please sign in to access your account.")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username_input")
                password = st.text_input("Password", type="password", key="login_password_input") 
                login_button = st.form_submit_button("Login")
                if login_button:
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()
                    c.execute("SELECT password, data FROM users WHERE username = ?", (username,))
                    result = c.fetchone()
                    if result and result[0] == hashed_password:
                        st.session_state.logged_in_user = username
                        initialize_and_load_session_state() # Reload session state with the new logged-in user's data
                        
                        st.success(f"Welcome back, {username}!")
                        logging.info(f"User {username} logged in successfully")
                        st.rerun() # Crucial for state refresh
                    else:
                        st.error("Invalid username or password.")
                        logging.warning(f"Failed login attempt for {username}")
        # --------------------------
        # SIGN UP TAB
        # --------------------------
        with tab_signup:
            st.subheader("Create a new account to start tracking your trades and progress.")
            with st.form("register_form"):
                new_username = st.text_input("New Username", key="register_username_input")
                new_password = st.text_input("New Password", type="password", key="register_password_input")
                confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password_input") 
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
                            initial_data = json.dumps({
                                "xp": DEFAULT_APP_STATE['xp'],
                                "level": DEFAULT_APP_STATE['level'],
                                "badges": DEFAULT_APP_STATE['badges'],
                                "streak": DEFAULT_APP_STATE['streak'],
                                "last_journal_date": DEFAULT_APP_STATE['last_journal_date'],
                                "last_login_xp_date": DEFAULT_APP_STATE['last_login_xp_date'],
                                "gamification_flags": DEFAULT_APP_STATE['gamification_flags'],
                                "drawings": DEFAULT_APP_STATE['drawings'],
                                "trade_journal": DEFAULT_APP_STATE['trade_journal'].to_dict('records'),
                                "strategies": DEFAULT_APP_STATE['strategies'].to_dict('records'),
                                "emotion_log": DEFAULT_APP_STATE['emotion_log'].to_dict('records'),
                                "reflection_log": DEFAULT_APP_STATE['reflection_log'].to_dict('records'),
                                "xp_log": DEFAULT_APP_STATE['xp_log'],
                                'chatroom_rules_accepted': DEFAULT_APP_STATE['chatroom_rules_accepted'], # New default
                                'chatroom_nickname': None # New default
                            })
                            try:
                                c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", (new_username, hashed_password, initial_data))
                                conn.commit()
                                st.session_state.logged_in_user = new_username
                                # This ensures the session state is properly set for the *new* user's defaults
                                initialize_and_load_session_state() 
                                st.success(f"Account created for {new_username}!")
                                logging.info(f"User {new_username} registered successfully")
                                st.rerun() # Crucial for state refresh
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
    else: # This block displays when a user IS logged in.
        # --------------------------
        # LOGGED-IN USER VIEW
        # --------------------------
        
        def handle_logout():
            if st.session_state.logged_in_user is not None:
                save_user_data(st.session_state.logged_in_user)
                logging.info(f"User {st.session_state.logged_in_user} data saved before logout.")
            
            # Explicitly reset ONLY the user-specific and transient states related to being logged in
            for key in ['logged_in_user', 'current_subpage', 'show_tools_submenu', 'temp_journal',
                        'xp', 'level', 'badges', 'streak', 'last_journal_date',
                        'last_login_xp_date', 'gamification_flags', 'xp_log', 
                        'chatroom_rules_accepted', 'user_nickname', 'forex_fundamentals_progress',
                        'edit_trade_metrics' # Add new UI-specific states
                        ]:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Re-initialize non-user specific global states and then the rest for logged-out context
            initialize_and_load_session_state()

            logging.info("User logged out successfully.")
            st.session_state.current_page = "account"
            st.rerun()

        st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")
        st.markdown("This is your personal dashboard. Track your progress and manage your account.")
        st.markdown("---")
        
        st.subheader("📈 Progress Snapshot")
        
        st.markdown("""
        <style>
        .kpi-card { background-color: rgba(45, 70, 70, 0.5); border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #58b3b1; margin-bottom: 10px; }
        .kpi-icon { font-size: 2.5em; margin-bottom: 10px; }
        .kpi-value { font-size: 1.8em; font-weight: bold; color: #FFFFFF; }
        .kpi-label { font-size: 0.9em; color: #A0A0A0; }
        .insights-card { background-color: rgba(45, 70, 70, 0.3); border-left: 5px solid #58b3b1; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .redeem-card { background-color: rgba(45, 70, 70, 0.5); border-radius: 10px; padding: 20px; border: 1px solid #58b3b1; text-align: center; height: 100%; }
        </style>
        """, unsafe_allow_html=True)

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            level = st.session_state.get('level', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🧙‍♂️</div><div class="kpi-value">Level {level}</div><div class="kpi-label">Trader\'s Rank</div></div>', unsafe_allow_html=True)
        with kpi_col2:
            streak = st.session_state.get('streak', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🔥</div><div class="kpi-value">{streak} Days</div><div class="kpi-label">Journaling Streak</div></div>', unsafe_allow_html=True)
        with kpi_col3:
            total_xp = st.session_state.get('xp', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">⭐</div><div class="kpi-value">{total_xp:,}</div><div class="kpi-label">Total Experience (XP)</div></div>', unsafe_allow_html=True)
        with kpi_col4:
            redeemable_xp = int(st.session_state.get('xp', 0) / 2)
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">💎</div><div class="kpi-value">{redeemable_xp:,}</div><div class="kpi-label">Redeemable XP (RXP)</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")

        chart_col, insights_col = st.columns([1, 2])

        with chart_col:
            st.markdown("<h5 style='text-align: center;'>Progress to Next Level</h5>", unsafe_allow_html=True)
            total_xp = st.session_state.get('xp', 0)
            xp_in_level = total_xp % 100
            xp_needed = 100 - xp_in_level

            fig = go.Figure(go.Pie(
                values=[xp_in_level, xp_needed],
                hole=0.6,
                marker_colors=['#58b3b1', '#2d4646'],
                textinfo='none',
                hoverinfo='label+value'
            ))
            fig.update_layout(
                showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text=f'<b>{xp_in_level}<span style="font-size:0.6em">/100</span></b>', x=0.5, y=0.5, font_size=18, showarrow=False, font_color="white")],
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with insights_col:
            st.markdown("<h5 style='text-align: center;'>Personalized Insights & Badges</h5>", unsafe_allow_html=True)
            insight_sub_col, badge_sub_col = st.columns(2)
            
            with insight_sub_col:
                st.markdown("<h6>💡 Insights</h6>", unsafe_allow_html=True)
                streak = st.session_state.get('streak', 0)
                insight_message = "Your journaling consistency is elite! This is a key trait of professional traders." if streak > 21 else "Over a week of consistent journaling! You're building a powerful habit." if streak > 7 else "Every trade journaled is a step forward. Stay consistent to build a strong foundation."
                st.markdown(f"<div class='insights-card'><p>{insight_message}</p></div>", unsafe_allow_html=True)
                
                num_trades = len(st.session_state.trade_journal) 
                if num_trades < 10: next_milestone = f"Log **{10 - num_trades} more trades** to earn the 'Ten Trades' badge!"
                elif num_trades < 50: next_milestone = f"You're **{50 - num_trades} trades** away from the '50 Club' badge. Keep it up!"
                else: next_milestone = "The next streak badge is at 30 days. You've got this!"
                st.markdown(f"<div class='insights-card'><p>🎯 **Next Up:** {next_milestone}</p></div>", unsafe_allow_html=True)

            with badge_sub_col:
                st.markdown("<h6>🏆 Badges Earned</h6>", unsafe_allow_html=True)
                badges = st.session_state.get('badges', [])
                if badges:
                    for badge in badges: st.markdown(f"- 🏅 {badge}")
                else:
                    st.info("No badges earned yet. Keep trading to unlock them!")
        
        st.markdown("<hr style='border-color: #4d7171;'>", unsafe_allow_html=True)
        # Removed "🚀 Your XP Journey" chart from here
        # Removed "---" separator as the chart above it is now gone.
        
        st.subheader("💎 Redeem Your RXP")
        current_rxp = int(st.session_state.get('xp', 0) / 2)
        st.info(f"You have **{current_rxp:,} RXP** available to spend.")
        
        items = {
            "1_month_access": {"name": "1 Month Free Access", "cost": 1000, "icon": "🗓️"},
            "consultation": {"name": "30-Min Pro Consultation", "cost": 2500, "icon": "🧑‍🏫"},
            "advanced_course": {"name": "Advanced Indicators Course", "cost": 5000, "icon": "📚"}
        }
        redeem_cols = st.columns(len(items))
        for i, (item_key, item_details) in enumerate(items.items()):
            with redeem_cols[i]:
                st.markdown(f'<div class="redeem-card"><h3>{item_details["icon"]}</h3><h5>{item_details["name"]}</h5><p>Cost: <strong>{item_details["cost"]:,} RXP</strong></p></div>', unsafe_allow_html=True)
                if st.button(f"Redeem {item_details['name']}", key=f"redeem_{item_key}", use_container_width=True):
                    if current_rxp >= item_details['cost']:
                        xp_cost = item_details['cost'] * 2
                        ta_update_xp(st.session_state.logged_in_user, -xp_cost, f"Redeemed '{item_details['name']}' ({item_details['cost']} RXP)")
                        
                        st.success(f"Successfully redeemed '{item_details['name']}'! Your RXP has been updated.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("You do not have enough RXP for this item.")

        st.markdown("---")

        # --- XP Transaction History (Now Above How to Earn XP) ---
        st.subheader("📜 Your XP Transaction History")
        
        xp_log_df = pd.DataFrame(st.session_state.get('xp_log', []))

        if not xp_log_df.empty:
            xp_log_df['Date'] = pd.to_datetime(xp_log_df['Date'])
            xp_log_df = xp_log_df.sort_values(by="Date", ascending=False).reset_index(drop=True)

            def style_amount_column_numeric(val):
                if val > 0:
                    return 'color: green; font-weight: bold;'
                elif val < 0:
                    return 'color: red; font-weight: bold;'
                return ''

            styled_xp_log = xp_log_df.style.applymap(style_amount_column_numeric, subset=['Amount'])
            styled_xp_log = styled_xp_log.format({'Amount': lambda x: f'+{int(x)}' if x > 0 else f'{int(x)}'})

            st.dataframe(styled_xp_log, use_container_width=True)
        else:
            st.info("Your XP transaction history is empty. Start interacting to earn XP!")
        # --- END XP Transaction History ---
        
        st.markdown("---")

        # --- How to Earn XP Section (Directly Visible) ---
        st.subheader("❓ How to Earn XP") 
        st.markdown("""
        Earn Experience Points (XP) and unlock new badges as you progress in your trading journey!

        -   **Daily Login**: Log in each day to earn **10 XP** for your consistency.
        -   **Log New Trades**: Get **10 XP** for every trade you meticulously log in your Trading Journal.
        -   **Detailed Notes**: Add substantive notes to your logged trades in the Trade Playbook to earn **5 XP**.
        -   **Trade Milestones**: Achieve trade volume milestones for bonus XP and special badges:
            *   Log 10 Trades: **+20 XP** + "Ten Trades Novice" Badge
            *   Log 50 Trades: **+50 XP** + "Fifty Trades Apprentice" Badge
            *   Log 100 Trades: **+100 XP** + "Centurion Trader" Badge
        -   **Performance Milestones**: Demonstrate trading skill for extra XP and recognition:
            *   Maintain a Profit Factor of 2.0 or higher: **+30 XP**
            *   Achieve an Average R:R of 1.5 or higher: **+25 XP**
            *   Reach a Win Rate of 60% or higher: **+20 XP**
        -   **Level Up!**: Every 100 XP earned levels up your Trader\'s Rank and rewards a new Level Badge.
        -   **Daily Journaling Streak**: Maintain your journaling consistency for streak badges and XP bonuses every 7 days!
        
        Keep exploring the dashboard and trading to earn more XP and climb the ranks!
        """, unsafe_allow_html=True)
        # --- END How to Earn XP Section ---
        
        st.markdown("---")

        # Removed "🏆 Global XP Leaderboard" from here
        # No more render_xp_leaderboard() call.

        st.markdown("---") # Retain a final separator before Manage Account if desired.

        with st.expander("⚙️ Manage Account"):
            st.write(f"**Username**: `{st.session_state.logged_in_user}`")
            st.write("**Email**: `trader.pro@email.com` (example)")
            if st.button("Log Out", key="logout_account_page", type="primary"):
                handle_logout()
# =========================================================
# COMMUNITY TRADE IDEAS PAGE
# =========================================================
elif st.session_state.current_page == 'community':
    if st.session_state.logged_in_user is None:
        st.warning("Please log in to participate in Community Trade Ideas.")
        st.session_state.current_page = 'account'
        st.rerun()

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
            if st.session_state.logged_in_user is not None:
                username = st.session_state.logged_in_user
                user_data_dir = os.path.join("user_data", username) 
                os.makedirs(os.path.join(user_data_dir, "community_images"), exist_ok=True)

                idea_id = _ta_hash()
                idea_data = {
                    "Username": username,
                    "Pair": trade_pair,
                    "Direction": trade_direction,
                    "Description": trade_description,
                    "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "IdeaID": idea_id,
                    "ImagePath": None
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
                    if st.session_state.logged_in_user is not None and st.session_state.logged_in_user == idea["Username"]:
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
    
    st.subheader("📄 Community Templates")
    with st.form("template_form"):
        template_type = st.selectbox("Template Type", ["Journaling Template", "Checklist", "Strategy Playbook"])
        template_name = st.text_input("Template Name")
        template_content = st.text_area("Template Content")
        submit_template = st.form_submit_button("Share Template")
        if submit_template:
            if st.session_state.logged_in_user is not None:
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
                    if st.session_state.logged_in_user is not None and st.session_state.logged_in_user == template["Username"]:
                        st.session_state.community_templates = st.session_state.community_templates.drop(idx).reset_index(drop=True)
                        _ta_save_community('templates', st.session_state.community_templates.to_dict('records'))
                        st.success("Template deleted successfully!")
                        logging.info(f"Template {template['ID']} deleted by {st.session_state.logged_in_user}")
                        st.rerun()
                    else:
                        st.error("You can only delete your own templates.")
    else:
        st.info("No templates shared yet. Share one above!")
    
    st.subheader("🏆 Leaderboard - Consistency")
    users = c.execute("SELECT username, data FROM users").fetchall()
    leader_data = []
    for u, d in users:
        user_d = json.loads(d) if d else {}
        trades = len(user_d.get("trade_journal", []))
        leader_data.append({"Username": u, "Journaled Trades": trades})
    if leader_data:
        leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
        leader_df["Rank"] = leader_df.index + 1
        st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]])
    else:
        st.info("No leaderboard data yet.")

# =========================================================
# COMMUNITY CHATROOM PAGE
# =========================================================
elif st.session_state.current_page == "Community Chatroom":
    # --------------------------
    # CHATROOM HELPER FUNCTIONS & CONFIG (local to this page)
    # --------------------------
    NICKNAME_COLORS = [
        "#8be9fd", # Cyan (Dracula theme inspired)
        "#50fa7b", # Green (Dracula theme inspired)
        "#ff79c6", # Pink (Dracula theme inspired)
        "#bd93f9", # Purple (Dracula theme inspired)
        "#ffb86c", # Orange (Dracula theme inspired)
        "#f1fa8c", # Yellow (Dracula theme inspired)
        "#A8C8F7", # Light Blue
        "#FFE599", # Light Gold
        "#A7D9B4", # Pastel Green
        "#CBA6C2", # Lavender
    ]

    def get_user_nickname_color(nickname):
        """
        Generates a consistent color for a given nickname based on its hash.
        Ensures the same user always has the same color.
        """
        hash_value = int(hashlib.sha256(nickname.encode('utf-8')).hexdigest(), 16)
        return NICKNAME_COLORS[hash_value % len(NICKNAME_COLORS)]

    def add_chat_message(channel, nickname, message_content, message_type="regular"):
        """Adds a new message to the specified chat channel's history and persists it."""
        timestamp = dt.datetime.now().strftime("[%H:%M]")
        message_data = {
            "timestamp": timestamp,
            "nickname": nickname,
            "content": message_content,
            "type": message_type
        }
        st.session_state.chat_messages[channel].append(message_data)
        _ta_save_community(f'chat_channel_{channel.lower().replace(" ", "_")}', st.session_state.chat_messages[channel])

    def check_chat_gamification_triggers(username, channel, message_content):
        """
        Awards XP and badges based on user activity in the chatroom.
        """
        if username is None:
            return # Cannot award XP/badges to an unknown user

        user_data = get_user_data(username) # Fetches user data directly to access total_messages if needed

        # Always award 1 XP for sending a message
        ta_update_xp(username, 1, "sending a chat message")

        # Check for "Active Chatter" badge
        # This count needs to be retrieved across ALL chat messages for this user
        total_messages_for_user = 0
        for ch_key, messages_list in st.session_state.chat_messages.items():
            # Ensure comparison is to the current user's *chat nickname*
            if st.session_state.user_nickname: # Only count if nickname is set
                total_messages_for_user += sum(1 for msg in messages_list if msg['nickname'] == st.session_state.user_nickname)

        if total_messages_for_user >= 10: # Assuming 10 messages for initial Active Chatter
            ta_award_badge(username, "Active Chatter")

        # XP/Badges specifically for "Trade Reviews" channel
        if channel == "Trade Reviews":
            message_lower = message_content.lower()
            if "winning trade:" in message_lower or "win:" in message_lower or "✅ win" in message_lower:
                ta_update_xp(username, 5, "posting a winning trade review in chat")
                # Count for Winning Trader badge specific to chat, can be merged with Journal's wins for higher level
                chat_winning_trades_count = sum(1 for msg in st.session_state.chat_messages["Trade Reviews"]
                                                if st.session_state.user_nickname and msg['nickname'] == st.session_state.user_nickname and (
                                                    "winning trade:" in msg['content'].lower() or "win:" in msg['content'].lower() or "✅ win" in msg['content'].lower()))
                if chat_winning_trades_count >= 3: # Example: 3 winning chat reviews for badge
                    ta_award_badge(username, "Community Winner") # New badge for chat activity

            elif "losing trade:" in message_lower or "loss:" in message_lower or "❌ loss" in message_lower:
                ta_update_xp(username, 2, "posting a detailed losing trade analysis in chat")
                # Count for Resilient Analyst badge specific to chat
                chat_losing_trades_count = sum(1 for msg in st.session_state.chat_messages["Trade Reviews"]
                                               if st.session_state.user_nickname and msg['nickname'] == st.session_state.user_nickname and (
                                                   "losing trade:" in msg['content'].lower() or "loss:" in msg['content'].lower() or "❌ loss" in msg['content'].lower()))
                if chat_losing_trades_count >= 5: # Example: 5 losing chat reviews for badge
                    ta_award_badge(username, "Community Resilient") # New badge for chat activity


    # --------------------------
    # CHATROOM MAIN LOGIC
    # --------------------------

    # Crucial check: Ensure user is logged in
    if st.session_state.logged_in_user is None:
        st.warning("Please log in to access the Community Chatroom.")
        st.session_state.current_page = 'account' # Redirect to your login page
        st.stop() # Stop execution of this block if not logged in

    st.title("💬 Community Chatroom")
    st.caption("Connect, collaborate, and grow your trading performance with fellow traders.")
    st.markdown('---')


    # 1. Mandatory Rules Acceptance
    if not st.session_state.chatroom_rules_accepted:
        st.subheader("Community Chatroom Rules")
        st.markdown("""
        Please read and accept the following rules to join the Zenvo Academy Community Chatroom. These rules are in place
        to maintain a **positive, focused, and professional environment** for all traders.

        1.  **Respect and Professionalism:** Treat all members with courtesy and respect. Personal attacks, harassment, or derogatory comments will not be tolerated.
        2.  **Stay On Topic:** Keep discussions relevant to the selected channel's theme.
        3.  **No Financial Advice:** Share *analysis* and *trade ideas*, but explicitly state these are not financial advice. Members are responsible for their own trading decisions.
        4.  **No Spam or Promotion:** Do not post unsolicited advertisements, promotional content, or engage in self-promotion unrelated to trade reviews.
        5.  **Constructive Criticism Only:** Provide helpful, objective feedback on trades and analysis. Avoid purely negative or dismissive comments.
        6.  **Positive and Supportive Environment:** Contribute to a growth-oriented mindset. Encourage and support fellow traders, especially when reviewing losses.
        7.  **Data Privacy:** Do not share personal information about yourself or others.

        ---
        **Violations of these rules may result in temporary suspension or permanent ban from the chatroom.**
        """)
        st.markdown("---")
        if st.button("I Agree to the Rules and Wish to Enter"):
            st.session_state.chatroom_rules_accepted = True
            save_user_data(st.session_state.logged_in_user) # Persist acceptance
            st.rerun()
        st.stop() # Prevent further rendering until rules are accepted


    # 2. One-time Nickname Setup
    if st.session_state.user_nickname is None:
        st.subheader("Set Your Chatroom Nickname")
        st.markdown("This nickname will be permanently visible to others in the chatroom. Choose wisely!")
        
        # Pre-fill with logged-in username suggestion
        suggested_nickname = st.session_state.logged_in_user 
        new_nickname = st.text_input("Enter your desired nickname (max 20 characters):", 
                                     value=suggested_nickname, max_chars=20)
        
        if st.button("Set Nickname", type="primary"):
            if new_nickname:
                # In a real app, you'd add backend logic to check for nickname uniqueness globally
                # For this example, we'll assume it's unique enough for session.
                st.session_state.user_nickname = new_nickname
                # Update and save the nickname in user's permanent data via save_user_data
                save_user_data(st.session_state.logged_in_user)

                st.success(f"Welcome, **{new_nickname}**! Your nickname has been set. You can now join the chat.")
                st.rerun()
            else:
                st.warning("Nickname cannot be empty.")
        st.stop() # Prevent further rendering until nickname is set

    # Fetch current user's XP for display (badges is covered by overall Account page)
    current_user_data = get_user_data(st.session_state.logged_in_user)
    current_user_xp = current_user_data.get('xp', 0)


    # Display user's current status and an encouragement
    st.markdown(f"""
        **👋 Welcome, {st.session_state.user_nickname}!**
        <small style='color: #888;'>Current XP: **{current_user_xp}**</small>
    """, unsafe_allow_html=True)


    # Define tabs for different chat channels
    channel_tabs_names = list(DEFAULT_APP_STATE['chat_messages'].keys())

    tabs = st.tabs(channel_tabs_names) # Create tabs using Streamlit

    for i, channel_name in enumerate(channel_tabs_names):
        with tabs[i]:
            # Channel-specific title and description
            channel_caption_map = {
                "General Discussion": "Casual conversation and community building.",
                "Trading Psychology": "Focus, mindset, and overcoming trading losses.",
                "Trade Reviews": "Post-trade analysis, sharing charts, and performance discussion.",
                "Market News": "Real-time economic news, forex events, and sentiment."
            }
            st.markdown(f"### {channel_name} <small style='color: #aaa;'>— {channel_caption_map.get(channel_name, 'Join the discussion!')}</small>", unsafe_allow_html=True)
            st.markdown("---")

            # Container for displaying chat messages (fixed height, scrollable)
            chat_history_container = st.container(height=450, border=True)

            with chat_history_container:
                # Display each message in the channel
                if not st.session_state.chat_messages[channel_name]:
                    st.info(f"No messages in the **{channel_name}** channel yet. Be the first to start a conversation!")
                else:
                    for message in st.session_state.chat_messages[channel_name]:
                        nickname = message['nickname']
                        timestamp = message['timestamp']
                        content = message['content']
                        msg_type = message.get('type', 'regular') # Default to regular
                        
                        user_color = get_user_nickname_color(nickname)

                        # Custom HTML for message rendering with professional trading aesthetics
                        if msg_type == 'regular':
                            st.markdown(
                                f"<div style='margin-bottom: 8px;'>"
                                f"  <p style='margin-bottom: 0px; font-size: 0.9em;'>"
                                f"    <strong style='color:{user_color};'>{nickname}</strong>"
                                f"    <small style='color:#777777; margin-left: 5px;'>{timestamp}</small>"
                                f"  </p>"
                                f"  <p style='margin-top: 2px; margin-left: 15px;'>{content}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        elif msg_type == 'win_trade':
                            st.markdown(
                                f"<div style='margin-bottom: 8px; padding: 5px; border-left: 3px solid #00c767; background-color: #0b1f15; border-radius: 4px;'>"
                                f"  <p style='margin-bottom: 0px; font-size: 0.9em;'>"
                                f"    <strong style='color:{user_color};'>{nickname}</strong>"
                                f"    <small style='color:#777777; margin-left: 5px;'>{timestamp}</small>"
                                f"  </p>"
                                f"  <p style='margin-top: 2px; margin-left: 15px; color:#50fa7b; font-weight:bold;'>✅ WINNING TRADE: {content}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        elif msg_type == 'loss_trade':
                            st.markdown(
                                f"<div style='margin-bottom: 8px; padding: 5px; border-left: 3px solid #ff4d4f; background-color: #260d0d; border-radius: 4px;'>"
                                f"  <p style='margin-bottom: 0px; font-size: 0.9em;'>"
                                f"    <strong style='color:{user_color};'>{nickname}</strong>"
                                f"    <small style='color:#777777; margin-left: 5px;'>{timestamp}</small>"
                                f"  </p>"
                                f"  <p style='margin-top: 2px; margin-left: 15px; color:#ff5555; font-weight:bold;'>❌ LEARNING FROM LOSS: {content}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        # Auto-scroll to bottom of chat_history_container could be added here via JS if needed

            # Message input area for the current channel
            current_message_content = st.chat_input(
                f"Message {channel_name} as {st.session_state.user_nickname}...",
                key=f"chat_input_{channel_name}", # Unique key for each chat_input
                max_chars=500 # Limit message length
            )

            # Logic to handle message submission
            if current_message_content:
                message_type_to_add = "regular"
                # Apply specific message types for 'Trade Reviews' for distinct styling and gamification
                if channel_name == "Trade Reviews":
                    content_lower = current_message_content.lower()
                    if "win:" in content_lower or "winning trade:" in content_lower or "✅ win" in content_lower:
                        message_type_to_add = "win_trade"
                    elif "loss:" in content_lower or "losing trade:" in content_lower or "❌ loss" in content_lower:
                        message_type_to_add = "loss_trade"
                
                # Add message to history (which also persists to DB)
                add_chat_message(channel_name, st.session_state.user_nickname, current_message_content, message_type_to_add)
                
                # Trigger gamification check
                check_chat_gamification_triggers(st.session_state.logged_in_user, channel_name, current_message_content)
                
                # Rerun the app to clear the chat input and display the new message
                st.rerun()

    # Encouraging footer message
    st.markdown("---")
    st.info("""
        💡 **Community Focus**: This chatroom is dedicated to improving trading performance through collaboration and positive support.
        Let's keep discussions constructive and relevant to our trading goals!
    """)


# =========================================================
# TOOLS PAGE
# =========================================================
elif st.session_state.current_page == 'tools':
    if st.session_state.logged_in_user is None:
        st.warning("Please log in to access the Tools section.")
        st.session_state.current_page = 'account'
        st.rerun()

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
            position_size = st.number_input("Position Size (lots)", min_value=0.01, value=0.1, step=0.01, format="%.2f", key="pl_position_size")
            close_price = st.number_input("Close Price", value=1.1050, step=0.0001, format="%.5f", key="pl_close_price")
        with col_calc2:
            account_currency = st.selectbox("Account Currency", ["USD", "EUR", "GBP", "JPY"], index=0, key="pl_account_currency")
            open_price = st.number_input("Open Price", value=1.1000, step=0.0001, format="%.5f", key="pl_open_price")
            trade_direction = st.radio("Trade Direction", ["Long", "Short"], key="pl_trade_direction")
        
        pip_size_for_pair_calc = 0.0001
        if "JPY" in currency_pair:
            pip_size_for_pair_calc = 0.01
        
        pip_movement = abs(close_price - open_price) / pip_size_for_pair_calc

        usd_per_pip_per_standard_lot = 10.0 # Universal assumed value for calculation simplicity

        price_change = close_price - open_price
        if trade_direction == "Short":
            price_change = -price_change
            
        profit_loss = (price_change / pip_size_for_pair_calc) * (position_size * usd_per_pip_per_standard_lot)
        
        if account_currency == "EUR":
            profit_loss *= 0.92
        elif account_currency == "GBP":
            profit_loss *= 0.80
        elif account_currency == "JPY":
            profit_loss *= 150 
        
        st.write(f"Pip Movement: {pip_movement:.2f} pips")
        value_per_pip_for_position = position_size * usd_per_pip_per_standard_lot
        st.write(f"Estimated Value Per Pip: {value_per_pip_for_position:.2f} USD (for {position_size:.2f} lots)")
        st.write(f"Potential Profit/Loss: {profit_loss:.2f} {account_currency}")

    # --------------------------
    # PRICE ALERTS
    # --------------------------
    with tabs[1]:
        st.header("⏰ Price Alerts")
        st.markdown("Set price alerts for your favourite forex pairs and get notified when the price hits your target.")
        st.markdown('---')
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY"]
        
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

        active_pairs = st.session_state.price_alerts["Pair"].unique().tolist() if not st.session_state.price_alerts.empty else []
        live_prices = {}
        if active_pairs:
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
        if not st.session_state.price_alerts.empty:
            for idx, row in st.session_state.price_alerts.iterrows():
                pair = row["Pair"]
                target = row["Target Price"]
                current_price = live_prices.get(pair)
                if isinstance(current_price, (int, float)):
                    tolerance = 0.0005
                    if "JPY" in pair:
                        tolerance = 0.01

                    if not row["Triggered"] and abs(current_price - target) <= tolerance:
                        st.session_state.price_alerts.at[idx, "Triggered"] = True
                        triggered_alerts.append((idx, f"{pair} reached {target:.5f} (Current: {current_price:.5f})"))
                        logging.info(f"Alert triggered: {pair} at {target}")

            if triggered_alerts:
                for idx, alert_msg in triggered_alerts:
                    st.balloons()
                    st.success(f"⚡ {alert_msg}")

        if not st.session_state.price_alerts.empty and any(not row["Triggered"] for _, row in st.session_state.price_alerts.iterrows()):
             st_autorefresh(interval=5000, key="price_alert_autorefresh")

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
        
        calculated_pip_value_per_lot = 10.0
        if "$7" in pip_value_currency:
            calculated_pip_value_per_lot = 7.0
            
        if st.button("Calculate Lot Size"):
            if stop_loss_pips <= 0 or calculated_pip_value_per_lot <= 0:
                st.error("Stop Loss (pips) and Pip Value must be positive numbers.")
            else:
                risk_amount = balance * (risk_percent / 100)
                lot_size = risk_amount / (stop_loss_pips * calculated_pip_value_per_lot)
                st.success(f"✅ Recommended Lot Size: {lot_size:.2f} lots")
                logging.info(f"Calculated lot size: {lot_size}")
        
        st.subheader('🔄 What-If Analyzer')
        base_equity = st.number_input('Starting Equity', value=10000.0, min_value=0.0, step=100.0, key='whatif_equity')
        risk_pct = st.slider('Risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key='whatif_risk') / 100.0
        winrate = st.slider('Win rate (%)', 10.0, 90.0, 50.0, 1.0, key='whatif_wr') / 100.0
        avg_r = st.slider('Average R multiple', 0.5, 5.0, 1.5, 0.1, key='whatif_avg_r')
        trades = st.slider('Number of trades', 10, 500, 100, 10, key='whatif_trades')
        
        E_R = (winrate * avg_r) - ((1 - winrate) * 1.0)
        
        if (1 + risk_pct * E_R) <= 0:
            exp_growth = 0.0
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
        
        if (1 + risk_pct * E_R) <= 0:
            sim_equity_base = [base_equity] + [0.0] * trades
        else:
            sim_equity_base = base_equity * (1 + risk_pct * E_R) ** np.arange(trades + 1)
        
        if (1 + alt_risk * E_R) <= 0:
            sim_equity_alt = [base_equity] + [0.0] * trades
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
        mt5_df = st.session_state.mt5_df
        
        current_journal_df = st.session_state.trade_journal.copy()

        def assign_session(timestamp):
            if pd.isna(timestamp):
                return 'Unknown'
            hour = timestamp.hour
            if 0 <= hour < 9: return 'Tokyo'
            if 8 <= hour < 17: return 'London'
            if 13 <= hour < 22: return 'New York'
            return 'Sydney'

        if not current_journal_df.empty:
            current_journal_df['datetime'] = pd.to_datetime(current_journal_df['Date'], errors='coerce')
            current_journal_df['r'] = pd.to_numeric(current_journal_df['RR'], errors='coerce')
            current_journal_df['session'] = current_journal_df['datetime'].apply(assign_session)
            current_journal_df = current_journal_df.dropna(subset=['datetime', 'r'])

        df_sessions_combined = current_journal_df.copy()
        
        if not mt5_df.empty:
            mt5_for_sessions = mt5_df.copy()
            mt5_for_sessions['datetime'] = pd.to_datetime(mt5_for_sessions['Close Time'], errors='coerce')
            mt5_for_sessions['r'] = pd.to_numeric(mt5_for_sessions['Profit'], errors='coerce')
            mt5_for_sessions['session'] = mt5_for_sessions['datetime'].apply(assign_session)
            mt5_for_sessions = mt5_for_sessions.dropna(subset=['datetime', 'r'])
            
            cols_to_combine = ['datetime', 'r', 'session']
            if 'Symbol' in current_journal_df.columns and 'Symbol' in mt5_for_sessions.columns:
                cols_to_combine.append('Symbol')
            
            filtered_journal = current_journal_df[current_journal_df.columns.intersection(cols_to_combine)]
            filtered_mt5 = mt5_for_sessions[mt5_for_sessions.columns.intersection(cols_to_combine)]

            df_sessions_combined = pd.concat([filtered_journal, filtered_mt5], ignore_index=True)

        if not df_sessions_combined.empty and 'session' in df_sessions_combined.columns and not df_sessions_combined['r'].isnull().all():
            
            def _ta_expectancy_by_group_session(df_input, group_cols):
                g = df_input.dropna(subset=["r"]).groupby(group_cols)
                res = g["r"].agg(
                    trades="count",
                    winrate=lambda s: (s>0).mean(),
                    avg_win=lambda s: s[s>0].mean() if (s>0).any() else 0.0,
                    avg_loss=lambda s: -s[s<0].mean() if (s<0).any() else 0.0,
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
                if local_hour < start:
                    time_diff_hours = start - local_hour
                else:
                    time_diff_hours = (24 - local_hour) + start
                time_until_label = "Opens in"
            else:
                if local_hour < end:
                    time_diff_hours = end - local_hour
                time_until_label = "Closes in"

            session_status.append({
                "Session": session["name"],
                "Status": "Open" if is_open else "Closed",
                "Local Time": local_time.strftime("%H:%M"),
                "Time Until": f"{time_diff_hours:.1f}" if (is_open or time_diff_hours > 0) else "0.0"
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
        recovery_pct_val = _ta_percent_gain_to_recover(drawdown_pct)
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

        expectancy_sim = (win_rate * avg_rr) - ((1 - win_rate) * 1.0)
        
        trades_needed = 0
        if (1 + risk_per_trade * expectancy_sim) <= 0:
            st.error("Expected growth factor is non-positive, recovery not possible under these conditions. Adjust risk or expectancy.")
            trades_needed = float('inf')
        elif drawdown_pct >= 1.0:
             trades_needed = float('inf')
        elif (1 - drawdown_pct) <= 0 :
            trades_needed = float('inf')
        elif (1 + risk_per_trade * expectancy_sim) == 1.0:
             trades_needed = float('inf')
        elif expectancy_sim <= 0:
            trades_needed = float('inf')
        else:
            try:
                target_multiplier = 1 / (1 - drawdown_pct)
                
                numerator = math.log(target_multiplier) 
                denominator = math.log(1 + risk_per_trade * expectancy_sim)
                trades_needed = math.ceil(numerator / denominator) if denominator != 0 else float('inf')

            except (ValueError, ZeroDivisionError):
                trades_needed = float('inf')

        st.write(f"Estimated Trades to Recover: {trades_needed if trades_needed != float('inf') else 'Infinite (Impossible)'}")

        sim_equity = [initial_equity * (1 - drawdown_pct)]
        if trades_needed != float('inf') :
            for _ in range(min(trades_needed + 10, 100)):
                if len(sim_equity) > 0 and (1 + risk_per_trade * expectancy_sim) > 0:
                    sim_equity.append(sim_equity[-1] * (1 + risk_per_trade * expectancy_sim))
                else:
                    sim_equity.append(0.0)
                    break 

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
                st.session_state.reflection_log = pd.concat([st.session_state.reflection_log, pd.DataFrame([log_entry])], ignore_index=True)
                if st.session_state.logged_in_user is not None:
                    username = st.session_state.logged_in_user
                    try:
                        save_user_data(username)
                    except Exception as e:
                        logging.error(f"Error saving reflection: {str(e)}")
                st.success("Reflection logged!")
        if not st.session_state.reflection_log.empty:
            st.dataframe(st.session_state.reflection_log)

# =========================================================
# ZENVO ACADEMY PAGE
# =========================================================
elif st.session_state.current_page == "Zenvo Academy":
    if st.session_state.logged_in_user is None:
        st.warning("Please log in to access the Zenvo Academy.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.title("📚 Zenvo Academy")
    st.caption("Your journey to trading mastery starts here. Explore interactive courses, track your progress, and unlock your potential.")
    st.markdown('---')

    tab1, tab2, tab3 = st.tabs(["🎓 Learning Path", "📈 My Progress", "🛠️ Resources"])

    with tab1:
        st.markdown("### 🗺️ Your Learning Path")
        st.write("Our Academy provides a clear learning path for traders of all levels. Start from the beginning or jump into a topic that interests you.")

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
                can_start = st.session_state.logged_in_user is not None
                if st.button("Start Learning", key="start_forex_fundamentals", disabled=not can_start):
                    if not can_start:
                        st.warning("Please log in to start learning!")
                    else:
                        st.info("Starting 'Forex Fundamentals' module!")
                        username = st.session_state.logged_in_user
                        user_data = get_user_data(username)
                        completed_courses = user_data.get('completed_courses', [])
                        
                        if "Forex Fundamentals" not in completed_courses:
                            completed_courses.append("Forex Fundamentals")
                            user_data['completed_courses'] = completed_courses
                            save_user_data(username)
                            ta_update_xp(username, 100, "Completed 'Forex Fundamentals' course")
                            st.rerun()
                        else:
                            st.info("You have already completed 'Forex Fundamentals'.")
                
                st.progress(st.session_state.get('forex_fundamentals_progress', 0))

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
                can_start_ta = st.session_state.logged_in_user is not None and st.session_state.get('level', 0) >= 1
                if st.button("Start Course", key="start_technical_analysis", disabled=not can_start_ta):
                    if not can_start_ta:
                        if st.session_state.logged_in_user is None:
                             st.warning("Please log in to start this course!")
                        else:
                            st.warning(f"You need to reach Level 1 (Current: Level {st.session_state.get('level', 0)}) to start this course!")
                    else:
                        st.info("Starting 'Technical Analysis 101' module!")
                        username = st.session_state.logged_in_user
                        user_data = get_user_data(username)
                        completed_courses = user_data.get('completed_courses', [])
                        
                        if "Technical Analysis 101" not in completed_courses:
                            completed_courses.append("Technical Analysis 101")
                            user_data['completed_courses'] = completed_courses
                            save_user_data(username)
                            ta_update_xp(username, 150, "Completed 'Technical Analysis 101' course")
                            st.rerun()
                        else:
                            st.info("You have already completed 'Technical Analysis 101'.")


    with tab2:
        st.markdown("### 🚀 Your Progress")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Level", st.session_state.get('level', 0))
        with col2:
            st.metric("Experience Points (XP)", f"{st.session_state.get('xp', 0)} / {(st.session_state.get('level', 0) + 1) * 100}")
        with col3:
            badges_count = len(st.session_state.get('badges', []))
            st.metric("Badges Earned", badges_count)

        st.markdown("---")
        st.markdown("#### 📜 Completed Courses")
        completed_courses = []
        if st.session_state.logged_in_user is not None:
             user_data_acad = get_user_data(st.session_state.logged_in_user)
             completed_courses = user_data_acad.get('completed_courses', [])

        if completed_courses:
            for course in completed_courses:
                st.success(f"**{course}** - Completed!")
        else:
            st.info("You haven't completed any courses yet. Get started on the Learning Path!")

        st.markdown("#### 🎖️ Your Badges")
        current_badges_list = st.session_state.get('badges', [])
        if current_badges_list:
            for badge in current_badges_list:
                st.markdown(f"- 🏅 {badge}")
        else:
            st.info("No badges earned yet. Complete courses to unlock them!")


    with tab3:
        st.markdown("### 🧰 Trading Resources")
        st.info("This section is under development. Soon you will find helpful tools, articles, and more to aid your trading journey!")
