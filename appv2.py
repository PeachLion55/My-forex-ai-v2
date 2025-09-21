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



import streamlit as st
import base64
import os
from streamlit_card import card  # Ensure this is installed (pip install streamlit-card)

# =========================================================
# URL-BASED NAVIGATION ROUTER (THE CRITICAL FIX)
# This code runs at the top and reads the URL to set the page.
# This makes navigation 100% reliable, overriding any other conflicting state changes.
# =========================================================
query_params = st.query_params.to_dict()
if "page" in query_params:
    # Set the session state from the URL's 'page' parameter
    st.session_state.current_page = query_params["page"][0]
elif 'current_page' not in st.session_state:
    # If there's no page in the URL and the app is just starting, set a default
    st.session_state.current_page = 'fundamentals'

# =========================================================
# HELPER FUNCTION TO ENCODE IMAGES
# =========================================================================
def get_image_as_base_64(path):
    """Encodes a local image file into a Base64 string for embedding."""
    if not os.path.exists(path):
        st.warning(f"Icon file not found, skipping icon: {path}")
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# =========================================================
# SIDEBAR (FINAL, 100% RELIABLE VERSION)
# =========================================================================
with st.sidebar:
    # --- LOGO DISPLAY ---
    logo_path = "logo22.png"
    if os.path.exists(logo_path):
        logo_base_64 = get_image_as_base_64(logo_path)
        if logo_base_64:
            st.markdown(
                f"""
                <div style='text-align: center; margin-top: -60px; margin-bottom: 25px;'>
                    <img src="data:image/png;base64,{logo_base_64}" width="60">
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- NAVIGATION ITEMS & ICONS ---
    nav_items = [
        ('fundamentals', 'Forex Fundamentals', 'forex_fundamentals.png'),
        ('watch list', 'My Watchlist', 'watchlist_icon.png'),
        ('trading_journal', 'My Trading Journal', 'trading_journal.png'),
        ('mt5', 'Performance Dashboard', 'performance_dashboard.png'),
        ('trading_tools', 'Trading Tools', 'trading_tools.png'),
        ('strategy', 'Manage My Strategy', 'manage_my_strategy.png'),
        ('Community Chatroom', 'Community Chatroom', 'community_chatroom.png'),
        ('Zenvo Academy', 'Zenvo Academy', 'zenvo_academy.png'),
        ('account', 'My Account', 'my_account.png'),
    ]

    # --- RELIABLE NAVIGATION FUNCTION ---
    # This function changes the URL, which is the most robust way to navigate.
    def navigate_to(page_key):
        # We need to set query_params which forces a rerun and changes the URL
        st.query_params["page"] = page_key

    # --- Generate Icon Buttons ---
    for page_key, page_name, icon_filename in nav_items:
        # Check if the page determined by the URL is this one
        is_active = (st.session_state.get('current_page') == page_key)
        icon_path = os.path.join("icons", icon_filename)
        icon_base_64 = get_image_as_base_64(icon_path)

        if icon_base_64:
            # --- CSS TO FIX BLURRY / DARKENED ICONS ---
            card_styles = {
                "card": {
                    "width": "55px", "height": "55px", "margin": "5px auto", "padding": "0",
                    "border-radius": "10px", "background-color": "transparent", "cursor": "pointer",
                    # Active state gets a bright border and an inner glow. Inactive is a clean, dark border.
                    "border": f"2px solid {'#FFFFFF' if is_active else 'rgba(88,179,177,0.4)'}",
                    "box-shadow": f"{'inset 0 0 8px rgba(255, 255, 255, 0.7)' if is_active else 'none'}",
                    "transition": "all 0.2s ease-in-out",
                },
                "div": {"padding": "0"},
                "img": { # CSS for the icon image
                    "width": "32px", "height": "32px",
                    "margin": "auto", "display": "block",
                    # Filter: brightness(1) ensures no darkening. Tweak if needed.
                    "filter": "brightness(1.0)",
                    # Force sharp, non-blurry rendering in browsers
                    "image-rendering": "-webkit-optimize-contrast",
                    "image-rendering": "pixelated",
                    "image-rendering": "crisp-edges",
                },
                "title": {"display": "none"}, "text": {"display": "none"}
            }

            # Use on_click to call our URL-changing function
            card(
                title=page_name,
                text="", image=f"data:image/png;base64,{icon_base_64}",
                styles=card_styles,
                key=page_key,
                on_click=lambda page=page_key: navigate_to(page)
            )
# =========================================================
# 1. IMPORTS
# =========================================================
import streamlit as st
import os
import io
import base64
import logging
import sqlite3
import json
import hashlib
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import time
import pytz
from datetime import datetime, timedelta

# =========================================================
# 2. LOGGING & DATABASE SETUP
# =========================================================
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_FILE = "users.db"

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
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
# 3. GLOBAL SESSION STATE INITIALIZATION
# =========================================================
if 'logged_in_user' not in st.session_state: st.session_state.logged_in_user = None
if 'current_page' not in st.session_state: st.session_state.current_page = 'account'
if 'user_nickname' not in st.session_state: st.session_state.user_nickname = None
if 'user_timezone' not in st.session_state: st.session_state.user_timezone = 'Europe/London'
if 'session_timings' not in st.session_state:
    st.session_state.session_timings = {
        "Sydney": {"start": 22, "end": 7},
        "Tokyo": {"start": 0, "end": 9},
        "London": {"start": 8, "end": 17},
        "New York": {"start": 13, "end": 22}
    }
if 'auth_view' not in st.session_state: st.session_state.auth_view = 'login'
# (Add any other global initializations your app needs for other pages)

# =========================================================
# 4. GLOBAL HELPER FUNCTIONS
# =========================================================
@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def handle_logout():
    """Handles user logout by clearing all session state variables."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    st.session_state.current_page = "account"
    st.rerun()

# --- FINAL, CORRECTED, PURE UTC get_active_market_sessions FUNCTION ---
def get_active_market_sessions():
    """
    Determines active forex sessions by directly comparing the current UTC hour
    against the session's defined UTC start/end hours. This is the most robust method.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    current_utc_hour = datetime.now(pytz.utc).hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        
        if start > end: # Overnight session (e.g., Sydney)
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else: # Same-day session (e.g., London)
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed"
    return ", ".join(active_sessions)

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
    (CORRECTED VERSION)
    Updates user XP, checks for level up, logs the transaction, and persists data
    WITHOUT causing a script rerun, which allows navigation to work.
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
        "Date": dt.date.today().isoformat(),
        "Amount": amount,
        "Description": description
    })

    save_user_data(username) # Persist all changes
    if amount != 0:
        show_xp_notification(amount) # This function is safe and does not rerun.

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
# MAIN APPLICATION LOGIC
# =========================================================

import streamlit as st
import os
import io
import base64
import pytz
from datetime import datetime, timedelta
import logging

# =========================================================
# HELPER FUNCTIONS (Included here for completeness, but should ideally be global)
# =========================================================

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end:
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else:
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    sessions_utc_hours = st.session_state.get('session_timings', {})
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Use corrected time
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if start_hour > end_hour and now_utc.hour >= end_hour:
                end_time_today += timedelta(days=1)
            elif now_utc > end_time_today:
                end_time_today += timedelta(days=1)

            next_end_times.append((end_time_today, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort()
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0:
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return soonest_session_name, time_str

# =========================================================
# FUNDAMENTALS PAGE
# =========================================================
if st.session_state.current_page == 'fundamentals':

    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to access Forex Fundamentals.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page-Specific Configuration ---
    page_info = {
        'title': 'Forex Fundamentals', 
        'icon': 'forex_fundamentals.png', 
        'caption': 'Macro snapshot, calendar highlights, and policy rates.'
    }

    # --- 2. Define CSS Styles for the New Header ---
    main_container_style = """
        background-color: black; padding: 20px 25px; border-radius: 10px; 
        display: flex; align-items: center; gap: 20px;
        border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);
    """
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"

    # --- 3. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    if icon_base64:
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">'
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 4. Build and Render Header ---
    header_html = (
        f'<div style="{main_container_style}">'
            f'<div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div>'
            f'<div style="{right_column_style}">'
                f'<div style="{info_tab_style}">{welcome_message}</div>'
                f'<div>'
                    f'<div style="{info_tab_style}">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown('---')
    st.markdown("### Upcoming Economic Events This Week")

    # (Your other Streamlit elements for this page go here...)
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

    st.markdown("### 💹 Current Major Central Bank Interest Rates")
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
    st.markdown("### 📊 Major High-Impact Forex Events Explained")
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
                background-color:#0000;
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

import streamlit as st
import os
import io
import base64
import pytz
from datetime import datetime, timedelta
import logging

# =========================================================
# HELPER FUNCTIONS (Included here for completeness, but should ideally be global)
# =========================================================

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end:
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else:
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    sessions_utc_hours = st.session_state.get('session_timings', {})
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Use corrected time
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if start_hour > end_hour and now_utc.hour >= end_hour:
                end_time_today += timedelta(days=1)
            elif now_utc > end_time_today:
                end_time_today += timedelta(days=1)

            next_end_times.append((end_time_today, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort()
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0:
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return soonest_session_name, time_str

# =========================================================
# TRADING JOURNAL PAGE
# =========================================================
if st.session_state.current_page == 'trading_journal':

    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to access your Trading Journal.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page-Specific Configuration ---
    page_info = {
        'title': 'My Trading Journal', 
        'icon': 'trading_journal.png', 
        'caption': 'A streamlined interface for professional trade analysis.'
    }

    # --- 2. Define CSS Styles for the New Header ---
    main_container_style = """
        background-color: black; padding: 20px 25px; border-radius: 10px; 
        display: flex; align-items: center; gap: 20px;
        border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);
    """
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"

    # --- 3. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    if icon_base64:
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">'
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 4. Build and Render Header ---
    header_html = (
        f'<div style="{main_container_style}">'
            f'<div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div>'
            f'<div style="{right_column_style}">'
                f'<div style="{info_tab_style}">{welcome_message}</div>'
                f'<div>'
                    f'<div style="{info_tab_style}">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")


    # --- 6. RETAINED CONTENT FROM ORIGINAL PAGE ---
    # The tab functionality from your original code is preserved and placed here.
    tab_entry, tab_playbook, tab_analytics = st.tabs(["**📝 Log New Trade**", "**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])
    
    # (Your code for each tab goes here...)
    
    # (Your code for each tab goes here...)

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
        
        # ========== START: GUARANTEED CSS OVERRIDE FOR st.button ==========
        # This CSS is now much more specific and forceful, and will work.
        st.markdown(
            """
            <style>
                div[data-testid="stColumn"] > div[data-testid="stHorizontalBlock"] {
                    position: relative;
                }

                /* Create a wrapper class to apply to the st.button's column */
                .st-emotion-cache-12w0qpk {
                    position: absolute;
                    top: 2px;
                    right: 3px;
                    z-index: 10;
                }
                
                /* Aggressively target the button inside the wrapper */
                .st-emotion-cache-12w0qpk button {
                    /* --- CRITICAL SIZE OVERRIDES --- */
                    font-size: 10px !important;
                    height: 1.1rem !important;  /* Make button vertically tiny */
                    min-height: 1.1rem !important;
                    width: 1.1rem !important;   /* Make button horizontally tiny */
                    min-width: 1.1rem !important;
                    padding: 0 !important;
                    line-height: 0 !important; /* Center the icon */

                    /* Appearance */
                    background: transparent !important;
                    color: #999 !important;
                    border: none !important;
                }

                .st-emotion-cache-12w0qpk button:hover {
                    color: #fff !important; 
                    background: rgba(100, 100, 100, 0.3) !important;
                }
            </style>
            """, unsafe_allow_html=True
        )
        # ========== END: CSS OVERRIDE ==========

        if df_playbook.empty:
            st.info("Your logged trades will appear here as playbook cards. Log your first trade to get started!")
        else:
            st.caption("Filter and review your past trades to refine your strategy and identify patterns.")
            
            if 'edit_state' not in st.session_state:
                st.session_state.edit_state = {}

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

            for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
                trade_id_key = row['TradeID']
                outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e", "No Trade/Study": "#58a6ff"}.get(row['Outcome'], "#30363d")

                with st.container(border=True):
                    # Trade Header - MODIFIED BLOCK
                    st.markdown(f"""
                    <div style="display: flex; flex-direction: row; align-items: stretch; gap: 15px; margin-left: -10px;">
                      <div style="width: 4px; background-color: {outcome_color}; border-radius: 3px;"></div>
                      <div style="padding-top: 2px; padding-bottom: 2px;">
                        <div style="font-size: 1.1em; font-weight: 600;">
                          {row['Symbol']} <span style="font-weight: 500; color: {outcome_color};">{row['Direction']} / {row['Outcome']}</span>
                        </div>
                        <div style="color: #8b949e; font-size: 0.9em; margin-top: 2px;">
                          {row['Date'].strftime('%A, %d %B %Y')} | {trade_id_key}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")

                    # Metrics Section
                    metric_cols = st.columns(3)
                    
                    pnl_val = float(pd.to_numeric(row.get('PnL', 0.0), errors='coerce') or 0.0)
                    rr_val = float(pd.to_numeric(row.get('RR', 0.0), errors='coerce') or 0.0)
                    lots_val = float(pd.to_numeric(row.get('Lots', 0.01), errors='coerce') or 0.01)

                    def render_metric_cell_or_form(col_obj, metric_label, db_column, current_value, key_suffix, format_str, is_pnl_metric=False):
                        is_editing = st.session_state.edit_state.get(f"{key_suffix}_{trade_id_key}", False)
                        
                        main_col, button_col = col_obj.columns([4, 1])

                        with main_col:
                            if is_editing:
                                with st.form(f"form_{key_suffix}_{trade_id_key}", clear_on_submit=False):
                                    st.markdown(f"**Edit {metric_label}**")
                                    new_value = st.number_input("", value=current_value, format=format_str, key=f"input_{key_suffix}_{trade_id_key}", label_visibility="collapsed")
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
                                
                                st.markdown(f"""
                                    <div class='playbook-metric-display' style='{border_style}'>
                                        <div class='label'>{metric_label}</div>
                                        {display_val_str}
                                    </div>""", unsafe_allow_html=True)
                        
                        with button_col:
                            st.markdown('<div class="st-emotion-cache-12w0qpk">', unsafe_allow_html=True)
                            if not is_editing:
                                if st.button("✏️", key=f"edit_btn_{key_suffix}_{trade_id_key}", help=f"Edit {metric_label}"):
                                    st.session_state.edit_state[f"{key_suffix}_{trade_id_key}"] = True
                                    st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

                    render_metric_cell_or_form(metric_cols[0], "Net PnL", "PnL", pnl_val, "pnl", "%.2f", is_pnl_metric=True)
                    render_metric_cell_or_form(metric_cols[1], "R-Multiple", "RR", rr_val, "rr", "%.2f")
                    render_metric_cell_or_form(metric_cols[2], "Position Size", "Lots", lots_val, "lots", "%.2f")
                    
                    st.markdown("---")

                    if row['EntryRationale']: st.markdown(f"**Entry Rationale:** *{row['EntryRationale']}*")
                    if row['Tags']:
                        tags_list = [f"`{tag.strip()}`" for tag in str(row['Tags']).split(',') if tag.strip()]
                        if tags_list: st.markdown(f"**Tags:** {', '.join(tags_list)}")
                    
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
                            st.session_state.trade_journal.loc[index, 'TradeJournalNotes'] = notes
                            
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
                            xp_deduction_amount += 10

                            gamification_flags = st.session_state.get('gamification_flags', {})
                            notes_award_key_for_deleted = f"xp_notes_for_trade_{trade_id_key}_content_hash"
                            if notes_award_key_for_deleted in gamification_flags:
                                xp_deduction_amount += 5
                                del gamification_flags[notes_award_key_for_deleted]
                            
                            if trade_id_key in st.session_state.edit_state:
                                for key in list(st.session_state.edit_state.keys()):
                                    if trade_id_key in key:
                                        del st.session_state.edit_state[key]
                            
                            st.session_state.gamification_flags = gamification_flags
                            
                            if xp_deduction_amount > 0:
                                ta_update_xp(username, -xp_deduction_amount, f"Deleted trade {row['TradeID']}")
                                st.toast(f"Trade {row['TradeID']} deleted. {xp_deduction_amount} XP deducted.", icon="🗑️")
                            else:
                                st.toast(f"Trade {row['TradeID']} deleted.", icon="🗑️")

                            st.session_state.trade_journal.drop(index, inplace=True)
                            st.session_state.trade_journal.reset_index(drop=True, inplace=True)
                            
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
                            
                    st.markdown("---") 

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

            pnl_metric_color = "#2da44e" if total_pnl >= 0 else "#cf222e"
            pnl_value_color_inner = "#50fa7b" if total_pnl >= 0 else "#ff5555"
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


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import calendar
import io
import base64
import pytz # For timezone-aware datetime operations
import requests # For Myfxbook API calls
import json # For handling JSON responses
from datetime import datetime, date, timedelta
import logging

# =========================================================
# GLOBAL/APP-WIDE HELPER FUNCTIONS
# =========================================================

# Ensure basic logging is set up if not already
if 'logging' not in globals():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        # Adjust path to be relative to the script's directory for robust deployment
        script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
        full_path = os.path.join(script_dir, path)

        if os.path.exists(full_path):
            with open(full_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        else:
            logging.warning(f"Warning: Image file not found at path: {full_path}")
            return None
    except Exception as e:
        logging.error(f"Error encoding image {path}: {e}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock, as per user's original request.
    """
    # Initialize session_timings in session_state if not present
    st.session_state.setdefault('session_timings', {
        'Sydney': {'start': 22, 'end': 7},    # 10 PM - 7 AM UTC
        'Tokyo': {'start': 0, 'end': 9},      # 12 AM - 9 AM UTC
        'London': {'start': 7, 'end': 16},    # 7 AM - 4 PM UTC
        'New York': {'start': 12, 'end': 21}  # 12 PM - 9 PM UTC
    })

    sessions_utc = st.session_state.session_timings
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1) # Apply user's specified 1-hour correction
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end: # Session crosses midnight UTC
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else: # Session within the same UTC day
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    # Initialize session_timings in session_state if not present
    st.session_state.setdefault('session_timings', {
        'Sydney': {'start': 22, 'end': 7},
        'Tokyo': {'start': 0, 'end': 9},
        'London': {'start': 7, 'end': 16},
        'New York': {'start': 12, 'end': 21}
    })

    sessions_utc_hours = st.session_state.session_timings
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Apply user's specified 1-hour correction
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_candidate = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            # Adjust end_time_candidate if the session has already ended for the current day
            if start_hour > end_hour: # Session crosses midnight
                if now_utc.hour >= start_hour: # Currently in the "start" part before midnight
                    end_time_candidate = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
                elif now_utc.hour < end_hour: # Currently in the "end" part after midnight
                    end_time_candidate = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            else: # Session within the same day
                if now_utc >= end_time_candidate: # If today's session already ended, consider tomorrow's
                    end_time_candidate += timedelta(days=1)
            
            # Final safeguard to ensure end_time_candidate is in the future
            while end_time_candidate < now_utc:
                end_time_candidate += timedelta(days=1)

            next_end_times.append((end_time_candidate, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort(key=lambda x: x[0]) # Sort by timestamp
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0: # Should not happen with corrected logic, but as a safeguard
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return soonest_session_name, time_str


# --- Global/App-wide Session State Initialization ---
# This ensures that 'mt5_df' and 'selected_calendar_month' always have a default
# even if the user hasn't loaded data yet.
if 'DEFAULT_APP_STATE' not in st.session_state:
    st.session_state.DEFAULT_APP_STATE = {
        'mt5_df': pd.DataFrame(columns=["Symbol", "Type", "Profit", "Volume", "Open Time", "Close Time", "Trade Duration"]),
        'selected_calendar_month': datetime.now().strftime('%B %Y')
    }
st.session_state.setdefault('mt5_df', st.session_state.DEFAULT_APP_STATE['mt5_df'].copy())
st.session_state.setdefault('selected_calendar_month', st.session_state.DEFAULT_APP_STATE['selected_calendar_month'])
st.session_state.setdefault('myfxbook_df_loaded', False) # Flag to indicate if Myfxbook data is active
st.session_state.setdefault('myfxbook_open_trades_df', pd.DataFrame()) # Initialize open trades DataFrame


# =========================================================
# PERFORMANCE DASHBOARD PAGE (MT5)
# =========================================================
if st.session_state.current_page == 'mt5':

    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to access the Performance Dashboard.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page-Specific Configuration ---
    page_info = {
        'title': 'Performance Dashboard',
        'icon': 'icons/performance_dashboard.png', # Ensure this path is correct relative to script
        'caption': 'Analyze your MT5 trading history with advanced metrics and visualizations.'
    }

    # --- 2. Define CSS Styles for the New Header and Expanders ---
    st.markdown(
        """
        <style>
        /* Main Header Container */
        .main-header-container {
            background-color: black; padding: 20px 25px; border-radius: 10px;
            display: flex; align-items: center; gap: 20px;
            border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);
        }
        .header-left-column {
            flex: 3; display: flex; align-items: center; gap: 20px;
        }
        .header-right-column {
            flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
        }
        .info-tab {
            background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;
        }
        .timer-display {
            font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;
        }
        .header-title {
            color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;
        }
        .icon-style {
            width: 130px; height: auto;
        }
        .header-caption {
            color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;
        }

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
        .metric-box .trade-info { /* Renamed for clarity, from .day-info */
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
        
        /* Streamlit Expander Styling */
        .streamlit-expanderHeader {
            background-color: #1a1a1a;
            color: #ffffff;
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #2d4646;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .streamlit-expanderHeader:hover,
        .streamlit-expanderHeader:focus,
        .streamlit-expanderHeader:active {
            color: #58b3b1 !important;
        }

        .streamlit-expanderContent {
            background-color: #262730;
            border-top: 1px solid #3d3d4b;
            padding: 15px 20px;
            border-radius: 0 0 8px 8px;
        }

        .streamlit-expanderContent .stDataFrame {
            background-color: transparent;
        }
        .streamlit-expanderContent .stDataFrame table {
            background-color: #2d2e37;
            border: 1px solid #3d3d4b;
        }
        .streamlit-expanderContent .stDataFrame th {
            background-color: #3d3d4b;
            color: #ffffff;
            font-weight: 600;
        }
        .streamlit-expanderContent .stDataFrame td {
            color: #e0e0e0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- 3. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_base64 = image_to_base_64(page_info['icon'])
    if icon_base64:
        # FIX: Corrected from class="{icon_style}" to style="{icon_style}"
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="width: 130px; height: auto;">'

    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'

    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div class="timer-display">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 4. Build and Render Header ---
    header_html = (
        f'<div class="main-header-container">'
            f'<div class="header-left-column">{icon_html}<div><h1 class="header-title">{page_info["title"]}</h1><p class="header-caption">{page_info["caption"]}</p></div></div>'
            f'<div class="header-right-column">'
                f'<div class="info-tab">{welcome_message}</div>'
                f'<div>'
                    f'<div class="info-tab">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")

    # --------------------------
    # Helper functions (MT5 page specific, updated for filtering)
    # --------------------------
    def _ta_human_pct_mt5(value):
        try:
            if value is None or pd.isna(value):
                return "N/A"
            return f"{float(value) * 100:.2f}%"
        except Exception:
            return "N/A"

    def _ta_human_num_mt5(value):
        try:
            if value is None or pd.isna(value):
                return "N/A"
            float_val = float(value)
            return f"{float_val:,.2f}"
        except (ValueError, TypeError):
            return "N/A"

    def _ta_compute_sharpe(df_trades, risk_free_rate=0.02):
        # Ensure data is filtered by symbol for Sharpe ratio
        df_filtered = df_trades[df_trades['Symbol'].notna()].copy()
        if "Profit" not in df_filtered.columns or df_filtered.empty:
            return np.nan

        df_for_sharpe = df_filtered.copy()
        df_for_sharpe["Close Time"] = pd.to_datetime(df_for_sharpe["Close Time"], errors='coerce')
        df_for_sharpe = df_for_sharpe.dropna(subset=["Close Time"])

        if df_for_sharpe.empty:
            return np.nan

        daily_pnl_series = df_for_sharpe.set_index("Close Time")["Profit"].resample('D').sum().fillna(0.0)

        if daily_pnl_series.empty or len(daily_pnl_series) < 2:
            return np.nan

        initial_capital = 10000 # Assume an initial capital for percentage calculation - this is just for example
        equity_curve = initial_capital + daily_pnl_series.cumsum()
        returns_pct = equity_curve.pct_change().dropna()

        if returns_pct.empty:
            return np.nan

        annualized_mean_return = returns_pct.mean() * 252
        annualized_std_dev = returns_pct.std() * np.sqrt(252)

        if annualized_std_dev == 0:
            return np.nan # Avoid division by zero

        sharpe = (annualized_mean_return - risk_free_rate) / annualized_std_dev
        return sharpe


    def _ta_daily_pnl_mt5(df_trades):
        """
        Returns a dictionary mapping datetime.date to total profit for that day.
        Only includes trades with a valid symbol.
        """
        df_copy = df_trades.copy()
        df_copy_filtered = df_copy[df_copy['Symbol'].notna()] # Ensure only trades with symbols are considered
        if "Close Time" in df_copy_filtered.columns and "Profit" in df_copy_filtered.columns and not df_copy_filtered.empty and not df_copy_filtered["Profit"].isnull().all():
            df_copy_filtered["date"] = pd.to_datetime(df_copy_filtered["Close Time"]).dt.date
            return df_copy_filtered.groupby("date")["Profit"].sum().to_dict()
        return {}

    def _ta_profit_factor_mt5(df_trades):
        # Ensure only trades with symbols are considered
        df_filtered = df_trades[df_trades['Symbol'].notna()]
        wins_sum = df_filtered[df_filtered["Profit"] > 0]["Profit"].sum()
        losses_sum = abs(df_filtered[df_filtered["Profit"] < 0]["Profit"].sum())
        return wins_sum / losses_sum if losses_sum != 0.0 else (np.inf if wins_sum > 0 else np.nan)


    def _ta_calculate_trading_score(trades_df):
        """
        Calculates a simple trading score based on filtered trade data (symbols only),
        excluding deposit amounts.
        """
        if trades_df.empty:
            return 0.0

        total_trades = len(trades_df)
        wins_df = trades_df[trades_df["Profit"] > 0]
        losses_df = trades_df[trades_df["Profit"] < 0]

        win_rate = len(wins_df) / total_trades if total_trades else 0.0
        profit_factor = _ta_profit_factor_mt5(trades_df) # Already uses trades_df, excludes deposits
        avg_win = wins_df["Profit"].mean() if not wins_df.empty else 0.0
        avg_loss = losses_df["Profit"].mean() if not losses_df.empty else 0.0
        
        # Avoid division by zero for R:R if avg_loss is 0
        avg_r_r = avg_win / abs(avg_loss) if avg_loss != 0.0 else (np.inf if avg_win > 0 else 0.0)

        # Scoring components - these weights are illustrative and can be tuned
        score_win_rate = win_rate * 40 # Max 40 points for 100% win rate
        
        score_profit_factor = 0
        if pd.notna(profit_factor) and profit_factor > 0:
            # Cap profit factor to prevent extreme values from dominating the score
            score_profit_factor = min(profit_factor, 3.0) * 20 # Max 60 points for PF >= 3
            
        score_r_r = 0
        if pd.notna(avg_r_r) and avg_r_r > 0:
            # Cap R:R for similar reasons
            score_r_r = min(avg_r_r, 2.0) * 10 # Max 20 points for R:R >= 2

        # Sum components
        raw_score = score_win_rate + score_profit_factor + score_r_r

        # Normalize to 0-100. Max possible raw score (40 + 60 + 20) = 120
        max_possible_raw_score = 120.0
        normalized_score = (raw_score / max_possible_raw_score) * 100 if max_possible_raw_score > 0 else 0

        # Ensure the score is within the 0-100 range
        return min(max(0.0, normalized_score), 100.0)


    # ----------------------------------------------------
    # Myfxbook API Integration Section
    # ----------------------------------------------------
    with st.expander("🔗 Integrate Myfxbook Account", expanded=not st.session_state.myfxbook_df_loaded):
        st.write("Login to your Myfxbook account to automatically fetch your trade data.")

        myfxbook_email = st.text_input("Myfxbook Email", key="myfxbook_email_input")
        myfxbook_password = st.text_input("Myfxbook Password", type="password", key="myfxbook_password_input")

        if st.button("Login to Myfxbook & Fetch Accounts"):
            if myfxbook_email and myfxbook_password:
                with st.spinner("Logging into Myfxbook..."):
                    try:
                        login_url = f"https://www.myfxbook.com/api/login.json?email={myfxbook_email}&password={myfxbook_password}"
                        response = requests.get(login_url)
                        response.raise_for_status()
                        login_data = response.json()

                        if not login_data["error"]:
                            session_id = login_data["session"]
                            st.session_state.myfxbook_session = session_id
                            st.success("Successfully logged into Myfxbook!")

                            accounts_url = f"https://www.myfxbook.com/api/get-my-accounts.json?session={session_id}"
                            accounts_response = requests.get(accounts_url)
                            accounts_response.raise_for_status()
                            accounts_data = accounts_response.json()

                            if not accounts_data["error"] and accounts_data.get("accounts"):
                                st.session_state.myfxbook_accounts = accounts_data["accounts"]
                                st.session_state.myfxbook_logged_in = True
                                st.rerun()
                            else:
                                st.error("Could not fetch Myfxbook accounts. " + accounts_data.get("message", "Unknown error."))
                                st.session_state.myfxbook_logged_in = False
                        else:
                            st.error("Myfxbook login failed: " + login_data.get("message", "Unknown error."))
                            st.session_state.myfxbook_logged_in = False
                    except requests.exceptions.RequestException as e:
                        st.error(f"Network or API error during Myfxbook login: {e}. Please check your internet connection and Myfxbook API status.")
                        logging.error(f"Myfxbook login error: {e}", exc_info=True)
                        st.session_state.myfxbook_logged_in = False
                    except json.JSONDecodeError:
                        st.error("Failed to parse Myfxbook login response. The API might be returning non-JSON data or is unavailable.")
                        logging.error("Myfxbook login JSONDecodeError", exc_info=True)
                        st.session_state.myfxbook_logged_in = False
                    except Exception as e:
                        st.error(f"An unexpected error occurred during Myfxbook login: {e}")
                        logging.error(f"Myfxbook unexpected login error: {e}", exc_info=True)
                        st.session_state.myfxbook_logged_in = False
            else:
                st.warning("Please enter your Myfxbook email and password.")

        if st.session_state.get('myfxbook_logged_in') and st.session_state.get('myfxbook_accounts'):
            account_names = {acc['name']: acc['id'] for acc in st.session_state.myfxbook_accounts}
            
            initial_account_name = None
            if 'myfxbook_selected_account_name' in st.session_state and st.session_state.myfxbook_selected_account_name in account_names:
                initial_account_name = st.session_state.myfxbook_selected_account_name
            elif account_names:
                initial_account_name = list(account_names.keys())[0]

            if account_names:
                selected_account_name = st.selectbox(
                    "Choose an account",
                    list(account_names.keys()),
                    index=list(account_names.keys()).index(initial_account_name) if initial_account_name else 0,
                    key="myfxbook_account_selector"
                )
                st.session_state.myfxbook_selected_account_name = selected_account_name
            else:
                st.warning("No Myfxbook accounts found or selectable.")
                selected_account_name = None

            if selected_account_name:
                selected_account_id = account_names[selected_account_name]
                st.session_state.myfxbook_selected_account_id = selected_account_id

                if st.button(f"Load Trade Data from {selected_account_name}"):
                    with st.spinner(f"Fetching data for {selected_account_name} from Myfxbook..."):
                        try:
                            session_id = st.session_state.myfxbook_session
                            account_id = st.session_state.myfxbook_selected_account_id

                            history_url = f"https://www.myfxbook.com/api/get-history.json?session={session_id}&id={account_id}"
                            history_response = requests.get(history_url)
                            history_response.raise_for_status()
                            history_data = history_response.json()

                            df_history = pd.DataFrame()
                            if not history_data["error"] and history_data.get("history"):
                                df_history = pd.DataFrame(history_data["history"])
                                df_history = df_history.rename(columns={
                                    "openTime": "Open Time",
                                    "closeTime": "Close Time",
                                    "symbol": "Symbol",
                                    "action": "Type",
                                    "profit": "Profit"
                                })
                                df_history['Volume'] = df_history['sizing'].apply(lambda x: x.get('value') if isinstance(x, dict) else pd.NA)
                                df_history['Volume'] = pd.to_numeric(df_history['Volume'], errors='coerce')
                                df_history["Type"] = df_history["Type"].apply(lambda x: x.split(" ")[0] if isinstance(x, str) and " " in x else x)
                                df_history = df_history[['Open Time', 'Close Time', 'Symbol', 'Type', 'Profit', 'Volume']]
                                logging.info(f"Fetched {len(df_history)} history trades.")
                            else:
                                st.info("No closed trade history found for this account via Myfxbook API (or API limit reached).")
                                logging.warning("Myfxbook history data empty or error: %s", history_data.get("message", "No message"))

                            open_trades_url = f"https://www.myfxbook.com/api/get-open-trades.json?session={session_id}&id={account_id}"
                            open_trades_response = requests.get(open_trades_url)
                            open_trades_response.raise_for_status()
                            open_trades_data = open_trades_response.json()

                            df_open_trades = pd.DataFrame()
                            if not open_trades_data["error"] and open_trades_data.get("openTrades"):
                                df_open_trades = pd.DataFrame(open_trades_data["openTrades"])
                                df_open_trades = df_open_trades.rename(columns={
                                    "openTime": "Open Time",
                                    "symbol": "Symbol",
                                    "action": "Type",
                                    "profit": "Profit"
                                })
                                df_open_trades['Close Time'] = pd.NaT
                                df_open_trades['closePrice'] = np.nan
                                df_open_trades['Volume'] = df_open_trades['sizing'].apply(lambda x: x.get('value') if isinstance(x, dict) else pd.NA)
                                df_open_trades['Volume'] = pd.to_numeric(df_open_trades['Volume'], errors='coerce')
                                df_open_trades["Type"] = df_open_trades["Type"].apply(lambda x: x.split(" ")[0] if isinstance(x, str) and " " in x else x)
                                
                                # Select and rename columns for display in "Open Trades" tab
                                df_open_trades_display = df_open_trades[['Open Time', 'Symbol', 'Type', 'openPrice', 'Profit', 'Volume']].copy()
                                st.session_state.myfxbook_open_trades_df = df_open_trades_display # Store open trades in session state
                                logging.info(f"Fetched {len(df_open_trades)} open trades.")
                            else:
                                st.info("No open trades found for this account.")
                                st.session_state.myfxbook_open_trades_df = pd.DataFrame() # Clear open trades if none found

                            df_to_process = df_history.copy()

                            # Ensure required columns exist and are correct type before filtering
                            required_cols_for_processing = ["Symbol", "Type", "Profit", "Volume", "Open Time", "Close Time"]
                            for col in required_cols_for_processing:
                                if col not in df_to_process.columns:
                                    df_to_process[col] = pd.NA

                            df_to_process["Open Time"] = pd.to_datetime(df_to_process["Open Time"], errors="coerce")
                            df_to_process["Close Time"] = pd.to_datetime(df_to_process["Close Time"], errors="coerce")
                            df_to_process["Profit"] = pd.to_numeric(df_to_process["Profit"], errors='coerce').fillna(0.0)
                            df_to_process["Volume"] = pd.to_numeric(df_to_process["Volume"], errors='coerce').fillna(0.0)

                            # CRITICAL: Filter out non-symbol (e.g., deposit/withdrawal) rows early
                            df_processed = df_to_process[df_to_process['Symbol'].notna()].copy()
                            df_processed = df_processed.dropna(subset=["Open Time", "Profit", "Close Time"]) # Only consider closed trades for main analysis

                            if df_processed.empty:
                                st.warning("No valid closed trading data found for the selected Myfxbook account after processing. Please check if the account has closed trades with symbols.")
                                st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                                st.session_state.myfxbook_df_loaded = False
                            else:
                                # Calculate Trade Duration for Myfxbook data if not present
                                if 'Trade Duration' not in df_processed.columns:
                                    df_processed['Trade Duration'] = (df_processed["Close Time"] - df_processed["Open Time"]).dt.total_seconds() / 3600
                                st.session_state.mt5_df = df_processed
                                st.session_state.myfxbook_df_loaded = True
                                st.success(f"Successfully loaded {len(df_processed)} closed trades from Myfxbook.")

                            st.rerun()

                        except requests.exceptions.RequestException as e:
                            st.error(f"Network or API error during Myfxbook data fetch: {e}. Please check your internet connection or Myfxbook API status.")
                            logging.error(f"Myfxbook data fetch error: {e}", exc_info=True)
                            st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                            st.session_state.myfxbook_df_loaded = False
                            st.session_state.myfxbook_open_trades_df = pd.DataFrame() # Clear open trades
                        except json.JSONDecodeError:
                            st.error("Failed to parse Myfxbook data response. The API might be returning non-JSON data or is unavailable.")
                            logging.error("Myfxbook data fetch JSONDecodeError", exc_info=True)
                            st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                            st.session_state.myfxbook_df_loaded = False
                            st.session_state.myfxbook_open_trades_df = pd.DataFrame() # Clear open trades
                        except Exception as e:
                            st.error(f"An unexpected error occurred during Myfxbook data processing: {e}")
                            logging.error(f"Myfxbook data processing error: {e}", exc_info=True)
                            st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                            st.session_state.myfxbook_df_loaded = False
                            st.session_state.myfxbook_open_trades_df = pd.DataFrame() # Clear open trades

    # --------------------------
    # File Uploader (MT5 page)
    # --------------------------
    df = st.session_state.get('mt5_df', st.session_state.DEFAULT_APP_STATE['mt5_df'].copy())

    # Only show CSV uploader if Myfxbook data is NOT currently loaded
    if not st.session_state.get('myfxbook_df_loaded', False):
        with st.expander("📁 Upload MT5 History CSV", expanded=df.empty):
            uploaded_file = st.file_uploader(
                "Upload your MT5 trading history CSV file here:",
                type=["csv"],
                help="Export your trading history from MetaTrader 5 as a CSV file.",
                label_visibility="collapsed" # Hide the default label
            )

            if uploaded_file:
                with st.spinner("Processing trading data from CSV..."):
                    try:
                        df_raw_csv = pd.read_csv(uploaded_file)
                        
                        # CRITICAL: Filter out any rows without a symbol immediately for CSV data
                        df_filtered_csv = df_raw_csv[df_raw_csv['Symbol'].notna()].copy()
                        
                        st.session_state.mt5_df = df_filtered_csv # Update session state with new df
                        st.session_state.myfxbook_open_trades_df = pd.DataFrame() # Clear open trades if CSV is uploaded

                        required_cols = ["Symbol", "Type", "Profit", "Volume", "Open Time", "Close Time"]
                        missing_cols = [col for col in required_cols if col not in df_filtered_csv.columns]
                        if missing_cols:
                            st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}. Please ensure your CSV has all necessary columns.")
                            st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                            st.session_state.selected_calendar_month = st.session_state.DEFAULT_APP_STATE['selected_calendar_month']
                            st.stop()

                        df_filtered_csv["Open Time"] = pd.to_datetime(df_filtered_csv["Open Time"], errors="coerce")
                        df_filtered_csv["Close Time"] = pd.to_datetime(df_filtered_csv["Close Time"], errors="coerce")
                        df_filtered_csv["Profit"] = pd.to_numeric(df_filtered_csv["Profit"], errors='coerce').fillna(0.0)
                        df_filtered_csv["Volume"] = pd.to_numeric(df_filtered_csv["Volume"], errors='coerce').fillna(0.0)
                        
                        # Only keep rows where both Open and Close Times are valid for trade duration and calendar
                        df_filtered_csv.dropna(subset=["Open Time", "Close Time"], inplace=True)

                        if df_filtered_csv.empty:
                            st.error("Uploaded CSV resulted in no valid trading data after processing timestamps or profits (ensure symbols are present).")
                            st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                            st.session_state.selected_calendar_month = st.session_state.DEFAULT_APP_STATE['selected_calendar_month']
                            st.stop()

                        df_filtered_csv["Trade Duration"] = (df_filtered_csv["Close Time"] - df_filtered_csv["Open Time"]).dt.total_seconds() / 3600
                        st.success("CSV data loaded and processed successfully!")
                        st.session_state.myfxbook_df_loaded = False # Ensure this is false if CSV is used
                        st.session_state.mt5_df = df_filtered_csv # Update session state with the fully processed DF
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing CSV: {str(e)}. Please check your CSV format and ensure it contains the required columns with valid data.")
                        logging.error(f"Error processing CSV: {str(e)}", exc_info=True)
                        st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                        st.session_state.selected_calendar_month = st.session_state.DEFAULT_APP_STATE['selected_calendar_month']
            else:
                if df.empty and not st.session_state.get('myfxbook_df_loaded', False):
                    st.info("👆 Upload your MT5 trading history CSV or login to Myfxbook above to explore advanced performance metrics.")
                    st.session_state.mt5_df = st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()
                    st.session_state.selected_calendar_month = st.session_state.DEFAULT_APP_STATE['selected_calendar_month']
                    st.session_state.myfxbook_open_trades_df = pd.DataFrame() # Ensure open trades are clear if no data loaded
    else:
        st.success("📊 Data is currently loaded from your Myfxbook account.")
        df = st.session_state.get('mt5_df', st.session_state.DEFAULT_APP_STATE['mt5_df'].copy()) # Ensure df is taken from session state if Myfxbook data was loaded


    # The main dashboard display logic (metrics, charts, calendar) starts here
    # It now operates on `df`, which is either from Myfxbook or CSV, and is guaranteed to be filtered for symbols.
    if df.empty and st.session_state.myfxbook_open_trades_df.empty:
        st.info("👆 Load your trading data (closed trades via Myfxbook/CSV, or open trades via Myfxbook) using one of the methods above to see your performance dashboard.")
    else:
        # Re-calculate daily_pnl_map and daily_pnl_df_for_stats with the loaded and filtered df
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
            # Fallback if daily_pnl_map is empty but Close Time exists (e.g., all 0 profit days or very sparse data)
            min_date_raw = df['Close Time'].min().date()
            max_date_raw = df['Close Time'].max().date()
            all_dates_raw_range = pd.date_range(start=min_date_raw, end=max_date_raw).date
            daily_pnl_df_for_stats = pd.DataFrame([
                {"date": d, "Profit": 0.0} for d in all_dates_raw_range
            ])

        tab_summary, tab_charts, tab_closed_trades, tab_open_trades, tab_edge, tab_export = st.tabs([
            "📈 Summary Metrics",
            "📊 Visualizations",
            "📜 Closed Trades",
            "📊 Open Trades",
            "🔍 Edge Finder",
            "📤 Export Reports"
        ])

        with tab_summary:
            st.subheader("Key Performance Metrics")

            # --- KEY CHANGE: Filter DataFrame for trades only ---
            # Create a new DataFrame that excludes deposits or other non-trade activities.
            # We assume that any row with a valid 'Symbol' is a trade.
            trades_df = df[df['Symbol'].notna() & (df['Symbol'] != '')].copy()

            # All calculations below now use 'trades_df' instead of the original 'df'.
            total_trades = len(trades_df)
            wins_df = trades_df[trades_df["Profit"] > 0]
            losses_df = trades_df[trades_df["Profit"] < 0]

            win_rate = len(wins_df) / total_trades if total_trades else 0.0
            net_profit = trades_df["Profit"].sum()
            profit_factor = _ta_profit_factor_mt5(trades_df)
            avg_win = wins_df["Profit"].mean() if not wins_df.empty else 0.0
            avg_loss = losses_df["Profit"].mean() if not losses_df.empty else 0.0
            total_losses_sum = abs(losses_df["Profit"].sum()) if not losses_df.empty else 0.0

            # Note: Ensure 'daily_pnl_df_for_stats' is also calculated using only trade data.
            # The calculation of max_drawdown correctly uses daily_pnl_df_for_stats which is derived from filtered trades.
            max_drawdown = (daily_pnl_df_for_stats["Profit"].cumsum() - daily_pnl_df_for_stats["Profit"].cumsum().cummax()).min() if not daily_pnl_df_for_stats.empty else 0.0
            sharpe_ratio = _ta_compute_sharpe(trades_df)
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if total_trades else 0.0

            def _ta_compute_streaks(df_pnl_daily):
                d = df_pnl_daily.sort_values(by="date")
                if d.empty:
                    return {"current_win": 0, "best_win": 0, "best_loss": 0}

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

            # --- TASK 1 FIX: TRADING SCORE CALCULATION (Not a Placeholder) ---
            # Now calculating a real trading score based on filtered trade data.
            trading_score_value = _ta_calculate_trading_score(trades_df)
            max_trading_score = 100 # The score is already normalized to 0-100 by the function
            trading_score_percentage = (trading_score_value / max_trading_score) * 100

            hit_rate = win_rate

            most_profitable_asset_calc = "N/A"
            if not trades_df.empty:
                symbol_profits = trades_df.groupby("Symbol")["Profit"].sum()
                if not symbol_profits.empty and symbol_profits.max() > 0:
                    most_profitable_asset_calc = symbol_profits.idxmax()
                else:
                    most_profitable_asset_calc = "None Profitable"

            best_trade_profit = 0.0
            best_trade_symbol = "N/A"
            worst_trade_loss = 0.0
            worst_trade_symbol = "N/A"

            if not wins_df.empty:
                best_trade = wins_df.loc[wins_df['Profit'].idxmax()]
                best_trade_profit = best_trade['Profit']
                best_trade_symbol = best_trade['Symbol']

            if not losses_df.empty:
                worst_trade = losses_df.loc[losses_df['Profit'].idxmin()]
                worst_trade_loss = worst_trade['Profit']
                worst_trade_symbol = worst_trade['Symbol']


            # --- DISPLAY LOGIC (No changes needed below this line in this block) ---
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
                        <strong>Avg Win</strong>
                        <span class='metric-value'>{avg_win_display}</span>
                    </div>
                """, unsafe_allow_html=True)

            with col7:
                best_trade_profit_formatted = _ta_human_num_mt5(best_trade_profit)
                if best_trade_profit > 0 and best_trade_symbol != "N/A":
                    trade_info_text = f"{best_trade_symbol} with profit of <span style='color: #5cb85c;'>${best_trade_profit_formatted}</span>."
                else:
                    trade_info_text = "No winning trades."

                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Best Single Trade</strong>
                        <span class='trade-info'>{trade_info_text}</span>
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

                formatted_total_loss_in_parentheses_val = _ta_human_num_mt5(total_losses_sum)

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
                worst_trade_loss_formatted = _ta_human_num_mt5(abs(worst_trade_loss))
                if worst_trade_loss < 0 and worst_trade_symbol != "N/A":
                     trade_info_text = f"{worst_trade_symbol} with loss of <span style='color: #d9534f;'>-${worst_trade_loss_formatted}</span>."
                else:
                    trade_info_text = "No losing trades."

                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Worst Single Trade</strong>
                        <span class='trade-info'>{trade_info_text}</span>
                    </div>
                """, unsafe_allow_html=True)

            with col10:
                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Most Profitable Asset</strong>
                        <span class='metric-value'>{most_profitable_asset_calc}</span>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            col11, col12, col13, col14, col15 = st.columns(5)
            
            with col11:
                avg_loss_formatted = _ta_human_num_mt5(abs(avg_loss))
                avg_loss_display = f"<span style='color: #d9534f;'>-${avg_loss_formatted}</span>" if avg_loss < 0.0 and avg_loss_formatted != "N/A" else f"${avg_loss_formatted}"
                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Avg Loss</strong>
                        <span class='metric-value'>{avg_loss_display}</span>
                    </div>
                """, unsafe_allow_html=True)

            with col12:
                total_losses_sum_formatted = _ta_human_num_mt5(total_losses_sum)
                total_losses_sum_display = f"<span style='color: #d9534f;'>-${total_losses_sum_formatted}</span>" if total_losses_sum > 0.0 else f"${total_losses_sum_formatted}"
                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Total Loss</strong>
                        <span class='metric-value'>{total_losses_sum_display}</span>
                    </div>
                """, unsafe_allow_html=True)

            with col13:
                profit_factor_formatted = _ta_human_num_mt5(profit_factor)
                profit_factor_color = "#5cb85c" if pd.notna(profit_factor) and profit_factor >= 1.0 else "#d9534f" if pd.notna(profit_factor) and profit_factor < 1.0 else "#cccccc"
                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Profit Factor</strong>
                        <span class='metric-value' style='color: {profit_factor_color};'>{profit_factor_formatted}</span>
                    </div>
                """, unsafe_allow_html=True)

            with col14:
                max_drawdown_formatted = _ta_human_num_mt5(abs(max_drawdown))
                max_drawdown_display = f"<span style='color: #d9534f;'>-${max_drawdown_formatted}</span>" if max_drawdown < 0.0 else f"${max_drawdown_formatted}"
                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Max Drawdown</strong>
                        <span class='metric-value'>{max_drawdown_display}</span>
                    </div>
                """, unsafe_allow_html=True)

            with col15:
                expectancy_formatted = _ta_human_num_mt5(expectancy)
                expectancy_color = "#5cb85c" if expectancy > 0.0 else "#d9534f" if expectancy < 0.0 else "#cccccc"
                expectancy_sign = "$" if expectancy >= 0 else "-$"
                st.markdown(f"""
                    <div class='metric-box'>
                        <strong>Expectancy</strong>
                        <span class='metric-value' style='color: {expectancy_color};'>{expectancy_sign}{_ta_human_num_mt5(abs(expectancy))}</span>
                    </div>
                """, unsafe_allow_html=True)


        with tab_charts:
            st.subheader("Visualizations")
            # Use 'trades_df' for visualizations to ensure they reflect trading activity only.
            if not daily_pnl_df_for_stats.empty:
                df_for_chart = daily_pnl_df_for_stats.copy()
                df_for_chart["Cumulative Profit"] = df_for_chart["Profit"].cumsum()
                fig_equity = px.line(df_for_chart, x="date", y="Cumulative Profit", title="Equity Curve")
                st.plotly_chart(fig_equity, use_container_width=True)

                profit_by_symbol = trades_df.groupby("Symbol")["Profit"].sum().reset_index()
                fig_symbol = px.bar(profit_by_symbol, x="Symbol", y="Profit", title="Profit by Symbol")
                st.plotly_chart(fig_symbol, use_container_width=True)

                trade_types = trades_df["Type"].value_counts().reset_index(name="Count")
                trade_types.columns = ['Type', 'Count']
                fig_type = px.pie(trade_types, names="Type", values="Count", title="Trades by Type")
                st.plotly_chart(fig_type, use_container_width=True)
            else:
                st.info("No closed trade data available to generate visualizations. Load your data or log in to Myfxbook.")
        
        with tab_closed_trades:
            st.subheader("Closed Trades History")
            # This tab correctly shows the original, unfiltered data from the session state.
            if not st.session_state.mt5_df.empty:
                st.dataframe(st.session_state.mt5_df, use_container_width=True)
            else:
                st.info("No closed trade history available. Please upload a CSV or connect your Myfxbook account to load closed trade data.")

        with tab_open_trades:
            st.subheader("Open Trades")
            if not st.session_state.myfxbook_open_trades_df.empty:
                st.dataframe(st.session_state.myfxbook_open_trades_df, use_container_width=True)
            else:
                st.info("No open trades currently available. This tab displays real-time open trades from your connected Myfxbook account.")


        with tab_edge:
            st.subheader("Edge Finder")
            st.write("Analyze your trading edge here by breaking down performance by various factors.")

            # Use 'trades_df' for edge analysis.
            if not trades_df.empty:
                analysis_options = ['Symbol', 'Type', 'Trade Duration']
                
                if 'Trade Duration' not in trades_df.columns or trades_df['Trade Duration'].isnull().all():
                    trades_df['Open Time'] = pd.to_datetime(trades_df['Open Time'], errors='coerce')
                    trades_df['Close Time'] = pd.to_datetime(trades_df['Close Time'], errors='coerce')
                    df_valid_duration = trades_df.dropna(subset=['Open Time', 'Close Time'])
                    if not df_valid_duration.empty:
                        trades_df.loc[df_valid_duration.index, 'Trade Duration'] = (df_valid_duration["Close Time"] - df_valid_duration["Open Time"]).dt.total_seconds() / 3600
                    else:
                        trades_df['Trade Duration'] = np.nan

                available_analysis_options = [opt for opt in analysis_options if opt in trades_df.columns and not trades_df[opt].isnull().all()]
                
                if not available_analysis_options:
                    st.info("Insufficient data columns to perform edge analysis (requires 'Symbol', 'Type', or calculated 'Trade Duration' with non-null values).")
                else:
                    analysis_by = st.selectbox("Analyze by:", available_analysis_options)

                    if analysis_by == 'Trade Duration':
                        df_for_edge = trades_df.copy()
                        df_for_edge = df_for_edge.dropna(subset=['Trade Duration'])
                        if not df_for_edge.empty and (df_for_edge['Trade Duration'].max() > df_for_edge['Trade Duration'].min() or len(df_for_edge['Trade Duration'].unique()) > 1):
                            df_for_edge['Duration Bin'] = pd.cut(df_for_edge['Trade Duration'], bins=5, labels=False, include_lowest=True)
                            grouped_data = df_for_edge.groupby('Duration Bin')['Profit'].agg(['sum', 'count', 'mean']).reset_index()
                            grouped_data['Duration Bin'] = grouped_data['Duration Bin'].apply(lambda x: f"Bin {x}")
                            fig_edge = px.bar(grouped_data, x='Duration Bin', y='sum', title=f"Profit by {analysis_by}")
                            st.plotly_chart(fig_edge, use_container_width=True)
                        else:
                            st.info("No sufficient variation in trade durations to analyze for Edge Finder.")
                    else:
                        grouped_data = trades_df.groupby(analysis_by)['Profit'].agg(['sum', 'count', 'mean']).reset_index()
                        fig_edge = px.bar(grouped_data, x=analysis_by, y='sum', title=f"Profit by {analysis_by}")
                        st.plotly_chart(fig_edge, use_container_width=True)
            else:
                st.info("No closed trade data available to use the Edge Finder. Load your data or log in to Myfxbook.")

        with tab_export:
            st.subheader("Export Reports")
            st.write("Export your trading data and reports.")

            # Provide an export for both processed trade data and the original data.
            if not trades_df.empty:
                csv_processed = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed Data CSV (Trades Only)",
                    data=csv_processed,
                    file_name="processed_trades_only_history.csv",
                    mime="text/csv",
                )
            
            if not st.session_state.myfxbook_open_trades_df.empty:
                csv_open_trades = st.session_state.myfxbook_open_trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Open Trades Data CSV",
                    data=csv_open_trades,
                    file_name="open_trades.csv",
                    mime="text/csv",
                )
            
            st.info("Further reporting options (e.g., custom PDF reports) could be integrated here.")


        st.markdown("---")
        try:
            _ta_show_badges_mt5(st.session_state.mt5_df)
        except Exception as e:
            logging.error(f"Error displaying badges: {str(e)}")
    
        st.markdown("---") # This markdown acts as a separator before the calendar.

        # --- TASK 2 FIX: CALENDAR LEAKING & PREFERRED VERSION ---
        # The entire calendar code block is now correctly placed INSIDE the 'if st.session_state.current_page == 'mt5':' block
        st.subheader("🗓️ Daily Performance Calendar")

        # --- 1. DEPOSIT INPUT ---
        deposit_amount = st.number_input(
            "Enter Initial Deposit/Balance for % Calculation",
            min_value=0.01,
            value=10000.0,
            step=100.0,
            key="deposit_input_calendar_v2", # Keep key unique to this widget
            help="This value is used to calculate the daily percentage gain or loss."
        )

        # --- 2. CSS STYLES FOR THE DARK THEME CALENDAR (Preferred Version) ---
        st.markdown("""
        <style>
            /* Main container for the calendar */
            .calendar-container {
                background-color: #1a1a1a; /* Dark background */
                border: 1px solid #444;
                border-radius: 8px;
                padding: 15px;
            }
            /* Grid for the days */
            .calendar-grid {
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 8px;
            }
            /* Styling for weekday headers (Sun, Mon, etc.) */
            .weekday-header {
                text-align: center;
                font-weight: bold;
                color: #a0a0a0;
                padding-bottom: 10px;
            }
            /* Base style for each day's box */
            .calendar-day { /* Reverted to .calendar-day as preferred */
                min-height: 110px;
                border-radius: 6px;
                padding: 8px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                font-family: sans-serif;
            }
            /* Day with no trades */
            .day-no-trade {
                background-color: #333333;
                border: 1px solid #555;
            }
            /* Day from another month */
            .day-other-month {
                background-color: #222222; /* Darker for other months */
                border: 1px solid #444;
                color: #666; /* Lighter text for other months */
            }
            /* Day with a profit */
            .day-profitable {
                background-color: #28a745; /* Green */
                color: white;
            }
            /* Day with a loss */
            .day-losing {
                background-color: #dc3545; /* Red */
                color: white;
            }
            /* Special border for today's date */
            .today {
                border: 2px solid #ff8c00 !important; /* Orange */
            }
            /* The day number (e.g., 1, 2, 3) */
            .day-number {
                font-weight: bold;
                font-size: 0.9em;
                color: inherit; /* Inherit color from parent box */
            }
            /* Container for the PnL details */
            .pnl-details { /* Reverted to .pnl-details as preferred */
                text-align: center;
            }
            .pnl-details .trade-count {
                font-size: 0.8em;
                opacity: 0.8;
            }
            .pnl-details .pnl-amount {
                font-size: 1.1em;
                font-weight: bold;
            }
            .pnl-details .pnl-percent {
                font-size: 0.9em;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)


        # --- 3. DATA PREPARATION ---
        if 'mt5_df' not in st.session_state or st.session_state.mt5_df.empty:
            st.warning("No closed trade data found. Please upload a file or connect your account.")
        else:
            df_cal = st.session_state.mt5_df.copy()
            df_cal['Close Time'] = pd.to_datetime(df_cal['Close Time'], errors='coerce')
            df_cal.dropna(subset=['Close Time'], inplace=True)

            daily_stats = pd.DataFrame()
            if not df_cal.empty:
                daily_stats = df_cal.groupby(df_cal['Close Time'].dt.date).agg(
                    daily_pnl=('Profit', 'sum'),
                    trade_count=('Profit', 'count')
                )
                daily_stats['pnl_percent'] = (daily_stats['daily_pnl'] / deposit_amount) * 100 if deposit_amount > 0 else 0

            # --- 4. STATE MANAGEMENT & NAVIGATION ---
            if 'calendar_date' not in st.session_state:
                st.session_state.calendar_date = daily_stats.index.max() if not daily_stats.empty else date.today()

            def go_to_prev_month():
                st.session_state.calendar_date = (st.session_state.calendar_date.replace(day=1) - timedelta(days=1))
            def go_to_next_month():
                # Get the last day of the current month
                last_day_of_current_month = calendar.monthrange(st.session_state.calendar_date.year, st.session_state.calendar_date.month)[1]
                # Set date to the first day of the next month
                st.session_state.calendar_date = (st.session_state.calendar_date.replace(day=last_day_of_current_month) + timedelta(days=1))

            col1_cal_nav, col2_cal_nav, col3_cal_nav = st.columns([2, 3, 2])
            with col1_cal_nav:
                st.button("◀ Previous Month", on_click=go_to_prev_month, use_container_width=True, key="calendar_prev_month_btn")
            with col2_cal_nav:
                st.markdown(f"<h4 style='text-align: center; margin-top: 10px;'>{st.session_state.calendar_date.strftime('%B %Y')}</h4>", unsafe_allow_html=True)
            with col3_cal_nav:
                st.button("Next Month ▶", on_click=go_to_next_month, use_container_width=True, key="calendar_next_month_btn")

            # --- 5. CALENDAR GRID GENERATION (ROBUST METHOD) ---
            html_parts = ["<div class='calendar-container'><div class='calendar-grid'>"]
            
            # Add weekday headers
            weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            for day_name in weekdays:
                html_parts.append(f"<div class='weekday-header'>{day_name}</div>")

            # Generate calendar days
            cal = calendar.Calendar(firstweekday=calendar.SUNDAY)
            month_days = cal.monthdatescalendar(st.session_state.calendar_date.year, st.session_state.calendar_date.month)
            today = date.today()

            for week in month_days:
                for day_date in week:
                    day_classes = ["calendar-day"] # Reverted to .calendar-day
                    inner_content = f"<div class='day-number'>{day_date.day}</div>"

                    if day_date.month != st.session_state.calendar_date.month:
                        day_classes.append("day-other-month")
                    else:
                        if day_date == today:
                            day_classes.append("today")

                        if day_date in daily_stats.index:
                            day_data = daily_stats.loc[day_date]
                            pnl, count, percent = day_data['daily_pnl'], day_data['trade_count'], day_data.get('pnl_percent', 0.0)
                            
                            if pnl > 0:
                                day_classes.append("day-profitable")
                                pnl_display, percent_display = f"+${pnl:,.2f}", f"+{percent:.2f}%"
                            else: # Includes pnl <= 0 (losses or exactly zero)
                                day_classes.append("day-losing") # Using 'losing' for zero PnL as well to show activity
                                pnl_display = f"-${abs(pnl):,.2f}"
                                percent_display = f"{percent:.2f}%" # Will show 0.00% or negative
                            
                            count_display = f"{count} {'Trade' if count == 1 else 'Trades'}"
                            
                            inner_content += f"""
                                <div class='pnl-details'>
                                    <div class='trade-count'>{count_display}</div>
                                    <div class='pnl-amount'>{pnl_display}</div>
                                    <div class='pnl-percent'>{percent_display}</div>
                                </div>
                            """
                        else:
                            day_classes.append("day-no-trade")
                            # Optionally add "No Trades" text for visual clarity
                            # inner_content += "<div class='pnl-details' style='opacity: 0.7;'>No Trades</div>"

                    # Assemble the final div for the day
                    html_parts.append(f"<div class='{' '.join(day_classes)}'>{inner_content}</div>")
            
            html_parts.append("</div></div>") # Close grid and container
            
            # Render the complete, clean HTML string
            st.markdown("".join(html_parts), unsafe_allow_html=True)

        st.markdown("---")
        if st.button("📄 Generate Performance Report"):
                    df_for_report = st.session_state.mt5_df.copy()
                    df_for_report = df_for_report[df_for_report['Symbol'].notna()].copy()

                    total_trades = len(df_for_report)
                    wins_df = df_for_report[df_for_report["Profit"] > 0]
                    losses_df = df_for_report[df_for_report["Profit"] < 0]
                    win_rate = len(wins_df) / total_trades if total_trades else 0.0
                    net_profit = df_for_report["Profit"].sum()
                    profit_factor = _ta_profit_factor_mt5(df_for_report)
                    avg_loss_report = losses_df["Profit"].mean() if not losses_df.empty else 0.0
                    total_losses_sum_report = abs(losses_df["Profit"].sum()) if not losses_df.empty else 0.0

                    daily_pnl_for_streaks = _ta_daily_pnl_mt5(df_for_report)
                    # Ensure daily_pnl_df_for_stats is available for max_drawdown if not already calculated for the report
                    daily_pnl_df_for_report_stats = pd.DataFrame(list(daily_pnl_for_streaks.items()), columns=['date', 'Profit'])
                    if not daily_pnl_df_for_report_stats.empty:
                        max_drawdown_report = (daily_pnl_df_for_report_stats["Profit"].cumsum() - daily_pnl_df_for_report_stats["Profit"].cumsum().cummax()).min()
                    else:
                        max_drawdown_report = 0.0

                    streaks = _ta_compute_streaks(daily_pnl_df_for_report_stats)
                    longest_win_streak = streaks['best_win']
                    longest_loss_streak = streaks['best_loss']

                    avg_trade_duration = df_for_report['Trade Duration'].mean() if 'Trade Duration' in df_for_report.columns and not df_for_report['Trade Duration'].isnull().all() else 0.0
                    total_volume = df_for_report['Volume'].sum() if 'Volume' in df_for_report.columns and not df_for_report['Volume'].isnull().all() else 0.0
                    avg_volume = df_for_report['Volume'].mean() if 'Volume' in df_for_report.columns and not df_for_report['Volume'].isnull().all() else 0.0
                    profit_per_trade = (net_profit / total_trades) if total_trades else 0.0
                    
                    expectancy_report = (win_rate * (wins_df["Profit"].mean() if not wins_df.empty else 0.0)) + ((1 - win_rate) * (losses_df["Profit"].mean() if not losses_df.empty else 0.0)) if total_trades else 0.0


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
                    <p><strong>Total Loss:</strong> <span class='negative'>-${_ta_human_num_mt5(total_losses_sum_report)}</span></p>
                    <p><strong>Average Loss:</strong> <span class='negative'>-${_ta_human_num_mt5(abs(avg_loss_report))}</span></p>
                    <p><strong>Profit Factor:</strong> {_ta_human_num_mt5(profit_factor)}</p>
                    <p><strong>Max Drawdown:</strong> <span class='negative'>-${_ta_human_num_mt5(abs(max_drawdown_report))}</span></p>
                    <p><strong>Expectancy:</strong> <span class='{'positive' if expectancy_report >= 0 else 'negative'}'>
                    {'$' if expectancy_report >= 0 else '-$'}{_ta_human_num_mt5(abs(expectancy_report))}</span></p>
                    <p><strong>Biggest Win:</strong> <span class='positive'>${_ta_human_num_mt5(wins_df["Profit"].max() if not wins_df.empty else 0.0)}</span></p>
                    <p><strong>Biggest Loss:</strong> <span class='negative'>-${_ta_human_num_mt5(abs(losses_df["Profit"].min()) if not losses_df.empty else 0.0)}</span></p>
                    <p><strong>Longest Win Streak:</strong> {_ta_human_num_mt5(longest_win_streak)}</p>
                    <p><strong>Longest Loss Streak:</strong> {_ta_human_num_mt5(longest_loss_streak)}</p>
                    <p><strong>Avg Trade Duration:</strong> {_ta_human_num_mt5(avg_trade_duration)}h</p>
                    <p><strong>Total Volume:</strong> {_ta_human_num_mt5(total_volume)}</p>
                    <p><strong>Avg Volume:</strong> {_ta_human_num_mt5(avg_volume)}</p>
                    <p><strong>Profit / Trade:</strong> <span class='{'positive' if profit_per_trade >= 0 else 'negative'}'>
                    {'$' if profit_per_trade >= 0 else '-$'}{_ta_human_num_mt5(abs(profit_per_trade))}</span></p>
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


import streamlit as st
import os
import io
import base64
import hashlib
import json
import pandas as pd
import plotly.graph_objects as go
import time
import logging
import pytz # Necessary for timezone handling
from datetime import datetime # Necessary for datetime objects

# NOTE: The helper functions (image_to_base_64, handle_logout, get_active_market_sessions),
# the global session_state initializations, and the database connection (conn, c)
# MUST be defined at the very top of your main app script, outside any page-specific blocks.
# This code assumes they are globally accessible.

# =========================================================
# ACCOUNT PAGE
# =========================================================
# This is the primary conditional block for the entire account page.
if st.session_state.current_page == 'account':

    # --- LOGGED-OUT VIEW (Login/Signup Forms) ---
    if st.session_state.get('logged_in_user') is None:

        # --- FINAL CSS FOR THE CONDENSED, CENTERED FORM ---
        st.markdown("""
        <style>
            [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
            div[data-testid="stAppViewContainer"] > .main { background-image: linear-gradient(to bottom, #050505, #000000); }
            div[data-testid="stAppViewContainer"] > .main .block-container {
                display: flex; flex-direction: column; justify-content: center; align-items: center;
                padding: 0; margin: 0; width: 100%; min-height: 100vh;
            }
            .login-wrapper {
                background: transparent; padding: 2.5rem 3rem; border-radius: 1rem; width: 470px;
                max-width: 95%; border: 1px solid rgba(255, 255, 255, 0.05);
            }
            .login-wrapper h1 {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 2.3rem; color: #FFFFFF; font-weight: 700; margin-top: 0; margin-bottom: 35px;
            }
            .login-wrapper input[type="text"], .login-wrapper input[type="password"] {
                background-color: #262730 !important; border: 1px solid #363741 !important;
                border-radius: 8px !important; color: #FFFFFF !important; padding: 1.3rem 1rem !important;
                margin-bottom: 0.75rem; box-shadow: none !important; transition: border 0.2s ease-in-out;
            }
            .login-wrapper input:focus { border: 1px solid #4a5fe2 !important; }
            .login-wrapper .stCheckbox p { color: #e0e0e0; font-size: 0.95rem; }
            .login-wrapper .stButton>button {
                background-color: #212229; color: #e0e0e0; border: 1px solid #363741;
                border-radius: 8px; padding: 0.75rem 1rem; font-size: 1rem; font-weight: 600;
                transition: all 0.2s ease;
            }
            .login-wrapper .stButton>button:hover { background-color: #2f303a; border-color: #4a5fe2; color: #FFFFFF; }
            .login-wrapper a { color: #4a5fe2; text-decoration: none; font-size: 0.95rem; font-weight: 500; }
            .login-wrapper a:hover { text-decoration: underline; }
            .bottom-container { display: flex; justify-content: flex-start; margin-top: 2rem; }
        </style>
        """, unsafe_allow_html=True)

        if 'auth_view' not in st.session_state:
            st.session_state.auth_view = 'login'
        
        st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
        
        # --- LOGIN VIEW ---
        if st.session_state.auth_view == 'login':
            st.markdown('<h1>Welcome back</h1>', unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
                col1, col2 = st.columns(2)
                with col1: st.checkbox("Remember for 30 days")
                with col2: st.markdown('<div style="text-align: right; padding-top: 8px;"><a href="#" target="_self">Forgot password</a></div>', unsafe_allow_html=True)
                login_button = st.form_submit_button("Sign In")
            if login_button:
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                c.execute("SELECT password, data FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                if result and result[0] == hashed_password:
                    st.session_state.logged_in_user = username
                    
                    # --- CORRECTLY LOAD ALL USER DATA FROM DB ---
                    user_data = json.loads(result[1])
                    st.session_state.user_nickname = user_data.get('user_nickname', username)
                    st.session_state.user_timezone = user_data.get('user_timezone', 'Europe/London')
                    st.session_state.session_timings = user_data.get('session_timings', st.session_state.session_timings) # Use global default if not in DB
                    st.session_state.xp = user_data.get('xp', 0)
                    st.session_state.level = user_data.get('level', 0)
                    st.session_state.badges = user_data.get('badges', [])
                    st.session_state.streak = user_data.get('streak', 0)
                    st.session_state.xp_log = user_data.get('xp_log', [])
                    st.session_state.trade_journal = user_data.get('trade_journal', [])
                    # --- END LOAD USER DATA ---
                    
                    st.success(f"Welcome back, {st.session_state.user_nickname}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
            if st.button("Sign up", key="signup_toggle_login_page"): st.session_state.auth_view = 'signup'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # --- SIGNUP VIEW ---
        elif st.session_state.auth_view == 'signup':
            st.markdown('<h1>Get Started</h1>', unsafe_allow_html=True)
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
                new_password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm Password", label_visibility="collapsed")
                register_button = st.form_submit_button("Sign up")
            if register_button:
                if new_password != confirm_password: st.error("Passwords do not match.")
                elif not new_username or not new_password: st.error("Username and password cannot be empty.")
                else:
                    c.execute("SELECT username FROM users WHERE username = ?", (new_username,))
                    if c.fetchone():
                        st.error("Username already exists.")
                    else:
                        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                        
                        # --- CORRECTLY INITIALIZE AND SAVE USER DATA ---
                        initial_user_data = {
                            "xp": 0, "level": 0, "badges": [], "streak": 0, "last_journal_date": None,
                            "last_login_xp_date": None, "gamification_flags": {}, "drawings": [],
                            "trade_journal": [], "strategies": [], "emotion_log": [],
                            "reflection_log": [], "xp_log": [], 'chatroom_rules_accepted': False,
                            'user_nickname': new_username, # Default nickname to username
                            'user_timezone': 'Europe/London', # FIXED DEFAULT TIMEZONE
                            'session_timings': { # Default UTC session timings
                                "Sydney": {"start": 22, "end": 7}, "Tokyo": {"start": 0, "end": 9},
                                "London": {"start": 8, "end": 17}, "New York": {"start": 13, "end": 22}
                            }
                        }
                        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", (new_username, hashed_password, json.dumps(initial_user_data, cls=CustomJSONEncoder)))
                        conn.commit()

                        # Set session state for the new user
                        st.session_state.logged_in_user = new_username
                        st.session_state.user_nickname = new_username
                        st.session_state.user_timezone = initial_user_data['user_timezone']
                        st.session_state.session_timings = initial_user_data['session_timings']
                        st.session_state.xp = initial_user_data['xp']
                        st.session_state.level = initial_user_data['level']
                        # ... set other initial session state vars
                        
                        st.success("Account created successfully! Logging you in...")
                        st.rerun()
            st.markdown('<div class="bottom-container"><span>Already have an account?</span>', unsafe_allow_html=True)
            if st.button("Sign In", key="signin_toggle_signup_page"): st.session_state.auth_view = 'login'; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


    # =========================================================
    # --- LOGGED-IN VIEW: DASHBOARD & SETTINGS ---
    # =========================================================
    else:
        
        
        # --- LOGGED-IN WELCOME HEADER ---
        icon_path = os.path.join("icons", "my_account.png")
        if os.path.exists(icon_path):
            icon_base64_welcome = image_to_base_64(icon_path)
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <img src="data:image/png;base64,{icon_base64_welcome}" width="100">
                    <h2 style="margin: 0;">Welcome back, {st.session_state.get('user_nickname', st.session_state.logged_in_user)}! 👋</h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.header(f"Welcome back, {st.session_state.get('user_nickname', st.session_state.logged_in_user)}! 👋")

        st.markdown("This is your personal dashboard. Track your progress and manage your account.")
        st.markdown("---")
        
        # --- PROGRESS SNAPSHOT & KPI CARDS ---
        st.subheader("📈 Progress Snapshot")
        st.markdown("""
        <style>
        .kpi-card { background-color: rgba(45, 70, 70, 0.5); border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #58b3b1; margin-bottom: 10px; }
        .kpi-icon { font-size: 2.5em; margin-bottom: 10px; }
        .kpi-value { font-size: 1.8em; font-weight: bold; color: #FFFFFF; }
        .kpi-label { font-size: 0.9em; color: #A0A0A0; }
        .insights-card { background-color: rgba(45, 70, 70, 0.3); border-left: 5px solid #58b3b1; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .redeem-card { background-color: rgba(45, 70, 70, 0.5); border-radius: 10px; padding: 20px; border: 1px solid #58b3b1; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: space-between;}
        </style>
        """, unsafe_allow_html=True)

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🧙‍♂️</div><div class="kpi-value">Level {st.session_state.get("level", 0)}</div><div class="kpi-label">Trader\'s Rank</div></div>', unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🔥</div><div class="kpi-value">{st.session_state.get("streak", 0)} Days</div><div class="kpi-label">Journaling Streak</div></div>', unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">⭐</div><div class="kpi-value">{st.session_state.get("xp", 0):,}</div><div class="kpi-label">Total XP</div></div>', unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-icon">💎</div><div class="kpi-value">{int(st.session_state.get("xp", 0) / 2):,}</div><div class="kpi-label">Redeemable XP (RXP)</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- CHARTS, INSIGHTS & BADGES ---
        chart_col, insights_col = st.columns([1, 2])
        with chart_col:
            st.markdown("<h5 style='text-align: center;'>Progress to Next Level</h5>", unsafe_allow_html=True)
            xp_in_level = st.session_state.get('xp', 0) % 100
            fig = go.Figure(go.Pie(values=[xp_in_level, 100 - xp_in_level], hole=0.6, marker_colors=['#58b3b1', '#2d4646'], textinfo='none', hoverinfo='label+value'))
            fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', annotations=[dict(text=f'<b>{xp_in_level}<span style="font-size:0.6em">/100</span></b>', x=0.5, y=0.5, font_size=18, showarrow=False, font_color="white")], margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with insights_col:
            st.markdown("<h5 style='text-align: center;'>Personalized Insights & Badges</h5>", unsafe_allow_html=True)
            insight_sub_col, badge_sub_col = st.columns(2)
            with insight_sub_col:
                st.markdown("<h6>💡 Insights</h6>", unsafe_allow_html=True)
                streak = st.session_state.get('streak', 0)
                insight_message = "Your journaling consistency is elite! This is a key trait of professional traders." if streak > 21 else "Over a week of consistent journaling! You're building a powerful habit." if streak > 7 else "Every trade journaled is a step forward. Stay consistent to build a strong foundation."
                st.markdown(f"<div class='insights-card'><p>{insight_message}</p></div>", unsafe_allow_html=True)
                num_trades = len(st.session_state.get('trade_journal', []))
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
        
        # --- REDEEM RXP SECTION ---
        st.subheader("💎 Redeem Your RXP")
        current_rxp = int(st.session_state.get('xp', 0) / 2)
        st.info(f"You have **{current_rxp:,} RXP** available to spend.")
        
        items = {"1_month_access": {"name": "6th Month Free Access", "cost": 300, "icon": "🗓️"}, "consultation": {"name": "Any Month Free Access", "cost": 750, "icon": "🗓️"}, "advanced_course": {"name": "Any 2 Month Free Access", "cost": 1400, "icon": "🗓️"}}
        redeem_cols = st.columns(len(items))
        for i, (item_key, item_details) in enumerate(items.items()):
            with redeem_cols[i]:
                st.markdown(f'<div><div class="redeem-card"><div><h3>{item_details["icon"]}</h3><h5>{item_details["name"]}</h5><p>Cost: <strong>{item_details["cost"]:,} RXP</strong></p></div>', unsafe_allow_html=True)
                if st.button(f"Redeem", key=f"redeem_{item_key}", use_container_width=True):
                    # Placeholder for redemption logic
                    pass 
                st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # --- XP TRANSACTION HISTORY ---
        st.subheader("📜 Your XP Transaction History")
        xp_log_df = pd.DataFrame(st.session_state.get('xp_log', []))
        if not xp_log_df.empty:
            xp_log_df['Date'] = pd.to_datetime(xp_log_df['Date'])
            xp_log_df = xp_log_df.sort_values(by="Date", ascending=False).reset_index(drop=True)
            styled_xp_log = xp_log_df.style.applymap(lambda val: 'color: green; font-weight: bold;' if val > 0 else 'color: red; font-weight: bold;' if val < 0 else '', subset=['Amount']).format({'Amount': lambda x: f'+{int(x)}' if x > 0 else f'{int(x)}'})
            st.dataframe(styled_xp_log, use_container_width=True)
        else:
            st.info("Your XP transaction history is empty. Start interacting to earn XP!")
        
        st.markdown("---")

        # --- HOW TO EARN XP (Dashboard integrated section) ---
        st.subheader("❓ How to Earn XP") 
        st.markdown("""
        Earn Experience Points (XP) and unlock new badges as you progress in your trading journey!
        - **Daily Login**: Log in each day to earn **10 XP** for your consistency.
        - **Log New Trades**: Get **10 XP** for every trade you meticulously log in your Trading Journal.
        - **Detailed Notes**: Add substantive notes to your logged trades in the Trade Playbook to earn **5 XP**.
        - **Trade Milestones**: Achieve trade volume milestones for bonus XP and special badges:
            * Log 10 Trades: **+20 XP** + "Ten Trades Novice" Badge
            * Log 50 Trades: **+50 XP** + "Fifty Trades Apprentice" Badge
            * Log 100 Trades: **+100 XP** + "Centurion Trader" Badge
        - **Performance Milestones**: Demonstrate trading skill for extra XP and recognition:
            * Maintain a Profit Factor of 2.0 or higher: **+30 XP**
            * Achieve an Average R:R of 1.5 or higher: **+25 XP**
            * Reach a Win rate of 60% or higher: **+20 XP**
        - **Level Up!**: Every 100 XP earned levels up your Trader's Rank and rewards a new Level Badge.
        - **Daily Journaling Streak**: Maintain your journaling consistency for streak badges and XP bonuses every 7 days!
        
        Keep exploring the dashboard and trading to earn more XP and climb the ranks!
        """)
        
        st.markdown("---")

        # =========================================================
        # ACCOUNT SETTINGS (All expanders grouped here)
        # =========================================================
        st.subheader("⚙️ Account Settings")

        # --- NICKNAME SETTINGS ---
        with st.expander("👤 Nickname"):
            st.caption("Set a custom nickname that will be displayed throughout the application.")
            with st.form("nickname_form_account_page"): # Unique key for form
                nickname = st.text_input(
                    "Your Nickname",
                    value=st.session_state.get('user_nickname', st.session_state.logged_in_user),
                    key="nickname_input_account_page" # Unique key for widget
                )
                if st.form_submit_button("Save Nickname", use_container_width=True, key="save_nickname_button"): # Unique key
                    st.session_state.user_nickname = nickname
                    st.success("Your nickname has been updated!")
                    
                    # --- SAVE NICKNAME TO DATABASE ---
                    try:
                        conn = sqlite3.connect(DB_FILE)
                        c = conn.cursor()
                        c.execute("SELECT data FROM users WHERE username = ?", (st.session_state.logged_in_user,))
                        current_data = json.loads(c.fetchone()[0])
                        current_data['user_nickname'] = nickname
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(current_data, cls=CustomJSONEncoder), st.session_state.logged_in_user))
                        conn.commit()
                        logging.info(f"Nickname updated for {st.session_state.logged_in_user} in DB.")
                    except Exception as e:
                        st.error(f"Failed to save nickname to database: {e}")
                        logging.error(f"DB error updating nickname: {e}")
                    finally:
                        if conn: conn.close()
                    # --- END SAVE ---
                    st.rerun()

        # --- ACCOUNT TIME SETTINGS (FIXED TO EUROPE/LONDON) ---
        with st.expander("🕒 Account Time"):
            st.caption("Your timezone is set to Europe/London to align with key trading hours.")
            
            fixed_timezone = 'Europe/London'
            st.session_state.user_timezone = fixed_timezone # Force this into session state
            
            st.info(f"Your current selected timezone is: **{fixed_timezone}**")
            
            # Display current local time in London
            london_tz = pytz.timezone(fixed_timezone)
            current_london_time = datetime.now(london_tz)
            st.info(f"Current time in London: **{current_london_time.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
            
            # This section no longer needs a form as the value is fixed and not user-selectable.

        # --- SESSION TIMINGS SETTINGS ---
        with st.expander("📈 Session Timings"):
            st.caption("Adjust the universal start and end hours (0-23) for each market session. These are always in UTC.")
            with st.form("session_timings_form_account_page"): # Unique key for form
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.markdown("**Session**")
                col2.markdown("**Start Hour (UTC)**")
                col3.markdown("**End Hour (UTC)**")
                
                new_timings = {}
                # This loop will now work because of the global initialization
                for session_name, timings in st.session_state.session_timings.items():
                    with st.container():
                        c1, c2, c3 = st.columns([2, 1, 1])
                        c1.write(f"**{session_name}**")
                        start_time = c2.number_input("Start", min_value=0, max_value=23, value=timings['start'], key=f"{session_name}_start_account", label_visibility="collapsed") # Unique key
                        end_time = c3.number_input("End", min_value=0, max_value=23, value=timings['end'], key=f"{session_name}_end_account", label_visibility="collapsed") # Unique key
                        new_timings[session_name] = {'start': start_time, 'end': end_time}
                
                if st.form_submit_button("Save Session Timings", use_container_width=True, key="save_session_timings_button"): # Unique key
                    st.session_state.session_timings.update(new_timings)
                    st.success("Session timings have been updated successfully!")
                    
                    # --- SAVE SESSION TIMINGS TO DATABASE ---
                    try:
                        conn = sqlite3.connect(DB_FILE)
                        c = conn.cursor()
                        c.execute("SELECT data FROM users WHERE username = ?", (st.session_state.logged_in_user,))
                        current_data = json.loads(c.fetchone()[0])
                        current_data['session_timings'] = new_timings # Save the new_timings dict directly
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(current_data, cls=CustomJSONEncoder), st.session_state.logged_in_user))
                        conn.commit()
                        logging.info(f"Session timings updated for {st.session_state.logged_in_user} in DB.")
                    except Exception as e:
                        st.error(f"Failed to save session timings to database: {e}")
                        logging.error(f"DB error updating session timings: {e}")
                    finally:
                        if conn: conn.close()
                    # --- END SAVE ---
                    st.rerun()

        # --- MANAGE ACCOUNT ---
        with st.expander("🔑 Manage Account"):
            st.write(f"**Username**: `{st.session_state.logged_in_user}`")
            st.write("**Email**: `trader.pro@email.com` (example)")
            if st.button("Log Out", key="logout_account_page", type="primary", use_container_width=True):
                handle_logout() # Ensure handle_logout() is defined globally
import streamlit as st
import os
import io
import base64
import pytz
from datetime import datetime, timedelta
import logging

# =========================================================
# HELPER FUNCTIONS (Included here for completeness, but should ideally be global)
# =========================================================

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end:
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else:
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    sessions_utc_hours = st.session_state.get('session_timings', {})
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Use corrected time
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if start_hour > end_hour and now_utc.hour >= end_hour:
                end_time_today += timedelta(days=1)
            elif now_utc > end_time_today:
                end_time_today += timedelta(days=1)

            next_end_times.append((end_time_today, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort()
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0:
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return soonest_session_name, time_str

# =========================================================
# COMMUNITY TRADE IDEAS PAGE
# =========================================================
if st.session_state.current_page == 'community':

    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to participate in Community Trade Ideas.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page-Specific Configuration ---
    page_info = {
        'title': 'Community Trade Ideas', 
        'icon': 'community_trade_ideas.png', 
        'caption': 'Share and explore trade ideas with the community.'
    }

    # --- 2. Define CSS Styles for the New Header ---
    main_container_style = """
        background-color: black; padding: 20px 25px; border-radius: 10px; 
        display: flex; align-items: center; gap: 20px;
        border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);
    """
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"

    # --- 3. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    if icon_base64:
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">'
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 4. Build and Render Header ---
    header_html = (
        f'<div style="{main_container_style}">'
            f'<div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div>'
            f'<div style="{right_column_style}">'
                f'<div style="{info_tab_style}">{welcome_message}</div>'
                f'<div>'
                    f'<div style="{info_tab_style}">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")

    # (The rest of your page code for uploading charts, discussions, etc., goes here...)
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

import streamlit as st
import os
import io
import base64
import pytz
from datetime import datetime, timedelta
import logging

# =========================================================
# HELPER FUNCTIONS (Included here for completeness, but should ideally be global)
# =========================================================

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end:
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else:
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    sessions_utc_hours = st.session_state.get('session_timings', {})
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Use corrected time
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if start_hour > end_hour and now_utc.hour >= end_hour:
                end_time_today += timedelta(days=1)
            elif now_utc > end_time_today:
                end_time_today += timedelta(days=1)

            next_end_times.append((end_time_today, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort()
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0:
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return soonest_session_name, time_str

# =========================================================
# COMMUNITY CHATROOM PAGE
# =========================================================
if st.session_state.current_page == "Community Chatroom":

    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to access the Community Chatroom.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page-Specific Configuration ---
    page_info = {
        'title': 'Community Chatroom', 
        'icon': 'community_chatroom.png', 
        'caption': 'Connect, collaborate, and grow with fellow traders.'
    }

    # --- 2. Define CSS Styles for the New Header ---
    main_container_style = """
        background-color: black; padding: 20px 25px; border-radius: 10px; 
        display: flex; align-items: center; gap: 20px;
        border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);
    """
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"

    # --- 3. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    if icon_base64:
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">'
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 4. Build and Render Header ---
    header_html = (
        f'<div style="{main_container_style}">'
            f'<div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div>'
            f'<div style="{right_column_style}">'
                f'<div style="{info_tab_style}">{welcome_message}</div>'
                f'<div>'
                    f'<div style="{info_tab_style}">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")
    # (The rest of your page code for the chatroom functionality goes here...)

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


import streamlit as st
import os
import io
import base64
import pytz
from datetime import datetime, timedelta
import logging

# =========================================================
# HELPER FUNCTIONS (Included here for completeness, but should ideally be global)
# =========================================================

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end:
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else:
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    sessions_utc_hours = st.session_state.get('session_timings', {})
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Use corrected time
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if start_hour > end_hour and now_utc.hour >= end_hour:
                end_time_today += timedelta(days=1)
            elif now_utc > end_time_today:
                end_time_today += timedelta(days=1)

            next_end_times.append((end_time_today, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort()
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0:
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return soonest_session_name, time_str

# =========================================================
# TRADING TOOLS PAGE
# =========================================================
if st.session_state.current_page == 'trading_tools':

    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to access the Tools section.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page-Specific Configuration ---
    page_info = {
        'title': 'Trading Tools', 
        'icon': 'trading_tools.png', 
        'caption': 'A complete suite of utilities to optimize your trading.'
    }

    # --- 2. Define CSS Styles for the New Header ---
    main_container_style = """
        background-color: black; padding: 20px 25px; border-radius: 10px; 
        display: flex; align-items: center; gap: 20px;
        border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);
    """
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"

    # --- 3. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    if icon_base64:
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">'
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 4. Build and Render Header ---
    header_html = (
        f'<div style="{main_container_style}">'
            f'<div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div>'
            f'<div style="{right_column_style}">'
                f'<div style="{info_tab_style}">{welcome_message}</div>'
                f'<div>'
                    f'<div style="{info_tab_style}">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")

    # --- 6. RETAINED CONTENT FROM ORIGINAL PAGE ---
    st.markdown("""
    Access a complete suite of utilities to optimize your trading. Features include a Profit/Loss Calculator, Price Alerts, Currency Correlation Heatmap, Risk Management Calculator, Trading Session Tracker, Drawdown Recovery Planner, Pre-Trade Checklist, and Pre-Market Checklist. Each tool is designed to help you manage risk, plan trades efficiently, and make data-driven decisions to maximize performance.
    """)

    # --- Universal Settings ---
    all_currency_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD",
        "EUR/GBP", "EUR/AUD", "EUR/JPY", "EUR/CHF", "EUR/CAD", "EUR/NZD",
        "GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD",
        "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
        "CAD/JPY", "CAD/CHF", "CHF/JPY", "NZD/JPY", "NZD/CHF", "NZD/CAD"
    ]
    all_account_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD"]

    st.markdown("""
    <style>
    div[data-testid="stTabs"] div[role="tablist"] > div { background-color: #5bb4b0 !important; }
    div[data-testid="stTabs"] button[data-baseweb="tab"] { color: #ffffff !important; transition: all 0.3s ease !important; }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover { color: #5bb4b0 !important; background-color: rgba(91, 180, 176, 0.2) !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #5bb4b0 !important; font-weight: bold !important; }
    </style>
    """, unsafe_allow_html=True)
    
    tools_options = [
        'Profit/Loss Calculator', 'Price Alerts', 'Currency Correlation Heatmap', 'Risk Management Calculator',
        'Trading Session Tracker', 'Drawdown Recovery Planner', 'Pre-Trade Checklist', 'Pre-Market Checklist'
    ]
    tabs = st.tabs(tools_options)
    
    # --------------------------
    # PROFIT / LOSS CALCULATOR (REFACTORED)
    # --------------------------
    with tabs[0]:
        st.header("💰 Profit / Loss Calculator")
        st.markdown("Calculate your potential profit or loss for a trade. The accuracy of the result in your account currency depends on the exchange rate you provide.")
        st.markdown('---')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pl_pair = st.selectbox("Currency Pair", all_currency_pairs, key="pl_pair")
            pl_position_size = st.number_input("Position Size (lots)", min_value=0.01, value=0.1, step=0.01, format="%.2f", key="pl_position_size")
        with col2:
            pl_account_currency = st.selectbox("Account Currency", all_account_currencies, key="pl_account_currency")
            pl_trade_direction = st.radio("Trade Direction", ["Long", "Short"], key="pl_trade_direction", horizontal=True)
        with col3:
            pl_open_price = st.number_input("Open Price", value=1.1000, step=0.0001, format="%.5f", key="pl_open_price")
            pl_close_price = st.number_input("Close Price", value=1.1050, step=0.0001, format="%.5f", key="pl_close_price")

        base_ccy, quote_ccy = pl_pair.split('/')
        
        # --- Smart Conversion Rate Logic ---
        is_conversion_needed = quote_ccy != pl_account_currency
        conversion_logic = None  # Will be 'multiply' or 'divide'
        pl_conversion_rate = 1.0

        if is_conversion_needed:
            # Conversion is needed from quote_ccy to pl_account_currency.
            
            # Case 1: The conversion pair has account_currency as BASE (e.g., USD/JPY). We DIVIDE the P/L by this rate.
            direct_conv_pair = f"{pl_account_currency}/{quote_ccy}"
            
            # Case 2: The conversion pair has account_currency as QUOTE (e.g., EUR/USD). We MULTIPLY the P/L by this rate.
            inverse_conv_pair = f"{quote_ccy}/{pl_account_currency}"

            if direct_conv_pair in all_currency_pairs:
                conversion_logic = 'divide'
                st.info(f"Your profit is calculated in **{quote_ccy}**. To convert to **{pl_account_currency}**, please provide the current **{direct_conv_pair}** exchange rate.")
                pl_conversion_rate = st.number_input(f"Current {direct_conv_pair} Rate", value=1.0, step=0.00001, format="%.5f", key="pl_conversion_rate_input")
            
            elif inverse_conv_pair in all_currency_pairs:
                conversion_logic = 'multiply'
                st.info(f"Your profit is calculated in **{quote_ccy}**. To convert to **{pl_account_currency}**, please provide the current **{inverse_conv_pair}** exchange rate.")
                pl_conversion_rate = st.number_input(f"Current {inverse_conv_pair} Rate", value=1.0, step=0.00001, format="%.5f", key="pl_conversion_rate_input")
            
            else: # Fallback for complex cross rates
                conversion_logic = 'multiply'
                st.warning(f"A direct conversion rate from **{quote_ccy}** to **{pl_account_currency}** is required.")
                pl_conversion_rate = st.number_input(f"Provide Rate (1 {quote_ccy} = X {pl_account_currency})", value=1.0, step=0.00001, format="%.5f", key="pl_conversion_rate_input")
        else:
            st.success(f"Profit is calculated directly in your account currency ({pl_account_currency}). No conversion needed.")

        if st.button("Calculate Profit/Loss"):
            # Determine pip size for display purposes only
            pip_size = 0.01 if "JPY" in quote_ccy else 0.0001
            
            # Calculate price difference
            price_difference = pl_close_price - pl_open_price
            if pl_trade_direction == "Short":
                price_difference *= -1
            
            pips_moved = price_difference / pip_size
            
            # Calculate P/L in the QUOTE currency using the direct contract size method for accuracy
            contract_size = pl_position_size * 100000
            profit_loss_quote_ccy = price_difference * contract_size
            
            # Convert to account currency using the predetermined logic
            profit_loss_account_ccy = profit_loss_quote_ccy
            if is_conversion_needed:
                if conversion_logic == 'divide':
                    # Avoid division by zero
                    profit_loss_account_ccy = profit_loss_quote_ccy / pl_conversion_rate if pl_conversion_rate != 0 else 0
                elif conversion_logic == 'multiply':
                    profit_loss_account_ccy = profit_loss_quote_ccy * pl_conversion_rate

            st.markdown("---")
            st.subheader("Results")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Pips Moved", f"{pips_moved:.1f}")
            col_res2.metric(f"Profit/Loss ({quote_ccy})", f"{profit_loss_quote_ccy:,.2f}")
            
            delta_color = "normal" if profit_loss_account_ccy >= 0 else "inverse"
            col_res3.metric(f"Profit/Loss ({pl_account_currency})", f"{profit_loss_account_ccy:,.2f}", delta_color=delta_color)
            
        st.markdown("""
        ---
        **Disclaimer:** This calculator provides a precise calculation based *only* on the values you input. Its accuracy is entirely dependent on you providing the correct open/close prices, position size, and the current, real-time conversion rate if applicable. It does not account for commissions, swaps, or slippage. Always verify with your broker's platform.
        """)

    # --------------------------
    # PRICE ALERTS (No Change)
    # --------------------------
    with tabs[1]:
        st.header("⏰ Price Alerts")
        st.markdown("Set price alerts for your favourite forex pairs and get notified when the price hits your target.")
        st.markdown('---')
        st.warning("Price alerts have been temporarily disabled in this version.")
        # Original price alert code is functional but relies on API calls.
        # It's kept here as a placeholder for non-API functionality or future implementation.
        
        st.markdown("""
        ---
        **Disclaimer:** This tool is for informational purposes. Price alert triggering depends on timely data, which is not guaranteed. Do not rely solely on these alerts for making trading decisions.
        """)

    # --------------------------
    # CURRENCY CORRELATION HEATMAP
    # --------------------------
    with tabs[2]:
        st.header("📊 Currency Correlation Heatmap")
        st.markdown("Understand how forex pairs move relative to each other using static, illustrative data.")
        st.markdown('---')
        
        st.info("The heatmap below is generated from static historical data for demonstration purposes only. It does not reflect live market correlations.")
        
        pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF"]
        data = np.array([
            [1.00, 0.75, -0.62, 0.88, -0.71, -0.92], # EUR/USD
            [0.75, 1.00, -0.45, 0.65, -0.58, -0.70], # GBP/USD
            [-0.62, -0.45, 1.00, -0.78, 0.85, 0.75], # USD/JPY
            [0.88, 0.65, -0.78, 1.00, -0.82, -0.80], # AUD/USD
            [-0.71, -0.58, 0.85, -0.82, 1.00, 0.78], # USD/CAD
            [-0.92, -0.70, 0.75, -0.80, 0.78, 1.00]  # USD/CHF
        ])
        corr_df = pd.DataFrame(data, columns=pairs, index=pairs)
        fig = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Illustrative Forex Pair Correlation")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ---
        **Disclaimer:** The data in this heatmap is **static and for educational purposes only**. It does not represent live market data. Correlations between currency pairs are dynamic and can change significantly based on market conditions, volatility, and geopolitical events. Always conduct your own up-to-date analysis.
        """)

    # --------------------------
    # RISK MANAGEMENT CALCULATOR
    # --------------------------
    with tabs[3]:
        st.header("🛡️ Risk Management Calculator")
        st.markdown("""Calculate the appropriate position size based on your account size, risk tolerance, and trade parameters. Accurate inputs are required.""")
        st.markdown('---')
        
        col1, col2 = st.columns(2)
        with col1:
            rm_balance = st.number_input("Account Balance", min_value=0.0, value=10000.0, key="rm_balance")
            rm_risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, key="rm_risk_percent")
            rm_pair = st.selectbox("Currency Pair", all_currency_pairs, key="rm_pair")
        with col2:
            rm_account_currency = st.selectbox("Account Currency", all_account_currencies, key="rm_account_currency")
            rm_stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=20.0, key="rm_stop_loss_pips")

        base_ccy_rm, quote_ccy_rm = rm_pair.split('/')
        
        # Add a manual conversion rate input if needed
        rm_conversion_rate = 1.0
        if quote_ccy_rm != rm_account_currency:
            st.info(f"Pip value is calculated in {quote_ccy_rm}. Please provide the exchange rate to convert it to your account currency ({rm_account_currency}).")
            rm_conversion_rate = st.number_input(f"Current {quote_ccy_rm}/{rm_account_currency} Rate", value=1.0, step=0.00001, format="%.5f", key="rm_conversion_rate")
        
        if st.button("Calculate Lot Size"):
            if rm_balance > 0 and rm_risk_percent > 0 and rm_stop_loss_pips > 0 and rm_conversion_rate > 0:
                # Risk amount in account currency
                risk_amount = rm_balance * (rm_risk_percent / 100)
                
                # Pip size
                pip_size_rm = 0.01 if "JPY" in quote_ccy_rm else 0.0001
                
                # Pip value per lot in quote currency
                pip_value_quote = pip_size_rm * 100000
                
                # Pip value per lot in account currency
                pip_value_account = pip_value_quote * rm_conversion_rate
                
                # Total risk per lot in account currency
                risk_per_lot = rm_stop_loss_pips * pip_value_account
                
                # Calculate lot size
                lot_size = risk_amount / risk_per_lot if risk_per_lot > 0 else 0
                
                st.markdown("---")
                st.success(f"Recommended Lot Size: **{lot_size:.2f} lots**")
                st.info(f"Amount at Risk: **{risk_amount:.2f} {rm_account_currency}**")
            else:
                st.error("Please ensure all inputs are greater than zero.")
        
        st.markdown("""
        ---
        **Disclaimer:** This calculation is 100% accurate based on the values you provide. Its relevance to live trading depends entirely on the accuracy of your input, especially the account balance and the required currency conversion rate. This tool does not account for slippage or commission costs. Double-check calculations with your broker's official information.
        """)
        
    # --------------------------
    # TRADING SESSION TRACKER
    # --------------------------
    with tabs[4]:
        st.header("🕒 Forex Market Sessions")
        st.markdown("""Stay aware of active trading sessions to trade when volatility is highest. This tool uses your system's current time to display session status.""")
        st.markdown('---')
        
        # This tool's logic is based on time calculations, not external APIs, so it works as intended.
        # It has been simplified slightly to ensure no external dependencies.
        st.subheader('Current Market Sessions')
        try:
            now_utc = dt.datetime.now(pytz.utc)
            sessions = {
                "Sydney": {"start_utc": 22, "end_utc": 7, "color": "#e74c3c"},
                "Tokyo": {"start_utc": 0, "end_utc": 9, "color": "#9b59b6"},
                "London": {"start_utc": 8, "end_utc": 17, "color": "#3498db"},
                "New York": {"start_utc": 13, "end_utc": 22, "color": "#2ecc71"}
            }

            for name, details in sessions.items():
                current_hour_utc = now_utc.hour
                is_open = False
                # Handle overnight sessions like Sydney
                if details['start_utc'] > details['end_utc']:
                    if current_hour_utc >= details['start_utc'] or current_hour_utc < details['end_utc']:
                        is_open = True
                else:
                    if details['start_utc'] <= current_hour_utc < details['end_utc']:
                        is_open = True
                
                status_text = "Open" if is_open else "Closed"
                bg_color = details['color'] if is_open else "#7f8c8d"
                
                st.markdown(
                    f"""<div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; color: white; margin-bottom: 10px; text-align: center;">
                    <strong>{name} Session: {status_text}</strong>
                    </div>""",
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error("Could not display session times. Please ensure your system time is set correctly.")
        
        st.markdown("""
        ---
        **Disclaimer:** Session times are based on standard market hours and your device's current UTC time. Daylight Saving Time changes can affect actual market open/close times. This tool is a guide and should be cross-referenced with official market calendars.
        """)
    
    # --------------------------
    # DRAWDOWN RECOVERY PLANNER
    # --------------------------
    with tabs[5]:
        st.header("📉 Drawdown Recovery Planner")
        st.markdown("""Plan your recovery from a drawdown. This tool models a mathematical recovery based on your trading parameters.""")
        st.markdown('---')
        
        # This is a mathematical calculator and works as intended.
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 99.0, 10.0, key="dd_pct") / 100.0
        
        recovery_gain_needed = (1 / (1 - drawdown_pct)) - 1
        st.metric("Gain Required to Recover", f"{recovery_gain_needed * 100:.2f}%")
        
        st.subheader("Recovery Projection")
        col1, col2 = st.columns(2)
        with col1:
            initial_equity = st.number_input("Pre-Drawdown Equity ($)", min_value=1.0, value=10000.0, key="dd_equity")
            win_rate = st.slider("Projected Win Rate (%)", 1, 100, 50, key="dd_wr") / 100.0
        with col2:
            avg_rr = st.slider("Projected Average R:R", 0.1, 10.0, 1.5, 0.1, key="dd_rr")
            risk_per_trade = st.slider("Risk per Trade on Remaining Equity (%)", 0.1, 5.0, 1.0, key="dd_risk") / 100.0
        
        expectancy = (win_rate * avg_rr) - ((1 - win_rate) * 1.0)
        
        if expectancy <= 0:
            st.error("With a negative or zero expectancy, mathematical recovery is impossible. Adjust win rate or R:R.")
        else:
            trades_needed = np.log(1 / (1 - drawdown_pct)) / np.log(1 + (risk_per_trade * expectancy))
            if trades_needed < 0 or np.isinf(trades_needed) or np.isnan(trades_needed):
                 st.metric("Estimated Trades to Recover", "N/A (Check parameters)")
            else:
                 st.metric("Estimated Trades to Recover", f"{math.ceil(trades_needed)}")

        st.markdown("""
        ---
        **Disclaimer:** This planner is a pure mathematical simulation based on the fixed parameters you provide. It is a theoretical model and does not represent a guarantee of future performance. Real-world trading involves fluctuating win rates and R:R multiples, which will alter the recovery path.
        """)

    # --------------------------
    # PRE-TRADE CHECKLIST
    # --------------------------
    with tabs[6]:
        st.header("✅ Pre-Trade Checklist")
        st.markdown("""Ensure discipline by running through this checklist before every trade. A structured approach reduces impulsive decisions and aligns trades with your strategy.""")
        st.markdown('---')
        checklist_items = [
            "My analysis confirms a clear entry signal according to my plan.",
            "I have identified and marked key support and resistance levels.",
            "The trade offers a risk-to-reward ratio that meets my minimum criteria (e.g., 1:1.5 or better).",
            "I have checked the economic calendar for high-impact news that could affect this trade.",
            "I have calculated the correct position size based on my risk management rules.",
            "I know exactly where my stop loss will be placed.",
            "I have a clear take profit target or a defined trade management strategy.",
            "I am emotionally neutral and not making a decision based on fear, greed, or boredom.",
            "This trade setup aligns with the current overall market trend or structure.",
            "I have reviewed my trading plan and this trade fits within its rules."
        ]
        
        all_checked = True
        for i, item in enumerate(checklist_items):
            if not st.checkbox(item, key=f"pre_trade_check_{i}"):
                all_checked = False

        st.markdown("---")
        if all_checked:
            st.success("✅ All checks complete. You are ready to execute the trade according to your plan.")
        else:
            st.warning("⚠️ Complete all checklist items before proceeding with the trade.")
            
        st.markdown("""
        ---
        **Disclaimer:** A checklist is a discipline tool, not a guarantee of a profitable trade. Its purpose is to ensure you adhere to your own pre-defined trading plan and rules.
        """)

    # --------------------------
    # PRE-MARKET CHECKLIST
    # --------------------------
    with tabs[7]:
        st.header("📅 Pre-Market Checklist")
        st.markdown("""Build consistent habits with this pre-market routine to prepare for the trading day.""")
        st.markdown('---')
        st.subheader("Morning Preparation")
        
        pre_market_items = [
            "Reviewed overnight price action on key pairs.",
            "Checked the economic calendar for today's major events and their times.",
            "Read top financial news headlines that may impact market sentiment.",
            "Defined my overall market bias for the day (e.g., bullish, bearish, neutral).",
            "Identified and marked the daily/4-hour key levels on my watchlist pairs.",
            "Mentally prepared and affirmed my commitment to following my trading plan."
        ]
        
        all_pre_market_checked = True
        for i, item in enumerate(pre_market_items):
            if not st.checkbox(item, key=f"pre_market_check_{i}"):
                all_pre_market_checked = False
                
        st.markdown("---")
        if all_pre_market_checked:
            st.success("✅ Pre-market routine complete. You are prepared for the day.")
            
        st.markdown("""
        ---
        **Disclaimer:** This checklist is a framework for building a professional trading routine. Completing it helps with preparation and mindset but does not predict or influence market outcomes.
        """)
import streamlit as st
import os
import io
import base64
import pytz
from datetime import datetime, timedelta
import logging

# =========================================================
# HELPER FUNCTIONS (These are assumed to be defined globally at the top of your main script)
# =========================================================

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end:
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else:
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

# --- NEW, SIMPLER HELPER FUNCTION TO GET THE NEXT SESSION END TIME ---
def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    sessions_utc_hours = st.session_state.get('session_timings', {})
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Use corrected time
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if start_hour > end_hour and now_utc.hour >= end_hour:
                end_time_today += timedelta(days=1)
            elif now_utc > end_time_today:
                end_time_today += timedelta(days=1)

            next_end_times.append((end_time_today, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort()
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    # Calculate remaining time
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0:
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format as HH:MM:SS
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    
    return soonest_session_name, time_str

# =========================================================
# ZENVO ACADEMY PAGE
# =========================================================
if st.session_state.current_page == "Zenvo Academy":
    
    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to access the Zenvo Academy.")
        st.session_state.current_page = 'account'
        st.rerun()

    # --- CSS fix to ensure sidebar is visible after login ---
    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page Configuration and CSS ---
    page_info = { 'title': 'Zenvo Academy', 'icon': 'zenvo_academy.png', 'caption': 'Your journey to trading mastery starts here.' }
    main_container_style = "background-color: black; padding: 20px 25px; border-radius: 10px; display: flex; align-items: center; gap: 20px; border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);"
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"

    # --- 2. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    if icon_base64:
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">'
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    
    # Get the timer string
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 3. Build and Render Header ---
    header_html = (
        f'<div style="{main_container_style}">'
            f'<div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div>'
            f'<div style="{right_column_style}">'
                f'<div style="{info_tab_style}">{welcome_message}</div>'
                # Group the session tab and the timer text together
                f'<div>'
                    f'<div style="{info_tab_style}">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")

    st.write("Welcome to the Zenvo Academy!")
    # (The rest of your page code for courses, progress tracking, etc., goes here...)

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

# =================================================================================
# FOREX WATCHLIST PAGE (Fully Self-Contained with All Helper Functions)
# =================================================================================
import streamlit as st
import base64
import os
import logging
from datetime import datetime, timedelta
import pytz
import json
import sqlite3
import pandas as pd # Required for CustomJSONEncoder if used generally
import numpy as np # Required for CustomJSONEncoder if used generally

# --- GLOBAL DATABASE CONNECTION (Assumed to be defined in the main app) ---
# The 'conn' and 'c' objects must be globally available from your main application's setup.
# Do NOT define them again here if they are already defined at the top level of your Streamlit app.

# If CustomJSONEncoder is used throughout your application for all data storage,
# ensure it's imported and potentially used in json.dumps.
# Based on your provided database context, it is defined.
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        if pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)): return None
        return super().default(obj)

# --- Initialize ALL session state variables at the very top ---
# This ensures consistent state across reruns and different pages.
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'editing_item_id' not in st.session_state:
    st.session_state.editing_item_id = None
if 'watchlist_loaded' not in st.session_state:
    st.session_state.watchlist_loaded = False
if 'new_analyses' not in st.session_state: # Temporary list for adding new analyses
    st.session_state.new_analyses = []


# Main conditional check to render the watchlist page
if st.session_state.get('current_page') in ('watch list', 'My Watchlist'):

    # --- 1. LOCAL HELPER FUNCTIONS (Database, Header, etc.) ---
    # All functions required for this page are now defined locally.

    @st.cache_data
    def image_to_base_64(path):
        """Converts a local image file to a base64 string for embedding."""
        try:
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except FileNotFoundError:
            logging.warning(f"Warning: Header icon file not found at path: {path}")
            return None
    
    def get_active_market_sessions():
        """Determines active forex sessions."""
        sessions_utc = st.session_state.get('session_timings', {})
        # Ensure we're working with timezone-aware datetime for robustness
        corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
        current_utc_hour = corrected_utc_time.hour
        active_sessions = []
        for session_name, timings in sessions_utc.items():
            start, end = timings['start'], timings['end']
            if start > end: # Session crosses midnight UTC
                if current_utc_hour >= start or current_utc_hour < end:
                    active_sessions.append(session_name)
            else: # Session within a single UTC day
                if start <= current_utc_hour < end:
                    active_sessions.append(session_name)
        if not active_sessions:
            return "Markets Closed", []
        return ", ".join(active_sessions), active_sessions

    def get_next_session_end_info(active_sessions_list):
        """Calculates the time remaining for the next session to close."""
        if not active_sessions_list: return None, None
        sessions_utc_hours = st.session_state.get('session_timings', {})
        now_utc = datetime.now(pytz.utc) + timedelta(hours=1)
        next_end_times = []
        for session_name in active_sessions_list:
            if session_name in sessions_utc_hours:
                end_hour = sessions_utc_hours[session_name]['end']
                start_hour = sessions_utc_hours[session_name]['start'] # Needed for cross-midnight logic
                
                # Calculate today's end time, assuming it's in the future
                end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)
                
                # Adjust for sessions crossing midnight or already passed today
                if start_hour > end_hour: # Session crosses midnight
                    if now_utc.hour >= start_hour: # If current time is in the 'start day' part
                        end_time_today += timedelta(days=1) # End is tomorrow
                elif now_utc > end_time_today: # If current time is past today's end time for a non-crossing session
                    end_time_today += timedelta(days=1) # End is tomorrow
                
                next_end_times.append((end_time_today, session_name))
        
        if not next_end_times: return None, None
        
        next_end_times.sort() # Sort to find the soonest ending session
        soonest_end_time, soonest_session_name = next_end_times[0]
        
        remaining = soonest_end_time - now_utc
        if remaining.total_seconds() < 0: # Should not happen with above logic, but as a safeguard
            return soonest_session_name, "Closing..."
        
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return soonest_session_name, f"{hours:02}:{minutes:02}:{seconds:02}"

    def load_user_data(username):
        """Fetches a user's data from the DB and decodes it from JSON."""
        # Ensure 'c' (cursor) is globally accessible from the main app's DB setup
        if 'c' not in globals() or c is None:
            logging.error("Database cursor 'c' not found or is None in load_user_data.")
            return {}
        try:
            c.execute("SELECT data FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result and result[0]: return json.loads(result[0])
            return {}
        except (json.JSONDecodeError, sqlite3.Error) as e:
            logging.error(f"Error loading data for user {username}: {e}")
            return {}

    def save_user_data(username, user_data):
        """Encodes user data to JSON and saves it to the DB."""
        # Ensure 'conn' and 'c' (connection and cursor) are globally accessible
        if 'conn' not in globals() or conn is None or 'c' not in globals() or c is None:
            logging.error("Database connection 'conn' or cursor 'c' not found or is None in save_user_data.")
            return False
        try:
            json_data = json.dumps(user_data, cls=CustomJSONEncoder) 
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json_data, username))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logging.error(f"Error saving data for user {username}: {e}")
            return False

    # Helper function for ordinal numbers (e.g., 1st, 2nd, 3rd)
    def ordinal(n):
        if 10 <= n % 100 <= 20:
            return str(n) + 'th'
        else:
            return str(n) + {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

    # --- 2. LOGIN CHECK & DATA LOADING ---
    current_user = st.session_state.get('logged_in_user')
    if not current_user:
        st.warning("Please log in to manage your Forex Watchlist.")
        st.session_state.current_page = 'account'
        st.session_state.watchlist_loaded = False
        st.rerun()

    if not st.session_state.watchlist_loaded:
        user_data = load_user_data(current_user)
        st.session_state.watchlist = user_data.get('watchlist', [])
        st.session_state.watchlist_loaded = True
        st.rerun()

    # --- 3. CSS STYLING ---
    st.markdown("""
        <style>
            /* Ensure the parent container of st.columns aligns items to the start (top) */
            div[data-testid="stHorizontalBlock"] {
                align-items: start; /* THIS IS CRUCIAL FOR TOP ALIGNMENT OF COLUMNS */
            }
            div[data-testid="column"] h3 { margin-top: 0.2rem; }
        </style>
        """, unsafe_allow_html=True)

    # --- 4. HEADER BANNER ---
    page_info = {'title': 'My Watchlist', 'icon': 'watchlist_icon.png', 'caption': 'Track potential trade setups and monitor key currency pairs.'}
    main_container_style = "background-color: black; padding: 20px 25px; border-radius: 10px; display: flex; align-items: center; gap: 20px; border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);"
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"
    
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">' if icon_base64 else ""
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", "Guest")}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>' if next_session_name and timer_str else ""
    
    header_html = ( f'<div style="{main_container_style}"><div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div><div style="{right_column_style}"><div style="{info_tab_style}">{welcome_message}</div><div><div style="{info_tab_style}">{market_sessions_display}</div>{timer_display}</div></div></div>' )
    st.markdown(header_html, unsafe_allow_html=True)
    
        # --- 5. MAIN 2-COLUMN LAYOUT ---
    # The horizontal rule below pushes the entire 2-column layout down from the header.
    st.markdown("---")
    
    add_col, display_col = st.columns([1, 2], gap="large")

    # --- COLUMN 1: ADD NEW PAIR FORM ---
    with add_col:
        st.markdown("<h3>➕ Add New Pair</h3>", unsafe_allow_html=True)

        # Wrap all inputs in a single form for a clean reset on submission.
        with st.form("new_pair_form", clear_on_submit=True):
            new_pair = st.text_input("Currency Pair", placeholder="e.g., EUR/USD")
            new_image = st.file_uploader("Upload Chart Image (Optional)", type=['png', 'jpg', 'jpeg'])
            st.markdown("---")

            st.markdown("<h5>Timeframe Analyses</h5>", unsafe_allow_html=True)
            
            # Display analyses that are pending submission
            if st.session_state.new_analyses:
                st.markdown("<h6>Pending Analyses:</h6>", unsafe_allow_html=True)
                for analysis in st.session_state.new_analyses:
                    with st.container(border=True):
                        st.markdown(f"**{analysis['timeframe']}:** {analysis['description']}")
            
            # Input fields for a new analysis entry
            timeframe_options = ["1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W", "1M"]
            analysis_tf = st.selectbox("Timeframe", options=timeframe_options, index=4)
            analysis_desc = st.text_area("Notes / Analysis", height=100)

            # Button to add analysis to the pending list
            add_analysis_button = st.form_submit_button("➕ Add Timeframe Analysis", use_container_width=True)
            
            st.markdown("---")

            # --- ADDED BACK: WHEN TO ENTER AND EXIT INPUTS ---
            st.markdown("<h5>When to enter and When to exit:</h5>", unsafe_allow_html=True)
            when_to_enter = st.text_area("When to enter", height=100)
            when_to_exit = st.text_area("When to exit", height=100)

            st.markdown("---")
            
            # Main submit button for the form
            save_button = st.form_submit_button("💾 Save Pair to Watchlist", use_container_width=True, type="primary")

        # --- LOGIC FOR HANDLING FORM BUTTONS (must be outside the 'with st.form' block) ---
        if add_analysis_button:
            if analysis_desc:
                st.session_state.new_analyses.append({"timeframe": analysis_tf, "description": analysis_desc})
                st.rerun() # Rerun to show pending analysis and clear inputs
            else:
                st.warning("Please add notes for the timeframe.")

        if save_button:
            if new_pair and st.session_state.new_analyses:
                new_item_data = {
                    "id": datetime.now().isoformat(),
                    "created_at": datetime.now().isoformat(),
                    "pair": new_pair.upper(),
                    "analyses": st.session_state.new_analyses,
                    "image": new_image.getvalue() if new_image else None,
                    "when_to_enter": when_to_enter,
                    "when_to_exit": when_to_exit
                }
                st.session_state.watchlist.insert(0, new_item_data)
                user_data = load_user_data(current_user)
                user_data['xp'] = user_data.get('xp', 0) + 5
                user_data['watchlist'] = st.session_state.watchlist
                save_user_data(current_user, user_data)
                st.session_state.new_analyses = []
                st.toast(f"{new_item_data['pair']} added! You gained 5 XP!", icon="⭐")
                st.balloons()
                st.rerun()
            else:
                st.warning("Currency Pair and at least one timeframe analysis are required.")

    # --- COLUMN 2: DISPLAY WATCHLIST ---
    with display_col:
        st.markdown("<h3>Your Watchlist This Week</h3>", unsafe_allow_html=True)
        if not st.session_state.watchlist:
            st.info("Your watchlist is empty. Add a new pair using the form on the left.")
            
        for index, item in enumerate(st.session_state.watchlist):
            item_id = item['id']
            is_editing = st.session_state.editing_item_id == item_id
            
            with st.expander(f"**{item.get('pair', 'N/A')}**", expanded=is_editing):
                if is_editing:
                    # --- EDIT FORM ---
                    with st.container(border=True):
                        with st.form(f"edit_form_{item_id}"):
                            st.subheader(f"Editing {item.get('pair', '')}")
                            original_analyses = item.get('analyses', [])
                            st.markdown("<h6>Mark any analysis for deletion and click Save.</h6>", unsafe_allow_html=True)
                            for i, analysis in enumerate(original_analyses):
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"**Notes for {analysis['timeframe']}**")
                                    new_desc = st.text_area("desc", value=analysis['description'], key=f"edit_desc_{item_id}_{i}", label_visibility="collapsed")
                                with col2:
                                    st.markdown("&nbsp;", unsafe_allow_html=True)
                                    st.checkbox("Delete", key=f"delete_flag_{item_id}_{i}")
                            st.markdown("---")
                            st.markdown("<h6>Edit Entry and Exit Points</h6>", unsafe_allow_html=True)
                            edit_enter = st.text_area("When to enter", value=item.get('when_to_enter', ''), key=f"edit_enter_{item_id}")
                            edit_exit = st.text_area("When to exit", value=item.get('when_to_exit', ''), key=f"edit_exit_{item_id}")
                            updated_img = st.file_uploader("Upload New Chart", type=['png', 'jpg', 'jpeg'], key=f"img_{item_id}")
                            c1, c2 = st.columns(2)
                            if c1.form_submit_button("✔️ Save Changes", use_container_width=True):
                                updated_analyses = []
                                for i, analysis in enumerate(original_analyses):
                                    if not st.session_state[f"delete_flag_{item_id}_{i}"]:
                                        updated_analyses.append({"timeframe": analysis['timeframe'], "description": st.session_state[f"edit_desc_{item_id}_{i}"]})
                                st.session_state.watchlist[index]['analyses'] = updated_analyses
                                st.session_state.watchlist[index]['when_to_enter'] = edit_enter
                                st.session_state.watchlist[index]['when_to_exit'] = edit_exit
                                if updated_img: st.session_state.watchlist[index]['image'] = updated_img.getvalue()
                                user_data = load_user_data(current_user)
                                user_data['watchlist'] = st.session_state.watchlist
                                save_user_data(current_user, user_data)
                                st.session_state.editing_item_id = None
                                st.toast("Item updated!")
                                st.rerun()
                            if c2.form_submit_button("❌ Cancel", use_container_width=True):
                                st.session_state.editing_item_id = None
                                st.rerun()
                else:
                    # --- NORMAL ITEM DISPLAY ---
                    with st.container(border=True):
                        st.subheader(f"{item.get('pair', 'N/A')}")
                        created_at_iso = item.get('created_at')
                        if created_at_iso and created_at_iso != 'unknown date':
                            try:
                                created_datetime = datetime.fromisoformat(created_at_iso)
                                day_with_ordinal = ordinal(created_datetime.day) 
                                formatted_date = created_datetime.strftime(f"%A {day_with_ordinal} %B %Y")
                                st.caption(f"Added on: {formatted_date}")
                            except ValueError: st.caption("Added on: invalid date format")
                        else: st.caption("Added on: unknown date")
                        
                        for analysis in item.get('analyses', []):
                            tf = analysis.get('timeframe', 'N/A')
                            desc = analysis.get('description', '').replace('\n', '<br>')
                            
                            # --- CSS GRID for robust alignment of box and description ---
                            st.markdown(f"""
                                <div style="display: grid; grid-template-columns: 40px auto; align-items: start; gap: 10px; margin-bottom: 10px;">
                                    <div style="
                                        border: 1px solid #444; border-radius: 4px; width: 40px; height: 40px;
                                        display: flex; align-items: center; justify-content: center;
                                        font-weight: bold; font-size: 0.9em;
                                    ">
                                        {tf}
                                    </div>
                                    <div style="word-wrap: break-word;">{desc}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        enter_point = item.get('when_to_enter', '').replace('\n', '  \n')
                        exit_point = item.get('when_to_exit', '').replace('\n', '  \n')
                        if enter_point or exit_point:
                            st.markdown("---")
                            if enter_point:
                                st.markdown("**When to enter:**")
                                st.success(enter_point)
                            if exit_point:
                                st.markdown("**When to exit:**")
                                st.error(exit_point)

                        if item.get('image'): st.image(item.get('image'), use_column_width=True)
                        st.markdown("<div style='height: 11px;'></div>", unsafe_allow_html=True) 

                        c1, c2 = st.columns(2)
                        if c1.button("✏️ Edit", key=f"edit_{item_id}", use_container_width=True):
                            st.session_state.editing_item_id = item_id
                            st.rerun()
                        if c2.button("🗑️ Delete Pair", key=f"delete_{item_id}", use_container_width=True):
                            deleted_pair = item.get('pair', 'Item')
                            del st.session_state.watchlist[index]
                            user_data = load_user_data(current_user)
                            user_data['watchlist'] = st.session_state.watchlist
                            save_user_data(current_user, user_data)
                            st.toast(f"Deleted {deleted_pair} from watchlist.")
                            st.rerun()
    
    # Place the horizontal rule after the columns
    st.markdown("---") 

import streamlit as st
import os
import io
import base64
import pytz
from datetime import datetime, timedelta
import logging

# =========================================================
# HELPER FUNCTIONS (Included here for completeness, but should ideally be global)
# =========================================================

@st.cache_data
def image_to_base_64(path):
    """Converts a local image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.warning(f"Warning: Image file not found at path: {path}")
        return None

def get_active_market_sessions():
    """
    Determines active forex sessions and returns a display string AND a list of active sessions.
    Includes a 1-hour correction for the server's clock.
    """
    sessions_utc = st.session_state.get('session_timings', {})
    corrected_utc_time = datetime.now(pytz.utc) + timedelta(hours=1)
    current_utc_hour = corrected_utc_time.hour
    
    active_sessions = []
    for session_name, timings in sessions_utc.items():
        start, end = timings['start'], timings['end']
        if start > end:
            if current_utc_hour >= start or current_utc_hour < end:
                active_sessions.append(session_name)
        else:
            if start <= current_utc_hour < end:
                active_sessions.append(session_name)

    if not active_sessions:
        return "Markets Closed", []
    return ", ".join(active_sessions), active_sessions

def get_next_session_end_info(active_sessions_list):
    """
    Calculates which active session will end next and returns its name
    and the remaining time as a formatted string (H:M:S).
    """
    if not active_sessions_list:
        return None, None

    sessions_utc_hours = st.session_state.get('session_timings', {})
    now_utc = datetime.now(pytz.utc) + timedelta(hours=1) # Use corrected time
    
    next_end_times = []

    for session_name in active_sessions_list:
        if session_name in sessions_utc_hours:
            end_hour = sessions_utc_hours[session_name]['end']
            start_hour = sessions_utc_hours[session_name]['start']
            
            end_time_today = now_utc.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            if start_hour > end_hour and now_utc.hour >= end_hour:
                end_time_today += timedelta(days=1)
            elif now_utc > end_time_today:
                end_time_today += timedelta(days=1)

            next_end_times.append((end_time_today, session_name))
    
    if not next_end_times:
        return None, None
        
    next_end_times.sort()
    soonest_end_time, soonest_session_name = next_end_times[0]
    
    remaining = soonest_end_time - now_utc
    if remaining.total_seconds() < 0:
        return soonest_session_name, "Closing..."

    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    return soonest_session_name, time_str

# =========================================================
# MANAGE MY STRATEGY PAGE
# =========================================================
if st.session_state.current_page == 'strategy':

    if st.session_state.get('logged_in_user') is None:
        st.warning("Please log in to manage your strategies.")
        st.session_state.current_page = 'account'
        st.rerun()

    st.markdown("""<style>[data-testid="stSidebar"] { display: block !important; }</style>""", unsafe_allow_html=True)

    # --- 1. Page-Specific Configuration ---
    page_info = {
        'title': 'Manage My Strategy', 
        'icon': 'manage_my_strategy.png', 
        'caption': 'Define, refine, and track your trading strategies.'
    }

    # --- 2. Define CSS Styles for the New Header ---
    main_container_style = """
        background-color: black; padding: 20px 25px; border-radius: 10px; 
        display: flex; align-items: center; gap: 20px;
        border: 1px solid #2d4646; box-shadow: 0 0 15px 5px rgba(45, 70, 70, 0.5);
    """
    left_column_style = "flex: 3; display: flex; align-items: center; gap: 20px;"
    right_column_style = "flex: 1; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;"
    info_tab_style = "background-color: #0E1117; border: 1px solid #2d4646; padding: 8px 15px; border-radius: 8px; color: white; text-align: center; font-family: sans-serif; font-size: 0.9rem; white-space: nowrap;"
    timer_style = "font-family: sans-serif; font-size: 0.8rem; color: #808495; text-align: right; margin-top: 4px;"
    title_style = "color: white; margin: 0; font-size: 2.2rem; line-height: 1.2;"
    icon_style = "width: 130px; height: auto;"
    caption_style = "color: #808495; margin: -15px 0 0 0; font-family: sans-serif; font-size: 1rem;"

    # --- 3. Prepare Dynamic Parts of the Header ---
    icon_html = ""
    icon_path = os.path.join("icons", page_info['icon'])
    icon_base64 = image_to_base_64(icon_path)
    if icon_base64:
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" style="{icon_style}">'
    
    welcome_message = f'Welcome, <b>{st.session_state.get("user_nickname", st.session_state.get("logged_in_user", "Guest"))}</b>!'
    active_sessions_str, active_sessions_list = get_active_market_sessions()
    market_sessions_display = f'Active Sessions: <b>{active_sessions_str}</b>'
    
    next_session_name, timer_str = get_next_session_end_info(active_sessions_list)
    timer_display = ""
    if next_session_name and timer_str:
        timer_display = f'<div style="{timer_style}">{next_session_name} session ends in <b>{timer_str}</b></div>'

    # --- 4. Build and Render Header ---
    header_html = (
        f'<div style="{main_container_style}">'
            f'<div style="{left_column_style}">{icon_html}<div><h1 style="{title_style}">{page_info["title"]}</h1><p style="{caption_style}">{page_info["caption"]}</p></div></div>'
            f'<div style="{right_column_style}">'
                f'<div style="{info_tab_style}">{welcome_message}</div>'
                f'<div>'
                    f'<div style="{info_tab_style}">{market_sessions_display}</div>'
                    f'{timer_display}'
                f'</div>'
            '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("---")
    # (The rest of your page code for managing strategies goes here...)
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
