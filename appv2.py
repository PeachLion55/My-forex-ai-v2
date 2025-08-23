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
from datetime import datetime, date, timedelta
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(
    filename='forex_dashboard.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom JSON Encoder for database storage
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd.isna(obj):
            return None
        return super().default(obj)

# Page configuration
st.set_page_config(
    page_title="Forex Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent theme
st.markdown("""
    <style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    [data-testid="stDecoration"] {display: none !important;}
    
    /* Remove top padding */
    .css-18e3th9, .css-1d391kg {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    .block-container {
        padding-top: 0rem !important;
    }
    
    /* Gridline background */
    .stApp {
        background-color: #000000;
        background-image:
            linear-gradient(rgba(88, 179, 177, 0.16) 1px, transparent 1px),
            linear-gradient(90deg, rgba(88, 179, 177, 0.16) 1px, transparent 1px);
        background-size: 40px 40px;
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
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
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: linear-gradient(to right, rgba(88, 179, 177, 1.0), rgba(0, 0, 0, 1.0)) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* XP Notification styling */
    .xp-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #58b3b1, #4d7171);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(88, 179, 177, 0.5);
        z-index: 10000;
        animation: slideIn 0.5s ease-out, fadeOut 3s ease-in-out;
        font-weight: bold;
        font-size: 16px;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeOut {
        0%, 70% { opacity: 1; }
        100% { opacity: 0; }
    }
    
    /* XP Progress Bar */
    .xp-progress-container {
        width: 100%;
        height: 30px;
        background-color: #2d4646;
        border-radius: 15px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .xp-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #58b3b1, #6fc5c3);
        border-radius: 15px;
        transition: width 0.5s ease;
        box-shadow: 0 2px 8px rgba(88, 179, 177, 0.5);
    }
    
    .xp-progress-text {
        position: absolute;
        width: 100%;
        text-align: center;
        line-height: 30px;
        color: white;
        font-weight: bold;
        font-size: 14px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Account details card */
    .account-card {
        background: linear-gradient(135deg, #2d4646, #1a2b2b);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        border: 1px solid #58b3b1;
    }
    
    .metric-box {
        background-color: #1a2b2b;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #58b3b1;
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(88, 179, 177, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced Database Management
class DatabaseManager:
    """Manages all database operations with proper error handling and data persistence"""
    
    def __init__(self, db_file="forex_dashboard.db"):
        self.db_file = db_file
        self.init_database()
    
    def get_connection(self):
        """Get a new database connection"""
        return sqlite3.connect(self.db_file, check_same_thread=False)
    
    def init_database(self):
        """Initialize database with all required tables"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Users table with enhanced structure
                c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    xp INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 0,
                    streak INTEGER DEFAULT 0,
                    last_journal_date TEXT
                )''')
                
                # Trading journal table
                c.execute('''CREATE TABLE IF NOT EXISTS trading_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date TIMESTAMP,
                    symbol TEXT,
                    weekly_bias TEXT,
                    daily_bias TEXT,
                    structure_4h TEXT,
                    structure_1h TEXT,
                    correlated_pair TEXT,
                    entry_points TEXT,
                    setup_5_15 TEXT,
                    entry_conditions TEXT,
                    planned_rr TEXT,
                    news_filter TEXT,
                    alerts TEXT,
                    concerns TEXT,
                    emotions TEXT,
                    confluence_score REAL,
                    outcome_rr TEXT,
                    notes TEXT,
                    entry_price REAL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    lots REAL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )''')
                
                # Strategies table
                c.execute('''CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    name TEXT,
                    description TEXT,
                    entry_rules TEXT,
                    exit_rules TEXT,
                    risk_management TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )''')
                
                # User badges table
                c.execute('''CREATE TABLE IF NOT EXISTS user_badges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    badge_name TEXT,
                    earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )''')
                
                # Drawings table
                c.execute('''CREATE TABLE IF NOT EXISTS drawings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    pair TEXT,
                    drawing_data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )''')
                
                # Emotion logs table
                c.execute('''CREATE TABLE IF NOT EXISTS emotion_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date TIMESTAMP,
                    emotion TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )''')
                
                # Community data table
                c.execute('''CREATE TABLE IF NOT EXISTS community_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                
                conn.commit()
                logging.info("Database initialized successfully")
                
        except Exception as e:
            logging.error(f"Database initialization error: {str(e)}")
            st.error(f"Database initialization failed: {str(e)}")
    
    def create_user(self, username, password):
        """Create a new user account"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                c.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, hashed_password)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            logging.error(f"Error creating user: {str(e)}")
            return False
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                c.execute(
                    "SELECT id FROM users WHERE username = ? AND password = ?",
                    (username, hashed_password)
                )
                result = c.fetchone()
                if result:
                    # Update last login
                    c.execute(
                        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                        (result[0],)
                    )
                    conn.commit()
                    return result[0]
                return None
        except Exception as e:
            logging.error(f"Error verifying user: {str(e)}")
            return None
    
    def get_user_data(self, user_id):
        """Get comprehensive user data"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Get user info
                c.execute(
                    "SELECT username, xp, level, streak, last_journal_date FROM users WHERE id = ?",
                    (user_id,)
                )
                user_info = c.fetchone()
                
                if not user_info:
                    return None
                
                # Get badges
                c.execute(
                    "SELECT badge_name FROM user_badges WHERE user_id = ?",
                    (user_id,)
                )
                badges = [row[0] for row in c.fetchall()]
                
                # Get trading journal
                c.execute(
                    "SELECT * FROM trading_journal WHERE user_id = ? ORDER BY date DESC",
                    (user_id,)
                )
                journal_data = c.fetchall()
                
                # Get strategies
                c.execute(
                    "SELECT * FROM strategies WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,)
                )
                strategies = c.fetchall()
                
                return {
                    'username': user_info[0],
                    'xp': user_info[1],
                    'level': user_info[2],
                    'streak': user_info[3],
                    'last_journal_date': user_info[4],
                    'badges': badges,
                    'journal': journal_data,
                    'strategies': strategies
                }
                
        except Exception as e:
            logging.error(f"Error getting user data: {str(e)}")
            return None
    
    def update_user_xp(self, user_id, xp_gained):
        """Update user XP and level"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Get current XP
                c.execute("SELECT xp, level FROM users WHERE id = ?", (user_id,))
                result = c.fetchone()
                
                if result:
                    current_xp = result[0]
                    current_level = result[1]
                    new_xp = current_xp + xp_gained
                    new_level = new_xp // 100
                    
                    # Update user
                    c.execute(
                        "UPDATE users SET xp = ?, level = ? WHERE id = ?",
                        (new_xp, new_level, user_id)
                    )
                    
                    # Check for level up
                    if new_level > current_level:
                        badge_name = f"Level {new_level}"
                        c.execute(
                            "INSERT INTO user_badges (user_id, badge_name) VALUES (?, ?)",
                            (user_id, badge_name)
                        )
                    
                    conn.commit()
                    return new_xp, new_level
                    
        except Exception as e:
            logging.error(f"Error updating XP: {str(e)}")
            return None, None
    
    def save_trade_to_journal(self, user_id, trade_data):
        """Save a trade to the journal"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                c.execute('''INSERT INTO trading_journal (
                    user_id, date, symbol, weekly_bias, daily_bias,
                    structure_4h, structure_1h, correlated_pair, entry_points,
                    setup_5_15, entry_conditions, planned_rr, news_filter,
                    alerts, concerns, emotions, confluence_score, outcome_rr,
                    notes, entry_price, stop_loss_price, take_profit_price,
                    lots, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    user_id,
                    trade_data.get('Date'),
                    trade_data.get('Symbol'),
                    trade_data.get('Weekly Bias'),
                    trade_data.get('Daily Bias'),
                    trade_data.get('4H Structure', ''),
                    trade_data.get('1H Structure', ''),
                    trade_data.get('Positive Correlated Pair & Bias', ''),
                    trade_data.get('Potential Entry Points', ''),
                    trade_data.get('5min/15min Setup?', ''),
                    trade_data.get('Entry Conditions'),
                    trade_data.get('Planned R:R'),
                    trade_data.get('News Filter', ''),
                    trade_data.get('Alerts', ''),
                    trade_data.get('Concerns', ''),
                    trade_data.get('Emotions'),
                    trade_data.get('Confluence Score 1-7', 0),
                    trade_data.get('Outcome / R:R Realised'),
                    trade_data.get('Notes/Journal'),
                    trade_data.get('Entry Price'),
                    trade_data.get('Stop Loss Price'),
                    trade_data.get('Take Profit Price'),
                    trade_data.get('Lots'),
                    trade_data.get('Tags', '')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Error saving trade: {str(e)}")
            return False

# Initialize database manager
db_manager = DatabaseManager()

# XP Notification System
def show_xp_notification(xp_gained):
    """Display XP gain notification"""
    notification_html = f"""
    <div class="xp-notification">
        ✨ +{xp_gained} XP Earned!
    </div>
    <script>
        setTimeout(function() {{
            var notification = document.querySelector('.xp-notification');
            if (notification) {{
                notification.style.display = 'none';
            }}
        }}, 3000);
    </script>
    """
    st.markdown(notification_html, unsafe_allow_html=True)

def render_xp_progress_bar(current_xp, current_level):
    """Render visual XP progress bar"""
    xp_in_level = current_xp % 100
    xp_needed = 100
    progress_percent = (xp_in_level / xp_needed) * 100
    
    progress_html = f"""
    <div class="xp-progress-container">
        <div class="xp-progress-bar" style="width: {progress_percent}%;"></div>
        <div class="xp-progress-text">
            Level {current_level} - {xp_in_level}/{xp_needed} XP
        </div>
    </div>
    """
    return progress_html

# Helper Functions
def detect_currency(title: str) -> str:
    """Detect currency from news title"""
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
    """Rate impact based on sentiment polarity"""
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
    """Fetch latest forex news from FXStreet RSS feed"""
    RSS_URL = "https://www.fxstreet.com/rss/news"
    try:
        feed = feedparser.parse(RSS_URL)
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
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=3)
            df = df[df["Date"] >= cutoff]
            return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Failed to fetch news: {str(e)}")
    
    return pd.DataFrame(columns=["Date","Currency","Headline","Polarity","Impact","Summary","Link"])

# Session State Initialization
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'fundamentals'
if 'logged_in_user_id' not in st.session_state:
    st.session_state.logged_in_user_id = None
if 'show_xp_notification' not in st.session_state:
    st.session_state.show_xp_notification = False
if 'xp_gained' not in st.session_state:
    st.session_state.xp_gained = 0

# Navigation Sidebar
with st.sidebar:
    # Logo display
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <div style='font-size: 48px; color: #58b3b1;'>💹</div>
            <h2 style='color: #58b3b1; margin: 0;'>Forex Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)
    
    nav_items = [
        ('fundamentals', '📊 Forex Fundamentals'),
        ('backtesting', '📈 Backtesting'),
        ('mt5', '📉 Performance Dashboard'),
        ('psychology', '🧠 Psychology'),
        ('strategy', '📋 Manage My Strategy'),
        ('account', '👤 My Account'),
        ('community', '👥 Community Trade Ideas'),
        ('tools', '🛠️ Tools')
    ]
    
    for page_key, page_name in nav_items:
        if st.button(page_name, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
            st.rerun()

# Show XP Notification if needed
if st.session_state.show_xp_notification and st.session_state.xp_gained > 0:
    show_xp_notification(st.session_state.xp_gained)
    st.session_state.show_xp_notification = False
    st.session_state.xp_gained = 0

# Main Application Pages
if st.session_state.current_page == 'fundamentals':
    st.title("📊 Forex Fundamentals")
    st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
    st.markdown('---')
    
    # Economic Calendar
    st.markdown("### 📅 Upcoming Economic Events")
    
    econ_calendar_data = [
        {"Date": "2025-08-22", "Time": "14:30", "Currency": "USD", "Event": "Non-Farm Payrolls", 
         "Actual": "", "Forecast": "200K", "Previous": "185K", "Impact": "High"},
        {"Date": "2025-08-23", "Time": "09:00", "Currency": "EUR", "Event": "CPI Flash Estimate YoY", 
         "Actual": "", "Forecast": "2.2%", "Previous": "2.1%", "Impact": "High"},
        {"Date": "2025-08-24", "Time": "12:00", "Currency": "GBP", "Event": "Bank of England Interest Rate Decision", 
         "Actual": "", "Forecast": "5.00%", "Previous": "5.00%", "Impact": "High"},
    ]
    
    econ_df = pd.DataFrame(econ_calendar_data)
    st.dataframe(econ_df, use_container_width=True, height=200)
    
    # News Feed
    st.markdown("### 📰 Latest Forex News")
    df_news = get_fxstreet_forex_news()
    if not df_news.empty:
        for idx, row in df_news.head(5).iterrows():
            with st.expander(f"{row['Currency']} - {row['Headline'][:80]}..."):
                st.write(f"**Impact:** {row['Impact']}")
                st.write(f"**Sentiment:** {row['Polarity']:.2f}")
                st.write(f"**Summary:** {row['Summary'][:200]}...")
                st.write(f"[Read More]({row['Link']})")
    else:
        st.info("No recent news available")

elif st.session_state.current_page == 'backtesting':
    st.title("📈 Backtesting")
    st.caption("Live TradingView chart for backtesting and enhanced trading journal")
    st.markdown('---')
    
    # Trading Journal tabs with renamed "Trade History"
    tab_entry, tab_analytics, tab_history = st.tabs(["📝 Log Trade", "📊 Analytics", "📜 Trade History"])
    
    with tab_entry:
        st.subheader("Log a New Trade")
        with st.form("trade_entry_form"):
            col1, col2 = st.columns(2)
            with col1:
                trade_date = st.date_input("Date", value=datetime.now().date())
                symbol = st.selectbox("Symbol", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
                weekly_bias = st.selectbox("Weekly Bias", ["Bullish", "Bearish", "Neutral"])
                daily_bias = st.selectbox("Daily Bias", ["Bullish", "Bearish", "Neutral"])
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, format="%.5f")
            
            with col2:
                stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.0001, format="%.5f")
                take_profit_price = st.number_input("Take Profit Price", min_value=0.0, step=0.0001, format="%.5f")
                lots = st.number_input("Lots", min_value=0.01, step=0.01, format="%.2f")
                emotions = st.selectbox("Emotions", ["Confident", "Anxious", "Fearful", "Excited", "Neutral"])
            
            notes = st.text_area("Notes/Journal")
            submit_button = st.form_submit_button("Save Trade")
            
            if submit_button:
                if st.session_state.logged_in_user_id:
                    trade_data = {
                        'Date': trade_date,
                        'Symbol': symbol,
                        'Weekly Bias': weekly_bias,
                        'Daily Bias': daily_bias,
                        'Entry Price': entry_price,
                        'Stop Loss Price': stop_loss_price,
                        'Take Profit Price': take_profit_price,
                        'Lots': lots,
                        'Emotions': emotions,
                        'Notes/Journal': notes,
                        'Planned R:R': f"1:{((take_profit_price - entry_price) / (entry_price - stop_loss_price)):.2f}" if stop_loss_price != entry_price else "1:0",
                        'Outcome / R:R Realised': f"1:{((take_profit_price - entry_price) / (entry_price - stop_loss_price)):.2f}" if stop_loss_price != entry_price else "1:0"
                    }
                    
                    if db_manager.save_trade_to_journal(st.session_state.logged_in_user_id, trade_data):
                        st.success("Trade saved successfully!")
                        
                        # Update XP
                        new_xp, new_level = db_manager.update_user_xp(st.session_state.logged_in_user_id, 10)
                        if new_xp is not None:
                            st.session_state.xp_gained = 10
                            st.session_state.show_xp_notification = True
                            st.rerun()
                    else:
                        st.error("Failed to save trade")
                else:
                    st.warning("Please log in to save trades")
    
    with tab_analytics:
        st.subheader("Trade Analytics")
        if st.session_state.logged_in_user_id:
            user_data = db_manager.get_user_data(st.session_state.logged_in_user_id)
            if user_data and user_data['journal']:
                st.info("Analytics coming soon...")
            else:
                st.info("No trades logged yet")
        else:
            st.info("Please log in to view analytics")
    
    with tab_history:
        st.subheader("Trade History")
        if st.session_state.logged_in_user_id:
            user_data = db_manager.get_user_data(st.session_state.logged_in_user_id)
            if user_data and user_data['journal']:
                trades_df = pd.DataFrame(user_data['journal'])
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trade history available")
        else:
            st.info("Please log in to view trade history")

elif st.session_state.current_page == 'mt5':
    st.title("📉 Performance Dashboard")
    st.caption("Analyze your MT5 trading history with advanced metrics")
    st.markdown('---')
    
    # Enhanced MT5 Dashboard with consistent color scheme
    uploaded_file = st.file_uploader("Upload MT5 History CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data preprocessing
            if "Open Time" in df.columns:
                df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
            if "Close Time" in df.columns:
                df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")
            
            # Calculate metrics
            total_trades = len(df)
            if "Profit" in df.columns:
                wins = df[df["Profit"] > 0]
                losses = df[df["Profit"] <= 0]
                win_rate = len(wins) / total_trades if total_trades else 0
                net_profit = df["Profit"].sum()
                avg_win = wins["Profit"].mean() if not wins.empty else 0
                avg_loss = losses["Profit"].mean() if not losses.empty else 0
            else:
                win_rate = 0
                net_profit = 0
                avg_win = 0
                avg_loss = 0
            
            # Display metrics with enhanced styling
            st.markdown("### Key Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h4 style="color: #58b3b1; margin: 0;">Total Trades</h4>
                    <h2 style="margin: 10px 0;">{total_trades}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                win_color = "#4CAF50" if win_rate >= 0.5 else "#f44336"
                st.markdown(f"""
                <div class="metric-box">
                    <h4 style="color: #58b3b1; margin: 0;">Win Rate</h4>
                    <h2 style="color: {win_color}; margin: 10px 0;">{win_rate:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                profit_color = "#4CAF50" if net_profit >= 0 else "#f44336"
                st.markdown(f"""
                <div class="metric-box">
                    <h4 style="color: #58b3b1; margin: 0;">Net Profit</h4>
                    <h2 style="color: {profit_color}; margin: 10px 0;">${net_profit:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <h4 style="color: #58b3b1; margin: 0;">Avg Win/Loss</h4>
                    <h2 style="margin: 10px 0;">
                        <span style="color: #4CAF50;">${avg_win:.2f}</span> / 
                        <span style="color: #f44336;">${abs(avg_loss):.2f}</span>
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts with consistent theme
            if "Profit" in df.columns:
                st.markdown("### Performance Visualizations")
                
                # Profit by Symbol
                if "Symbol" in df.columns:
                    profit_by_symbol = df.groupby("Symbol")["Profit"].sum().reset_index()
                    fig_symbol = px.bar(
                        profit_by_symbol,
                        x="Symbol",
                        y="Profit",
                        title="Profit by Instrument",
                        color="Profit",
                        color_continuous_scale=["#f44336", "#58b3b1", "#4CAF50"],
                        template="plotly_dark"
                    )
                    fig_symbol.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff"
                    )
                    st.plotly_chart(fig_symbol, use_container_width=True)
                
                # Equity Curve
                df["Cumulative"] = df["Profit"].cumsum()
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=df.index,
                    y=df["Cumulative"],
                    mode='lines',
                    name='Equity',
                    line=dict(color='#58b3b1', width=2)
                ))
                fig_equity.update_layout(
                    title="Equity Curve",
                    xaxis_title="Trade Number",
                    yaxis_title="Cumulative Profit ($)",
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff"
                )
                st.plotly_chart(fig_equity, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("📤 Upload your MT5 trading history CSV to explore performance metrics")

elif st.session_state.current_page == 'account':
    st.title("👤 My Account")
    st.markdown("Manage your account and track your progress")
    st.markdown('---')
    
    if st.session_state.logged_in_user_id is None:
        # Login/Register tabs
        tab_signin, tab_signup = st.tabs(["🔐 Sign In", "📝 Sign Up"])
        
        with tab_signin:
            st.subheader("Welcome back!")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button:
                    user_id = db_manager.verify_user(username, password)
                    if user_id:
                        st.session_state.logged_in_user_id = user_id
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab_signup:
            st.subheader("Create a new account")
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Register")
                
                if register_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif db_manager.create_user(new_username, new_password):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists")
    
    else:
        # Enhanced account details display
        user_data = db_manager.get_user_data(st.session_state.logged_in_user_id)
        
        if user_data:
            st.markdown(f"""
            <div class="account-card">
                <h2 style="color: #58b3b1; margin-bottom: 20px;">Welcome, {user_data['username']}!</h2>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #ffffff; margin-bottom: 10px;">Level {user_data['level']}</h4>
                    {render_xp_progress_bar(user_data['xp'], user_data['level'])}
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px;">
                    <div style="text-align: center;">
                        <h3 style="color: #58b3b1; margin: 0;">{user_data['xp']}</h3>
                        <p style="color: #999; margin: 5px 0;">Total XP</p>
                    </div>
                    <div style="text-align: center;">
                        <h3 style="color: #58b3b1; margin: 0;">{user_data['streak']}</h3>
                        <p style="color: #999; margin: 5px 0;">Day Streak</p>
                    </div>
                    <div style="text-align: center;">
                        <h3 style="color: #58b3b1; margin: 0;">{len(user_data['badges'])}</h3>
                        <p style="color: #999; margin: 5px 0;">Badges</p>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <h4 style="color: #ffffff;">Badges Earned</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
                        {' '.join([f'<span style="background: #58b3b1; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">{badge}</span>' for badge in user_data['badges']]) if user_data['badges'] else '<span style="color: #999;">No badges earned yet</span>'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Logout", key="logout_account"):
                st.session_state.logged_in_user_id = None
                st.success("Logged out successfully!")
                st.rerun()

# Additional pages would continue here...
# The code continues with remaining pages (psychology, strategy, community, tools)
# Due to length constraints, I'm showing the key improvements implemented

# Continuing from the account page...

elif st.session_state.current_page == 'psychology':
    st.title("🧠 Psychology")
    st.caption("Trading psychology is critical to success. Track emotions and maintain discipline.")
    st.markdown('---')
    
    st.subheader("📝 Emotion Tracker")
    with st.form("emotion_form"):
        emotion = st.selectbox("Current Emotion", ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral"])
        notes = st.text_area("Notes on Your Mindset")
        submit_emotion = st.form_submit_button("Log Emotion")
        
        if submit_emotion and st.session_state.logged_in_user_id:
            try:
                with db_manager.get_connection() as conn:
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO emotion_logs (user_id, date, emotion, notes) VALUES (?, ?, ?, ?)",
                        (st.session_state.logged_in_user_id, datetime.now(), emotion, notes)
                    )
                    conn.commit()
                    st.success("Emotion logged successfully!")
                    
                    # Award XP for logging emotion
                    new_xp, new_level = db_manager.update_user_xp(st.session_state.logged_in_user_id, 5)
                    if new_xp is not None:
                        st.session_state.xp_gained = 5
                        st.session_state.show_xp_notification = True
                        st.rerun()
            except Exception as e:
                st.error(f"Failed to log emotion: {str(e)}")
        elif submit_emotion:
            st.warning("Please log in to track emotions")
    
    # Display emotion history
    if st.session_state.logged_in_user_id:
        try:
            with db_manager.get_connection() as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT date, emotion, notes FROM emotion_logs WHERE user_id = ? ORDER BY date DESC LIMIT 10",
                    (st.session_state.logged_in_user_id,)
                )
                emotions = c.fetchall()
                
                if emotions:
                    st.subheader("Recent Emotion Log")
                    emotion_df = pd.DataFrame(emotions, columns=["Date", "Emotion", "Notes"])
                    st.dataframe(emotion_df, use_container_width=True)
                    
                    # Emotion distribution chart
                    c.execute(
                        "SELECT emotion, COUNT(*) as count FROM emotion_logs WHERE user_id = ? GROUP BY emotion",
                        (st.session_state.logged_in_user_id,)
                    )
                    emotion_stats = c.fetchall()
                    
                    if emotion_stats:
                        stats_df = pd.DataFrame(emotion_stats, columns=["Emotion", "Count"])
                        fig = px.pie(
                            stats_df, 
                            values="Count", 
                            names="Emotion", 
                            title="Emotion Distribution",
                            color_discrete_sequence=["#58b3b1", "#4d7171", "#6fc5c3", "#3d5757", "#7fd5d3", "#2d4646"]
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#ffffff"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load emotion history: {str(e)}")
    
    # Mindset Tips
    st.subheader("💡 Mindset Tips")
    tips = [
        "Stick to your trading plan to avoid impulsive decisions",
        "Take breaks after losses to reset your mindset",
        "Focus on process, not profits, to stay disciplined",
        "Journal every trade to identify emotional patterns",
        "Practice mindfulness to manage stress during volatile markets"
    ]
    for tip in tips:
        st.markdown(f"• {tip}")
    
    # 30-Day Challenge
    st.subheader("🏆 30-Day Discipline Challenge")
    if st.session_state.logged_in_user_id:
        user_data = db_manager.get_user_data(st.session_state.logged_in_user_id)
        if user_data:
            streak = user_data.get('streak', 0)
            progress = min(streak / 30.0, 1.0)
            st.progress(progress)
            st.write(f"Current Streak: {streak} days")
            
            if progress >= 1.0:
                st.success("🎉 Challenge completed! You've mastered consistency!")

elif st.session_state.current_page == 'strategy':
    st.title("📋 Manage My Strategy")
    st.caption("Define, refine, and track your trading strategies")
    st.markdown('---')
    
    st.subheader("➕ Add New Strategy")
    with st.form("strategy_form"):
        strategy_name = st.text_input("Strategy Name")
        description = st.text_area("Strategy Description")
        entry_rules = st.text_area("Entry Rules")
        exit_rules = st.text_area("Exit Rules")
        risk_management = st.text_area("Risk Management Rules")
        submit_strategy = st.form_submit_button("Save Strategy")
        
        if submit_strategy and st.session_state.logged_in_user_id:
            try:
                with db_manager.get_connection() as conn:
                    c = conn.cursor()
                    c.execute(
                        """INSERT INTO strategies 
                        (user_id, name, description, entry_rules, exit_rules, risk_management) 
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (st.session_state.logged_in_user_id, strategy_name, description, 
                         entry_rules, exit_rules, risk_management)
                    )
                    conn.commit()
                    st.success(f"Strategy '{strategy_name}' saved successfully!")
                    
                    # Award XP for creating strategy
                    new_xp, new_level = db_manager.update_user_xp(st.session_state.logged_in_user_id, 15)
                    if new_xp is not None:
                        st.session_state.xp_gained = 15
                        st.session_state.show_xp_notification = True
                        st.rerun()
            except Exception as e:
                st.error(f"Failed to save strategy: {str(e)}")
        elif submit_strategy:
            st.warning("Please log in to save strategies")
    
    # Display strategies
    if st.session_state.logged_in_user_id:
        user_data = db_manager.get_user_data(st.session_state.logged_in_user_id)
        if user_data and user_data['strategies']:
            st.subheader("Your Strategies")
            for strategy in user_data['strategies']:
                with st.expander(f"📑 {strategy[2]}"):  # strategy[2] is the name
                    st.markdown(f"**Description:** {strategy[3]}")
                    st.markdown(f"**Entry Rules:** {strategy[4]}")
                    st.markdown(f"**Exit Rules:** {strategy[5]}")
                    st.markdown(f"**Risk Management:** {strategy[6]}")
                    st.caption(f"Created: {strategy[7]}")
        else:
            st.info("No strategies defined yet. Add one above!")

elif st.session_state.current_page == 'community':
    st.title("👥 Community Trade Ideas")
    st.caption("Share and explore trade ideas with the community")
    st.markdown('---')
    
    st.subheader("➕ Share a Trade Idea")
    with st.form("trade_idea_form"):
        trade_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"])
        trade_direction = st.radio("Direction", ["Long", "Short"])
        trade_description = st.text_area("Trade Description")
        uploaded_image = st.file_uploader("Upload Chart Screenshot (optional)", type=["png", "jpg", "jpeg"])
        submit_idea = st.form_submit_button("Share Idea")
        
        if submit_idea and st.session_state.logged_in_user_id:
            try:
                with db_manager.get_connection() as conn:
                    c = conn.cursor()
                    
                    # Get username
                    c.execute("SELECT username FROM users WHERE id = ?", (st.session_state.logged_in_user_id,))
                    username = c.fetchone()[0]
                    
                    # Save to community data
                    c.execute("SELECT data FROM community_data WHERE key = 'trade_ideas'")
                    result = c.fetchone()
                    
                    ideas = json.loads(result[0]) if result else []
                    new_idea = {
                        'id': str(uuid.uuid4()),
                        'username': username,
                        'pair': trade_pair,
                        'direction': trade_direction,
                        'description': trade_description,
                        'timestamp': datetime.now().isoformat(),
                        'image': None  # Could implement image storage if needed
                    }
                    ideas.append(new_idea)
                    
                    if result:
                        c.execute("UPDATE community_data SET data = ? WHERE key = 'trade_ideas'", (json.dumps(ideas),))
                    else:
                        c.execute("INSERT INTO community_data (key, data) VALUES ('trade_ideas', ?)", (json.dumps(ideas),))
                    
                    conn.commit()
                    st.success("Trade idea shared successfully!")
                    
                    # Award XP for sharing
                    new_xp, new_level = db_manager.update_user_xp(st.session_state.logged_in_user_id, 20)
                    if new_xp is not None:
                        st.session_state.xp_gained = 20
                        st.session_state.show_xp_notification = True
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Failed to share idea: {str(e)}")
        elif submit_idea:
            st.warning("Please log in to share trade ideas")
    
    # Display community ideas
    st.subheader("📊 Community Trade Ideas")
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT data FROM community_data WHERE key = 'trade_ideas'")
            result = c.fetchone()
            
            if result:
                ideas = json.loads(result[0])
                if ideas:
                    for idea in sorted(ideas, key=lambda x: x['timestamp'], reverse=True)[:10]:
                        with st.expander(f"{idea['pair']} - {idea['direction']} by {idea['username']}"):
                            st.write(f"**Description:** {idea['description']}")
                            st.caption(f"Posted: {idea['timestamp'][:19]}")
                else:
                    st.info("No trade ideas shared yet. Be the first!")
            else:
                st.info("No trade ideas shared yet. Be the first!")
                
    except Exception as e:
        st.error(f"Failed to load community ideas: {str(e)}")

elif st.session_state.current_page == 'tools':
    st.title("🛠️ Tools")
    st.caption("Professional trading tools and calculators")
    st.markdown('---')
    
    tool_tabs = st.tabs([
        "💰 P/L Calculator",
        "⏰ Price Alerts", 
        "🔗 Correlations",
        "📊 Risk Manager",
        "🕐 Sessions",
        "📉 Drawdown",
        "✅ Checklists"
    ])
    
    with tool_tabs[0]:  # P/L Calculator
        st.header("💰 Profit/Loss Calculator")
        st.markdown("Calculate potential profit or loss for your trades")
        
        col1, col2 = st.columns(2)
        with col1:
            currency_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY"], key="pl_pair")
            position_size = st.number_input("Position Size (lots)", min_value=0.01, value=0.1, step=0.01)
            open_price = st.number_input("Open Price", value=1.1000, step=0.0001, format="%.4f")
        
        with col2:
            close_price = st.number_input("Close Price", value=1.1050, step=0.0001, format="%.4f")
            account_currency = st.selectbox("Account Currency", ["USD", "EUR", "GBP"], key="pl_acc")
            direction = st.radio("Trade Direction", ["Long", "Short"])
        
        if st.button("Calculate P/L"):
            pip_multiplier = 100 if "JPY" in currency_pair else 10000
            pips = abs(close_price - open_price) * pip_multiplier
            
            if direction == "Long":
                profit = (close_price - open_price) * position_size * 100000
            else:
                profit = (open_price - close_price) * position_size * 100000
            
            color = "green" if profit >= 0 else "red"
            st.markdown(f"""
            <div class="metric-box" style="margin-top: 20px;">
                <h4>Result</h4>
                <h2 style="color: {color};">{profit:+.2f} {account_currency}</h2>
                <p>Pip Movement: {pips:.1f} pips</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tool_tabs[1]:  # Price Alerts
        st.header("⏰ Price Alerts")
        st.markdown("Set alerts for key price levels")
        
        with st.form("alert_form"):
            alert_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY"])
            alert_price = st.number_input("Alert Price", min_value=0.0, format="%.5f")
            alert_type = st.radio("Alert Type", ["Above", "Below"])
            submit_alert = st.form_submit_button("Set Alert")
            
            if submit_alert:
                st.success(f"Alert set: {alert_pair} {alert_type} {alert_price}")
                if st.session_state.logged_in_user_id:
                    # Award XP for setting alert
                    new_xp, new_level = db_manager.update_user_xp(st.session_state.logged_in_user_id, 3)
                    if new_xp is not None:
                        st.session_state.xp_gained = 3
                        st.session_state.show_xp_notification = True
                        st.rerun()
    
    with tool_tabs[2]:  # Correlations
        st.header("🔗 Currency Correlation Heatmap")
        
        # Sample correlation data
        pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF"]
        corr_data = np.array([
            [1.00, 0.87, -0.72, 0.68, -0.55, -0.60],
            [0.87, 1.00, -0.65, 0.74, -0.58, -0.62],
            [-0.72, -0.65, 1.00, -0.55, 0.69, 0.71],
            [0.68, 0.74, -0.55, 1.00, -0.61, -0.59],
            [-0.55, -0.58, 0.69, -0.61, 1.00, 0.88],
            [-0.60, -0.62, 0.71, -0.59, 0.88, 1.00],
        ])
        
        corr_df = pd.DataFrame(corr_data, columns=pairs, index=pairs)
        
        fig = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=["#f44336", "#2d4646", "#58b3b1"],
            title="Forex Pair Correlations"
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Strong positive correlation (>0.7): Pairs move together | Strong negative correlation (<-0.7): Pairs move inversely")
    
    with tool_tabs[3]:  # Risk Manager
        st.header("📊 Risk Management Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=10000.0)
            risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0)
        
        with col2:
            stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=20.0)
            pip_value = st.number_input("Pip Value per Lot ($)", min_value=0.01, value=10.0)
        
        if st.button("Calculate Position Size"):
            risk_amount = account_balance * (risk_percent / 100)
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            
            st.markdown(f"""
            <div class="metric-box" style="margin-top: 20px;">
                <h4>Recommended Position Size</h4>
                <h2 style="color: #58b3b1;">{lot_size:.2f} lots</h2>
                <p>Risk Amount: ${risk_amount:.2f}</p>
                <p>Per Pip Risk: ${risk_amount/stop_loss_pips:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tool_tabs[4]:  # Trading Sessions
        st.header("🕐 Trading Sessions")
        
        # Get current time in different zones
        now_utc = datetime.now(pytz.UTC)
        
        sessions = [
            {"name": "Sydney", "start": 22, "end": 7, "tz": "Australia/Sydney"},
            {"name": "Tokyo", "start": 0, "end": 9, "tz": "Asia/Tokyo"},
            {"name": "London", "start": 8, "end": 17, "tz": "Europe/London"},
            {"name": "New York", "start": 13, "end": 22, "tz": "America/New_York"},
        ]
        
        session_status = []
        for session in sessions:
            tz = pytz.timezone(session["tz"])
            local_time = now_utc.astimezone(tz)
            local_hour = local_time.hour
            
            # Check if session is open
            if session["start"] <= session["end"]:
                is_open = session["start"] <= local_hour < session["end"]
            else:  # Session crosses midnight
                is_open = local_hour >= session["start"] or local_hour < session["end"]
            
            session_status.append({
                "Session": session["name"],
                "Status": "🟢 Open" if is_open else "🔴 Closed",
                "Local Time": local_time.strftime("%H:%M")
            })
        
        session_df = pd.DataFrame(session_status)
        st.dataframe(session_df, use_container_width=True, hide_index=True)
        
        # Session overlaps
        st.subheader("Session Overlaps (Highest Volatility)")
        st.info("🔥 London/New York Overlap: 13:00-17:00 UTC (Most liquid)")
        st.info("⚡ Tokyo/London Overlap: 07:00-09:00 UTC")
        st.info("💫 Sydney/Tokyo Overlap: 00:00-07:00 UTC")
    
    with tool_tabs[5]:  # Drawdown Recovery
        st.header("📉 Drawdown Recovery Planner")
        
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 50.0, 10.0) / 100
        
        # Calculate required gain
        if drawdown_pct < 0.99:
            recovery_gain = drawdown_pct / (1 - drawdown_pct)
        else:
            recovery_gain = float("inf")
        
        st.metric("Required Gain to Recover", f"{recovery_gain*100:.2f}%" if recovery_gain != float("inf") else "∞")
        
        # Recovery simulation
        st.subheader("Recovery Simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            initial_equity = st.number_input("Initial Equity ($)", value=10000.0)
            win_rate = st.slider("Expected Win Rate (%)", 30, 70, 50) / 100
        
        with col2:
            avg_rr = st.slider("Average R:R", 0.5, 3.0, 1.5, 0.1)
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0) / 100
        
        # Calculate expected value
        ev = win_rate * avg_rr - (1 - win_rate)
        
        if ev > 0:
            trades_to_recover = int(np.log(1/(1-drawdown_pct)) / np.log(1 + risk_per_trade * ev))
            st.success(f"Estimated trades to recover: {trades_to_recover}")
            
            # Simulation
            equity_curve = [initial_equity * (1 - drawdown_pct)]
            for i in range(min(trades_to_recover * 2, 100)):
                equity_curve.append(equity_curve[-1] * (1 + risk_per_trade * ev))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(equity_curve))),
                y=equity_curve,
                mode='lines',
                name='Projected Equity',
                line=dict(color='#58b3b1', width=2)
            ))
            fig.add_hline(
                y=initial_equity,
                line_dash="dash",
                line_color="green",
                annotation_text="Recovery Target"
            )
            fig.update_layout(
                title="Recovery Projection",
                xaxis_title="Trade Number",
                yaxis_title="Equity ($)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Negative expectancy! Improve win rate or R:R ratio")
    
    with tool_tabs[6]:  # Checklists
        st.header("✅ Trading Checklists")
        
        checklist_type = st.radio("Select Checklist", ["Pre-Trade", "Pre-Market", "End-of-Day"])
        
        if checklist_type == "Pre-Trade":
            st.subheader("Pre-Trade Checklist")
            checklist_items = [
                "Market structure aligns with bias",
                "Key levels identified",
                "Entry trigger confirmed",
                "Risk-reward ratio ≥ 1:2",
                "No high-impact news imminent",
                "Position size calculated",
                "Stop loss set",
                "Take profit set",
                "Emotionally calm and focused"
            ]
        elif checklist_type == "Pre-Market":
            st.subheader("Pre-Market Routine")
            checklist_items = [
                "Reviewed economic calendar",
                "Analyzed major news events",
                "Set weekly/daily biases",
                "Identified key levels on charts",
                "Prepared watchlist",
                "Checked correlations",
                "Reviewed previous trades"
            ]
        else:  # End-of-Day
            st.subheader("End-of-Day Review")
            checklist_items = [
                "Logged all trades",
                "Updated trading journal",
                "Reviewed winning trades",
                "Analyzed losing trades",
                "Identified lessons learned",
                "Set goals for tomorrow",
                "Practiced gratitude"
            ]
        
        checked_items = []
        for i, item in enumerate(checklist_items):
            if st.checkbox(item, key=f"check_{checklist_type}_{i}"):
                checked_items.append(item)
        
        progress = len(checked_items) / len(checklist_items)
        st.progress(progress)
        
        if progress == 1.0:
            st.success(f"✅ {checklist_type} checklist complete!")
            if st.session_state.logged_in_user_id:
                # Award XP for completing checklist
                new_xp, new_level = db_manager.update_user_xp(st.session_state.logged_in_user_id, 5)
                if new_xp is not None:
                    st.session_state.xp_gained = 5
                    st.session_state.show_xp_notification = True
        else:
            remaining = len(checklist_items) - len(checked_items)
            st.warning(f"Complete {remaining} more items")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Forex Dashboard v2.0 | Built with Streamlit</p>
    <p style="color: #58b3b1;">Trade Smart. Manage Risk. Stay Disciplined.</p>
</div>
""", unsafe_allow_html=True)

# Error handler for database cleanup
def cleanup_on_exit():
    """Cleanup function to ensure database connections are closed"""
    try:
        if hasattr(db_manager, 'conn'):
            db_manager.conn.close()
    except:
        pass

# Register cleanup
import atexit
atexit.register(cleanup_on_exit)
