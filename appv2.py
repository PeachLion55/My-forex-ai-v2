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

# =========================================================
# PAGE CONFIG AND STYLING
# =========================================================
st.set_page_config(
    page_title="Forex Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional sidebar and layout
st.markdown("""
<style>
/* Hide default sidebar */
.css-1d391kg {
    display: none;
}

/* Main app styling */
.stApp {
    background: linear-gradient(135deg, #0b0b0b 0%, #0a0a0a 100%);
    color: white;
}

/* Custom sidebar */
.custom-sidebar {
    position: fixed;
    left: 0;
    top: 0;
    width: 250px;
    height: 100vh;
    background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
    border-right: 2px solid #333;
    z-index: 999;
    overflow-y: auto;
    padding: 20px 0;
}

.sidebar-logo {
    text-align: center;
    padding: 20px;
    border-bottom: 1px solid #333;
    margin-bottom: 20px;
}

.sidebar-logo h1 {
    color: #ff6b35;
    font-size: 28px;
    font-weight: bold;
    margin: 0;
}

.nav-item {
    display: block;
    width: 90%;
    margin: 5px auto;
    padding: 12px 20px;
    background: linear-gradient(45deg, #ff6b35, #f7931e);
    color: white;
    text-decoration: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: left;
}

.nav-item:hover {
    background: linear-gradient(45deg, #e55a2b, #d87419);
    transform: translateX(5px);
}

.nav-item.active {
    background: linear-gradient(45deg, #ff8c42, #ffab42);
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4);
}

.nav-item i {
    margin-right: 10px;
    width: 20px;
}

.sub-nav {
    margin-left: 20px;
    margin-top: 5px;
    display: none;
}

.sub-nav.show {
    display: block;
}

.sub-nav-item {
    display: block;
    width: 85%;
    margin: 3px auto;
    padding: 8px 15px;
    background: rgba(255, 107, 53, 0.2);
    color: #ccc;
    text-decoration: none;
    border-radius: 6px;
    font-size: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: left;
}

.sub-nav-item:hover {
    background: rgba(255, 107, 53, 0.4);
    color: white;
}

.sub-nav-item.active {
    background: rgba(255, 107, 53, 0.6);
    color: white;
}

/* Main content area */
.main-content {
    margin-left: 270px;
    padding: 20px;
    min-height: 100vh;
}

/* Strategy selector at bottom */
.strategy-selector {
    position: absolute;
    bottom: 80px;
    left: 20px;
    right: 20px;
}

.strategy-select {
    width: 100%;
    padding: 10px;
    background: #2a2a2a;
    color: white;
    border: 1px solid #444;
    border-radius: 6px;
}

/* Settings and logout buttons */
.bottom-nav {
    position: absolute;
    bottom: 20px;
    left: 0;
    right: 0;
    padding: 0 20px;
}

.bottom-nav .nav-item {
    margin-bottom: 5px;
}

/* Responsive design */
@media (max-width: 768px) {
    .custom-sidebar {
        width: 200px;
    }
    .main-content {
        margin-left: 220px;
    }
}

/* Button styling */
.stButton button {
    background: linear-gradient(45deg, rgba(255,107,53,0.9), rgba(247,147,30,0.9));
    color: white;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 6px;
    border: none;
    transition: all 0.2s ease;
}

.stButton button:hover {
    background: linear-gradient(45deg, rgba(229,90,43,0.9), rgba(216,116,25,0.9));
    transform: translateY(-1px);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
}

/* Data tables */
.dataframe {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
}

.dataframe th {
    background-color: #2a2a2a;
    color: #ff6b35;
}

.dataframe td {
    background-color: rgba(255,255,255,0.02);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATABASE SETUP
# =========================================================
DB_FILE = "users.db"

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

# =========================================================
# HELPER FUNCTIONS
# =========================================================
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
def get_fxstreet_forex_news():
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

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'

if 'current_subpage' not in st.session_state:
    st.session_state.current_subpage = None

if 'show_tools_submenu' not in st.session_state:
    st.session_state.show_tools_submenu = False

# Initialize trading journal
journal_cols = [
    "Date", "Symbol", "Weekly Bias", "Daily Bias", "4H Structure", "1H Structure",
    "Positive Correlated Pair & Bias", "Potential Entry Points", "5min/15min Setup?",
    "Entry Conditions", "Planned R:R", "News Filter", "Alerts", "Concerns",
    "Emotions", "Confluence Score 1-7", "Outcome / R:R Realised", "Notes/Journal",
    "Entry Price", "Stop Loss Price", "Take Profit Price", "Lots"
]

journal_dtypes = {
    "Date": "datetime64[ns]", "Symbol": str, "Weekly Bias": str, "Daily Bias": str,
    "4H Structure": str, "1H Structure": str, "Positive Correlated Pair & Bias": str,
    "Potential Entry Points": str, "5min/15min Setup?": str, "Entry Conditions": str,
    "Planned R:R": str, "News Filter": str, "Alerts": str, "Concerns": str,
    "Emotions": str, "Confluence Score 1-7": float, "Outcome / R:R Realised": str,
    "Notes/Journal": str, "Entry Price": float, "Stop Loss Price": float,
    "Take Profit Price": float, "Lots": float
}

if "tools_trade_journal" not in st.session_state:
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
def render_sidebar():
    st.markdown("""
    <div class="custom-sidebar">
        <div class="sidebar-logo">
            <h1>TD</h1>
        </div>
        
        <div class="nav-menu">
    """, unsafe_allow_html=True)
    
    # Navigation items
    nav_items = [
        ('dashboard', 'DASHBOARD', '📊'),
        ('markets', 'MARKETS', '💹'),
        ('calendar', 'CALENDAR', '📅'),
        ('analytics', 'ANALYTICS', '📈'),
        ('calculator', 'CALCULATOR', '🔢'),
        ('mental', 'MENTAL', '🧠'),
        ('backtest', 'BACKTEST', '⚡'),
        ('trades', 'TRADES', '🔄'),
        ('account', 'ACCOUNT', '👤'),
        ('community', 'COMMUNITY', '🌐')
    ]
    
    for page_key, page_name, icon in nav_items:
        active_class = 'active' if st.session_state.current_page == page_key else ''
        
        if st.button(f"{icon} {page_name}", key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
            st.session_state.current_subpage = None
            st.session_state.show_tools_submenu = False
            st.rerun()
    
    # Tools submenu
    tools_active = 'active' if st.session_state.current_page == 'tools' else ''
    if st.button(f"🛠 TOOLS", key="nav_tools"):
        st.session_state.show_tools_submenu = not st.session_state.show_tools_submenu
        st.session_state.current_page = 'tools'
        if not st.session_state.show_tools_submenu:
            st.session_state.current_subpage = None
        st.rerun()
    
    # Tools submenu items
    if st.session_state.show_tools_submenu:
        st.markdown('<div class="sub-nav show">', unsafe_allow_html=True)
        
        tools_subitems = [
            ('profit_loss', 'Profit/Loss Calculator'),
            ('alerts', 'Price Alerts'),
            ('correlation', 'Currency Correlation'),
            ('risk_mgmt', 'Risk Management'),
            ('sessions', 'Trading Sessions'),
            ('drawdown', 'Drawdown Recovery'),
            ('checklist', 'Pre-Trade Checklist'),
            ('premarket', 'Pre-Market Checklist')
        ]
        
        for sub_key, sub_name in tools_subitems:
            sub_active = 'active' if st.session_state.current_subpage == sub_key else ''
            if st.button(sub_name, key=f"sub_{sub_key}"):
                st.session_state.current_subpage = sub_key
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy selector
    st.markdown("""
        <div class="strategy-selector">
            <label style="color: #ccc; font-size: 12px; margin-bottom: 5px; display: block;">Strategy</label>
            <select class="strategy-select">
                <option>Strategy 1</option>
                <option>Strategy 2</option>
                <option>Strategy 3</option>
            </select>
        </div>
        
        <div class="bottom-nav">
    """, unsafe_allow_html=True)
    
    # Bottom navigation
    if st.button("⚙ SETTINGS", key="nav_settings"):
        st.session_state.current_page = 'settings'
        st.rerun()
    
    if st.button("🚪 LOGOUT", key="nav_logout"):
        if 'logged_in_user' in st.session_state:
            del st.session_state.logged_in_user
            st.success("Logged out successfully!")
            st.rerun()
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE CONTENT FUNCTIONS
# =========================================================
def show_dashboard():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("📊 Trading Dashboard")
    st.markdown("Welcome to your professional forex trading dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff6b35; margin: 0;">Total Trades</h3>
            <h2 style="margin: 10px 0;">127</h2>
            <p style="color: #ccc; margin: 0;">This Month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff6b35; margin: 0;">Win Rate</h3>
            <h2 style="margin: 10px 0; color: #00ff00;">73.5%</h2>
            <p style="color: #ccc; margin: 0;">Last 30 Days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff6b35; margin: 0;">Profit Factor</h3>
            <h2 style="margin: 10px 0; color: #00ff00;">2.1</h2>
            <p style="color: #ccc; margin: 0;">Risk Adjusted</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff6b35; margin: 0;">Account Balance</h3>
            <h2 style="margin: 10px 0;">$12,450</h2>
            <p style="color: #00ff00; margin: 0;">+5.2% MTD</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_markets():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("💹 Live Markets")
    
    # Major currency pairs with live rates
    pairs_data = {
        'Pair': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF'],
        'Price': [1.0856, 1.2734, 149.23, 0.6698, 1.3542, 0.8923],
        'Change': ['+0.0012', '-0.0045', '+0.34', '+0.0023', '-0.0078', '+0.0015'],
        'Change %': ['+0.11%', '-0.35%', '+0.23%', '+0.34%', '-0.57%', '+0.17%']
    }
    
    df = pd.DataFrame(pairs_data)
    st.dataframe(df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_calendar():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("📅 Economic Calendar")
    
    # Get news data
    df_news = get_fxstreet_forex_news()
    
    if not df_news.empty:
        st.dataframe(df_news[['Date', 'Currency', 'Headline', 'Impact']], use_container_width=True)
    else:
        st.info("Loading economic calendar data...")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_analytics():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("📈 Analytics")
    
    # Sample analytics chart
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    cumulative_returns = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=cumulative_returns, mode='lines', name='Portfolio Performance'))
    fig.update_layout(
        title='Portfolio Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_calculator():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("🔢 Position Size Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        account_balance = st.number_input("Account Balance ($)", value=10000.0, step=100.0)
        risk_percent = st.number_input("Risk per Trade (%)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
    
    with col2:
        stop_loss_pips = st.number_input("Stop Loss (pips)", value=20.0, min_value=1.0, step=1.0)
        pip_value = st.number_input("Pip Value ($)", value=10.0, min_value=0.01, step=0.01)
    
    if st.button("Calculate Position Size"):
        risk_amount = account_balance * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        st.success(f"Recommended Position Size: {position_size:.2f} lots")
        st.info(f"Risk Amount: ${risk_amount:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_mental():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("🧠 Trading Psychology")
    
    st.markdown("""
    Trading psychology is crucial for long-term success. Track your emotions and mindset here.
    """)
    
    # Emotion tracker
    with st.form("emotion_form"):
        emotion = st.selectbox("Current Emotion", ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral"])
        notes = st.text_area("Notes on Your Mindset")
        
        if st.form_submit_button("Log Emotion"):
            st.success("Emotion logged successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_backtest():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("⚡ Strategy Backtesting")
    
    # TradingView widget placeholder
    st.info("TradingView chart integration would go here")
    
    # Sample backtest results
    st.subheader("Backtest Results")
    results_data = {
        'Metric': ['Total Trades', 'Win Rate', 'Profit Factor', 'Max Drawdown', 'Sharpe Ratio'],
        'Value': [150, '68.7%', 1.85, '-12.3%', 1.42]
    }
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_trades():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("🔄 Trade Management")
    
    # Add new trade button
    if st.button("➕ Add New Trade"):
        st.session_state.show_add_trade = True
    
    # Trade journal
    st.subheader("Trading Journal")
    if not st.session_state.tools_trade_journal.empty:
        st.dataframe(st.session_state.tools_trade_journal, use_container_width=True)
    else:
        st.info("No trades recorded yet. Click 'Add New Trade' to get started.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_tools():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if st.session_state.current_subpage == 'profit_loss':
        st.title("💰 Profit/Loss Calculator")
        # Profit/Loss calculator content
        
    elif st.session_state.current_subpage == 'alerts':
        st.title("⏰ Price Alerts")
        # Price alerts content
        
    elif st.session_state.current_subpage == 'correlation':
        st.title("📊 Currency Correlation")
        # Correlation heatmap content
        
    elif st.session_state.current_subpage == 'risk_mgmt':
        st.title("🛡️ Risk Management")
        # Risk management tools
        
    elif st.session_state.current_subpage == 'sessions':
        st.title("🕒 Trading Sessions")
        # Trading session tracker
        
    elif st.session_state.current_subpage == 'drawdown':
        st.title("📉 Drawdown Recovery")
        # Drawdown recovery planner
        
    elif st.session_state.current_subpage == 'checklist':
        st.title("✅ Pre-Trade Checklist")
        # Pre-trade checklist
        
    elif st.session_state.current_subpage == 'premarket':
        st.title("📅 Pre-Market Checklist")
        # Pre-market checklist
        
    else:
        st.title("🛠 Trading Tools")
        st.markdown("Select a tool from the sidebar to get started.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_account():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("👤 My Account")
    
    if "logged_in_user" not in st.session_state:
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    # Authentication logic here
                    st.session_state.logged_in_user = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                if st.form_submit_button("Register"):
                    if new_password == confirm_password:
                        st.session_state.logged_in_user = new_username
                        st.success(f"Account created for {new_username}!")
                        st.rerun()
                    else:
                        st.error("Passwords do not match.")
    else:
        st.success(f"Logged in as: {st.session_state.logged_in_user}")
        
        if st.button("Logout"):
            del st.session_state.logged_in_user
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_community():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("🌐 Community")
    
    st.markdown("Connect with other traders and share insights.")
    
    # Placeholder for community features
    st.info("Community features coming soon!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_settings():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("⚙ Settings")
    
    st.subheader("Application Settings")
    
    # Theme settings
    st.selectbox("Theme", ["Dark", "Light"], index=0)
    
    # Notification settings
    st.checkbox("Email Notifications", value=True)
    st.checkbox("Push Notifications", value=False)
    
    # Risk settings
    st.number_input("Default Risk per Trade (%)", value=1.0, min_value=0.1, max_value=5.0)
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)
