# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import os
import json
import hashlib
import numpy as np

# Path to accounts JSON file
ACCOUNTS_FILE = "user_accounts.json"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Elite Forex Dashboard", layout="wide")

# ----------------- SIDEBAR CONTROLS -----------------
bg_opacity = 0.7  # Background FX opacity
tv_height = 800  # TradingView chart height in px

# Theme toggle
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == 'dark' else 1)
st.session_state.theme = theme.lower()

# ----------------- CUSTOM CSS (Sleek, Professional BG + Enhanced Styling) -----------------
st.markdown(
    f"""
<style>
/* Advanced forex dashboard background */
.stApp {{
    background:
        linear-gradient(145deg, #0a0e1a 0%, #1a2236 50%, #0a0e1a 100%),
        linear-gradient(90deg, rgba(10, 20, 50, 0.9) 0%, rgba(20, 40, 80, 0.7) 100%);
    background-attachment: fixed;
}}
/* Subtle animated grid overlay */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(90deg, rgba(0, 255, 255, {bg_opacity*0.03}) 1px, transparent 1px),
        linear-gradient(0deg, rgba(0, 255, 255, {bg_opacity*0.03}) 1px, transparent 1px);
    background-size: 20px 20px, 20px 20px;
    animation: moveGrid 60s linear infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 0.8;
}}
/* Subtle particle effect */
.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    background: url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg"%3E%3Cfilter id="noise"%3E%3CfeTurbulence type="fractalNoise" baseFrequency="0.7" numOctaves="1" stitchTiles="stitch"/%3E%3C/filter%3E%3Crect width="100%25" height="100%25" filter="url(%23noise)" opacity="0.05"/%3E%3C/svg%3E');
    animation: flicker 10s infinite;
    pointer-events: none;
    z-index: 0;
}}
@keyframes moveGrid {{
    0% {{ transform: translateY(0px); }}
    100% {{ transform: translateY(20px); }}
}}
@keyframes flicker {{
    0% {{ opacity: 0.05; }}
    50% {{ opacity: 0.08; }}
    100% {{ opacity: 0.05; }}
}}
/* Theme adjustments */
.stApp {{
    color: {'#ffffff' if st.session_state.theme == 'dark' else '#000000'};
    background-color: {'#0a0e1a' if st.session_state.theme == 'dark' else '#f5f5f5'};
}}
/* Lift content above background */
.main, .block-container, .stTabs, .stMarkdown, .css-ffhzg2, .css-1d391kg {{ position: relative; z-index: 1; }}
/* Tab styling */
div[data-baseweb="tab-list"] {{
    gap: 8px;
    padding-bottom: 4px;
    background-color: {'#0a0e1a' if st.session_state.theme == 'dark' else '#e0e0e0'};
    border-bottom: 1px solid {'#2a2a2a' if st.session_state.theme == 'dark' else '#cccccc'};
}}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background-color: #FFD700 !important;
    color: black !important;
    font-weight: 700;
    padding: 14px 26px !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #FFD700 !important;
    border-bottom: none !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
    background-color: {'#1b1b1b' if st.session_state.theme == 'dark' else '#ffffff'} !important;
    color: {'#bbb' if st.session_state.theme == 'dark' else '#333333'} !important;
    padding: 14px 26px !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid {'#2a2a2a' if st.session_state.theme == 'dark' else '#cccccc'} !important;
    border-bottom: none !important;
}}
div[data-baseweb="tab-list"] button:hover {{
    background-color: {'#2a2a2a' if st.session_state.theme == 'dark' else '#f0f0f0'} !important;
    color: {'white' if st.session_state.theme == 'dark' else '#000000'} !important;
    border-color: #FFD700 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}}
/* Card styling */
.card {{
    background: linear-gradient(180deg, {'rgba(255,255,255,0.04), rgba(255,255,255,0.02)' if st.session_state.theme == 'dark' else 'rgba(0,0,0,0.04), rgba(0,0,0,0.02)'});
    border: 1px solid {'rgba(255,255,255,0.1)' if st.session_state.theme == 'dark' else 'rgba(0,0,0,0.1)'};
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.3), 0 0 8px rgba(255,215,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}
.card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.4), 0 0 12px rgba(255,215,0,0.15);
}}
/* Dataframe styling */
.dataframe th {{
    background-color: {'#1f1f1f' if st.session_state.theme == 'dark' else '#e0e0e0'};
    color: {'#FFD700' if st.session_state.theme == 'dark' else '#000000'};
    border-bottom: 1px solid {'#2a2a2a' if st.session_state.theme == 'dark' else '#cccccc'};
}}
.dataframe td {{
    background-color: {'#121212' if st.session_state.theme == 'dark' else '#ffffff'};
    color: {'white' if st.session_state.theme == 'dark' else '#000000'};
    border-bottom: 1px solid {'#2a2a2a' if st.session_state.theme == 'dark' else '#cccccc'};
}}
/* Input styling */
.stSelectbox, .stNumberInput, .stTextInput, .stRadio {{
    background-color: {'#1b1b1b' if st.session_state.theme == 'dark' else '#ffffff'};
    border-radius: 8px;
    padding: 8px;
    border: 1px solid {'#2a2a2a' if st.session_state.theme == 'dark' else '#cccccc'};
    box-shadow: 0 0 4px rgba(255,215,0,0.1);
}}
/* Button styling */
.stButton button {{
    background-color: #FFD700;
    color: black;
    border-radius: 8px;
    font-weight: bold;
    border: 1px solid #FFD700;
    box-shadow: 0 0 4px rgba(255,215,0,0.2);
}}
.stButton button:hover {{
    background-color: #E6C200;
    border-color: #E6C200;
    box-shadow: 0 0 8px rgba(255,215,0,0.3);
}}
/* Expander styling */
.stExpander {{
    border: 1px solid {'#2a2a2a' if st.session_state.theme == 'dark' else '#cccccc'};
    border-radius: 8px;
    background-color: {'#1b1b1b' if st.session_state.theme == 'dark' else '#ffffff'};
    box-shadow: 0 0 4px rgba(255,215,0,0.1);
}}
/* Tooltip styling */
.tooltip {{
    position: relative;
    display: inline-block;
    cursor: pointer;
}}
.tooltip .tooltiptext {{
    visibility: hidden;
    width: 200px;
    background-color: {'#1b1b1b' if st.session_state.theme == 'dark' else '#ffffff'};
    color: {'white' if st.session_state.theme == 'dark' else '#000000'};
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 10;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
    border: 1px solid {'#2a2a2a' if st.session_state.theme == 'dark' else '#cccccc'};
}}
.tooltip:hover .tooltiptext {{
    visibility: visible;
    opacity: 1;
}}
/* Gauge styling */
.gauge {{
    width: 200px;
    height: 100px;
    position: relative;
    margin: 0 auto;
}}
.gauge::before {{
    content: '';
    position: absolute;
    width: 180px;
    height: 90px;
    background: conic-gradient(#ff4d4d 0% 25%, #FFD700 25% 50%, #4CAF50 50% 75%, #ff4d4d 75% 100%);
    border-radius: 90px 90px 0 0;
    transform: rotate(-90deg);
}}
.gauge::after {{
    content: '';
    position: absolute;
    width: 160px;
    height: 80px;
    background: {'#0a0e1a' if st.session_state.theme == 'dark' else '#f5f5f5'};
    border-radius: 80px 80px 0 0;
    top: 10px;
    left: 10px;
}}
.gauge-needle {{
    position: absolute;
    width: 2px;
    height: 80px;
    background: white;
    left: 50%;
    top: 50%;
    transform-origin: bottom;
    transition: transform 0.5s ease;
}}
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
        return pd.DataFrame(columns=["Date", "Currency", "Headline", "Polarity", "Impact", "Summary", "Link"])
    except Exception:
        return pd.DataFrame(columns=["Date", "Currency", "Headline", "Polarity", "Impact", "Summary", "Link"])

# Static economic calendar
econ_calendar_data = [
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP Price Index y/y", "Actual": "3.0%", "Forecast": "3.1%", "Previous": "3.3%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP q/q", "Actual": "0.3%", "Forecast": "0.1%", "Previous": "0.0%", "Impact": "High"},
    {"Date": "2025-08-15", "Time": "02:30", "Currency": "CNY", "Event": "New Home Prices m/m", "Actual": "-0.31%", "Forecast": "", "Previous": "-0.27%", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Industrial Production y/y", "Actual": "5.7%", "Forecast": "6.0%", "Previous": "6.8%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Retail Sales y/y", "Actual": "3.7%", "Forecast": "4.6%", "Previous": "4.8%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Fixed Asset Investment ytd/y", "Actual": "1.6%", "Forecast": "2.7%", "Previous": "2.8%", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "NBS Press Conference", "Actual": "", "Forecast": "", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Unemployment Rate", "Actual": "5.2%", "Forecast": "5.1%", "Previous": "5.0%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "05:30", "Currency": "JPY", "Event": "Revised Industrial Production m/m", "Actual": "2.1%", "Forecast": "1.7%", "Previous": "1.7%", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "French Bank Holiday", "Actual": "", "Forecast": "", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "Italian Bank Holiday", "Actual": "", "Forecast": "", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "ECOFIN Meetings", "Actual": "", "Forecast": "", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "CAD", "Event": "Manufacturing Sales m/m", "Actual": "0.3%", "Forecast": "0.4%", "Previous": "-1.5%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "CAD", "Event": "Wholesale Sales m/m", "Actual": "0.7%", "Forecast": "0.7%", "Previous": "0.0%", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Core Retail Sales m/m", "Actual": "0.3%", "Forecast": "0.3%", "Previous": "0.8%", "Impact": "High"},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.6%", "Previous": "0.9%", "Impact": "High"},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Empire State Manufacturing Index", "Actual": "11.9", "Forecast": "-1.2", "Previous": "5.5", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Import Prices m/m", "Actual": "0.4%", "Forecast": "0.1%", "Previous": "-0.1%", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "14:15", "Currency": "USD", "Event": "Capacity Utilization Rate", "Actual": "77.5%", "Forecast": "77.6%", "Previous": "77.7%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "14:15", "Currency": "USD", "Event": "Industrial Production m/m", "Actual": "-0.1%", "Forecast": "0.0%", "Previous": "0.4%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Prelim UoM Consumer Sentiment", "Actual": "58.6", "Forecast": "61.9", "Previous": "61.7", "Impact": "High"},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Prelim UoM Inflation Expectations", "Actual": "4.9%", "Forecast": "", "Previous": "4.5%", "Impact": "Medium"},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Business Inventories m/m", "Actual": "0.2%", "Forecast": "0.2%", "Previous": "0.0%", "Impact": "Low"},
    {"Date": "2025-08-15", "Time": "21:00", "Currency": "USD", "Event": "TIC Long-Term Purchases", "Actual": "150.8B", "Forecast": "", "Previous": "266.8B", "Impact": "Low"},
    {"Date": "2025-08-16", "Time": "Tentative", "Currency": "USD", "Event": "President Trump Speaks", "Actual": "", "Forecast": "", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-17", "Time": "23:30", "Currency": "NZD", "Event": "BusinessNZ Services Index", "Actual": "47.3", "Forecast": "", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-18", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-1.2%", "Forecast": "", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-18", "Time": "05:30", "Currency": "JPY", "Event": "Tertiary Industry Activity m/m", "Actual": "0.1%", "Forecast": "0.6%", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-18", "Time": "10:00", "Currency": "EUR", "Event": "Trade Balance", "Actual": "18.1B", "Forecast": "16.2B", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-18", "Time": "13:15", "Currency": "CAD", "Event": "Housing Starts", "Actual": "270K", "Forecast": "284K", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-18", "Time": "13:30", "Currency": "CAD", "Event": "Foreign Securities Purchases", "Actual": "-4.75B", "Forecast": "-2.79B", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-18", "Time": "15:00", "Currency": "USD", "Event": "NAHB Housing Market Index", "Actual": "34", "Forecast": "33", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Input q/q", "Actual": "1.4%", "Forecast": "2.9%", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Output q/q", "Actual": "1.0%", "Forecast": "2.1%", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-19", "Time": "01:30", "Currency": "AUD", "Event": "Westpac Consumer Sentiment", "Actual": "0.6%", "Forecast": "", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-19", "Time": "Tentative", "Currency": "CNY", "Event": "Foreign Direct Investment ytd/y", "Actual": "-15.2%", "Forecast": "", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-19", "Time": "09:00", "Currency": "EUR", "Event": "Current Account", "Actual": "33.4B", "Forecast": "32.3B", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "CPI m/m", "Actual": "0.4%", "Forecast": "0.1%", "Previous": "", "Impact": "High"},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "Median CPI y/y", "Actual": "3.1%", "Forecast": "3.1%", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "BoC Business Outlook Survey", "Actual": "", "Forecast": "", "Previous": "", "Impact": "Medium"},
    {"Date": "2025-08-20", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-0.2%", "Forecast": "", "Previous": "", "Impact": "Low"},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "CPI y/y", "Actual": "2.5%", "Forecast": "2.6%", "Previous": "2.7%", "Impact": "High"},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "PPI y/y", "Actual": "-3.3%", "Forecast": "-3.0%", "Previous": "-3.1%", "Impact": "Medium"},
    {"Date": "2025-08-20", "Time": "08:00", "Currency": "EUR", "Event": "German PPI m/m", "Actual": "0.2%", "Forecast": "0.1%", "Previous": "0.1%", "Impact": "Low"},
    {"Date": "2025-08-20", "Time": "13:30", "Currency": "CAD", "Event": "Manufacturing Sales m/m", "Actual": "0.3%", "Forecast": "0.5%", "Previous": "-1.2%", "Impact": "Medium"},
    {"Date": "2025-08-20", "Time": "14:30", "Currency": "USD", "Event": "Crude Oil Inventories", "Actual": "-5.3M", "Forecast": "-1.2M", "Previous": "-0.6M", "Impact": "Medium"},
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Employment Change", "Actual": "36.1K", "Forecast": "30.0K", "Previous": "-10.0K", "Impact": "High"},
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Unemployment Rate", "Actual": "3.6%", "Forecast": "3.7%", "Previous": "3.8%", "Impact": "High"},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI y/y", "Actual": "3.2%", "Forecast": "3.3%", "Previous": "3.0%", "Impact": "High"},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI m/m", "Actual": "0.3%", "Forecast": "0.4%", "Previous": "0.1%", "Impact": "Medium"},
    {"Date": "2025-08-21", "Time": "14:00", "Currency": "EUR", "Event": "ECB Interest Rate Decision", "Actual": "0.50%", "Forecast": "0.50%", "Previous": "0.25%", "Impact": "High"},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Initial Jobless Claims", "Actual": "218K", "Forecast": "220K", "Previous": "217K", "Impact": "Medium"},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Continuing Claims", "Actual": "1445K", "Forecast": "1450K", "Previous": "1440K", "Impact": "Medium"},
    {"Date": "2025-08-21", "Time": "15:00", "Currency": "USD", "Event": "Existing Home Sales", "Actual": "4.25M", "Forecast": "4.23M", "Previous": "4.19M", "Impact": "Medium"},
    {"Date": "2025-08-22", "Time": "09:30", "Currency": "GBP", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.3%", "Previous": "0.2%", "Impact": "Medium"},
]
econ_df = pd.DataFrame(econ_calendar_data)

# Static market data (simulated)
market_data = [
    {"Pair": "EUR/USD", "Bid": 1.1050, "Ask": 1.1052, "Change": 0.45, "Volatility": 0.85},
    {"Pair": "USD/JPY", "Bid": 147.20, "Ask": 147.22, "Change": -0.32, "Volatility": 1.10},
    {"Pair": "GBP/USD", "Bid": 1.2950, "Ask": 1.2953, "Change": 0.28, "Volatility": 0.95},
    {"Pair": "USD/CHF", "Bid": 0.8650, "Ask": 0.8652, "Change": -0.15, "Volatility": 0.70},
    {"Pair": "AUD/USD", "Bid": 0.6700, "Ask": 0.6703, "Change": 0.62, "Volatility": 1.20},
    {"Pair": "NZD/USD", "Bid": 0.6050, "Ask": 0.6052, "Change": 0.18, "Volatility": 0.90},
    {"Pair": "USD/CAD", "Bid": 1.3750, "Ask": 1.3753, "Change": -0.25, "Volatility": 0.80},
    {"Pair": "EUR/GBP", "Bid": 0.8520, "Ask": 0.8522, "Change": 0.10, "Volatility": 0.65},
]
market_df = pd.DataFrame(market_data)

# Correlation matrix (static)
correlation_data = {
    "EUR/USD": {"EUR/USD": 1.00, "USD/JPY": -0.65, "GBP/USD": 0.85, "USD/CHF": -0.70, "AUD/USD": 0.75, "NZD/USD": 0.70, "USD/CAD": -0.60, "EUR/GBP": 0.55},
    "USD/JPY": {"EUR/USD": -0.65, "USD/JPY": 1.00, "GBP/USD": -0.60, "USD/CHF": 0.80, "AUD/USD": -0.55, "NZD/USD": -0.50, "USD/CAD": 0.45, "EUR/GBP": -0.40},
    "GBP/USD": {"EUR/USD": 0.85, "USD/JPY": -0.60, "GBP/USD": 1.00, "USD/CHF": -0.65, "AUD/USD": 0.80, "NZD/USD": 0.75, "USD/CAD": -0.55, "EUR/GBP": 0.30},
    "USD/CHF": {"EUR/USD": -0.70, "USD/JPY": 0.80, "GBP/USD": -0.65, "USD/CHF": 1.00, "AUD/USD": -0.60, "NZD/USD": -0.55, "USD/CAD": 0.50, "EUR/GBP": -0.45},
    "AUD/USD": {"EUR/USD": 0.75, "USD/JPY": -0.55, "GBP/USD": 0.80, "USD/CHF": -0.60, "AUD/USD": 1.00, "NZD/USD": 0.90, "USD/CAD": -0.65, "EUR/GBP": 0.40},
    "NZD/USD": {"EUR/USD": 0.70, "USD/JPY": -0.50, "GBP/USD": 0.75, "USD/CHF": -0.55, "AUD/USD": 0.90, "NZD/USD": 1.00, "USD/CAD": -0.60, "EUR/GBP": 0.35},
    "USD/CAD": {"EUR/USD": -0.60, "USD/JPY": 0.45, "GBP/USD": -0.55, "USD/CHF": 0.50, "AUD/USD": -0.65, "NZD/USD": -0.60, "USD/CAD": 1.00, "EUR/GBP": -0.30},
    "EUR/GBP": {"EUR/USD": 0.55, "USD/JPY": -0.40, "GBP/USD": 0.30, "USD/CHF": -0.45, "AUD/USD": 0.40, "NZD/USD": 0.35, "USD/CAD": -0.30, "EUR/GBP": 1.00},
}
correlation_df = pd.DataFrame(correlation_data)

# Load news
df_news = get_fxstreet_forex_news()

# =========================================================
# NAVIGATION
# =========================================================
tabs = ["Market Overview", "Economic Calendar", "Technical Analysis", "Tools", "My Account"]
selected_tab = st.tabs(tabs)

# =========================================================
# TAB 1: MARKET OVERVIEW
# =========================================================
with selected_tab[0]:
    st.title("📈 Market Overview")
    st.caption("Real-time snapshot of major currency pairs and market sentiment.")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 💹 Currency Pairs Snapshot")
        def highlight_change(row):
            color = '#4CAF50' if row['Change'] > 0 else '#FF4D4D'
            return ['background-color: %s' % color if col == 'Change' else '' for col in row.index]
        st.dataframe(market_df.style.format({"Bid": "{:.4f}", "Ask": "{:.4f}", "Change": "{:.2f}%", "Volatility": "{:.2f}%"}).apply(highlight_change, axis=1), use_container_width=True)
    with col2:
        st.markdown("### 📊 Sentiment Gauge")
        pair = st.selectbox("Select Pair for Sentiment", market_df["Pair"].tolist(), key="sentiment_pair")
        base, quote = pair.split("/")
        filtered_news = df_news[df_news["Currency"].isin([base, quote])]
        avg_polarity = filtered_news["Polarity"].mean() if not filtered_news.empty else 0
        needle_rotation = (avg_polarity + 1) * 90  # Map [-1,1] to [0,180] degrees
        st.markdown(
            f"""
            <div class="gauge">
                <div class="gauge-needle" style="transform: rotate({needle_rotation}deg);"></div>
            </div>
            <div style="text-align: center; margin-top: 10px;">
                Sentiment: <b>{rate_impact(avg_polarity)}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# TAB 2: ECONOMIC CALENDAR
# =========================================================
with selected_tab[1]:
    st.title("📅 Economic Calendar")
    st.caption("Track high-impact events and filter by currency or importance.")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        currency_filter = st.multiselect("Filter by Currency", sorted(econ_df["Currency"].unique()), key="econ_currency")
    with col2:
        impact_filter = st.multiselect("Filter by Impact", ["High", "Medium", "Low"], key="econ_impact")
    with col3:
        sort_by = st.selectbox("Sort By", ["Date", "Currency", "Impact"], key="econ_sort")
    filtered_df = econ_df.copy()
    if currency_filter:
        filtered_df = filtered_df[filtered_df["Currency"].isin(currency_filter)]
    if impact_filter:
        filtered_df = filtered_df[filtered_df["Impact"].isin(impact_filter)]
    filtered_df = filtered_df.sort_values(sort_by)
    def highlight_impact(row):
        colors = {"High": "#FF4D4D", "Medium": "#FFD700", "Low": "#4CAF50"}
        return ['background-color: %s' % colors.get(row['Impact'], '') if col == 'Impact' else '' for col in row.index]
    st.dataframe(filtered_df.style.apply(highlight_impact, axis=1), use_container_width=True, height=400)
    st.markdown("### ⏰ Event Timeline")
    timeline_data = filtered_df[["Date", "Time", "Event"]].head(5)
    for _, row in timeline_data.iterrows():
        st.markdown(f"**{row['Date']} {row['Time']}**: {row['Event']}")

# =========================================================
# TAB 3: TECHNICAL ANALYSIS
# =========================================================
with selected_tab[2]:
    st.title("📊 Technical Analysis")
    st.caption("Advanced charts with indicators and curated news.")
    pairs_map = {
        "EUR/USD": "FX:EURUSD",
        "USD/JPY": "FX:USDJPY",
        "GBP/USD": "FX:GBPUSD",
        "USD/CHF": "OANDA:USDCHF",
        "AUD/USD": "FX:AUDUSD",
        "NZD/USD": "OANDA:NZDUSD",
        "USD/CAD": "CMCMARKETS:USDCAD",
        "EUR/GBP": "FX:EURGBP",
    }
    col1, col2 = st.columns([3, 1])
    with col1:
        pair = st.selectbox("Select Pair", list(pairs_map.keys()), key="ta_pair")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1H", "4H", "D"], key="ta_timeframe")
        tv_symbol = pairs_map[pair]
        tv_html = f"""
        <div class="tradingview-widget-container" style="height:{tv_height}px; width:100%">
          <div id="tradingview_chart" class="tradingview-widget-container__widget" style="height:{tv_height}px; width:100%"></div>
          <div class="tradingview-widget-copyright" style="padding-top:6px">
            <a href="https://www.tradingview.com/symbols/{tv_symbol.replace(':','-')}/" rel="noopener" target="_blank">
              <span class="blue-text">{pair} chart by TradingView</span>
            </a>
          </div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "{timeframe}",
            "timezone": "Etc/UTC",
            "theme": "{st.session_state.theme}",
            "style": "1",
            "locale": "en",
            "hide_top_toolbar": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "save_image": true,
            "calendar": false,
            "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies"],
            "watchlist": {list(pairs_map.values())}
          }}
          </script>
        </div>
        """
        components.html(tv_html, height=tv_height + 50, scrolling=False)
    with col2:
        st.markdown("### 📰 News & Sentiment")
        if not df_news.empty:
            base, quote = pair.split("/")
            filtered_df = df_news[df_news["Currency"].isin([base, quote])].copy()
            filtered_df["HighProb"] = filtered_df.apply(
                lambda row: "🔥" if (row["Impact"] in ["Significantly Bullish", "Significantly Bearish"]) and
                                    (pd.to_datetime(row["Date"]) >= pd.Timestamp.utcnow() - pd.Timedelta(days=1))
                else "", axis=1
            )
            filtered_df_display = filtered_df.copy()
            filtered_df_display["HeadlineDisplay"] = filtered_df["HighProb"] + " " + filtered_df["Headline"]
            if not filtered_df_display.empty:
                selected_headline = st.selectbox(
                    "Select Headline", filtered_df_display["HeadlineDisplay"].tolist(), key="ta_headline_select"
                )
                selected_row = filtered_df_display[filtered_df_display["HeadlineDisplay"] == selected_headline].iloc[0]
                st.markdown(f"**[{selected_row['Headline']}]({selected_row['Link']})**")
                st.write(f"**Published:** {selected_row['Date'].date() if isinstance(selected_row['Date'], pd.Timestamp) else selected_row['Date']}")
                st.write(f"**Currency:** {selected_row['Currency']} | **Impact:** {selected_row['Impact']}")
                with st.expander("Summary"):
                    st.write(selected_row["Summary"])
            else:
                st.info("No pair-specific headlines found.")
        else:
            st.info("News feed unavailable.")

# =========================================================
# TAB 4: TOOLS
# =========================================================
with selected_tab[3]:
    st.title("🛠 Tools")
    tools_subtabs = st.tabs(["Position Sizing", "Backtesting", "Correlation Matrix"])
    # ---------------- Position Sizing Calculator ----------------
    with tools_subtabs[0]:
        st.header("💰 Position Sizing Calculator")
        st.markdown(
            """
            <div class="tooltip">Calculate position size and risk
                <span class="tooltiptext">Enter your account details and trade parameters to calculate optimal position size and risk-reward ratio.</span>
            </div>
            """, unsafe_allow_html=True
        )
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            currency_pair = st.selectbox("Currency Pair", list(pairs_map.keys()), key="ps_currency_pair")
            account_balance = st.number_input("Account Balance", min_value=100.0, value=10000.0, step=100.0, key="ps_balance")
            risk_percentage = st.number_input("Risk Per Trade (%)", min_value=0.1, value=1.0, step=0.1, key="ps_risk")
        with col_calc2:
            stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=50.0, step=1.0, key="ps_stop_loss")
            take_profit_pips = st.number_input("Take Profit (pips)", min_value=0.0, value=100.0, step=1.0, key="ps_take_profit")
            account_currency = st.selectbox("Account Currency", ["USD", "EUR", "GBP", "JPY"], key="ps_account_currency")
        pip_multiplier = 100 if "JPY" in currency_pair else 10000
        pip_value = (0.0001 if "JPY" not in currency_pair else 0.01) * 100000
        risk_amount = account_balance * (risk_percentage / 100)
        position_size = risk_amount / (stop_loss_pips * pip_value)
        profit = position_size * take_profit_pips * pip_value if take_profit_pips > 0 else 0
        risk_reward = take_profit_pips / stop_loss_pips if take_profit_pips > 0 else 0
        st.markdown(f"**Position Size**: {position_size:.2f} lots")
        st.markdown(f"**Risk Amount**: {risk_amount:.2f} {account_currency}")
        st.markdown(f"**Potential Profit**: {profit:.2f} {account_currency}")
        st.markdown(f"**Risk-Reward Ratio**: {risk_reward:.2f}:1")
    # ---------------- Backtesting ----------------
    with tools_subtabs[1]:
        st.header("📊 Backtesting")
        st.markdown("Backtest strategies with historical data and track performance.")
        tv_widget = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_advanced_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget({{
                    "width": "100%",
                    "height": 600,
                    "symbol": "FX:EURUSD",
                    "interval": "D",
                    "timezone": "Etc/UTC",
                    "theme": "{st.session_state.theme}",
                    "style": "1",
                    "toolbar_bg": "{'#f1f3f6' if st.session_state.theme == 'light' else '#1b1b1b'}",
                    "withdateranges": true,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true,
                    "save_image": false,
                    "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies"],
                    "container_id": "tradingview_advanced_chart"
                }});
            </script>
        </div>
        """
        st.components.v1.html(tv_widget, height=620)
        journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "P/L", "Notes"]
        if "tools_trade_journal" not in st.session_state or st.session_state.tools_trade_journal.empty:
            st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols)
        updated_journal = st.data_editor(
            data=st.session_state.tools_trade_journal.copy(),
            num_rows="dynamic",
            key="tools_backtesting_journal_unique",
            column_config={
                "P/L": st.column_config.NumberColumn("P/L", format="%.2f")
            }
        )
        st.session_state.tools_trade_journal = updated_journal
        if not updated_journal.empty:
            st.markdown("### 📈 Trade Performance")
            win_rate = (updated_journal["P/L"] > 0).mean() * 100 if not updated_journal["P/L"].isna().all() else 0
            avg_pl = updated_journal["P/L"].mean() if not updated_journal["P/L"].isna().all() else 0
            total_trades = len(updated_journal)
            st.markdown(f"**Total Trades**: {total_trades}")
            st.markdown(f"**Win Rate**: {win_rate:.2f}%")
            st.markdown(f"**Average P/L**: {avg_pl:.2f}")
        if "logged_in_user" in st.session_state:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save Journal", key="save_journal_button"):
                    username = st.session_state.logged_in_user
                    accounts = {}
                    if os.path.exists(ACCOUNTS_FILE):
                        with open(ACCOUNTS_FILE, "r") as f:
                            accounts = json.load(f)
                    accounts.setdefault(username, {})["tools_trade_journal"] = st.session_state.tools_trade_journal.to_dict(orient="records")
                    with open(ACCOUNTS_FILE, "w") as f:
                        json.dump(accounts, f, indent=4)
                    st.success("Trading journal saved!")
            with col2:
                if st.button("📂 Load Journal", key="load_journal_button"):
                    username = st.session_state.logged_in_user
                    accounts = {}
                    if os.path.exists(ACCOUNTS_FILE):
                        with open(ACCOUNTS_FILE, "r") as f:
                            accounts = json.load(f)
                    saved_journal = accounts.get(username, {}).get("tools_trade_journal", [])
                    if saved_journal:
                        st.session_state.tools_trade_journal = pd.DataFrame(saved_journal, columns=journal_cols)
                        st.success("Trading journal loaded!")
                    else:
                        st.info("No saved journal found.")
        else:
            st.info("Sign in to save your trading journal.")
    # ---------------- Correlation Matrix ----------------
    with tools_subtabs[2]:
        st.header("🔗 Correlation Matrix")
        st.markdown("Analyze correlations to optimize your portfolio.")
        st.dataframe(correlation_df.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1), use_container_width=True)

# =========================================================
# TAB 5: MY ACCOUNT
# =========================================================
with selected_tab[4]:
    st.title("👤 My Account")
    st.subheader("Account Management")
    if not os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "w") as f:
            json.dump({}, f)
    # ---------------- Login ----------------
    login_expander = st.expander("Login")
    with login_expander:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            with open(ACCOUNTS_FILE, "r") as f:
                accounts = json.load(f)
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            if username in accounts and accounts[username]["password"] == hashed_password:
                st.session_state.logged_in_user = username
                st.success(f"Logged in as {username}")
                saved_journal = accounts.get(username, {}).get("tools_trade_journal", [])
                saved_watchlist = accounts.get(username, {}).get("watchlist", [])
                saved_alerts = accounts.get(username, {}).get("alerts", [])
                journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "P/L", "Notes"]
                st.session_state.tools_trade_journal = pd.DataFrame(saved_journal, columns=journal_cols) if saved_journal else pd.DataFrame(columns=journal_cols)
                st.session_state.watchlist = saved_watchlist if saved_watchlist else list(pairs_map.keys())[:3]
                st.session_state.alerts = saved_alerts if saved_alerts else []
            else:
                st.error("Invalid username or password")
    # ---------------- Sign Up ----------------
    signup_expander = st.expander("Sign Up")
    with signup_expander:
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            with open(ACCOUNTS_FILE, "r") as f:
                accounts = json.load(f)
            if new_username in accounts:
                st.error("Username already exists")
            else:
                accounts[new_username] = {"password": hashlib.sha256(new_password.encode()).hexdigest()}
                with open(ACCOUNTS_FILE, "w") as f:
                    json.dump(accounts, f, indent=4)
                st.success(f"Account created for {new_username}")
    # ---------------- Account Settings ----------------
    if "logged_in_user" in st.session_state:
        st.subheader("Profile Settings")
        colA, colB = st.columns(2)
        with colA:
            name = st.text_input("Name", value=st.session_state.get("name", ""), key="account_name")
            base_ccy = st.selectbox("Preferred Base Currency", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"], key="account_base_ccy")
        with colB:
            email = st.text_input("Email", value=st.session_state.get("email", ""), key="account_email")
            alerts_enabled = st.checkbox("Enable High-Impact Event Alerts", value=st.session_state.get("alerts_enabled", True), key="account_alerts")
        if st.button("Save Preferences", key="account_save_prefs"):
            st.session_state.name = name
            st.session_state.email = email
            st.session_state.base_ccy = base_ccy
            st.session_state.alerts_enabled = alerts_enabled
            accounts = {}
            if os.path.exists(ACCOUNTS_FILE):
                with open(ACCOUNTS_FILE, "r") as f:
                    accounts = json.load(f)
            accounts.setdefault(st.session_state.logged_in_user, {})["preferences"] = {
                "name": name, "email": email, "base_ccy": base_ccy, "alerts_enabled": alerts_enabled
            }
            with open(ACCOUNTS_FILE, "w") as f:
                json.dump(accounts, f, indent=4)
            st.success("Preferences saved!")
        # ---------------- Watchlist ----------------
        st.subheader("Watchlist")
        if "watchlist" not in st.session_state:
            st.session_state.watchlist = list(pairs_map.keys())[:3]
        watchlist = st.multiselect("Select Pairs to Watch", list(pairs_map.keys()), default=st.session_state.watchlist, key="watchlist_select")
        if st.button("Save Watchlist"):
            st.session_state.watchlist = watchlist
            accounts = {}
            if os.path.exists(ACCOUNTS_FILE):
                with open(ACCOUNTS_FILE, "r") as f:
                    accounts = json.load(f)
            accounts.setdefault(st.session_state.logged_in_user, {})["watchlist"] = watchlist
            with open(ACCOUNTS_FILE, "w") as f:
                json.dump(accounts, f, indent=4)
            st.success("Watchlist saved!")
        for pair in watchlist:
            st.markdown(f"**{pair}**")
        # ---------------- Alerts ----------------
        st.subheader("Event Alerts")
        high_impact_events = econ_df[econ_df["Impact"] == "High"][["Date", "Time", "Currency", "Event"]]
        if not high_impact_events.empty and st.session_state.get("alerts_enabled", False):
            st.markdown("### Upcoming High-Impact Events")
            st.dataframe(high_impact_events, use_container_width=True)
            selected_events = st.multiselect("Select Events for Alerts", high_impact_events["Event"].tolist(), key="alerts_select")
            if st.button("Save Alerts"):
                st.session_state.alerts = selected_events
                accounts = {}
                if os.path.exists(ACCOUNTS_FILE):
                    with open(ACCOUNTS_FILE, "r") as f:
                        accounts = json.load(f)
                accounts.setdefault(st.session_state.logged_in_user, {})["alerts"] = selected_events
                with open(ACCOUNTS_FILE, "w") as f:
                    json.dump(accounts, f, indent=4)
                st.success("Alerts saved!")
            if st.session_state.get("alerts", []):
                st.markdown("**Your Alerts**:")
                for event in st.session_state.alerts:
                    st.markdown(f"- {event}")
