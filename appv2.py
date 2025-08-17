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
import yfinance as yf
import matplotlib.pyplot as plt
import altair as alt
# Path to your accounts JSON file
ACCOUNTS_FILE = "user_accounts.json"  # Consistent file name
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Ultimate Forex Dashboard", layout="wide", initial_sidebar_state="collapsed")
# ----------------- FIXED SETTINGS -----------------
bg_opacity = 0.6  # Enhanced background opacity for better visuals
tv_height = 1000  # Increased TradingView chart height for better usability

# ----------------- CUSTOM CSS (Enhanced Dark Futuristic BG + Tabs + More) -----------------
st.markdown(
    f"""
<style>
/* Enhanced futuristic dark background with animated grid and subtle glow */
.stApp {{
    background:
        radial-gradient(circle at 10% 15%, rgba(255,215,0,{bg_opacity*0.2}) 0%, transparent 20%),
        radial-gradient(circle at 90% 25%, rgba(0,170,255,{bg_opacity*0.15}) 0%, transparent 20%),
        radial-gradient(circle at 50% 80%, rgba(255,0,255,{bg_opacity*0.1}) 0%, transparent 30%),
        linear-gradient(135deg, #0b0b0b 0%, #0a0a0a 100%);
    color: #e0e0e0;
    font-family: 'Arial', sans-serif;
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(90deg, rgba(255,255,255,{bg_opacity*0.06}) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,{bg_opacity*0.06}) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: moveGrid 40s linear infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 1;
}}
@keyframes moveGrid {{
    0% {{ transform: translate(-25px, -25px); }}
    100% {{ transform: translate(25px, 25px); }}
}}
/* Lift content above bg */
.main, .block-container, .stTabs, .stMarkdown, .css-ffhzg2, .css-1d391kg {{ position: relative; z-index: 1; }}
/* Enhanced Tab styling with glow */
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background-color: #FFD700 !important;
    color: black !important;
    font-weight: bold;
    padding: 16px 28px !important;
    border-radius: 12px;
    margin-right: 12px !important;
    box-shadow: 0 0 10px rgba(255,215,0,0.5);
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
    background-color: #1f1f1f !important;
    color: #ccc !important;
    padding: 16px 28px !important;
    border-radius: 12px;
    margin-right: 12px !important;
    border: 1px solid #282828 !important;
    transition: all 0.3s;
}}
div[data-baseweb="tab-list"] button[aria-selected="false"]:hover {{
    background-color: #282828 !important;
    box-shadow: 0 0 5px rgba(255,215,0,0.3);
}}
/* Enhanced Card styling with hover effect */
.card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    transition: transform 0.3s, box-shadow 0.3s;
}}
.card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.4);
}}
/* Metric styling */
.stMetric {{
    background-color: #1f1f1f;
    border-radius: 10px;
    padding: 10px;
}}
/* Dataframe enhancements */
.dataframe th {{
    background-color: #FFD700 !important;
    color: black !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS / FUNCTIONS
# =========================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

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

def assign_impact(event: str) -> str:
    event_upper = event.upper()
    high_impact_keywords = ["GDP", "CPI", "PPI", "INTEREST RATE", "UNEMPLOYMENT RATE", "RETAIL SALES", "FOMC", "ECB", "BOE", "BOJ", "NFP"]
    medium_impact_keywords = ["MANUFACTURING", "INDUSTRIAL PRODUCTION", "HOUSING", "CONSUMER SENTIMENT"]
    if any(kw in event_upper for kw in high_impact_keywords):
        return "High"
    elif any(kw in event_upper for kw in medium_impact_keywords):
        return "Medium"
    return "Low"

def get_pip_value(pair: str, position_size: float, account_currency: str, current_price: float) -> float:
    base, quote = pair.split("/")
    if quote == "JPY":
        pip = 0.01
    else:
        pip = 0.0001
    pip_value = (pip / current_price) * (position_size * 100000)
    # Simplify conversion assuming account_currency == quote for now; extend later if needed
    return pip_value

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_price(pair: str) -> float:
    ticker = f"{pair.replace('/', '')}=X"
    data = yf.download(ticker, period="1d", interval="1m")
    if not data.empty:
        return data['Close'].iloc[-1]
    return 1.0  # Fallback

@st.cache_data(ttl=600, show_spinner=False)
def get_fxstreet_forex_news() -> pd.DataFrame:
    RSS_URL = "https://www.fxstreet.com/rss/news"
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
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=3)
            df = df[df["Date"] >= cutoff]
        except Exception:
            pass
        return df.reset_index(drop=True)
    return pd.DataFrame(columns=["Date","Currency","Headline","Polarity","Impact","Summary","Link"])

# =========================================================
# DATA LOADING (Enhanced with live prices and impact assignment)
# =========================================================
# Static calendar (enhanced with assigned impacts if empty)
econ_calendar_data = [
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP Price Index y/y", "Actual": "3.0%", "Forecast": "3.1%", "Previous": "3.3%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP q/q", "Actual": "0.3%", "Forecast": "0.1%", "Previous": "0.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "02:30", "Currency": "CNY", "Event": "New Home Prices m/m", "Actual": "-0.31%", "Forecast": "", "Previous": "-0.27%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Industrial Production y/y", "Actual": "5.7%", "Forecast": "6.0%", "Previous": "6.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Retail Sales y/y", "Actual": "3.7%", "Forecast": "4.6%", "Previous": "4.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Fixed Asset Investment ytd/y", "Actual": "1.6%", "Forecast": "2.7%", "Previous": "2.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "NBS Press Conference", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Unemployment Rate", "Actual": "5.2%", "Forecast": "5.1%", "Previous": "5.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "05:30", "Currency": "JPY", "Event": "Revised Industrial Production m/m", "Actual": "2.1%", "Forecast": "1.7%", "Previous": "1.7%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "French Bank Holiday", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "Italian Bank Holiday", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "ECOFIN Meetings", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "CAD", "Event": "Manufacturing Sales m/m", "Actual": "0.3%", "Forecast": "0.4%", "Previous": "-1.5%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "CAD", "Event": "Wholesale Sales m/m", "Actual": "0.7%", "Forecast": "0.7%", "Previous": "0.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Core Retail Sales m/m", "Actual": "0.3%", "Forecast": "0.3%", "Previous": "0.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.6%", "Previous": "0.9%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Empire State Manufacturing Index", "Actual": "11.9", "Forecast": "-1.2", "Previous": "5.5", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Import Prices m/m", "Actual": "0.4%", "Forecast": "0.1%", "Previous": "-0.1%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "14:15", "Currency": "USD", "Event": "Capacity Utilization Rate", "Actual": "77.5%", "Forecast": "77.6%", "Previous": "77.7%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "14:15", "Currency": "USD", "Event": "Industrial Production m/m", "Actual": "-0.1%", "Forecast": "0.0%", "Previous": "0.4%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Prelim UoM Consumer Sentiment", "Actual": "58.6", "Forecast": "61.9", "Previous": "61.7", "Impact": ""},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Prelim UoM Inflation Expectations", "Actual": "4.9%", "Forecast": "", "Previous": "4.5%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Business Inventories m/m", "Actual": "0.2%", "Forecast": "0.2%", "Previous": "0.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "21:00", "Currency": "USD", "Event": "TIC Long-Term Purchases", "Actual": "150.8B", "Forecast": "", "Previous": "266.8B", "Impact": ""},
    {"Date": "2025-08-16", "Time": "Tentative", "Currency": "USD", "Event": "President Trump Speaks", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-17", "Time": "23:30", "Currency": "NZD", "Event": "BusinessNZ Services Index", "Actual": "47.3", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-1.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "05:30", "Currency": "JPY", "Event": "Tertiary Industry Activity m/m", "Actual": "0.1%", "Forecast": "0.6%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "10:00", "Currency": "EUR", "Event": "Trade Balance", "Actual": "18.1B", "Forecast": "16.2B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "13:15", "Currency": "CAD", "Event": "Housing Starts", "Actual": "270K", "Forecast": "284K", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "13:30", "Currency": "CAD", "Event": "Foreign Securities Purchases", "Actual": "-4.75B", "Forecast": "-2.79B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "15:00", "Currency": "USD", "Event": "NAHB Housing Market Index", "Actual": "34", "Forecast": "33", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Input q/q", "Actual": "1.4%", "Forecast": "2.9%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Output q/q", "Actual": "1.0%", "Forecast": "2.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "01:30", "Currency": "AUD", "Event": "Westpac Consumer Sentiment", "Actual": "0.6%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "Tentative", "Currency": "CNY", "Event": "Foreign Direct Investment ytd/y", "Actual": "-15.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "09:00", "Currency": "EUR", "Event": "Current Account", "Actual": "33.4B", "Forecast": "32.3B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "CPI m/m", "Actual": "0.4%", "Forecast": "0.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "Median CPI y/y", "Actual": "3.1%", "Forecast": "3.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "BoC Business Outlook Survey", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-20", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-0.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "CPI y/y", "Actual": "2.5%", "Forecast": "2.6%", "Previous": "2.7%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "PPI y/y", "Actual": "-3.3%", "Forecast": "-3.0%", "Previous": "-3.1%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "08:00", "Currency": "EUR", "Event": "German PPI m/m", "Actual": "0.2%", "Forecast": "0.1%", "Previous": "0.1%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "13:30", "Currency": "CAD", "Event": "Manufacturing Sales m/m", "Actual": "0.3%", "Forecast": "0.5%", "Previous": "-1.2%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "14:30", "Currency": "USD", "Event": "Crude Oil Inventories", "Actual": "-5.3M", "Forecast": "-1.2M", "Previous": "-0.6M", "Impact": ""},
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Employment Change", "Actual": "36.1K", "Forecast": "30.0K", "Previous": "-10.0K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Unemployment Rate", "Actual": "3.6%", "Forecast": "3.7%", "Previous": "3.8%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI y/y", "Actual": "3.2%", "Forecast": "3.3%", "Previous": "3.0%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI m/m", "Actual": "0.3%", "Forecast": "0.4%", "Previous": "0.1%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "14:00", "Currency": "EUR", "Event": "ECB Interest Rate Decision", "Actual": "0.50%", "Forecast": "0.50%", "Previous": "0.25%", "Impact": "High"},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Initial Jobless Claims", "Actual": "218K", "Forecast": "220K", "Previous": "217K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Continuing Claims", "Actual": "1445K", "Forecast": "1450K", "Previous": "1440K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "15:00", "Currency": "USD", "Event": "Existing Home Sales", "Actual": "4.25M", "Forecast": "4.23M", "Previous": "4.19M", "Impact": ""},
    {"Date": "2025-08-22", "Time": "09:30", "Currency": "GBP", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.3%", "Previous": "0.2%", "Impact": "Medium"},
]
econ_df = pd.DataFrame(econ_calendar_data)
econ_df['Impact'] = econ_df.apply(lambda row: assign_impact(row['Event']) if not row['Impact'] else row['Impact'], axis=1)
# Interest rates (enhanced with delta calculation)
interest_rates = [
    {"Currency": "USD", "Current": "4.50%", "Previous": "4.75%", "Changed": "12-18-2024"},
    {"Currency": "GBP", "Current": "4.00%", "Previous": "4.25%", "Changed": "08-07-2025"},
    {"Currency": "EUR", "Current": "2.15%", "Previous": "2.40%", "Changed": "06-05-2025"},
    {"Currency": "JPY", "Current": "0.50%", "Previous": "0.25%", "Changed": "01-24-2025"},
    {"Currency": "AUD", "Current": "3.60%", "Previous": "3.85%", "Changed": "08-12-2025"},
    {"Currency": "CAD", "Current": "2.75%", "Previous": "3.00%", "Changed": "03-12-2025"},
    {"Currency": "NZD", "Current": "3.25%", "Previous": "3.50%", "Changed": "05-28-2025"},
    {"Currency": "CHF", "Current": "0.00%", "Previous": "0.25%", "Changed": "06-19-2025"},
]
for rate in interest_rates:
    curr = float(rate['Current'][:-1])
    prev = float(rate['Previous'][:-1])
    rate['Delta'] = curr - prev
# Load news
df_news = get_fxstreet_forex_news()

# =========================================================
# NAVIGATION (Enhanced with better tab names and icons)
# =========================================================
tabs = ["📅 Forex Fundamentals", "📖 Understanding Forex", "📊 Technical Analysis", "🛠️ Tools", "👤 My Account"]
selected_tab = st.tabs(tabs)

# =========================================================
# TAB 1: FOREX FUNDAMENTALS (Enhanced with live prices and metrics)
# =========================================================
with selected_tab[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📅 Forex Fundamentals")
        st.caption("Macro snapshot: live prices, sentiment, calendar highlights, and policy rates.")
    with col2:
        st.info("Use **Technical Analysis** for charts and **Tools** for calculators.")
    # -------- Live Prices Metrics --------
    st.markdown("### 💹 Live Major Pair Prices")
    pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
    cols = st.columns(len(pairs))
    for i, pair in enumerate(pairs):
        price = fetch_live_price(pair)
        delta = round((price - price * 0.001), 5)  # Simulated delta for demo
        with cols[i]:
            st.metric(label=pair, value=f"{price:.4f}", delta=f"{delta:.4f}")
    # -------- Economic Calendar (with filters and color-coded impacts) --------
    st.markdown("### 🗓️ Upcoming Economic Events")
    if 'selected_currency_1' not in st.session_state:
        st.session_state.selected_currency_1 = None
    if 'selected_currency_2' not in st.session_state:
        st.session_state.selected_currency_2 = None
    uniq_ccy = sorted(set(list(econ_df["Currency"].unique()) + list(df_news["Currency"].unique())))
    currency_filter_1 = st.selectbox(
        "Primary currency to highlight", options=["None"] + uniq_ccy, key="cal_curr_1"
    )
    st.session_state.selected_currency_1 = None if currency_filter_1 == "None" else currency_filter_1
    currency_filter_2 = st.selectbox(
        "Secondary currency to highlight", options=["None"] + uniq_ccy, key="cal_curr_2"
    )
    st.session_state.selected_currency_2 = None if currency_filter_2 == "None" else currency_filter_2
    def highlight_currency(row):
        styles = [''] * len(row)
        impact_color = {'High': 'background-color: #ff4d4d', 'Medium': 'background-color: #ffd700', 'Low': 'background-color: #4d4dff'}
        if row['Impact']:
            styles[row.index.get_loc('Impact')] = impact_color.get(row['Impact'], '')
        if st.session_state.selected_currency_1 and row['Currency'] == st.session_state.selected_currency_1:
            styles = ['background-color: #171447; color: white' if col == 'Currency' else 'background-color: #171447' for col in row.index]
        if st.session_state.selected_currency_2 and row['Currency'] == st.session_state.selected_currency_2:
            styles = ['background-color: #471414; color: white' if col == 'Currency' else 'background-color: #471414' for col in row.index]
        return styles
    st.dataframe(econ_df.style.apply(highlight_currency, axis=1), use_container_width=True, height=400)
    # -------- Interest rate tiles (with delta metrics) --------
    st.markdown("### 🏦 Major Central Bank Interest Rates")
    boxes_per_row = 4
    colors = ["#171447", "#471414", "#144714", "#471447"]
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
                            <p style="margin: 2px 0;"><b>Delta:</b> {rate['Delta']:.2f}%</p>
                            <p style="margin: 2px 0;"><b>Changed On:</b> {rate['Changed']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# =========================================================
# TAB 2: UNDERSTANDING FOREX FUNDAMENTALS (Enhanced with more sections and visuals)
# =========================================================
with selected_tab[1]:
    st.title("📖 Understanding Forex Fundamentals")
    st.caption("Core drivers of currencies, explained simply with examples and tips.")
    with st.expander("Interest Rates & Central Banks", expanded=True):
        st.write("""
- Central banks adjust rates to control inflation and growth.
- Higher rates tend to attract capital → stronger currency.
- Example: FED rate hike often strengthens USD.
- Watch: FOMC (USD), ECB (EUR), BoE (GBP), BoJ (JPY), RBA (AUD), BoC (CAD), SNB (CHF), RBNZ (NZD).
- Tip: Track rate decision dates in the calendar.
        """)
    with st.expander("Inflation & Growth"):
        st.write("""
- Inflation (CPI/PPI) impacts real yields and policy expectations.
- Growth indicators (GDP, PMIs, employment) shift risk appetite and rate paths.
- Example: High CPI can lead to rate hikes, boosting currency.
- Tip: Compare actual vs. forecast for market reactions.
        """)
    with st.expander("Risk Sentiment & Commodities"):
        st.write("""
- Risk-on often lifts AUD/NZD; risk-off supports USD/JPY/CHF.
- Oil impacts CAD; gold sometimes correlates with AUD.
- Example: Geopolitical tensions strengthen safe-havens like JPY.
- Tip: Use news sentiment to gauge risk.
        """)
    with st.expander("Geopolitical Factors"):
        st.write("""
- Trade wars, elections, and conflicts can cause volatility.
- Example: US-China tensions weaken CNY/AUD.
- Tip: Monitor news for sudden shifts.
        """)
    with st.expander("How to Use the Economic Calendar"):
        st.write("""
1) Filter by the currency you trade.
2) Note forecast vs. actual – deviations cause moves.
3) Expect volatility around high-impact events; widen stops or reduce size.
4) Combine with technicals for better entries.
        """)
    # Add a simple visual: Sentiment distribution chart
    if not df_news.empty:
        sentiment_chart = alt.Chart(df_news).mark_bar().encode(
            x='Impact',
            y='count()',
            color='Impact'
        ).properties(title="News Sentiment Distribution")
        st.altair_chart(sentiment_chart, use_container_width=True)

# =========================================================
# TAB 3: TECHNICAL ANALYSIS (Enhanced with more pairs, studies, and sentiment chart)
# =========================================================
with selected_tab[2]:
    st.title("📊 Technical Analysis")
    st.caption("Live TradingView chart + curated news and sentiment for the selected pair.")
    # ---- Pair selector & symbol map (added more pairs) ----
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
    }
    pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
    # ---- TradingView Widget (enhanced with more studies and watchlist) ----
    watchlist = list(pairs_map.values())
    tv_symbol = pairs_map[pair]
    tv_html = f"""
    <div class="tradingview-widget-container" style="height:900px; width:100%">
      <div id="tradingview_chart" class="tradingview-widget-container__widget" style="height:900px; width:100%"></div>
      <div class="tradingview-widget-copyright" style="padding-top:6px">
        <a href="https://www.tradingview.com/symbols/{tv_symbol.replace(':','-')}/" rel="noopener" target="_blank">
          <span class="blue-text">{pair} chart by TradingView</span>
        </a>
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "hide_top_toolbar": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "save_image": true,
        "calendar": true,
        "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies", "BB@tv-basicstudies"],
        "watchlist": {json.dumps(watchlist)}
      }}
      </script>
    </div>
    """
    components.html(tv_html, height=950, scrolling=False)
    # -------- News selector (enhanced with search) --------
    st.markdown("### 📰 News & Sentiment for Selected Pair")
    if not df_news.empty:
        base, quote = pair.split("/")
        filtered_df = df_news[df_news["Currency"].isin([base, quote])].copy()
        search_term = st.text_input("Search headlines", key="news_search")
        if search_term:
            filtered_df = filtered_df[filtered_df["Headline"].str.contains(search_term, case=False)]
        if not filtered_df.empty:
            try:
                filtered_df["HighProb"] = filtered_df.apply(
                    lambda row: "🔥" if (row["Impact"] in ["Significantly Bullish", "Significantly Bearish"]) and
                                         (pd.to_datetime(row["Date"]) >= pd.Timestamp.utcnow() - pd.Timedelta(days=1))
                    else "", axis=1
                )
            except Exception:
                filtered_df["HighProb"] = ""
            filtered_df_display = filtered_df.copy()
            filtered_df_display["HeadlineDisplay"] = filtered_df["HighProb"] + " " + filtered_df["Headline"]
            selected_headline = st.selectbox(
                "Select a headline for details",
                filtered_df_display["HeadlineDisplay"].tolist(),
                key="ta_headline_select"
            )
            selected_row = filtered_df_display[filtered_df_display["HeadlineDisplay"] == selected_headline].iloc[0]
            st.markdown(f"**[{selected_row['Headline']}]({selected_row['Link']})**")
            st.write(f"**Published:** {selected_row['Date'].date() if isinstance(selected_row['Date'], pd.Timestamp) else selected_row['Date']}")
            st.write(f"**Detected currency:** {selected_row['Currency']} | **Impact:** {selected_row['Impact']}")
            with st.expander("Summary"):
                st.write(selected_row["Summary"])
            # Add sentiment bar chart for filtered news
            sentiment_chart = alt.Chart(filtered_df).mark_bar().encode(
                x='Impact',
                y='count()',
                color='Impact'
            ).properties(title="Sentiment Distribution for Pair")
            st.altair_chart(sentiment_chart, use_container_width=True)
        else:
            st.info("No pair-specific headlines found.")
    else:
        st.info("News feed unavailable right now.")

# =========================================================
# TAB 4: TOOLS (Enhanced with more subtabs, accurate calcs, position sizing)
# =========================================================
with selected_tab[3]:
    st.title("🛠️ Tools")
    tools_subtabs = st.tabs(["💰 Profit/Loss Calculator", "📏 Position Sizing", "📊 Backtesting"])
    # ---------------- Profit/Stop-loss Calculator (enhanced with live prices) ----------------
    with tools_subtabs[0]:
        st.header("💰 Profit / Stop-loss Calculator")
        st.markdown("Calculate potential profit/loss with live pip values.")
        currency_pair = st.selectbox(
            "Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"], key="pl_currency_pair"
        )
        account_currency = st.selectbox(
            "Account Currency", ["USD", "EUR", "GBP", "JPY"], key="pl_account_currency"
        )
        position_size = st.number_input(
            "Position Size (lots)", min_value=0.01, value=0.1, step=0.01, key="pl_position_size"
        )
        current_price = fetch_live_price(currency_pair)
        open_price = st.number_input("Open Price", value=current_price, step=0.0001, key="pl_open_price")
        close_price = st.number_input("Close Price", value=current_price + 0.0050, step=0.0001, key="pl_close_price")
        trade_direction = st.radio("Trade Direction", ["Long", "Short"], key="pl_trade_direction")
        pip_multiplier = 100 if "JPY" in currency_pair else 10000
        pip_movement = abs(close_price - open_price) * pip_multiplier
        if trade_direction == "Short":
            pip_movement = -pip_movement if close_price < open_price else pip_movement
        pip_value = get_pip_value(currency_pair, position_size, account_currency, current_price)
        profit_loss = pip_movement * pip_value
        st.write(f"**Current Price**: {current_price:.4f}")
        st.write(f"**Pip Movement**: {pip_movement:.2f} pips")
        st.write(f"**Pip Value**: {pip_value:.2f} {account_currency}")
        st.write(f"**Potential Profit/Loss**: {profit_loss:.2f} {account_currency}")
    # ---------------- Position Sizing (new tool) ----------------
    with tools_subtabs[1]:
        st.header("📏 Position Sizing Calculator")
        st.markdown("Determine optimal lot size based on risk.")
        account_balance = st.number_input("Account Balance", min_value=100.0, value=10000.0, step=100.0)
        risk_percent = st.number_input("Risk % per Trade", min_value=0.1, value=1.0, step=0.1)
        stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1, value=50, step=1)
        currency_pair_ps = st.selectbox(
            "Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"], key="ps_currency_pair"
        )
        account_currency_ps = st.selectbox(
            "Account Currency", ["USD", "EUR", "GBP", "JPY"], key="ps_account_currency"
        )
        current_price_ps = fetch_live_price(currency_pair_ps)
        pip_value_ps = get_pip_value(currency_pair_ps, 1.0, account_currency_ps, current_price_ps)  # For 1 lot
        risk_amount = account_balance * (risk_percent / 100)
        position_size_calc = risk_amount / (stop_loss_pips * (pip_value_ps / 100000))  # Adjust for micro lots
        st.write(f"**Risk Amount**: {risk_amount:.2f} {account_currency_ps}")
        st.write(f"**Recommended Position Size**: {position_size_calc:.2f} lots")
    # ---------------- Backtesting (enhanced with journal stats and historical chart) ----------------
    with tools_subtabs[2]:
        st.header("📊 Backtesting")
        st.markdown("Backtest strategies with advanced chart, journal, and performance stats.")
        # TradingView Advanced Chart
        tv_widget = """
        <div class="tradingview-widget-container">
            <div id="tradingview_advanced_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget({
                    "width": "100%",
                    "height": 650,
                    "symbol": "FX:EURUSD",
                    "interval": "D",
                    "timezone": "Etc/UTC",
                    "theme": "dark",
                    "style": "1",
                    "toolbar_bg": "#1f1f1f",
                    "withdateranges": true,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true,
                    "save_image": true,
                    "studies": ["RSI", "MACD"],
                    "container_id": "tradingview_advanced_chart"
                });
            </script>
        </div>
        """
        st.components.v1.html(tv_widget, height=670)
        # Backtesting Journal (enhanced with P/L column)
        journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "P/L", "Notes"]
        if "tools_trade_journal" not in st.session_state or st.session_state.tools_trade_journal.empty:
            st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols)
        # Editable journal
        updated_journal_tools = st.data_editor(
            data=st.session_state.tools_trade_journal.copy(),
            num_rows="dynamic",
            key="tools_backtesting_journal_unique",
            column_config={"P/L": st.column_config.NumberColumn("P/L (Calculated)", disabled=True)}
        )
        # Calculate P/L if possible
        for idx, row in updated_journal_tools.iterrows():
            if pd.notnull(row['Entry']) and pd.notnull(row['Exit']) and pd.notnull(row['Lots']) and pd.notnull(row['Symbol']):
                pip_mult = 100 if "JPY" in row['Symbol'] else 10000
                pips = (row['Exit'] - row['Entry']) * pip_mult if row['Direction'] == "Long" else (row['Entry'] - row['Exit']) * pip_mult
                pip_val = get_pip_value(row['Symbol'], row['Lots'], "USD", row['Entry'])
                updated_journal_tools.at[idx, 'P/L'] = pips * pip_val
        st.session_state.tools_trade_journal = updated_journal_tools
        # Journal Stats
        if not st.session_state.tools_trade_journal.empty:
            total_pl = st.session_state.tools_trade_journal['P/L'].sum()
            win_rate = (st.session_state.tools_trade_journal['P/L'] > 0).sum() / len(st.session_state.tools_trade_journal) * 100
            st.metric("Total P/L", f"{total_pl:.2f}")
            st.metric("Win Rate", f"{win_rate:.2f}%")
        # Save / Load
        if "logged_in_user" in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Journal"):
                    username = st.session_state.logged_in_user
                    accounts = json.load(open(ACCOUNTS_FILE, "r")) if os.path.exists(ACCOUNTS_FILE) else {}
                    accounts.setdefault(username, {})["tools_trade_journal"] = st.session_state.tools_trade_journal.to_dict(orient="records")
                    json.dump(accounts, open(ACCOUNTS_FILE, "w"), indent=4)
                    st.success("Journal saved!")
            with col2:
                if st.button("📂 Load Journal"):
                    username = st.session_state.logged_in_user
                    accounts = json.load(open(ACCOUNTS_FILE, "r")) if os.path.exists(ACCOUNTS_FILE) else {}
                    saved_journal = accounts.get(username, {}).get("tools_trade_journal", [])
                    if saved_journal:
                        st.session_state.tools_trade_journal = pd.DataFrame(saved_journal)
                        st.success("Journal loaded!")
                    else:
                        st.info("No saved journal.")
        else:
            st.info("Sign in to save/load journal.")
        # Historical Chart using yfinance
        hist_pair = st.selectbox("View Historical Data for Pair", list(pairs_map.keys()), key="hist_pair")
        data = yf.download(f"{hist_pair.replace('/', '')}=X", period="1y")
        if not data.empty:
            fig, ax = plt.subplots()
            ax.plot(data['Close'])
            ax.set_title(f"{hist_pair} Historical Prices")
            st.pyplot(fig)

            # =========================================================
# TAB 5: MY ACCOUNT (Enhanced with hashing, logout, more preferences)
# =========================================================
with selected_tab[4]:
    st.title("👤 My Account")
    if not os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "w") as f:
            json.dump({}, f)
    if "logged_in_user" in st.session_state:
        st.subheader(f"Welcome, {st.session_state.logged_in_user}")
        if st.button("Logout"):
            del st.session_state.logged_in_user
            if "tools_trade_journal" in st.session_state:
                del st.session_state.tools_trade_journal
            st.success("Logged out successfully.")
            st.experimental_rerun()
    else:
        # ---------------- LOGIN ----------------
        login_expander = st.expander("Login", expanded=True)
        with login_expander:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                with open(ACCOUNTS_FILE, "r") as f:
                    accounts = json.load(f)
                hashed_pw = hash_password(password)
                if username in accounts and accounts[username]["password"] == hashed_pw:
                    st.session_state.logged_in_user = username
                    st.success(f"Logged in as {username}")
                    # Load journal
                    saved_journal = accounts.get(username, {}).get("tools_trade_journal", [])
                    if saved_journal:
                        st.session_state.tools_trade_journal = pd.DataFrame(saved_journal)
                    else:
                        st.session_state.tools_trade_journal = pd.DataFrame(columns=["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "P/L", "Notes"])
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        # ---------------- SIGN UP ----------------
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
                    hashed_pw = hash_password(new_password)
                    accounts[new_username] = {"password": hashed_pw, "tools_trade_journal": []}
                    with open(ACCOUNTS_FILE, "w") as f:
                        json.dump(accounts, f, indent=4)
                    st.success(f"Account created for {new_username}")
    # ---------------- ACCOUNT SETTINGS (enhanced) ----------------
    if "logged_in_user" in st.session_state:
        st.subheader("Profile Settings")
        colA, colB = st.columns(2)
        with colA:
            name = st.text_input("Name", value=st.session_state.get("name",""), key="account_name")
            base_ccy = st.selectbox("Preferred Base Currency", ["USD","EUR","GBP","JPY","AUD","CAD","NZD","CHF"], index=0, key="account_base_ccy")
            theme = st.selectbox("App Theme", ["Dark Futuristic", "Light Modern"], key="account_theme")
        with colB:
            email = st.text_input("Email", value=st.session_state.get("email",""), key="account_email")
            alerts = st.checkbox("Email me before high-impact events", value=st.session_state.get("alerts", True), key="account_alerts")
            notifications = st.checkbox("Push notifications for news", value=st.session_state.get("notifications", False), key="account_notifications")
        if st.button("Save Preferences", key="account_save_prefs"):
            st.session_state.name = name
            st.session_state.email = email
            st.session_state.base_ccy = base_ccy
            st.session_state.alerts = alerts
            st.session_state.notifications = notifications
            # Save to accounts file (extend persistence)
            username = st.session_state.logged_in_user
            accounts = json.load(open(ACCOUNTS_FILE, "r"))
            accounts[username]["preferences"] = {
                "name": name,
                "email": email,
                "base_ccy": base_ccy,
                "alerts": alerts,
                "notifications": notifications
            }
            json.dump(accounts, open(ACCOUNTS_FILE, "w"), indent=4)
            st.success("Preferences saved!")
