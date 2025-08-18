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

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Path to SQLite DB
DB_FILE = "users.db"

# Connect to SQLite with error handling
try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    conn.commit()
    logging.info("SQLite database initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize SQLite database: {str(e)}")
    st.error(f"Database initialization failed: {str(e)}")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide")

# ----------------- CUSTOM CSS -----------------
bg_opacity = 0.5
st.markdown(
    f"""
<style>
/* Futuristic dark background with animated grid */
.stApp {{
    background:
        radial-gradient(circle at 15% 20%, rgba(255,215,0,{bg_opacity*0.18}) 0%, transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(0,170,255,{bg_opacity*0.12}) 0%, transparent 25%),
        linear-gradient(135deg, #0b0b0b 0%, #0a0a0a 100%);
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(90deg, rgba(255,255,255,{bg_opacity*0.05}) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,{bg_opacity*0.05}) 1px, transparent 1px);
    background-size: 42px 42px, 42px 42px;
    animation: moveGrid 38s linear infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 1;
}}
@keyframes moveGrid {{
    0% {{ transform: translateY(0px); }}
    100% {{ transform: translateY(42px); }}
}}
/* Content above bg */
.main, .block-container, .stTabs, .stMarkdown, .css-ffhzg2, .css-1d391kg {{ position: relative; z-index: 1; }}
/* Enhanced tab styling without black background */
div[data-baseweb="tab-list"] {{
    gap: 12px;
    padding-bottom: 8px;
    background: transparent;
    border-radius: 12px;
    padding: 8px;
}}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background: linear-gradient(45deg, #FFD700, #FFA500) !important;
    color: #000 !important;
    font-weight: 600;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
    background: #2a2a2a !important;
    color: #ccc !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: 1px solid #3a3a3a !important;
}}
div[data-baseweb="tab-list"] button:hover {{
    background: #3a3a3a !important;
    color: #fff !important;
    transform: translateY(-2px);
}}
/* Original Card look */
.card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    transition: transform 0.3s ease;
}}
.card:hover {{
    transform: translateY(-4px);
}}
/* Original dataframe styling */
.dataframe th {{
    background-color: #1f1f1f;
    color: #FFD700;
}}
.dataframe td {{
    background-color: #121212;
    color: white;
}}
/* Original Selectbox and input styling */
.stSelectbox, .stNumberInput, .stTextInput, .stRadio {{
    background-color: #1b1b1b;
    border-radius: 8px;
    padding: 8px;
}}
/* Original Button styling */
.stButton button {{
    background-color: #FFD700;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}}
.stButton button:hover {{
    background-color: #E6C200;
}}
/* Original Expander styling */
.stExpander {{
    border: 1px solid #242424;
    border-radius: 8px;
    background-color: #1b1b1b;
}}
/* small utility */
.small-muted {{ color:#9e9e9e; font-size:0.9rem; }}
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

# =========================================================
# NAVIGATION
# =========================================================
tabs = ["Forex Fundamentals", "Backtesting", "Tools", "My Account", "MT5 Stats Dashboard"]
selected_tab = st.tabs(tabs)

# =========================================================
# TAB 1: FOREX FUNDAMENTALS
# =========================================================
with selected_tab[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📅 Forex Fundamentals")
        st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
    with col2:
        st.info("See the **Backtesting** tab for live charts + detailed news.")
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
# TAB 2: BACKTESTING
# =========================================================
with selected_tab[1]:
    st.title("📊 Backtesting")
    st.caption("Live TradingView chart for backtesting, curated news, and trading journal for the selected pair.")
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
    # TradingView widget with debug
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
    # Backtesting Journal
    st.markdown("### 📝 Trading Journal")
    journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "Notes"]
    if "tools_trade_journal" not in st.session_state or st.session_state.tools_trade_journal.empty:
        st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols)
    updated_journal_tools = st.data_editor(
        data=st.session_state.tools_trade_journal.copy(),
        num_rows="dynamic",
        key="bt_backtesting_journal_unique"
    )
    st.session_state.tools_trade_journal = updated_journal_tools
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
                            st.session_state.tools_trade_journal = pd.DataFrame(saved_journal)
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
    # News & Sentiment
    st.markdown("### 📰 News & Sentiment for Selected Pair")
    if not df_news.empty:
        base, quote = pair.split("/")
        filtered_df = df_news[df_news["Currency"].isin([base, quote])].copy()
        try:
            filtered_df["HighProb"] = filtered_df.apply(
                lambda row: "🔥" if (row["Impact"] in ["Significantly Bullish", "Significantly Bearish"]) and
                                    (pd.to_datetime(row["Date"]) >= pd.Timestamp.utcnow() - pd.Timedelta(days=1))
                else "", axis=1
            )
        except Exception as e:
            filtered_df["HighProb"] = ""
            logging.error(f"Error processing news high probability: {str(e)}")
        filtered_df_display = filtered_df.copy()
        filtered_df_display["HeadlineDisplay"] = filtered_df["HighProb"] + " " + filtered_df["Headline"]
        if not filtered_df_display.empty:
            selected_headline = st.selectbox(
                "Select a headline for details",
                filtered_df_display["HeadlineDisplay"].tolist(),
                key="bt_headline_select"
            )
            selected_row = filtered_df_display[filtered_df_display["HeadlineDisplay"] == selected_headline].iloc[0]
            st.markdown(f"**[{selected_row['Headline']}]({selected_row['Link']})**")
            st.write(f"**Published:** {selected_row['Date'].date() if isinstance(selected_row['Date'], pd.Timestamp) else selected_row['Date']}")
            st.write(f"**Detected currency:** {selected_row['Currency']} | **Impact:** {selected_row['Impact']}")
            with st.expander("Summary"):
                st.write(selected_row["Summary"])
        else:
            st.info("No pair-specific headlines found in the recent feed.")
    else:
        st.info("News feed unavailable right now.")
# =========================================================
# TAB 3: TOOLS
# =========================================================
with selected_tab[2]:
    st.title("🛠 Tools")
    tools_subtabs = st.tabs(["Profit/Loss Calculator", "Backtesting", "Price alerts", "Currency Correlation Heatmap", "Risk Management Calculator", "Trading Session Tracker"])
    with tools_subtabs[0]:
        st.header("💰 Profit / Loss Calculator")
        st.markdown("Calculate your potential profit or loss for a trade.")
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
    with tools_subtabs[1]:
        st.header("📊 Backtesting")
        st.markdown("Backtest your trading strategies here.")
        pair = "EURUSD"
        tv_symbol = "FX:EURUSD"
        if "logged_in_user" in st.session_state and pair not in st.session_state.drawings:
            username = st.session_state.logged_in_user
            logging.info(f"Loading drawings for user {username}, pair {pair} (Backtesting)")
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
        tv_widget = f"""
        <div class="tradingview-widget-container" style="height:780px; width:100%">
            <div id="tradingview_advanced_chart" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            console.log("Initializing TradingView widget for Backtesting");
            try {{
              const widget = new TradingView.widget({{
                "autosize": true,
                "symbol": "{tv_symbol}",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "toolbar_bg": "#f1f3f6",
                "withdateranges": true,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "save_image": false,
                "studies": [],
                "container_id": "tradingview_advanced_chart"
              }});
              widget.onChartReady(() => {{
                console.log("Chart ready for Backtesting");
                const chart = widget.activeChart();
                window.chart = chart;
                const initialContent = {initial_content};
                if (Object.keys(initialContent).length > 0) {{
                  console.log("Loading initial content:", initialContent);
                  chart.setContent(initialContent);
                }} else {{
                  console.log("No initial content to load for Backtesting");
                }}
              }});
            }} catch (error) {{
              console.error("Error initializing TradingView widget:", error);
            }}
            </script>
        </div>
        """
        components.html(tv_widget, height=820, scrolling=False)
        if "logged_in_user" in st.session_state:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Save Drawings", key="back_save_drawings"):
                    logging.info("Save Drawings button clicked for Backtesting")
                    save_script = """
                    <script>
                    try {
                      console.log("Attempting to save drawings for Backtesting");
                      window.parent.chart.getContent((content) => {
                        console.log("Drawing content received:", content);
                        window.parent.postMessage({
                          type: 'streamlit:setComponentValue',
                          value: content,
                          dataType: 'json',
                          key: 'back_drawings_key'
                        }, '*');
                      });
                    } catch (error) {
                      console.error("Error saving drawings:", error);
                    }
                    </script>
                    """
                    components.html(save_script, height=0)
                    logging.info("Triggered save script for Backtesting")
                    st.session_state["back_save_trigger"] = True
            with col2:
                if st.button("Load Drawings", key="back_load_drawings"):
                    username = st.session_state.logged_in_user
                    logging.info(f"Load Drawings button clicked for user {username}, Backtesting")
                    try:
                        c.execute("SELECT data FROM users WHERE username = ?", (username,))
                        result = c.fetchone()
                        if result:
                            user_data = json.loads(result[0])
                            content = user_data.get("drawings", {}).get(pair, {})
                            if content:
                                load_script = f"""
                                <script>
                                console.log("Loading drawings for Backtesting:", {json.dumps(content)});
                                window.parent.chart.setContent({json.dumps(content)});
                                </script>
                                """
                                components.html(load_script, height=0)
                                st.success("Drawings loaded successfully!")
                                logging.info(f"Successfully loaded drawings for Backtesting")
                            else:
                                st.info("No saved drawings for this pair.")
                                logging.info(f"No saved drawings found for Backtesting")
                        else:
                            st.error("Failed to load user data.")
                            logging.error(f"No user data found for {username}")
                    except Exception as e:
                        st.error(f"Failed to load drawings: {str(e)}")
                        logging.error(f"Error loading drawings for {username}: {str(e)}")
            with col3:
                if st.button("Refresh Account", key="back_refresh_account"):
                    username = st.session_state.logged_in_user
                    logging.info(f"Refresh Account button clicked for user {username} (Backtesting)")
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
            if 'back_drawings_key' in st.session_state and st.session_state.get("back_save_trigger", False):
                content = st.session_state['back_drawings_key']
                logging.info(f"Received drawing content for Backtesting: {content}")
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
                        logging.info(f"Drawings saved to database for Backtesting: {content}")
                    except Exception as e:
                        st.error(f"Failed to save drawings: {str(e)}")
                        logging.error(f"Database error saving drawings for Backtesting: {str(e)}")
                    finally:
                        del st.session_state['back_drawings_key']
                        del st.session_state["back_save_trigger"]
                else:
                    st.warning("No valid drawing content received. Ensure you have drawn on the chart.")
                    logging.warning(f"No valid drawing content received for Backtesting: {content}")
        else:
            st.info("Sign in via the My Account tab to save/load drawings.")
            logging.info("User not logged in, save/load drawings disabled (Backtesting)")
        # Backtesting Journal
        journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "Notes"]
        if "tools_trade_journal" not in st.session_state or st.session_state.tools_trade_journal.empty:
            st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols)
        updated_journal_tools = st.data_editor(
            data=st.session_state.tools_trade_journal.copy(),
            num_rows="dynamic",
            key="tools_backtesting_journal_unique"
        )
        st.session_state.tools_trade_journal = updated_journal_tools
        if "logged_in_user" in st.session_state:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save to My Account", key="save_journal_button"):
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
                if st.button("📂 Load Journal", key="load_journal_button"):
                    username = st.session_state.logged_in_user
                    logging.info(f"Loading journal for user {username}")
                    try:
                        c.execute("SELECT data FROM users WHERE username = ?", (username,))
                        result = c.fetchone()
                        if result:
                            user_data = json.loads(result[0])
                            saved_journal = user_data.get("tools_trade_journal", [])
                            if saved_journal:
                                st.session_state.tools_trade_journal = pd.DataFrame(saved_journal)
                                st.success("Trading journal loaded from your account!")
                                logging.info(f"Journal loaded for {username}")
                            else:
                                st.info("No saved journal found in your account.")
                                logging.info(f"No journal found for {username}")
                    except Exception as e:
                        st.error(f"Failed to load journal: {str(e)}")
                        logging.error(f"Error loading journal for {username}: {str(e)}")
        else:
            st.info("Sign in to save your trading journal to your account.")
    with tools_subtabs[2]:
        st.header("⏰ Price Alerts")
        st.markdown("Set price alerts for your favourite forex pairs and get notified when the price hits your target.")
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
    with tools_subtabs[3]:
        st.header("📊 Currency Correlation Heatmap")
        st.markdown("Understand how forex pairs move relative to each other. ")
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
    with tools_subtabs[4]:
        st.header("🛡️ Risk Management Calculator")
        st.markdown("Proper position sizing keeps your account safe. ")
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
    with tools_subtabs[5]:
        st.header("🕒 Forex Market Sessions")
        st.markdown("Stay aware of active trading sessions to trade when volatility is highest.")
        sessions = {
            "Sydney": (22, 7),
            "Tokyo": (0, 9),
            "London": (8, 17),
            "New York": (13, 22),
        }
        now_utc = dt.datetime.now(pytz.UTC).hour
        cols = st.columns(len(sessions))
        for i, (session, (start, end)) in enumerate(sessions.items()):
            active = start <= now_utc < end if start < end else (now_utc >= start or now_utc < end)
            color = "#144714" if active else "#171447"
            with cols[i]:
                st.markdown(f"""
                    <div style="
                        background-color:{color};
                        border-radius:10px;
                        padding:15px;
                        text-align:center;
                        color:white;
                        box-shadow:2px 2px 8px rgba(0,0,0,0.5);
                    ">
                        <h3 style="margin:0;">{session}</h3>
                        <p style="margin:0;">{start}:00 - {end}:00 UTC</p>
                        <b>{'ACTIVE' if active else 'Closed'}</b>
                    </div>
                """, unsafe_allow_html=True)

# =========================================================
# TAB 4: MY ACCOUNT
# =========================================================
with selected_tab[3]:
    st.title("👤 My Account")
    st.markdown("""
    Welcome to your personal account dashboard. Manage your login credentials, customize your preferences, and access your saved trading data. Sign in to save your chart drawings and trading journal, or sign up to create a new account.
    """)
    # LOGIN
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        logging.info(f"Login attempt for user {username}")
        try:
            c.execute("SELECT password, data FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result and result[0] == password:
                st.session_state.logged_in_user = username
                user_data = json.loads(result[1]) if result[1] else {}
                st.session_state.drawings = user_data.get("drawings", {})
                journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "Notes"]
                saved_journal = user_data.get("tools_trade_journal", [])
                st.session_state.tools_trade_journal = pd.DataFrame(saved_journal, columns=journal_cols) if saved_journal else pd.DataFrame(columns=journal_cols)
                st.success(f"Logged in as {username}")
                logging.info(f"Login successful for {username}")
            else:
                st.error("Invalid username or password")
                logging.warning(f"Login failed for {username}: Invalid credentials")
        except Exception as e:
            st.error(f"Login error: {str(e)}")
            logging.error(f"Login error for {username}: {str(e)}")
    # SIGN UP
    st.subheader("Sign Up")
    new_username = st.text_input("New Username", key="signup_username")
    new_password = st.text_input("New Password", type="password", key="signup_password")
    if st.button("Sign Up"):
        logging.info(f"Sign up attempt for user {new_username}")
        try:
            c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", (new_username, new_password, json.dumps({})))
            conn.commit()
            st.success(f"Account created for {new_username}")
            logging.info(f"Account created for {new_username}")
        except sqlite3.IntegrityError:
            st.error("Username already exists")
            logging.warning(f"Sign up failed for {new_username}: Username exists")
        except Exception as e:
            st.error(f"Sign up error: {str(e)}")
            logging.error(f"Sign up error for {new_username}: {str(e)}")
    # ACCOUNT SETTINGS
    if "logged_in_user" in st.session_state:
        st.subheader("Profile Settings")
        colA, colB = st.columns(2)
        with colA:
            name = st.text_input("Name", value=st.session_state.get("name", ""), key="account_name")
            base_ccy = st.selectbox("Preferred Base Currency", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"], index=0, key="account_base_ccy")
        with colB:
            email = st.text_input("Email", value=st.session_state.get("email", ""), key="account_email")
            alerts = st.checkbox("Email me before high-impact events", value=st.session_state.get("alerts", True), key="account_alerts")
        if st.button("Save Preferences", key="account_save_prefs"):
            username = st.session_state.logged_in_user
            logging.info(f"Saving preferences for user {username}")
            try:
                prefs = {
                    "name": name,
                    "email": email,
                    "base_ccy": base_ccy,
                    "alerts": alerts
                }
                c.execute("SELECT data FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                user_data = json.loads(result[0]) if result else {}
                user_data["preferences"] = prefs
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
                conn.commit()
                st.session_state.name = name
                st.session_state.email = email
                st.session_state.base_ccy = base_ccy
                st.session_state.alerts = alerts
                st.success("Preferences saved.")
                logging.info(f"Preferences saved for {username}")
            except Exception as e:
                st.error(f"Failed to save preferences: {str(e)}")
                logging.error(f"Error saving preferences for {username}: {str(e)}")

# =========================================================
# TAB 5: MT5 STATS DASHBOARD
# =========================================================
with selected_tab[4]:
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
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            padding: 20px 0;
        }
        .metric-card {
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            color: #333333;
            background: transparent;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .metric-title {
            font-size: 14px;
            font-weight: 500;
            color: #666666;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #333333;
        }
        .positive .metric-value {
            color: #2e7d32;
        }
        .negative .metric-value {
            color: #d32f2f;
        }
        .neutral .metric-value {
            color: #333333;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            color: #333333;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .upload-container {
            background: transparent !important;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .stFileUploader > div > div > div {
            background-color: transparent !important;
            border-radius: 8px;
            color: #333333 !important;
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
                    # Display metrics
                    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
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
                    for title, value, style in metrics:
                        st.markdown(f"""
                            <div class="metric-card {style}">
                                <div class="metric-title">{title}</div>
                                <div class="metric-value">{value}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
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
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}.")
    else:
        st.info("👆 Upload your MT5 trading history CSV to explore your performance metrics.")
