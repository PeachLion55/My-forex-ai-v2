import streamlit as st
import pandas as pd
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
from datetime import datetime, timedelta

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide")

# ----------------- SIDEBAR CONTROLS -----------------
# Fixed settings (no sidebar controls)
bg_opacity = 0.5     # Background FX opacity
tv_height = 950      # TradingView chart height in px

# ----------------- CUSTOM CSS (Dark Futuristic BG + Tabs) -----------------
st.markdown(
    f"""
<style>
/* Futuristic dark background with animated grid */
.stApp {{
    background:
        radial-gradient(circle at 15%% 20%%, rgba(255,215,0,{bg_opacity*0.18}) 0%%, transparent 25%%),
        radial-gradient(circle at 85%% 30%%, rgba(0,170,255,{bg_opacity*0.12}) 0%%, transparent 25%%),
        linear-gradient(135deg, #0b0b0b 0%%, #0a0a0a 100%%);
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(90deg, rgba(255,255,255,{bg_opacity*0.05}) 1px, transparent 1px),
        linear-gradient(0deg,  rgba(255,255,255,{bg_opacity*0.05}) 1px, transparent 1px);
    background-size: 42px 42px, 42px 42px;
    animation: moveGrid 38s linear infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 1;
}}
@keyframes moveGrid {{
    0%   {{ transform: translateY(0px); }}
    100% {{ transform: translateY(42px); }}
}}
/* lift the app content above bg layer */
.main, .block-container, .stTabs, .stMarkdown, .css-ffhzg2, .css-1d391kg {{ position: relative; z-index: 1; }}

/* Tab styling */
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background-color: #FFD700 !important;
    color: black !important;
    font-weight: 700;
    padding: 14px 26px !important;
    border-radius: 10px;
    margin-right: 10px !important;
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
    background-color: #1b1b1b !important;
    color: #bbb !important;
    padding: 14px 26px !important;
    border-radius: 10px;
    margin-right: 10px !important;
    border: 1px solid #242424 !important;
}}
/* Card look for info boxes */
.card {{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
}}
</style>
""",
    unsafe_allow_html=True,
)

import streamlit as st

# =========================================================
# NAVIGATION
# =========================================================
tabs = ["Forex Fundamentals", "Understanding Forex Fundamentals", "Technical Analysis", "Tools", "My Account"]
selected_tab = st.tabs(tabs)

# ---------- Forex Fundamentals ----------


    # Your existing code here

# ---------- Understanding Forex Fundamentals ----------

    # Your existing code here

# ---------- Technical Analysis ----------

    # Your existing code here

selected_tab[3]:
    st.title("🛠 Tools")
    tools_subtabs = st.tabs(["Profit/Stop-loss Calculator", "Backtesting"])

    # ---------------- Profit/Stop-loss Calculator ----------------
    with tools_subtabs[0]:
        st.header("💰 Profit / Stop-loss Calculator")
        st.markdown("Calculate your potential profit or loss for a trade.")

        currency_pair = st.selectbox(
            "Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY"], key="tools_currency_pair"
        )
        account_currency = st.selectbox(
            "Account Currency", ["USD", "EUR", "GBP", "JPY"], key="tools_account_currency"
        )
        position_size = st.number_input(
            "Position Size (lots)", min_value=0.01, value=0.1, step=0.01, key="tools_position_size"
        )
        open_price = st.number_input(
            "Open Price", value=1.1000, step=0.0001, key="tools_open_price"
        )
        close_price = st.number_input(
            "Close Price", value=1.1050, step=0.0001, key="tools_close_price"
        )
        trade_direction = st.radio(
            "Trade Direction", ["Long", "Short"], key="tools_trade_direction"
        )

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

    # ---------------- Backtesting ----------------
    with tools_subtabs[1]:
        st.header("📈 Backtesting")
        st.markdown(
            "Analyze historical candlestick charts manually and log trades below."
        )

        # TradingView widget
        st.components.v1.html(
            """
            <div class="tradingview-widget-container">
              <div id="tradingview_advanced"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({
                "width": "1000",
                "height": "600",
                "symbol": "EURUSD",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "save_image": false,
                "studies": [],
                "hideideas": true,
                "withdateranges": true,
                "details": true,
                "hotlist": true,
                "calendar": true,
                "news": ["headlines"]
              });
              </script>
            </div>
            """,
            height=650,
        )

        # ---------------- Trading Journal ----------------
        st.subheader("📝 Trading Journal")
        journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "Notes"]

        # Initialize journal in session_state if missing
        if "trade_journal" not in st.session_state:
            st.session_state.trade_journal = pd.DataFrame(columns=journal_cols)

        # Load journal from account if logged in and first load
        if "logged_in_user" in st.session_state and st.session_state.trade_journal.empty:
            import json, os
            ACCOUNTS_FILE = "user_accounts.json"
            if os.path.exists(ACCOUNTS_FILE):
                with open(ACCOUNTS_FILE, "r") as f:
                    accounts = json.load(f)
                username = st.session_state.logged_in_user
                saved_journal = accounts.get(username, {}).get("trade_journal", [])
                # Convert to DataFrame to keep it editable
                st.session_state.trade_journal = pd.DataFrame(saved_journal, columns=journal_cols)

        # Ensure journal is always a DataFrame
        if not isinstance(st.session_state.trade_journal, pd.DataFrame):
            st.session_state.trade_journal = pd.DataFrame(st.session_state.trade_journal, columns=journal_cols)

        # Display editable journal
        journal_editor = st.data_editor(
            st.session_state.trade_journal,
            num_rows="dynamic",
            key="tools_backtesting_journal"
        )

        # Update session state with edits
        st.session_state.trade_journal = journal_editor

        # Save to account button
        if "logged_in_user" in st.session_state:
            if st.button("💾 Save to My Account"):
                import json, os
                ACCOUNTS_FILE = "user_accounts.json"
                if os.path.exists(ACCOUNTS_FILE):
                    with open(ACCOUNTS_FILE, "r") as f:
                        accounts = json.load(f)
                    username = st.session_state.logged_in_user
                    accounts.setdefault(username, {})["trade_journal"] = st.session_state.trade_journal.to_dict(orient="records")
                    with open(ACCOUNTS_FILE, "w") as f:
                        json.dump(accounts, f, indent=4)
                    st.success("Trading journal saved to your account!")
                else:
                    st.error("Accounts file not found.")
        else:
            st.info("Sign in to save your trading journal to your account.")
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
        # Keep only last 3 days to stay relevant
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=3)
            df = df[df["Date"] >= cutoff]
        except Exception:
            pass
        return df.reset_index(drop=True)
    return pd.DataFrame(columns=["Date","Currency","Headline","Polarity","Impact","Summary","Link"])

# Static calendar (your provided data)
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

# Load news once for all tabs
df_news = get_fxstreet_forex_news()

# =========================================================
# TAB 1: FOREX FUNDAMENTALS
# =========================================================
with selected_tab[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📅 Forex Fundamentals")
        st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
    with col2:
        st.info("See the **Technical Analysis** tab for live charts + detailed news.")

    # -------- Economic Calendar (with currency highlight filters) --------
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
        if st.session_state.selected_currency_1 and row['Currency'] == st.session_state.selected_currency_1:
            styles = ['background-color: #171447; color: white' if col == 'Currency' else 'background-color: #171447' for col in row.index]
        if st.session_state.selected_currency_2 and row['Currency'] == st.session_state.selected_currency_2:
            styles = ['background-color: #471414; color: white' if col == 'Currency' else 'background-color: #471414' for col in row.index]
        return styles

    st.dataframe(econ_df.style.apply(highlight_currency, axis=1), use_container_width=True)

    # -------- Interest rate tiles --------
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
    colors = ["#171447", "#471414"]
    for i in range(0, len(interest_rates), boxes_per_row):
        cols = st.columns(boxes_per_row)
        for j, rate in enumerate(interest_rates[i:i+boxes_per_row]):
            color = colors[j % 2]
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

# =========================================================
# TAB 2: UNDERSTANDING FOREX FUNDAMENTALS
# =========================================================
with selected_tab[1]:
    st.title("📖 Understanding Forex Fundamentals")
    st.caption("Core drivers of currencies, explained simply.")

    with st.expander("Interest Rates & Central Banks"):
        st.write("""
- Central banks adjust rates to control inflation and growth.
- Higher rates tend to attract capital → stronger currency.
- Watch: FOMC (USD), ECB (EUR), BoE (GBP), BoJ (JPY), RBA (AUD), BoC (CAD), SNB (CHF), RBNZ (NZD).
        """)

    with st.expander("Inflation & Growth"):
        st.write("""
- Inflation (CPI/PPI) impacts real yields and policy expectations.
- Growth indicators (GDP, PMIs, employment) shift risk appetite and rate paths.
        """)

    with st.expander("Risk Sentiment & Commodities"):
        st.write("""
- Risk-on often lifts AUD/NZD; risk-off supports USD/JPY/CHF.
- Oil impacts CAD; gold sometimes correlates with AUD.
        """)

    with st.expander("How to Use the Economic Calendar"):
        st.write("""
1) Filter by the currency you trade.
2) Note forecast vs. actual.
3) Expect volatility around high-impact events; widen stops or reduce size.
        """)

# =========================================================
# TAB 3: TECHNICAL ANALYSIS
# =========================================================
with selected_tab[2]:
    st.title("📊 Technical Analysis")
    st.caption("Live TradingView chart + curated news for the selected pair.")

    # ---- Pair selector & symbol map ----
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

    # ---- TradingView Widget (only in Tab 3) ----
    watchlist = list(pairs_map.values())
    tv_symbol = pairs_map[pair]

    tv_html = f"""
    <div class="tradingview-widget-container" style="height:800px; width:100%">
      <div id="tradingview_chart" class="tradingview-widget-container__widget" style="height:800px; width:100%"></div>
      <div class="tradingview-widget-copyright" style="padding-top:6px">
        <a href="https://www.tradingview.com/symbols/{tv_symbol.replace(':','-')}/" rel="noopener" target="_blank">
          <span class="blue-text">{pair} chart by TradingView</span>
        </a>
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "hide_top_toolbar": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "save_image": true,
        "calendar": false,
        "studies": [],
        "watchlist": {watchlist}
      }}
      </script>
    </div>
    """

    components.html(tv_html, height=850, scrolling=False)

    # -------- News selector --------
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
        except Exception:
            filtered_df["HighProb"] = ""

        filtered_df_display = filtered_df.copy()
        filtered_df_display["HeadlineDisplay"] = filtered_df["HighProb"] + " " + filtered_df["Headline"]

        if not filtered_df_display.empty:
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
        else:
            st.info("No pair-specific headlines found in the recent feed.")
    else:
        st.info("News feed unavailable right now.")

# =========================================================
# TAB 4: MY ACCOUNT
# =========================================================
with selected_tab[4]:
    st.title("👤 My Account")
    st.caption("Create an account or sign in to save your preferences and trading journal.")

    # File to store user accounts (JSON)
    import os, json

    ACCOUNTS_FILE = "user_accounts.json"
    if not os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "w") as f:
            json.dump({}, f)

    # Load existing accounts
    with open(ACCOUNTS_FILE, "r") as f:
        accounts = json.load(f)

    # --- Account Actions: Sign In / Create ---
    account_action = st.radio("Select Action", ["Sign In", "Create Account"], horizontal=True)

    if account_action == "Create Account":
        st.subheader("Create New Account")
        new_username = st.text_input("Username", key="create_username")
        new_password = st.text_input("Password", type="password", key="create_password")
        new_name = st.text_input("Full Name", key="create_name")
        new_base_ccy = st.selectbox("Preferred Base Currency", ["USD","EUR","GBP","JPY","AUD","CAD","NZD","CHF"], index=0)
        save_alerts = st.checkbox("Email me before high-impact events", value=True)

        if st.button("Create Account"):
            if new_username in accounts:
                st.error("Username already exists. Please choose another.")
            elif not new_username or not new_password:
                st.error("Username and password cannot be empty.")
            else:
                # Save new account
                accounts[new_username] = {
                    "password": new_password,
                    "name": new_name,
                    "base_ccy": new_base_ccy,
                    "alerts": save_alerts,
                    "trade_journal": []
                }
                with open(ACCOUNTS_FILE, "w") as f:
                    json.dump(accounts, f)
                st.success("Account created successfully! You can now sign in.")

    elif account_action == "Sign In":
        st.subheader("Sign In")
        username = st.text_input("Username", key="signin_username")
        password = st.text_input("Password", type="password", key="signin_password")

        if st.button("Sign In"):
            if username not in accounts:
                st.error("Username does not exist.")
            elif accounts[username]["password"] != password:
                st.error("Incorrect password.")
            else:
                st.session_state.logged_in_user = username
                st.session_state.name = accounts[username].get("name", "")
                st.session_state.base_ccy = accounts[username].get("base_ccy", "USD")
                st.session_state.alerts = accounts[username].get("alerts", True)
                st.session_state.trade_journal = pd.DataFrame(accounts[username].get("trade_journal", []))
                st.success(f"Signed in as {username}!")

    # --- If logged in, show profile and option to save journal ---
    if "logged_in_user" in st.session_state:
        st.markdown(f"**Current Profile:** {st.session_state.name} | {st.session_state.base_ccy} | Alerts: {'On' if st.session_state.alerts else 'Off'}")

        colA, colB = st.columns(2)
        with colA:
            st.session_state.name = st.text_input("Name", value=st.session_state.get("name", ""))
            st.session_state.base_ccy = st.selectbox(
                "Preferred Base Currency",
                ["USD","EUR","GBP","JPY","AUD","CAD","NZD","CHF"],
                index=["USD","EUR","GBP","JPY","AUD","CAD","NZD","CHF"].index(st.session_state.base_ccy)
            )
        with colB:
            st.session_state.alerts = st.checkbox(
                "Email me before high-impact events", value=st.session_state.get("alerts", True)
            )

        if st.button("Save Preferences"):
            username = st.session_state.logged_in_user
            accounts[username]["name"] = st.session_state.name
            accounts[username]["base_ccy"] = st.session_state.base_ccy
            accounts[username]["alerts"] = st.session_state.alerts
            with open(ACCOUNTS_FILE, "w") as f:
                json.dump(accounts, f)
            st.success("Preferences saved!")

        # Button to save the backtesting journal
        if st.button("💾 Save Trading Journal"):
            username = st.session_state.logged_in_user
            if "trade_journal" in st.session_state:
                # Convert DataFrame to list of dicts
                accounts[username]["trade_journal"] = st.session_state.trade_journal.to_dict(orient="records")
                with open(ACCOUNTS_FILE, "w") as f:
                    json.dump(accounts, f)
                st.success("Trading journal saved to your account!")
            else:
                st.warning("No journal data to save.")
