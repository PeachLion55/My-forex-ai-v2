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
import requests
from streamlit_autorefresh import st_autorefresh

# Path to your accounts JSON file
ACCOUNTS_FILE = "accounts.json"  # or a full path if needed

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide")

# ----------------- CUSTOM CSS (Dark Futuristic BG + Tabs) -----------------
bg_opacity = 0.5  # Background FX opacity
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
/* lift the app content above bg layer */
.main, .block-container, .stTabs, .stMarkdown, .css-ffhzg2, .css-1d391kg {{ position: relative; z-index: 1; }}
/* Tab styling */
div[data-baseweb="tab-list"] {{
    gap: 8px;
    padding-bottom: 4px;
}}
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background-color: #FFD700 !important;
    color: black !important;
    font-weight: 700;
    padding: 14px 26px !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: none !important;
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
    background-color: #1b1b1b !important;
    color: #bbb !important;
    padding: 14px 26px !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #242424 !important;
    border-bottom: none !important;
}}
div[data-baseweb="tab-list"] button:hover {{
    background-color: #2a2a2a !important;
    color: white !important;
}}
/* Card look for info boxes */
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
/* Improve dataframe styling */
.dataframe th {{
    background-color: #1f1f1f;
    color: #FFD700;
}}
.dataframe td {{
    background-color: #121212;
    color: white;
}}
/* Selectbox and input styling */
.stSelectbox, .stNumberInput, .stTextInput, .stRadio {{
    background-color: #1b1b1b;
    border-radius: 8px;
    padding: 8px;
}}
/* Button styling */
.stButton button {{
    background-color: #FFD700;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}}
.stButton button:hover {{
    background-color: #E6C200;
}}
/* Expander styling */
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
# NAVIGATION
# =========================================================
tabs = ["Forex Fundamentals", "Technical Analysis", "Tools", "My Account", "MT5 Stats Dashboard"]
selected_tab = st.tabs(tabs)

# =========================================================
# HELPERS / DATA (existing functions preserved)
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
    except Exception:
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
        except Exception:
            pass
        return df.reset_index(drop=True)
    return pd.DataFrame(columns=["Date","Currency","Headline","Polarity","Impact","Summary","Link"])

# Static calendar (your provided data)
econ_calendar_data = [
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP Price Index y/y", "Actual": "3.0%", "Forecast": "3.1%", "Previous": "3.3%", "Impact": ""},
    # ... (keep your full list as in the original) ...
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

    # Economic Calendar (with currency highlight filters)
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

    # Add spacing
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Major High-Impact Events (clean, visually distinct)
    st.markdown("### 📊 Major High-Impact Forex Events")
    forex_high_impact_events = [
        {
            "event": "Non-Farm Payrolls (NFP)",
            "description": "Monthly change in U.S. non-farm payroll employment. Extremely influential for USD.",
            "impact_positive": {"USD": "↑", "EUR/USD": "↓", "GBP/USD": "↓", "USD/JPY": "↑"},
            "impact_negative": {"USD": "↓", "EUR/USD": "↑", "GBP/USD": "↑", "USD/JPY": "↓"},
        },
        {
            "event": "Consumer Price Index (CPI)",
            "description": "Measures consumer inflation. Central banks use it to gauge inflationary pressures.",
            "impact_positive": {"Currency": "↑ (higher rates likely)", "MAJOR PAIRS": "Currency strengthens vs majors"},
            "impact_negative": {"Currency": "↓ (dovish expectations)", "MAJOR PAIRS": "Currency weakens vs majors"},
        },
        {
            "event": "Interest Rate Decision",
            "description": "Official policy interest rate — immediate and direct effect on the currency.",
            "impact_positive": {"Currency": "↑ if hike or hawkish guidance", "Pairs": "Currency strengthens across pairings"},
            "impact_negative": {"Currency": "↓ if cut or dovish guidance", "Pairs": "Currency weakens across pairings"},
        },
        {
            "event": "GDP (YoY / QoQ)",
            "description": "Broadest measure of economic activity. Strong GDP -> stronger currency.",
            "impact_positive": {"Currency": "↑", "Pairs": "Currency strengthens vs majors"},
            "impact_negative": {"Currency": "↓", "Pairs": "Currency weakens vs majors"},
        },
        {
            "event": "Retail Sales",
            "description": "A key gauge of consumer spending and near-term growth.",
            "impact_positive": {"Currency": "↑", "Pairs": "Currency strengthens vs majors"},
            "impact_negative": {"Currency": "↓", "Pairs": "Currency weakens vs majors"},
        },
        # Add or extend other events as needed
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
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="color:#FFD700; margin:0;">{ev['event']}</h4>
                    <span class="small-muted">{ev.get('currency','')}</span>
                </div>
                <p style="margin:6px 0 12px 0;">{ev['description']}</p>
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
# TAB 2: UNDERSTANDING FOREX FUNDAMENTALS
# =========================================================

# =========================================================
# TAB 3: TECHNICAL ANALYSIS
# =========================================================
with selected_tab[1]:
    st.title("📊 Technical Analysis")
    st.caption("Live TradingView chart + curated news for the selected pair.")

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
    watchlist = list(pairs_map.values())

    # TradingView Advanced Chart (widget)
    tv_html = f"""
    <div class="tradingview-widget-container" style="height:780px; width:100%">
      <div id="tradingview_chart_{tv_symbol.replace(':','_')}" style="height:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
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
        "container_id": "tradingview_chart_{tv_symbol.replace(':','_')}"
      }});
      </script>
    </div>
    """
    components.html(tv_html, height=820, scrolling=False)

    # News & Sentiment for Selected Pair
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
# TAB 4: TOOLS (Calculator, Backtesting, Price Alerts)
# =========================================================
with selected_tab[2]:
    st.title("🛠 Tools")
    tools_subtabs = st.tabs(["Profit/Stop-loss Calculator", "Backtesting", "Price alerts", "Currency Correlation Heatmap", "Risk Calculator", "Trading Session Tracker", "Candlestick Pattern Cheat Sheet"])

    # Profit/Stop-loss Calculator
    with tools_subtabs[0]:
        st.header("💰 Profit / Stop-loss Calculator")
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

    # Backtesting
    with tools_subtabs[1]:
        st.header("📊 Backtesting")
        st.markdown("Backtest your trading strategies here.")
        tv_widget = """
        <div class="tradingview-widget-container">
            <div id="tradingview_advanced_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget({
                    "width": "100%",
                    "height": 600,
                    "symbol": "FX:EURUSD",
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
                });
            </script>
        </div>
        """
        st.components.v1.html(tv_widget, height=620)

        # Backtesting Journal (editable)
        journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "Notes"]
        if "tools_trade_journal" not in st.session_state or st.session_state.tools_trade_journal.empty:
            st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols)
        updated_journal_tools = st.data_editor(
            data=st.session_state.tools_trade_journal.copy(),
            num_rows="dynamic",
            key="tools_backtesting_journal_unique"
        )
        st.session_state.tools_trade_journal = updated_journal_tools

        # Save / Load Buttons
        if "logged_in_user" in st.session_state:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save to My Account", key="save_journal_button"):
                    username = st.session_state.logged_in_user
                    accounts = {}
                    if os.path.exists(ACCOUNTS_FILE):
                        with open(ACCOUNTS_FILE, "r") as f:
                            accounts = json.load(f)
                    accounts.setdefault(username, {})["tools_trade_journal"] = st.session_state.tools_trade_journal.to_dict(orient="records")
                    with open(ACCOUNTS_FILE, "w") as f:
                        json.dump(accounts, f, indent=4)
                    st.success("Trading journal saved to your account!")
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
                        st.success("Trading journal loaded from your account!")
                    else:
                        st.info("No saved journal found in your account.")
        else:
            st.info("Sign in to save your trading journal to your account.")

    # Price Alerts
    with tools_subtabs[2]:
        st.header("⏰ Price Alerts")
        st.markdown("Set price alerts for your favourite forex pairs and get notified when the price hits your target.")

        # popular pairs (exchangerate.host uses base/symbol)
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY"]

        # Initialize session state for alerts (no Direction — triggers when target hit)
        if "price_alerts" not in st.session_state:
            st.session_state.price_alerts = pd.DataFrame(columns=["Pair", "Target Price", "Triggered"])

        # Input form for new alert (pair + price)
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

        # show current alerts table
        st.subheader("Your Alerts")
        st.dataframe(st.session_state.price_alerts, use_container_width=True, height=220)

        # Auto-refresh for live prices (every 5 seconds) - reduces chance of public API throttling
        st_autorefresh(interval=5000, key="price_alert_autorefresh")

        # Fetch live prices only for pairs with active alerts
        active_pairs = st.session_state.price_alerts["Pair"].unique().tolist()
        live_prices = {}
        for p in active_pairs:
            if not p:
                continue
            base, quote = p[:3], p[3:]
            try:
                # exchangerate.host: no API key, simple JSON
                r = requests.get(f"https://api.exchangerate.host/latest?base={base}&symbols={quote}", timeout=6)
                data = r.json()
                price_val = data.get("rates", {}).get(quote)
                live_prices[p] = float(price_val) if price_val is not None else None
            except Exception:
                live_prices[p] = None

        # Check and trigger alerts (no direction - hit equals target within tolerance)
        triggered_alerts = []
        for idx, row in st.session_state.price_alerts.iterrows():
            pair = row["Pair"]
            target = row["Target Price"]
            current_price = live_prices.get(pair)
            if isinstance(current_price, (int, float)):
                if not row["Triggered"] and abs(current_price - target) < (0.0005 if "JPY" not in pair else 0.01):
                    st.session_state.price_alerts.at[idx, "Triggered"] = True
                    triggered_alerts.append((idx, f"{pair} reached {target} (Current: {current_price:.5f})"))

        # Show triggered alerts and allow dismiss (cancel remains)
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
                        st.experimental_rerun()
        else:
            st.info("No price alerts set. Add one above to start monitoring prices.")

# ---------------- TOOLS: NEW SUBTABS ----------------
import pandas as pd
import numpy as np
import datetime as dt

# ----------- Tool 3: Currency Correlation Heatmap -----------
with tools_subtabs[3]:
    st.header("📊 Currency Correlation Heatmap")
    st.markdown("Understand how forex pairs move relative to each other. "
                "Positive correlation means pairs move in the same direction, "
                "while negative correlation means they move opposite.")

    # Example static correlation values (could be replaced with real data later)
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

    import plotly.express as px
    fig = px.imshow(corr_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title="Forex Pair Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)


# ----------- Tool 4: Risk Management Calculator -----------
with tools_subtabs[4]:
    st.header("🛡️ Risk Management Calculator")
    st.markdown("Proper position sizing keeps your account safe. "
                "Use this calculator to determine your lot size.")

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


# ----------- Tool 5: Economic Session Clock -----------
with tools_subtabs[5]:
    st.header("🕒 Forex Market Sessions")
    st.markdown("Stay aware of active trading sessions to trade when volatility is highest.")

    # Session times (UTC)
    sessions = {
        "Sydney": (22, 7),
        "Tokyo": (0, 9),
        "London": (8, 17),
        "New York": (13, 22),
    }

    now_utc = dt.datetime.utcnow().hour
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


# ----------- Tool 6: Candlestick Pattern Cheat Sheet -----------
with tools_subtabs[6]:
    st.header("📑 Candlestick Pattern Cheat Sheet")
    st.markdown("Learn to recognize powerful reversal and continuation signals.")

    patterns = [
        {"name": "Hammer", "type": "Bullish", "meaning": "Potential reversal after downtrend"},
        {"name": "Shooting Star", "type": "Bearish", "meaning": "Potential reversal after uptrend"},
        {"name": "Engulfing", "type": "Bullish/Bearish", "meaning": "Strong reversal depending on direction"},
        {"name": "Doji", "type": "Neutral", "meaning": "Market indecision, possible reversal"},
    ]

    cols = st.columns(2)
    for i, pattern in enumerate(patterns):
        with cols[i % 2]:
            st.markdown(f"""
                <div style="
                    background-color:#1e1e2f;
                    border-radius:12px;
                    padding:15px;
                    margin-bottom:10px;
                    box-shadow:2px 2px 8px rgba(0,0,0,0.4);
                    color:white;
                ">
                    <h3 style="margin:0; color:#FFD700;">{pattern['name']}</h3>
                    <p style="margin:2px 0;"><b>Type:</b> {pattern['type']}</p>
                    <p style="margin:2px 0;"><b>Meaning:</b> {pattern['meaning']}</p>
                </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 5: MY ACCOUNT
# =========================================================
with selected_tab[3]:
    st.title("👤 My Account")
    st.subheader("Account Login / Sign Up")
    ACCOUNTS_FILE = "user_accounts.json"
    if not os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "w") as f:
            json.dump({}, f)

    # LOGIN
    login_expander = st.expander("Login")
    with login_expander:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            with open(ACCOUNTS_FILE, "r") as f:
                accounts = json.load(f)
            if username in accounts and accounts[username]["password"] == password:
                st.session_state.logged_in_user = username
                st.success(f"Logged in as {username}")
                saved_journal = accounts.get(username, {}).get("tools_trade_journal", [])
                journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "Notes"]
                if saved_journal:
                    st.session_state.tools_trade_journal = pd.DataFrame(saved_journal, columns=journal_cols)
                else:
                    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols)
            else:
                st.error("Invalid username or password")

    # SIGN UP
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
                accounts[new_username] = {"password": new_password}
                with open(ACCOUNTS_FILE, "w") as f:
                    json.dump(accounts, f, indent=4)
                st.success(f"Account created for {new_username}")

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
            st.session_state.name = name
            st.session_state.email = email
            st.session_state.base_ccy = base_ccy
            st.session_state.alerts = alerts
            st.success("Preferences saved for this session.")

# ==================== TAB 5: MT5 STATS DASHBOARD ====================
with selected_tab[4]:
    st.markdown("## 📊 MT5 Stats Dashboard")
    st.write("Upload your MT5 trading history CSV to view a detailed performance dashboard.")

    uploaded_file = st.file_uploader("Upload MT5 History CSV", type=["csv"])

    if uploaded_file:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go

        df = pd.read_csv(uploaded_file)

        # Ensure required columns exist
        required_cols = ["Symbol", "Type", "Profit", "Volume", "Open Time", "Close Time", "Balance"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV is missing required columns: {', '.join(missing_cols)}")
        else:
            # Convert time columns
            df["Open Time"] = pd.to_datetime(df["Open Time"], errors="coerce")
            df["Close Time"] = pd.to_datetime(df["Close Time"], errors="coerce")

            # Basic performance metrics
            total_trades = len(df)
            wins = df[df["Profit"] > 0]
            losses = df[df["Profit"] <= 0]
            win_rate = (len(wins)/total_trades*100) if total_trades else 0
            avg_win = wins["Profit"].mean() if not wins.empty else 0
            avg_loss = losses["Profit"].mean() if not losses.empty else 0
            profit_factor = (wins["Profit"].sum() / abs(losses["Profit"].sum())) if not losses.empty else np.inf
            net_profit = df["Profit"].sum()
            biggest_win = df["Profit"].max()
            biggest_loss = df["Profit"].min()
            max_drawdown = df["Balance"].max() - df["Balance"].min() if "Balance" in df else 0
            # Longest win/loss streaks
            longest_win_streak = max((len(list(g)) for k,g in df["Profit"].gt(0).groupby(df["Profit"].gt(0)) if k), default=0)
            longest_loss_streak = max((len(list(g)) for k,g in df["Profit"].lt(0).groupby(df["Profit"].lt(0)) if k), default=0)
            # Avg trade duration (hours)
            df["Trade Duration"] = (df["Close Time"] - df["Open Time"]).dt.total_seconds()/3600
            avg_trade_duration = df["Trade Duration"].mean()

            # --- Function to determine text color ---
            def metric_color(title, value):
                try:
                    val = float(value)
                    if title in ["Max Drawdown", "Longest Loss Streak"]:
                        return "#ff4c4c"  # Always red
                    if title == "Profit Factor":
                        if val > 1:
                            return "lightgreen"
                        elif val == 1:
                            return "yellow"
                        else:
                            return "#ff4c4c"
                    elif val > 0:
                        return "lightgreen"
                    elif val < 0:
                        return "#ff4c4c"
                    else:
                        return "white"
                except:
                    return "white"

            # --- Metric Data with Icons ---
            metrics = [
                ("Total Trades", total_trades, "📈"),
                ("Win Rate", f"{win_rate:.2f}%", "✅"),
                ("Net Profit", f"${net_profit:,.2f}", "💰"),
                ("Profit Factor", round(profit_factor,2), "⚡"),
                ("Biggest Win", f"${biggest_win:,.2f}", "🏆"),
                ("Biggest Loss", f"${biggest_loss:,.2f}", "💀"),
                ("Max Drawdown", f"${max_drawdown:,.2f}", "📉"),
                ("Longest Win Streak", longest_win_streak, "🔥"),
                ("Longest Loss Streak", longest_loss_streak, "❌"),
                ("Avg Trade Duration (h)", f"{avg_trade_duration:.2f}", "⏱️")
            ]

            # --- Display Metrics in Two Rows ---
            st.markdown("### ⚡ Key Performance Metrics")
            first_row = st.columns(5)
            second_row = st.columns(5)

            for i, (title, value, icon) in enumerate(metrics):
                color = metric_color(title, value)
                text = f"{icon} **{title}:** {value}"
                if i < 5:
                    first_row[i].markdown(f"<p style='color:{color}; font-size:18px'>{text}</p>", unsafe_allow_html=True)
                else:
                    second_row[i-5].markdown(f"<p style='color:{color}; font-size:18px'>{text}</p>", unsafe_allow_html=True)

            st.markdown("---")

            # --- Balance / Equity Curve ---
            st.markdown("### 💵 Balance / Equity Curve")
            fig_balance = px.line(df, x="Close Time", y="Balance", title="Equity / Balance Curve", template="plotly_dark")
            st.plotly_chart(fig_balance, use_container_width=True)

            # --- Profit by Symbol ---
            st.markdown("### 📊 Profit by Symbol")
            profit_symbol = df.groupby("Symbol")["Profit"].sum().reset_index()
            fig_symbol = px.bar(profit_symbol, x="Symbol", y="Profit", color="Profit",
                                title="Profit by Instrument", template="plotly_dark")
            st.plotly_chart(fig_symbol, use_container_width=True)

            # --- Trade Distribution ---
            st.markdown("### 🔎 Trade Distribution")
            col1, col2 = st.columns(2)
            with col1:
                fig_types = px.pie(df, names="Type", title="Buy vs Sell Distribution", template="plotly_dark")
                st.plotly_chart(fig_types, use_container_width=True)
            with col2:
                df["Weekday"] = df["Open Time"].dt.day_name()
                fig_weekday = px.histogram(df, x="Weekday", color="Type",
                                           title="Trades by Day of Week", template="plotly_dark")
                st.plotly_chart(fig_weekday, use_container_width=True)

            # --- Cumulative PnL ---
            st.markdown("### 📈 Cumulative Profit & Loss")
            df["Cumulative PnL"] = df["Profit"].cumsum()
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=df["Close Time"], y=df["Cumulative PnL"], mode="lines", name="Cumulative PnL"))
            fig_pnl.update_layout(template="plotly_dark", title="Cumulative PnL Over Time")
            st.plotly_chart(fig_pnl, use_container_width=True)

            st.success("✅ MT5 Stats Dashboard Loaded Successfully!")

    else:
        st.info("👆 Please upload your MT5 trading history CSV to view the dashboard.")
