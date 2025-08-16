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

# =========================================================
# NAVIGATION
# =========================================================
tabs = ["Forex Fundamentals", "Understanding Forex Fundamentals", "Technical Analysis", "My Account"]
selected_tab = st.tabs(tabs)

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
import requests

with selected_tab[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📅 Forex Fundamentals")
        st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")

# -------- Live Currency Strength Meter --------
st.markdown("### 💪 Live Currency Strength")
try:
    import requests

    # Get API key from secrets
    api_key = st.secrets["EXCHANGERATE_HOST_API_KEY"]

    # Fetch latest exchange rates relative to USD
    url = f"https://api.exchangerate.host/latest?base=USD&access_key={api_key}"
    response = requests.get(url)
    data = response.json()

    if not data.get("success", True):
        raise ValueError(data.get("error", {}).get("info", "Unknown API error"))

    rates = data.get("rates", {})
    if not rates:
        raise ValueError("No rates returned")

    # Major currencies
    major_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

    # Build currency strength relative to all pairs
    currency_strength = {}
    for ccy in major_currencies:
        # Average of ccy vs all other major currencies
        values = []
        for other in major_currencies:
            if ccy == other:
                continue
            pair_rate = rates.get(other)
            if pair_rate is not None:
                values.append(pair_rate)
        # Mean of all rates as a rough "strength"
        currency_strength[ccy] = sum(values) / len(values) if values else 0

    # Normalize strength to 0–1 scale
    max_rate = max(currency_strength.values())
    min_rate = min(currency_strength.values())
    if max_rate == min_rate:
        normalized_strength = {ccy: 0.5 for ccy in currency_strength}  # all equal
    else:
        normalized_strength = {ccy: (val - min_rate) / (max_rate - min_rate) for ccy, val in currency_strength.items()}

    # Display as horizontal bars like LiveCharts
    st.markdown("<div style='display:flex; gap:10px;'>", unsafe_allow_html=True)
    colors = ["#171447", "#471414"]
    for i, ccy in enumerate(major_currencies):
        strength_pct = int(normalized_strength[ccy] * 100)
        color = colors[i % 2]
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                border-radius:10px;
                padding:10px;
                text-align:center;
                color:white;
                flex:1;
            ">
                <h4 style='margin:5px 0'>{ccy}</h4>
                <div style="
                    background-color: #f1f1f1;
                    border-radius:5px;
                    height:15px;
                    margin:5px 0;
                ">
                    <div style="
                        width:{strength_pct}%;
                        background-color:#00ff00;
                        height:15px;
                        border-radius:5px;
                    "></div>
                </div>
                <p style='margin:0'>{strength_pct}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Failed to load currency strength: {e}")

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
with selected_tab[3]:
    st.title("👤 My Account")
    st.caption("Your preferences are stored in-session.")

    colA, colB = st.columns(2)
    with colA:
        name = st.text_input("Name", value=st.session_state.get("name", ""))
        base_ccy = st.selectbox("Preferred Base Currency", ["USD","EUR","GBP","JPY","AUD","CAD","NZD","CHF"], index=0)
    with colB:
        email = st.text_input("Email", value=st.session_state.get("email", ""))
        alerts = st.checkbox("Email me before high-impact events", value=st.session_state.get("alerts", True))

    if st.button("Save Preferences"):
        st.session_state.name = name
        st.session_state.email = email
        st.session_state.base_ccy = base_ccy
        st.session_state.alerts = alerts
        st.success("Preferences saved for this session.")

    if "name" in st.session_state:
        st.markdown(
            f"**Current Profile:** {st.session_state.name} | {st.session_state.get('base_ccy','USD')} | Alerts: "
            f"{'On' if st.session_state.get('alerts', False) else 'Off'}"
        )
