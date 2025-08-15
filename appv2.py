import streamlit as st
import pandas as pd
import feedparser
import re
from textblob import TextBlob

st.set_page_config(page_title="Forex Dashboard", layout="wide")

# ----------------- HORIZONTAL NAVIGATION -----------------
tabs = ["Forex Fundamentals", "My Account"]
selected_tab = st.tabs(tabs)

# ----------------- CUSTOM CSS FOR TABS AND PADDING -----------------
st.markdown("""
<style>
    /* Active tab styling */
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #FFD700 !important;
        color: black !important;
        font-weight: bold;
        padding: 15px 30px !important;
        border-radius: 8px;
        margin-right: 10px !important;
    }
    /* Inactive tab styling */
    div[data-baseweb="tab-list"] button[aria-selected="false"] {
        background-color: #f0f0f0 !important;
        color: #555 !important;
        padding: 15px 30px !important;
        border-radius: 8px;
        margin-right: 10px !important;
    }
    /* Page content padding */
    .css-1d391kg { 
        padding: 30px 40px !important; 
    }
</style>
""", unsafe_allow_html=True)

# ----------------- FUNCTIONS -----------------
def detect_currency(title):
    title_upper = title.upper()
    currency_map = {
        "USD": ["USD", "US", "FED", "FEDERAL RESERVE", "AMERICA"],
        "GBP": ["GBP", "UK", "BRITAIN", "BOE", "POUND", "STERLING"],
        "EUR": ["EUR", "EURO", "EUROZONE", "ECB"],
        "JPY": ["JPY", "JAPAN", "BOJ", "YEN"],
        "AUD": ["AUD", "AUSTRALIA", "RBA"],
        "CAD": ["CAD", "CANADA", "BOC"],
        "CHF": ["CHF", "SWITZERLAND", "SNB"],
        "NZD": ["NZD", "NEW ZEALAND", "RBNZ"],
    }
    for curr, keywords in currency_map.items():
        for kw in keywords:
            if kw in title_upper:
                return curr
    return "Unknown"

def rate_impact(polarity):
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

def get_fxstreet_forex_news():
    RSS_URL = "https://www.fxstreet.com/rss/news"
    feed = feedparser.parse(RSS_URL)
    rows = []

    for entry in feed.entries:
        title = entry.title
        date = entry.published[:10] if hasattr(entry, "published") else ""
        currency = detect_currency(title)
        sentiment_score = TextBlob(title).sentiment.polarity
        impact = rate_impact(sentiment_score)
        summary = entry.summary

        rows.append({
            "Date": date,
            "Currency": currency,
            "Headline": title,
            "Impact": impact,
            "Summary": summary,
            "Link": entry.link
        })

    return pd.DataFrame(rows)

# ----------------- ECONOMIC CALENDAR DATA -----------------
# Replace with your actual data
econ_calendar_data = [
    # Fri Aug 15
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

    # Sat Aug 16
    {"Date": "2025-08-16", "Time": "Tentative", "Currency": "USD", "Event": "President Trump Speaks", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},

    # Sun Aug 17
    {"Date": "2025-08-17", "Time": "23:30", "Currency": "NZD", "Event": "BusinessNZ Services Index", "Actual": "47.3", "Forecast": "", "Previous": "", "Impact": ""},

    # Mon Aug 18
    {"Date": "2025-08-18", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-1.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "05:30", "Currency": "JPY", "Event": "Tertiary Industry Activity m/m", "Actual": "0.1%", "Forecast": "0.6%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "10:00", "Currency": "EUR", "Event": "Trade Balance", "Actual": "18.1B", "Forecast": "16.2B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "13:15", "Currency": "CAD", "Event": "Housing Starts", "Actual": "270K", "Forecast": "284K", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "13:30", "Currency": "CAD", "Event": "Foreign Securities Purchases", "Actual": "-4.75B", "Forecast": "-2.79B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "15:00", "Currency": "USD", "Event": "NAHB Housing Market Index", "Actual": "34", "Forecast": "33", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Input q/q", "Actual": "1.4%", "Forecast": "2.9%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Output q/q", "Actual": "1.0%", "Forecast": "2.1%", "Previous": "", "Impact": ""},

    # Tue Aug 19
    {"Date": "2025-08-19", "Time": "01:30", "Currency": "AUD", "Event": "Westpac Consumer Sentiment", "Actual": "0.6%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "Tentative", "Currency": "CNY", "Event": "Foreign Direct Investment ytd/y", "Actual": "-15.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "09:00", "Currency": "EUR", "Event": "Current Account", "Actual": "33.4B", "Forecast": "32.3B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "CPI m/m", "Actual": "0.4%", "Forecast": "0.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "Median CPI y/y", "Actual": "3.1%", "Forecast": "3.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "BoC Business Outlook Survey", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},

    # Wed Aug 20
    {"Date": "2025-08-20", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-0.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "CPI y/y", "Actual": "2.5%", "Forecast": "2.6%", "Previous": "2.7%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "PPI y/y", "Actual": "-3.3%", "Forecast": "-3.0%", "Previous": "-3.1%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "08:00", "Currency": "EUR", "Event": "German PPI m/m", "Actual": "0.2%", "Forecast": "0.1%", "Previous": "0.1%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "13:30", "Currency": "CAD", "Event": "Manufacturing Sales m/m", "Actual": "0.3%", "Forecast": "0.5%", "Previous": "-1.2%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "14:30", "Currency": "USD", "Event": "Crude Oil Inventories", "Actual": "-5.3M", "Forecast": "-1.2M", "Previous": "-0.6M", "Impact": ""},

    # Thu Aug 21
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Employment Change", "Actual": "36.1K", "Forecast": "30.0K", "Previous": "-10.0K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Unemployment Rate", "Actual": "3.6%", "Forecast": "3.7%", "Previous": "3.8%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI y/y", "Actual": "3.2%", "Forecast": "3.3%", "Previous": "3.0%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI m/m", "Actual": "0.3%", "Forecast": "0.4%", "Previous": "0.1%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "14:00", "Currency": "EUR", "Event": "ECB Interest Rate Decision", "Actual": "0.50%", "Forecast": "0.50%", "Previous": "0.25%", "Impact": "High"},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Initial Jobless Claims", "Actual": "218K", "Forecast": "220K", "Previous": "217K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Continuing Claims", "Actual": "1445K", "Forecast": "1450K", "Previous": "1440K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "15:00", "Currency": "USD", "Event": "Existing Home Sales", "Actual": "4.25M", "Forecast": "4.23M", "Previous": "4.19M", "Impact": ""},

    # Fri Aug 22
    {"Date": "2025-08-22", "Time": "09:30", "Currency": "GBP", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.3%", "Previous": "0.2%", "Impact": "Medium"},
]
econ_df = pd.DataFrame(econ_calendar_data)

# ----------------- PAGE CONTENT -----------------
with selected_tab[0]:
    st.title("📅 Forex News Sentiment")
    st.caption("Click a headline to view detailed summary and sentiment")

    df = get_fxstreet_forex_news()

    if not df.empty:
        # Initialize session state
        if 'selected_currency_1' not in st.session_state:
            st.session_state.selected_currency_1 = None
        if 'selected_currency_2' not in st.session_state:
            st.session_state.selected_currency_2 = None

        # First currency dropdown
        currency_filter_1 = st.selectbox(
            "What primary currency pair would you like to track?", 
            options=["All"] + sorted(df["Currency"].unique()),
            key="currency1_dropdown"
        )
        st.session_state.selected_currency_1 = None if currency_filter_1 == "All" else currency_filter_1

        # Second currency dropdown
        currency_filter_2 = st.selectbox(
            "What secondary currency pair would you like to track?", 
            options=["None"] + sorted(df["Currency"].unique()),
            key="currency2_dropdown"
        )
        st.session_state.selected_currency_2 = None if currency_filter_2 in ["None", "All"] else currency_filter_2

        # Now filter dataframe for headlines based on BOTH dropdowns if needed
        filtered_df = df.copy()
        if st.session_state.selected_currency_1:
            filtered_df = filtered_df[filtered_df["Currency"] == st.session_state.selected_currency_1]

        # Flag high-probability headlines
        filtered_df["HighProb"] = filtered_df.apply(
            lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] 
            and pd.to_datetime(row["Date"]) >= pd.Timestamp.now() - pd.Timedelta(days=1)
            else "", axis=1
        )

        filtered_df_display = filtered_df.copy()
        filtered_df_display["Headline"] = filtered_df["HighProb"] + " " + filtered_df["Headline"]

        selected_headline = st.selectbox(
            "Select a headline for details", 
            filtered_df_display["Headline"].tolist()
        )
        selected_row = filtered_df_display[filtered_df_display["Headline"] == selected_headline].iloc[0]

        st.markdown(f"### [{selected_row['Headline']}]({selected_row['Link']})")
        st.write(f"**Published:** {selected_row['Date']}")

        # ----------------- ECONOMIC CALENDAR -----------------
        st.markdown("### 🗓️ Upcoming Economic Events")

        def highlight_currency(row):
            styles = [''] * len(row)
            
            if st.session_state.selected_currency_1 and row['Currency'] == st.session_state.selected_currency_1:
                styles = ['background-color: #171447; color: white' if col == 'Currency' else 'background-color: #171447' for col in row.index]
            
            if st.session_state.selected_currency_2 and row['Currency'] == st.session_state.selected_currency_2:
                styles = ['background-color: #471414; color: white' if col == 'Currency' else 'background-color: #471414' for col in row.index]

            return styles

        st.dataframe(
            econ_df.style.apply(highlight_currency, axis=1)
        )

# ----------------- INTEREST RATES -----------------
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

# Number of boxes per row
boxes_per_row = 4

for i in range(0, len(interest_rates), boxes_per_row):
    cols = st.columns(boxes_per_row)
    for j, rate in enumerate(interest_rates[i:i+boxes_per_row]):
        with cols[j]:
            st.markdown(
                f"""
                <div style="
                    background-color:#f0f2f6;
                    border-radius:10px;
                    padding:15px;
                    text-align:center;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                ">
                    <h3>{rate['Currency']}</h3>
                    <p><b>Current:</b> {rate['Current']}</p>
                    <p><b>Previous:</b> {rate['Previous']}</p>
                    <p><b>Changed On:</b> {rate['Changed']}</p>
                </div>
                """, unsafe_allow_html=True
            )
# ----------------- BEGINNER-FRIENDLY TRADE OUTLOOK -----------------
if not df.empty:
    st.markdown("## 🧭 Beginner-Friendly Trade Outlook")
    if "Bullish" in selected_row["Impact"]:
        st.info(f"🟢 Sentiment on **{selected_row['Currency']}** is bullish. Look for buying setups on H1/H4.")
    elif "Bearish" in selected_row["Impact"]:
        st.warning(f"🔴 Sentiment on **{selected_row['Currency']}** is bearish. Look for selling setups on H1/H4.")
    else:
        st.write("⚪ No strong directional sentiment detected right now.")
else:
    st.info("No forex news available at the moment.")

# ----------------- IMPACT RATING -----------------
if not df.empty:
    st.markdown("### 🔥 Impact Rating")
    impact = selected_row["Impact"]
    if "Bullish" in impact:
        st.success(impact)
    elif "Bearish" in impact:
        st.error(impact)
    else:
        st.warning(impact)

    # ----------------- TIMEFRAMES LIKELY AFFECTED -----------------
    st.markdown("### ⏱️ Timeframes Likely Affected")
    if "Significantly" in impact:
        timeframes = ["H4", "Daily"]
    elif impact in ["Bullish", "Bearish"]:
        timeframes = ["H1", "H4"]
    else:
        timeframes = ["H1"]
    st.write(", ".join(timeframes))

    # ----------------- LIKELY AFFECTED CURRENCY PAIRS -----------------
    st.markdown("### 💱 Likely Affected Currency Pairs")
    base = selected_row["Currency"]
    if base != "Unknown":
        pairs = [f"{base}/USD", f"EUR/{base}", f"{base}/JPY", f"{base}/CHF", f"{base}/CAD", f"{base}/NZD", f"{base}/AUD"]
        st.write(", ".join(pairs))
    else:
        st.write("Cannot determine affected pairs.")

    # ----------------- BEGINNER-FRIENDLY TRADE OUTLOOK -----------------
    st.markdown("## 🧭 Beginner-Friendly Trade Outlook")
    if "Bullish" in impact:
        st.info(f"🟢 Sentiment on **{base}** is bullish. Look for buying setups on H1/H4.")
    elif "Bearish" in impact:
        st.warning(f"🔴 Sentiment on **{base}** is bearish. Look for selling setups on H1/H4.")
    else:
        st.write("⚪ No strong directional sentiment detected right now.")
else:
    st.info("No forex news available at the moment.")
