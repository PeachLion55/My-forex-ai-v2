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
    {"Date": "2025-08-15", "Time": "12:30", "Currency": "USD", "Event": "CPI m/m", "Actual": "0.3%", "Forecast": "0.2%", "Previous": "0.4%", "Impact": "High"},
    {"Date": "2025-08-15", "Time": "14:00", "Currency": "EUR", "Event": "ECB Interest Rate Decision", "Actual": "0.50%", "Forecast": "0.50%", "Previous": "0.25%", "Impact": "High"},
    {"Date": "2025-08-16", "Time": "09:30", "Currency": "GBP", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.3%", "Previous": "0.2%", "Impact": "Medium"},
]
econ_df = pd.DataFrame(econ_calendar_data)

# ----------------- PAGE CONTENT -----------------
with selected_tab[0]:
    st.title("📅 Forex News Sentiment")
    st.caption("Click a headline to view detailed summary and sentiment")

    df = get_fxstreet_forex_news()

    if not df.empty:
        currency_filter = st.selectbox(
            "What currency pair would you like to track?", 
            options=["All"] + sorted(df["Currency"].unique())
        )
        if currency_filter != "All":
            df = df[df["Currency"] == currency_filter]

        # Flag high-probability headlines
        df["HighProb"] = df.apply(
            lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] 
            and pd.to_datetime(row["Date"]) >= pd.Timestamp.now() - pd.Timedelta(days=1)
            else "", axis=1
        )

        df_display = df.copy()
        df_display["Headline"] = df["HighProb"] + " " + df["Headline"]

        selected_headline = st.selectbox("Select a headline for details", df_display["Headline"].tolist())
        selected_row = df_display[df_display["Headline"] == selected_headline].iloc[0]

        st.markdown(f"### [{selected_row['Headline']}]({selected_row['Link']})")
        st.write(f"**Published:** {selected_row['Date']}")

        # ----------------- BLUE BOX DETAILED SUMMARY -----------------
        summary_text = selected_row["Summary"]
        sentences = re.split(r'(?<=[.!?]) +', summary_text)
        bullet_points = "\n".join([f"- {s}" for s in sentences[:10]])  # first 10 sentences
        st.info(bullet_points)

        # ----------------- ECONOMIC CALENDAR BELOW SUMMARY -----------------
        st.markdown("### 🗓️ Upcoming Economic Events")
        st.dataframe(econ_df)

        # ----------------- IMPACT RATING -----------------
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

        # ----------------- CURRENCY SENTIMENT BIAS TABLE -----------------
        st.markdown("---")
        st.markdown("## 📈 Currency Sentiment Bias Table")
        bias_df = df.groupby("Currency")["Impact"].value_counts().unstack().fillna(0)
        st.dataframe(bias_df)

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

with selected_tab[1]:
    st.title("👤 My Account")
    st.write("This is your account page. You can add user settings, subscription info, or API key management here.")
