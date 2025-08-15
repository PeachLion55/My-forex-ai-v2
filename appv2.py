import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta

st.set_page_config(page_title="Forex AI Dashboard", layout="wide")

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
        font-weight: bold !important;
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
def get_gnews_forex_sentiment():
    API_KEY = st.secrets["GNEWS_API_KEY"]
    url = f"https://gnews.io/api/v4/search?q=forex+OR+inflation+OR+interest+rate+OR+CPI+OR+GDP+OR+Fed+OR+ECB&lang=en&token={API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"GNews API error: {response.status_code}")
        return pd.DataFrame()

    articles = response.json().get("articles", [])
    rows = []

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

    for article in articles:
        title = article.get("title", "")
        date = article.get("publishedAt", "")[:10]
        currency = detect_currency(title)
        sentiment_score = TextBlob(title).sentiment.polarity
        impact = rate_impact(sentiment_score)
        summary = article.get("description", "") or title.split(":")[-1].strip()

        rows.append({
            "Date": date,
            "Currency": currency,
            "Headline": title,
            "Impact": impact,
            "Summary": summary
        })

    df = pd.DataFrame(rows)
    # Convert Date to datetime for filtering
    df["DateTime"] = pd.to_datetime(df["Date"])
    return df

# ----------------- TAB 1: Forex Fundamentals -----------------
with selected_tab[0]:
    st.title("📅 Forex Economic Calendar & News Sentiment")
    st.caption("Click a headline to view detailed summary and sentiment")

    df = get_gnews_forex_sentiment()

    if not df.empty:
        # Filter by today and next 24 hours
        now = datetime.now()
        next_24h = now + timedelta(hours=24)
        default_df = df[(df["DateTime"] >= now) & (df["DateTime"] <= next_24h)]

        currency_filter = st.selectbox("Filter by Currency", options=["All"] + sorted(df["Currency"].unique()))
        if currency_filter != "All":
            default_df = default_df[default_df["Currency"] == currency_filter]

        st.markdown("### 📌 Economic Calendar (Today + Next 24h)")

        # Flag high-probability headlines
        default_df["HighProb"] = default_df.apply(
            lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] else "", axis=1
        )
        default_df["HeadlineDisplay"] = default_df["HighProb"] + " " + default_df["Headline"]

        selected_headline = st.selectbox("Select a headline for details", default_df["HeadlineDisplay"].tolist())
        st.dataframe(default_df[["Date", "Currency", "HeadlineDisplay"]].sort_values(by="DateTime", ascending=True), use_container_width=True)

        # Show detailed info
        selected_row = default_df[default_df["HeadlineDisplay"] == selected_headline].iloc[0]
        st.markdown("### 🧠 Summary")
        st.info(selected_row["Summary"])
        st.markdown("### 🔥 Impact Rating")
        impact = selected_row["Impact"]
        if "Bullish" in impact:
            st.success(impact)
        elif "Bearish" in impact:
            st.error(impact)
        else:
            st.warning(impact)

        st.markdown("### ⏱️ Timeframes Likely Affected")
        if "Significantly" in impact:
            timeframes = ["H4", "Daily"]
        elif impact in ["Bullish", "Bearish"]:
            timeframes = ["H1", "H4"]
        else:
            timeframes = ["H1"]
        st.write(", ".join(timeframes))

        st.markdown("### 💱 Likely Affected Currency Pairs")
        base = selected_row["Currency"]
        if base != "Unknown":
            pairs = [f"{base}/USD", f"EUR/{base}", f"{base}/JPY", f"{base}/CHF", f"{base}/CAD", f"{base}/NZD", f"{base}/AUD"]
            st.write(", ".join(pairs))
        else:
            st.write("Cannot determine affected pairs.")

        # Load more data button
        if st.button("Load More Data (All Articles)"):
            full_df = df.copy()
            if currency_filter != "All":
                full_df = full_df[full_df["Currency"] == currency_filter]

            full_df["HighProb"] = full_df.apply(
                lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] else "", axis=1
            )
            full_df["HeadlineDisplay"] = full_df["HighProb"] + " " + full_df["Headline"]
            st.markdown("### 📌 Full Economic Calendar")
            st.dataframe(full_df[["Date", "Currency", "HeadlineDisplay"]].sort_values(by="DateTime", ascending=True), use_container_width=True)

    else:
        st.info("No forex news available or API limit reached.")

# ----------------- TAB 2: My Account -----------------
with selected_tab[1]:
    st.title("👤 My Account")
    st.write("This is your account page. You can add user settings, subscription info, or API key management here.")
