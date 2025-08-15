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
        background-color: #FFD700 !important;  /* Gold color */
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

def get_gnews_forex_sentiment():
    """Fetch Forex-related news and compute sentiment/impact"""
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
        published_at = article.get("publishedAt", "")
        # Ensure valid datetime
        try:
            date_time = pd.to_datetime(published_at)
        except:
            date_time = pd.NaT

        currency = detect_currency(title)
        sentiment_score = TextBlob(title).sentiment.polarity
        impact = rate_impact(sentiment_score)
        summary = article.get("description", "") or title.split(":")[-1].strip()

        rows.append({
            "DateTime": date_time,
            "Date": date_time.date() if pd.notna(date_time) else None,
            "Currency": currency,
            "Headline": title,
            "Impact": impact,
            "Summary": summary
        })

    df = pd.DataFrame(rows)
    # Drop rows where DateTime failed
    df = df.dropna(subset=["DateTime"])
    return df

# ----------------- PAGE CONTENT -----------------
with selected_tab[0]:
    st.title("📅 Forex Economic Calendar & News Sentiment")
    st.caption("Click a headline to view detailed summary and sentiment")

    df = get_gnews_forex_sentiment()

    if not df.empty:
        now = datetime.now()
        next_24h = now + timedelta(hours=24)

        # Show only today + next 24h
        default_df = df[(df["DateTime"] >= now) & (df["DateTime"] <= next_24h)]

        load_more = st.button("Load More Historical Data")

        if load_more:
            # Show all data sorted by most recent
            default_df = df.sort_values(by="DateTime", ascending=False)

        if default_df.empty:
            st.info("No forex news available for the next 24 hours.")
        else:
            default_df["HighProb"] = default_df.apply(
                lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] else "", axis=1
            )
            default_df_display = default_df.copy()
            default_df_display["Headline"] = default_df_display["HighProb"] + " " + default_df_display["Headline"]

            selected_headline = st.selectbox(
                "Select a headline for details", default_df_display["Headline"].tolist()
            )

            st.dataframe(
                default_df_display[["DateTime", "Currency", "Headline"]].sort_values(by="DateTime", ascending=False),
                use_container_width=True
            )

            selected_row = default_df_display[default_df_display["Headline"] == selected_headline].iloc[0]

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

            st.markdown("---")
            st.markdown("## 📈 Currency Sentiment Bias Table")
            bias_df = default_df.groupby("Currency")["Impact"].value_counts().unstack().fillna(0)
            st.dataframe(bias_df)

            st.markdown("## 🧭 Beginner-Friendly Trade Outlook")
            if "Bullish" in impact:
                st.info(f"🟢 Sentiment on **{base}** is bullish. Look for buying setups on H1/H4.")
            elif "Bearish" in impact:
                st.warning(f"🔴 Sentiment on **{base}** is bearish. Look for selling setups on H1/H4.")
            else:
                st.write("⚪ No strong directional sentiment detected right now.")
    else:
        st.info("No forex news available or API limit reached.")

with selected_tab[1]:
    st.title("👤 My Account")
    st.write("This is your account page. You can add user settings, subscription info, or API key management here.")
