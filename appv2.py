import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from bs4 import BeautifulSoup

st.set_page_config(page_title="Gold & Forex AI Dashboard", layout="wide")

# ----------------- SIDEBAR -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Gold", "Forex", "Forex Fundamentals"])

# ----------------- FUNCTIONS -----------------

# Sentiment from GNews or other provider can go here

def get_gnews_forex_sentiment():
    import re
    from textblob import TextBlob

    API_KEY = st.secrets["GNEWS_API_KEY"]
    url = f"https://gnews.io/api/v4/search?q=forex+OR+inflation+OR+interest+rate+OR+CPI+OR+GDP+OR+Fed+OR+ECB&lang=en&token={API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        st.error("GNews API error.")
        return pd.DataFrame()

    articles = response.json().get("articles", [])
    rows = []

    for article in articles:
        title = article.get("title", "")
        date = article.get("publishedAt", "")[:10]

        # Detect currency in title
        match = re.search(r'\b(USD|GBP|EUR|JPY|AUD|CAD|CHF|NZD)\b', title.upper())
        currency = match.group(1) if match else "Unknown"

        # Sentiment score
        sentiment_score = TextBlob(title).sentiment.polarity
        sentiment = "Bullish" if sentiment_score > 0.1 else "Bearish" if sentiment_score < -0.1 else "Neutral"

        rows.append({"Date": date, "Currency": currency, "Headline": title, "Sentiment": sentiment})

    df = pd.DataFrame(rows)
    return df

# ----------------- PAGE CONTENT -----------------

if page == "Gold":
    st.title("🟡 Gold News & Macro View")
    st.write("Coming soon: Gold sentiment & fundamentals")

elif page == "Forex":
    st.title("💱 Forex News & Macro View")
    st.write("Coming soon: Forex news sentiment and rate analysis")

elif page == "Forex Fundamentals":
    st.title("📰 Live Forex News Sentiment")
    st.caption("Data from [GNews.io](https://gnews.io) with basic NLP sentiment tagging")

    df = get_gnews_forex_sentiment()

    if not df.empty:
        currency_filter = st.selectbox("Filter by Currency", options=["All"] + sorted(df["Currency"].unique()))
        if currency_filter != "All":
            df = df[df["Currency"] == currency_filter]

        st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)
    else:
        st.info("No news data available or API limit reached.")
