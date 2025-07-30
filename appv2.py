import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob

st.set_page_config(page_title="Gold & Forex AI Dashboard", layout="wide")

# ----------------- SIDEBAR -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Gold", "Forex", "Forex Fundamentals"])

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

    return pd.DataFrame(rows)

def get_gold_news_sentiment():
    API_KEY = st.secrets["GNEWS_API_KEY"]
    url = f"https://gnews.io/api/v4/search?q=gold+OR+gold+price+OR+precious+metals+OR+inflation+OR+interest+rates+OR+Fed&lang=en&token={API_KEY}"

    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"GNews API error (Gold): {response.status_code}")
        return pd.DataFrame()

    articles = response.json().get("articles", [])
    rows = []

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
        sentiment_score = TextBlob(title).sentiment.polarity
        impact = rate_impact(sentiment_score)
        summary = article.get("description", "") or title.split(":")[-1].strip()

        rows.append({
            "Date": date,
            "Headline": title,
            "Impact": impact,
            "Summary": summary
        })

    return pd.DataFrame(rows)

# ----------------- PAGE CONTENT -----------------

if page == "Gold":
    st.title("🟡 Gold News & Sentiment Dashboard")
    st.caption("Gold-related macro news from GNews.io")

    df = get_gold_news_sentiment()

    if not df.empty:
        # Flag high-probability events
        df["HighProb"] = df.apply(
            lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] and pd.to_datetime(row["Date"]) >= pd.Timestamp.now() - pd.Timedelta(days=1)
            else "", axis=1
        )

        df_display = df.copy()
        df_display["Headline"] = df["HighProb"] + " " + df["Headline"]

        selected_headline = st.selectbox("Select a headline for details", df_display["Headline"].tolist())

        st.dataframe(df_display[["Date", "Headline"]].sort_values(by="Date", ascending=False), use_container_width=True)

        selected_row = df_display[df_display["Headline"] == selected_headline].iloc[0]

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

        st.markdown("### 🪙 Affected Gold Markets")
        st.write("XAU/USD, XAU/EUR, XAU/JPY, XAU/GBP")

        st.markdown("---")
        st.markdown("## 🧭 Beginner-Friendly Trade Outlook")
        if "Bullish" in impact:
            st.info("🟢 Based on the news sentiment, **gold is expected to rise**. Traders may look for **buying opportunities** on higher timeframes such as H4 and Daily.")
        elif "Bearish" in impact:
            st.warning("🔴 The sentiment suggests **bearish pressure on gold**. Traders should be cautious and may consider **selling setups** on H1 or H4 timeframes.")
        else:
            st.write("⚪ Current news sentiment on gold is **neutral**. No strong direction is expected in the short term.")

    else:
        st.info("No gold news available or API limit reached.")

elif page == "Forex":
    st.title("💱 Forex News & Macro View")
    st.write("Coming soon: Forex news sentiment and rate analysis")

elif page == "Forex Fundamentals":
    st.title("📰 Live Forex News Sentiment")
    st.caption("Click a headline to view detailed summary and sentiment")

    df = get_gnews_forex_sentiment()

    if not df.empty:
        # Currency filter
        currency_filter = st.selectbox("Filter by Currency", options=["All"] + sorted(df["Currency"].unique()))
        if currency_filter != "All":
            df = df[df["Currency"] == currency_filter]

        # Flag high-probability headlines
        df["HighProb"] = df.apply(
            lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] and pd.to_datetime(row["Date"]) >= pd.Timestamp.now() - pd.Timedelta(days=1)
            else "", axis=1
        )

        df_display = df.copy()
        df_display["Headline"] = df["HighProb"] + " " + df["Headline"]

        selected_headline = st.selectbox("Select a headline for details", df_display["Headline"].tolist())

        st.dataframe(df_display[["Date", "Currency", "Headline"]].sort_values(by="Date", ascending=False), use_container_width=True)

        selected_row = df_display[df_display["Headline"] == selected_headline].iloc[0]

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

        bias_df = df.groupby("Currency")["Impact"].value_counts().unstack().fillna(0)
        st.dataframe(bias_df)

        st.markdown("## 🧭 Beginner-Friendly Trade Outlook")
        if "Bullish" in impact:
            st.info(f"🟢 Sentiment on **{base}** is bullish. Look for buying setups on H1/H4, especially in pairs like {base}/USD or {base}/JPY.")
        elif "Bearish" in impact:
            st.warning(f"🔴 Sentiment on **{base}** is bearish. Look for selling opportunities on H1/H4 in pairs like {base}/USD or EUR/{base}.")
        else:
            st.write("⚪ No strong directional sentiment detected right now.")

    else:
        st.info("No forex news available or API limit reached.")

        # 🕒 Timeframe Impact
        st.markdown("### ⏱️ Timeframes Likely Affected")
        if "Significantly" in impact:
            timeframes = ["H4", "Daily"]
        elif impact in ["Bullish", "Bearish"]:
            timeframes = ["H1", "H4"]
        else:
            timeframes = ["H1"]
        st.write(", ".join(timeframes))

        # 💱 Affected Forex Pairs
        st.markdown("### 💱 Likely Affected Currency Pairs")
        base = selected_row["Currency"]
        if base != "Unknown":
            pairs = [f"{base}/USD", f"EUR/{base}", f"{base}/JPY", f"{base}/CHF", f"{base}/CAD", f"{base}/NZD", f"{base}/AUD"]
            st.write(", ".join(pairs))
        else:
            st.write("Cannot determine affected pairs.")
