import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob

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

# ----------------- PAGE CONTENT -----------------
elif page == "Forex Fundamentals":
    st.title("📰 Live Forex News Sentiment")
    st.caption("Click a headline to view detailed summary and sentiment")

    df = get_gnews_forex_sentiment()

    if not df.empty:
        # ---------------- Filter and Search ----------------
        currency_filter = st.selectbox(
            "Filter by Currency", options=["All"] + sorted(df["Currency"].unique())
        )
        if currency_filter != "All":
            df = df[df["Currency"] == currency_filter]

        keyword = st.text_input("Search headlines for keyword")
        if keyword:
            df = df[df["Headline"].str.contains(keyword, case=False)]

        # ---------------- High-Probability Highlights ----------------
        df["HighProb"] = df.apply(
            lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] 
            and pd.to_datetime(row["Date"]) >= pd.Timestamp.now() - pd.Timedelta(days=1)
            else "", axis=1
        )
        df_display = df.copy()
        df_display["Headline"] = df["HighProb"] + " " + df["Headline"]

        # ---------------- Display Headlines ----------------
        selected_headline = st.selectbox("Select a headline for details", df_display["Headline"].tolist())

        # Custom styling for dataframe
        st.markdown(
            """
            <style>
            .dataframe th, .dataframe td {
                padding: 12px 10px !important;
            }
            .dataframe tbody tr:hover {
                background-color: #f9f9f9;
            }
            </style>
            """, unsafe_allow_html=True
        )

        st.dataframe(df_display[["Date", "Currency", "Headline", "Impact"]].sort_values(by="Date", ascending=False), use_container_width=True)

        selected_row = df_display[df_display["Headline"] == selected_headline].iloc[0]

        # ---------------- Headline Details ----------------
        st.markdown("### 🧠 Summary")
        st.info(selected_row["Summary"])

        st.markdown("### 🔥 Impact Rating & Sentiment Score")
        impact = selected_row["Impact"]
        sentiment_score = TextBlob(selected_row["Headline"]).sentiment.polarity
        if "Bullish" in impact:
            st.success(f"{impact} ({sentiment_score:.2f})")
        elif "Bearish" in impact:
            st.error(f"{impact} ({sentiment_score:.2f})")
        else:
            st.warning(f"{impact} ({sentiment_score:.2f})")

        # ---------------- Timeframes ----------------
        st.markdown("### ⏱️ Timeframes Likely Affected")
        if "Significantly" in impact:
            timeframes = ["H4", "Daily"]
        elif impact in ["Bullish", "Bearish"]:
            timeframes = ["H1", "H4"]
        else:
            timeframes = ["H1"]
        st.write(", ".join(timeframes))

        # ---------------- Likely Affected Pairs ----------------
        st.markdown("### 💱 Likely Affected Currency Pairs")
        base = selected_row["Currency"]
        if base != "Unknown":
            pairs = [f"{base}/USD", f"EUR/{base}", f"{base}/JPY", f"{base}/CHF", f"{base}/CAD", f"{base}/NZD", f"{base}/AUD"]
            st.write(", ".join(pairs))
        else:
            st.write("Cannot determine affected pairs.")

        # ---------------- Bias Table ----------------
        st.markdown("---")
        st.markdown("## 📈 Currency Sentiment Bias Table")
        bias_df = df.groupby("Currency")["Impact"].value_counts().unstack().fillna(0)
        st.dataframe(bias_df)

        # ---------------- Beginner-Friendly Outlook ----------------
        st.markdown("## 🧭 Beginner-Friendly Trade Outlook")
        outlook_table = []
        for curr in df["Currency"].unique():
            major_impact = df[df["Currency"]==curr]["Impact"].value_counts().idxmax()
            outlook_table.append({"Currency": curr, "Sentiment": major_impact})
        outlook_df = pd.DataFrame(outlook_table)
        st.dataframe(outlook_df)

        # ---------------- CSV Download ----------------
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Headlines as CSV",
            data=csv,
            file_name="forex_news_sentiment.csv",
            mime="text/csv",
        )

    else:
        st.info("No forex news available or API limit reached.")
