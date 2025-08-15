import streamlit as st
import pandas as pd
from textblob import TextBlob
import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

st.set_page_config(page_title="Forex AI Dashboard", layout="wide")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

def fetch_full_article(url):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Get main text; adjust the selector if needed
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print("Error fetching article:", e)
        return None

def get_gpt_summary(article_text):
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=f"Summarize this news in detailed paragraphs and key points:\n\n{article_text}",
            max_output_tokens=400
        )
        return response.output_text
    except Exception as e:
        print("GPT summary error:", e)
        return None

# ----------------- PAGE CONTENT -----------------
tabs = ["Forex Fundamentals", "My Account"]
selected_tab = st.tabs(tabs)

with selected_tab[0]:
    st.title("📅 Forex Economic Calendar & News Sentiment")
    st.caption("Click a headline to view detailed summary and sentiment")

    df = get_fxstreet_forex_news()

    if not df.empty:
        currency_filter = st.selectbox("What currency pair would you like to track?", options=["All"] + sorted(df["Currency"].unique()))
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
        selected_row = df_display[df_display["Headline"] == selected_headline].iloc[0]

        st.markdown(f"### [{selected_row['Headline']}]({selected_row['Link']})")
        st.write(f"**Published:** {selected_row['Date']}")

        st.markdown("### 🧠 Original Summary")
        st.info(selected_row["Summary"])  # Blue box

        # ----------------- GPT Summary -----------------
        st.markdown("### 🟡 GPT Summary")
        full_article = fetch_full_article(selected_row["Link"])
        if full_article:
            gpt_summary = get_gpt_summary(full_article)
            if gpt_summary:
                st.warning(gpt_summary)  # Yellow box
            else:
                st.warning("GPT summary unavailable, showing original text.")
        else:
            st.warning("Could not fetch full article to summarize.")

with selected_tab[1]:
    st.title("👤 My Account")
    st.write("This is your account page. You can add user settings, subscription info, or API key management here.")
