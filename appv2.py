import streamlit as st
import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

st.set_page_config(page_title="Forex AI Dashboard", layout="wide")

# ----------------- HORIZONTAL NAVIGATION -----------------
tabs = ["Forex Fundamentals", "My Account"]
selected_tab = st.tabs(tabs)

# ----------------- CUSTOM CSS FOR TABS AND PADDING -----------------
st.markdown("""
<style>
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #FFD700 !important;
        color: black !important;
        font-weight: bold;
        padding: 15px 30px !important;
        border-radius: 8px;
        margin-right: 10px !important;
    }
    div[data-baseweb="tab-list"] button[aria-selected="false"] {
        background-color: #f0f0f0 !important;
        color: #555 !important;
        padding: 15px 30px !important;
        border-radius: 8px;
        margin-right: 10px !important;
    }
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

@st.cache_resource
def get_fxstreet_forex_news():
    RSS_URL = "https://www.fxstreet.com/rss/news"
    feed = feedparser.parse(RSS_URL)
    rows = []

    for entry in feed.entries:
        title = entry.title
        date = entry.published[:10] if hasattr(entry, "published") else ""
        currency = detect_currency(title)
        summary = entry.summary

        rows.append({
            "Date": date,
            "Currency": currency,
            "Headline": title,
            "Summary": summary,
            "Link": entry.link
        })
    return pd.DataFrame(rows)

# Initialize Hugging Face summarizer
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = get_summarizer()

def summarize_article(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # Extract main content text (may need adjustment per site structure)
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        if len(text) > 2000:  # Hugging Face models have a max token limit
            text = text[:2000]
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return "Summary unavailable."

# ----------------- PAGE CONTENT -----------------
with selected_tab[0]:
    st.title("📅 Forex Economic Calendar & News Sentiment")
    st.caption("Click a headline to view detailed summary")

    df = get_fxstreet_forex_news()

    if not df.empty:
        currency_filter = st.selectbox("Select a currency:", options=["All"] + sorted(df["Currency"].unique()))
        if currency_filter != "All":
            df = df[df["Currency"] == currency_filter]

        selected_headline = st.selectbox("Select a headline:", df["Headline"].tolist())
        selected_row = df[df["Headline"] == selected_headline].iloc[0]

        st.markdown(f"### [{selected_row['Headline']}]({selected_row['Link']})")
        st.write(f"**Published:** {selected_row['Date']}")
        
        # Blue box - original FXStreet summary
        st.markdown("### 🧠 Original FXStreet Summary")
        st.info(selected_row["Summary"])  # Blue box

        # Yellow box - AI-generated summary
        st.markdown("### 💛 AI-Generated Summary")
        ai_summary = summarize_article(selected_row["Link"])
        st.warning(ai_summary)  # Yellow box

with selected_tab[1]:
    st.title("👤 My Account")
    st.write("Account settings, subscription info, or API keys go here.")
