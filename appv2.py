import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]
from datetime import datetime, timedelta

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Forex AI Dashboard", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ----------------- HORIZONTAL NAVIGATION -----------------
tabs = ["Forex Fundamentals", "My Account"]
selected_tab = st.tabs(tabs)

# ----------------- CUSTOM CSS -----------------
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

def get_fxstreet_rss():
    """Fetch latest forex news from FXStreet RSS feed"""
    rss_url = "https://www.fxstreet.com/rss/news"
    df_list = pd.read_xml(rss_url, xpath="//item")
    df = df_list[['title', 'pubDate', 'description', 'link']]
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df.rename(columns={'title':'Headline', 'pubDate':'Date', 'description':'Summary', 'link':'URL'}, inplace=True)
    return df

def detect_currency(title):
    """Basic currency detection"""
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
        if any(kw in title_upper for kw in keywords):
            return curr
    return "Unknown"

def rate_impact(polarity):
    """Assign sentiment impact"""
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

def gpt_summary(text):
    """Generate a detailed summary using OpenAI GPT"""
    prompt = f"Summarize this forex news article in detailed paragraphs and bullet key points:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.5,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# ----------------- PAGE CONTENT -----------------

with selected_tab[0]:
    st.title("📅 Forex Economic Calendar & News Sentiment")
    st.caption("Click a headline to view detailed summary and GPT-enhanced insights")

    df = get_fxstreet_rss()
    if not df.empty:
        # Filter only today + next 24 hours
        now = datetime.utcnow()
        next_24h = now + timedelta(hours=24)
        df = df[(df['Date'] >= now) & (df['Date'] <= next_24h)]

        if df.empty:
            st.info("No news in the next 24 hours.")
        else:
            df['Currency'] = df['Headline'].apply(detect_currency)
            df['Polarity'] = df['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
            df['Impact'] = df['Polarity'].apply(rate_impact)

            currency_filter = st.selectbox("Filter by Currency", options=["All"] + sorted(df["Currency"].unique()))
            if currency_filter != "All":
                df = df[df["Currency"] == currency_filter]

            df_display = df.copy()
            df_display["HeadlineDisplay"] = df_display.apply(lambda row: f"{'🔥 ' if 'Significantly' in row['Impact'] else ''}{row['Headline']}", axis=1)

            selected_headline = st.selectbox("Select a headline for details", df_display["HeadlineDisplay"].tolist())
            selected_row = df_display[df_display["HeadlineDisplay"] == selected_headline].iloc[0]

            st.dataframe(df_display[['Date','Currency','Impact','Headline']].sort_values(by='Date', ascending=False), use_container_width=True)

            st.markdown("### 🧠 GPT Detailed Summary & Key Points")
            with st.spinner("Generating GPT summary..."):
                detailed_summary = gpt_summary(selected_row['Summary'])
                st.markdown(detailed_summary)

            st.markdown("### 🔥 Impact Rating")
            impact = selected_row["Impact"]
            if "Bullish" in impact:
                st.success(impact)
            elif "Bearish" in impact:
                st.error(impact)
            else:
                st.warning(impact)

with selected_tab[1]:
    st.title("👤 My Account")
    st.write("This is your account page. You can add user settings, subscription info, or API key management here.")
