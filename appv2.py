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

def parse_economic_calendar(raw_text):
    lines = raw_text.split('\n')
    dates, times, currencies, events, actuals, forecasts, previous, impacts = [], [], [], [], [], [], [], []
    current_date = ""

    for line in lines:
        if not line.strip():
            continue
        if re.match(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s\w+\s\d+', line.strip()):
            current_date = line.strip()
            continue
        parts = re.split(r'\t+|\s{2,}', line.strip())
        if len(parts) < 2:
            continue
        time = parts[0] if re.match(r'\d{1,2}:\d{2}[ap]m', parts[0]) else ""
        currency = parts[1] if time else parts[0]
        rest = parts[2:] if time else parts[1:]
        event = rest[0] if len(rest) > 0 else ""
        actual = rest[1] if len(rest) > 1 else ""
        forecast = rest[2] if len(rest) > 2 else ""
        previous = rest[3] if len(rest) > 3 else ""
        impact = rest[4] if len(rest) > 4 else ""
        dates.append(current_date)
        times.append(time)
        currencies.append(currency)
        events.append(event)
        actuals.append(actual)
        forecasts.append(forecast)
        previous.append(previous)
        impacts.append(impact)

    return pd.DataFrame({
        "Date": dates,
        "Time": times,
        "Currency": currencies,
        "Event": events,
        "Actual": actuals,
        "Forecast": forecasts,
        "Previous": previous,
        "Impact": impacts
    })

# ----------------- PAGE CONTENT -----------------
with selected_tab[0]:
    st.title("📅 Forex Economic Calendar & News Sentiment")
    st.caption("Click a headline or event to view details")

    # --- SECTION 1: Forex News ---
    st.subheader("📰 Forex News Sentiment")
    df_news = get_fxstreet_forex_news()
    if not df_news.empty:
        currency_filter = st.selectbox(
            "Filter by currency for news:", 
            options=["All"] + sorted(df_news["Currency"].unique())
        )
        if currency_filter != "All":
            df_news = df_news[df_news["Currency"] == currency_filter]

        df_news["HighProb"] = df_news.apply(
            lambda row: "🔥" if row["Impact"] in ["Significantly Bullish", "Significantly Bearish"] 
            and pd.to_datetime(row["Date"]) >= pd.Timestamp.now() - pd.Timedelta(days=1)
            else "", axis=1
        )
        df_display = df_news.copy()
        df_display["Headline"] = df_news["HighProb"] + " " + df_news["Headline"]

        selected_headline = st.selectbox("Select a headline for details", df_display["Headline"].tolist())
        selected_row = df_display[df_display["Headline"] == selected_headline].iloc[0]

        st.markdown(f"### [{selected_row['Headline']}]({selected_row['Link']})")
        st.write(f"**Published:** {selected_row['Date']}")
        summary_text = selected_row["Summary"]
        sentences = re.split(r'(?<=[.!?]) +', summary_text)
        bullet_points = "\n".join([f"- {s}" for s in sentences[:10]])
        st.info(bullet_points)

        impact = selected_row["Impact"]
        if "Bullish" in impact:
            st.success(impact)
        elif "Bearish" in impact:
            st.error(impact)
        else:
            st.warning(impact)

        st.markdown("### 💱 Likely Affected Currency Pairs")
        base = selected_row["Currency"]
        if base != "Unknown":
            pairs = [f"{base}/USD", f"EUR/{base}", f"{base}/JPY", f"{base}/CHF", f"{base}/CAD", f"{base}/NZD", f"{base}/AUD"]
            st.write(", ".join(pairs))
        else:
            st.write("Cannot determine affected pairs.")

    else:
        st.info("No forex news available at the moment.")

    # --- SECTION 2: Economic Calendar ---
    st.subheader("📆 Economic Calendar")
    raw_calendar = st.text_area(
        "Paste your Myfxbook or similar economic calendar text here",
        height=300
    )
    if raw_calendar:
        df_calendar = parse_economic_calendar(raw_calendar)
        currency_filter_cal = st.selectbox(
            "Filter economic events by currency:",
            options=["All"] + sorted(df_calendar["Currency"].unique())
        )
        if currency_filter_cal != "All":
            df_calendar = df_calendar[df_calendar["Currency"] == currency_filter_cal]

        st.dataframe(df_calendar, use_container_width=True)

        st.subheader("Event Details")
        for idx, row in df_calendar.iterrows():
            with st.expander(f"{row['Date']} {row['Time']} - {row['Currency']} - {row['Event']}"):
                st.write(f"**Currency:** {row['Currency']}")
                st.write(f"**Event:** {row['Event']}")
                st.write(f"**Actual:** {row['Actual']}")
                st.write(f"**Forecast:** {row['Forecast']}")
                st.write(f"**Previous:** {row['Previous']}")
                if row['Impact']:
                    st.write(f"**Impact:** {row['Impact']}")

with selected_tab[1]:
    st.title("👤 My Account")
    st.write("This is your account page. You can add user settings, subscription info, or API key management here.")
