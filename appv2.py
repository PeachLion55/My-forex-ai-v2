# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
import datetime

# ================== UTILITY FUNCTIONS ==================
@st.cache_data
def get_fxstreet_forex_news():
    """Fetch latest Forex news from FXStreet RSS feed."""
    feed_url = "https://www.fxstreet.com/rss/news"
    feed = feedparser.parse(feed_url)
    items = []
    for entry in feed.entries:
        currency = "N/A"
        impact = "N/A"
        if hasattr(entry, "tags"):
            tags = [tag.term for tag in entry.tags]
            currency = tags[0] if tags else "N/A"
        items.append({
            "Headline": entry.title,
            "Link": entry.link,
            "Date": pd.to_datetime(entry.published),
            "Currency": currency,
            "Impact": impact
        })
    return pd.DataFrame(items)

@st.cache_data
def get_economic_calendar():
    """Placeholder for future economic calendar data."""
    # Can be replaced with real API later
    now = pd.Timestamp.now()
    return pd.DataFrame({
        "Date": [now + pd.Timedelta(days=i) for i in range(5)],
        "Event": ["GDP Release", "CPI Data", "Interest Rate", "Employment Data", "Retail Sales"],
        "Currency": ["USD", "EUR", "GBP", "JPY", "AUD"],
        "Impact": ["High", "Medium", "High", "Low", "Medium"]
    })

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Forex AI Dashboard", layout="wide")

tabs = ["Forex Fundamentals", "Technical Analysis"]
selected_tab = st.tabs(tabs)

# ================== TAB 1: FOREX FUNDAMENTALS ==================
with selected_tab[0]:
    st.title("📅 Forex Fundamentals")
    st.caption("Economic news, sentiment, and live rates.")

    # --- News Feed ---
    news_df = get_fxstreet_forex_news()
    if not news_df.empty:
        currency_filter = st.selectbox(
            "Filter news by currency (optional)",
            options=["All"] + sorted(news_df["Currency"].unique())
        )
        filtered_news = news_df.copy()
        if currency_filter != "All":
            filtered_news = filtered_news[filtered_news["Currency"] == currency_filter]

        if not filtered_news.empty:
            st.dataframe(filtered_news[["Date", "Currency", "Headline", "Link"]])
        else:
            st.info("No headlines for this filter.")
    else:
        st.info("No Forex news available.")

    # --- Economic Calendar ---
    st.subheader("📅 Economic Calendar (Sample)")
    calendar_df = get_economic_calendar()
    st.dataframe(calendar_df)

# ================== TAB 2: TECHNICAL ANALYSIS ==================
with selected_tab[1]:
    st.title("📊 Technical Analysis")
    st.caption("Live TradingView chart and Forex headlines.")

    # --- TradingView Widget ---
    tradingview_widget = """
    <div class="tradingview-widget-container" style="height:900px; width:100%">
      <div class="tradingview-widget-container__widget"></div>
      <div class="tradingview-widget-copyright">
        <a href="https://www.tradingview.com/symbols/CMCMARKETS-USDCAD/?exchange=CMCMARKETS" rel="noopener" target="_blank">
          <span class="blue-text">USDCAD chart by TradingView</span>
        </a>
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {
        "allow_symbol_change": true,
        "calendar": false,
        "details": false,
        "hide_side_toolbar": false,
        "hide_top_toolbar": false,
        "hide_legend": true,
        "hide_volume": true,
        "hotlist": false,
        "interval": "D",
        "locale": "en",
        "save_image": true,
        "style": "1",
        "symbol": "CMCMARKETS:USDCAD",
        "theme": "dark",
        "timezone": "Europe/London",
        "backgroundColor": "#0F0F0F",
        "gridColor": "rgba(242, 242, 242, 0.06)",
        "watchlist": [
          "FX:EURUSD",
          "FX:USDJPY",
          "FX:GBPUSD",
          "OANDA:USDCHF",
          "FX:AUDUSD",
          "OANDA:NZDUSD"
        ],
        "withdateranges": false,
        "compareSymbols": [],
        "studies": [
          "STD;Divergence%1Indicator"
        ],
        "autosize": true
      }
      </script>
    </div>
    """
    components.html(tradingview_widget, height=900, width=1000)

    # --- Forex Headlines Below Widget ---
    st.subheader("📢 Forex Headlines")
    headlines_df = get_fxstreet_forex_news()
    if not headlines_df.empty:
        currency_filter = st.selectbox(
            "Filter headlines by currency (optional)",
            options=["All"] + sorted(headlines_df["Currency"].unique()),
            key="ta_currency_filter"
        )
        filtered_headlines = headlines_df.copy()
        if currency_filter != "All":
            filtered_headlines = filtered_headlines[filtered_headlines["Currency"] == currency_filter]

        if not filtered_headlines.empty:
            selected_headline = st.selectbox(
                "Select a headline for details",
                filtered_headlines["Headline"].tolist(),
                key="ta_headline_select"
            )
            selected_row = filtered_headlines[filtered_headlines["Headline"] == selected_headline].iloc[0]
            st.markdown(f"### [{selected_row['Headline']}]({selected_row['Link']})")
            st.write(f"**Published:** {selected_row['Date']}")
            if "Impact" in selected_row:
                st.write(f"**Impact:** {selected_row['Impact']}")
        else:
            st.info("No headlines for this filter.")
    else:
        st.info("No Forex news available.")

# ================== END OF APP ==================
