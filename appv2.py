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

def get_finnhub_calendar():
    API_KEY = "YOUR_FINNHUB_API_KEY"
    url = f"https://finnhub.io/api/v1/calendar/economic?token={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("economicCalendar", [])
        if not data:
            st.warning("No economic calendar data found.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df = df.rename(columns={
            "date": "Date",
            "symbol": "Symbol",
            "country": "Country",
            "actual": "Actual",
            "forecast": "Forecast",
            "previous": "Previous",
            "impact": "Impact",
            "event": "Event"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "Event", "Country", "Impact", "Actual", "Forecast", "Previous"]]
    else:
        st.error("Failed to fetch data from Finnhub.")
        return pd.DataFrame()

# ----------------- PAGE CONTENT -----------------

if page == "Gold":
    st.title("🟡 Gold News & Macro View")
    st.write("Coming soon: Gold sentiment & fundamentals")

elif page == "Forex":
    st.title("💱 Forex News & Macro View")
    st.write("Coming soon: Forex news sentiment and rate analysis")

elif page == "Forex Fundamentals":
    st.title("📅 Forex Economic Calendar (via Finnhub)")
    st.caption("Data from [finnhub.io](https://finnhub.io)")
    country = st.selectbox("Filter by Country", options=["All", "US", "EU", "JP", "GB", "CN", "CA"])
    df = get_finnhub_calendar()
    if not df.empty:
        if country != "All":
            df = df[df["Country"] == country]
        st.dataframe(df.sort_values(by="Date"))