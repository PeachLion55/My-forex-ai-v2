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

def get_eodhd_calendar():
    API_KEY = st.secrets["EODHD_API_KEY"]
    url = f"https://eodhd.com/api/economic-events?api_token={API_KEY}&from=2024-01-01&to=2025-12-31"
    
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if not data:
            st.warning("No economic calendar data found.")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.rename(columns={
            "event_date": "Date",
            "country": "Country",
            "event": "Event",
            "actual": "Actual",
            "previous": "Previous",
            "estimate": "Forecast"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        return df[["Date", "Event", "Country", "Actual", "Forecast", "Previous"]]
    
    else:
        st.error(f"Failed to fetch data from EODHD. Status: {response.status_code}")
        st.text(f"Response text: {response.text}")
        return pd.DataFrame()

# ----------------- PAGE CONTENT -----------------

if page == "Gold":
    st.title("🟡 Gold News & Macro View")
    st.write("Coming soon: Gold sentiment & fundamentals")

elif page == "Forex":
    st.title("💱 Forex News & Macro View")
    st.write("Coming soon: Forex news sentiment and rate analysis")

elif page == "Forex Fundamentals":
    st.title("📅 Forex Economic Calendar (via EODHD)")
    st.caption("Data from [eodhd.com](https://eodhd.com)")
    country = st.selectbox("Filter by Country", options=["All", "United States", "United Kingdom", "Germany", "Japan", "China", "Canada"])
    df = get_eodhd_calendar()
    if not df.empty:
        if country != "All":
            df = df[df["Country"] == country]
        st.dataframe(df.sort_values(by="Date"))
