# =========================================
# Forex Pro — Streamlit App (Launch-Ready)
# =========================================
# Dependencies: streamlit, pandas, numpy, textblob, feedparser, plotly
# Run: streamlit run app.py

# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import os
import json
import hashlib
from typing import Dict, Any, List

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Forex Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===================== CONSTANTS ======================
DATA_DIR = "."
ACCOUNTS_FILE = os.path.join(DATA_DIR, "accounts.json")
APP_NAME = "Forex Pro"
THEME_PRIMARY = "#FFD700"
TV_HEIGHT = 820
BACKGROUND_OPACITY = 0.55

# ===================== GLOBAL CSS =====================
st.markdown(
    f"""
<style>
/* Futuristic dark background with animated grid */
.stApp {{
  background:
    radial-gradient(circle at 12% 18%, rgba(255,215,0,{BACKGROUND_OPACITY*0.20}) 0%, transparent 26%),
    radial-gradient(circle at 90% 30%, rgba(0,170,255,{BACKGROUND_OPACITY*0.14}) 0%, transparent 24%),
    linear-gradient(135deg, #0a0a0a 0%, #0b0b0b 100%);
}}
.stApp::before {{
  content: "";
  position: fixed; inset: 0;
  background-image:
    linear-gradient(90deg, rgba(255,255,255,{BACKGROUND_OPACITY*0.05}) 1px, transparent 1px),
    linear-gradient(0deg,  rgba(255,255,255,{BACKGROUND_OPACITY*0.05}) 1px, transparent 1px);
  background-size: 42px 42px, 42px 42px;
  animation: moveGrid 38s linear infinite;
  pointer-events: none; z-index: 0; opacity: 1;
}
@keyframes moveGrid {{
  0% {{ transform: translateY(0px); }}
  100% {{ transform: translateY(42px); }}
}}
.main, .block-container, .stTabs, .stMarkdown {{ position: relative; z-index: 1; }}

/* Header bar */
.header {{
  display: flex; align-items:center; justify-content: space-between;
  padding: 8px 14px; margin-bottom: 6px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  box-shadow: 0 6px 16px rgba(0,0,0,0.25);
}}
.app-title {{
  font-weight: 800; font-size: 22px; letter-spacing: 0.2px;
}}
.badge {{
  padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 12px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
}}

/* Tabs */
div[data-baseweb="tab-list"] button[aria-selected="true"] {{
  background-color: {THEME_PRIMARY} !important;
  color: black !important;
  font-weight: 800;
  padding: 12px 22px !important;
  border-radius: 12px;
  margin-right: 10px !important;
}}
div[data-baseweb="tab-list"] button[aria-selected="false"] {{
  background-color: #151515 !important;
  color: #cfcfcf !important;
  padding: 12px 22px !important;
  border-radius: 12px;
  margin-right: 10px !important;
  border: 1px solid #232323 !important;
}}

/* Cards */
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.25);
}}
.kpi {{
  display:flex; flex-direction:column; gap:4px; padding:12px 14px; border-radius:14px;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
}}
.kpi .label {{ font-size: 12px; opacity: .8; }}
.kpi .value {{ font-size: 20px; font-weight: 800; }}

a, .stMarkdown a {{ color: {THEME_PRIMARY}; text-decoration: none; }}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== UTILITIES =====================

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def ensure_accounts_file():
    if not os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "w") as f:
            json.dump({}, f)

def read_accounts() -> Dict[str, Any]:
    ensure_accounts_file()
    with open(ACCOUNTS_FILE, "r") as f:
        return json.load(f)

def write_accounts(data: Dict[str, Any]) -> None:
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_user() -> str | None:
    return st.session_state.get("logged_in_user")

def save_user_blob(username: str, key: str, value: Any):
    acc = read_accounts()
    user = acc.setdefault(username, {})
    user.setdefault("data", {})[key] = value
    write_accounts(acc)

def load_user_blob(username: str, key: str, default=None):
    acc = read_accounts()
    return acc.get(username, {}).get("data", {}).get(key, default)

@st.cache_data(ttl=600, show_spinner=False)
def get_fxstreet_forex_news() -> pd.DataFrame:
    RSS_URL = "https://www.fxstreet.com/rss/news"
    feed = feedparser.parse(RSS_URL)
    rows = []
    for entry in getattr(feed, "entries", []):
        title = entry.title
        published = getattr(entry, "published", "")
        date = published[:10] if published else ""
        currency = detect_currency(title)
        polarity = TextBlob(title).sentiment.polarity
        impact = rate_impact(polarity)
        summary = getattr(entry, "summary", "")
        rows.append({
            "Date": date,
            "Currency": currency,
            "Headline": title,
            "Polarity": polarity,
            "Impact": impact,
            "Summary": summary,
            "Link": entry.link
        })
    df = pd.DataFrame(rows, columns=["Date","Currency","Headline","Polarity","Impact","Summary","Link"])
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=3)
        df = df[df["Date"] >= cutoff]
    return df.reset_index(drop=True)

def detect_currency(title: str) -> str:
    t = title.upper()
    currency_map = {
        "USD": ["USD", "US ", " US:", "FED", "FEDERAL RESERVE", "AMERICA", "U.S."],
        "GBP": ["GBP", "UK", " BRITAIN", "BOE", "POUND", "STERLING"],
        "EUR": ["EUR", "EURO", "EUROZONE", "ECB"],
        "JPY": ["JPY", "JAPAN", "BOJ", "YEN"],
        "AUD": ["AUD", "AUSTRALIA", "RBA"],
        "CAD": ["CAD", "CANADA", "BOC"],
        "CHF": ["CHF", "SWITZERLAND", "SNB"],
        "NZD": ["NZD", "NEW ZEALAND", "RBNZ"],
    }
    for curr, kws in currency_map.items():
        for kw in kws:
            if kw in t:
                return curr
    return "Unknown"

def rate_impact(polarity: float) -> str:
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

# ---------- STATIC ECON CALENDAR (yours) ----------
econ_calendar_data = [
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP Price Index y/y", "Actual": "3.0%", "Forecast": "3.1%", "Previous": "3.3%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "00:50", "Currency": "JPY", "Event": "Prelim GDP q/q", "Actual": "0.3%", "Forecast": "0.1%", "Previous": "0.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "02:30", "Currency": "CNY", "Event": "New Home Prices m/m", "Actual": "-0.31%", "Forecast": "", "Previous": "-0.27%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Industrial Production y/y", "Actual": "5.7%", "Forecast": "6.0%", "Previous": "6.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Retail Sales y/y", "Actual": "3.7%", "Forecast": "4.6%", "Previous": "4.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Fixed Asset Investment ytd/y", "Actual": "1.6%", "Forecast": "2.7%", "Previous": "2.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "NBS Press Conference", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "03:00", "Currency": "CNY", "Event": "Unemployment Rate", "Actual": "5.2%", "Forecast": "5.1%", "Previous": "5.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "05:30", "Currency": "JPY", "Event": "Revised Industrial Production m/m", "Actual": "2.1%", "Forecast": "1.7%", "Previous": "1.7%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "French Bank Holiday", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "Italian Bank Holiday", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "All Day", "Currency": "EUR", "Event": "ECOFIN Meetings", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "CAD", "Event": "Manufacturing Sales m/m", "Actual": "0.3%", "Forecast": "0.4%", "Previous": "-1.5%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "CAD", "Event": "Wholesale Sales m/m", "Actual": "0.7%", "Forecast": "0.7%", "Previous": "0.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Core Retail Sales m/m", "Actual": "0.3%", "Forecast": "0.3%", "Previous": "0.8%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.6%", "Previous": "0.9%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Empire State Manufacturing Index", "Actual": "11.9", "Forecast": "-1.2", "Previous": "5.5", "Impact": ""},
    {"Date": "2025-08-15", "Time": "13:30", "Currency": "USD", "Event": "Import Prices m/m", "Actual": "0.4%", "Forecast": "0.1%", "Previous": "-0.1%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "14:15", "Currency": "USD", "Event": "Capacity Utilization Rate", "Actual": "77.5%", "Forecast": "77.6%", "Previous": "77.7%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "14:15", "Currency": "USD", "Event": "Industrial Production m/m", "Actual": "-0.1%", "Forecast": "0.0%", "Previous": "0.4%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Prelim UoM Consumer Sentiment", "Actual": "58.6", "Forecast": "61.9", "Previous": "61.7", "Impact": ""},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Prelim UoM Inflation Expectations", "Actual": "4.9%", "Forecast": "", "Previous": "4.5%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "15:00", "Currency": "USD", "Event": "Business Inventories m/m", "Actual": "0.2%", "Forecast": "0.2%", "Previous": "0.0%", "Impact": ""},
    {"Date": "2025-08-15", "Time": "21:00", "Currency": "USD", "Event": "TIC Long-Term Purchases", "Actual": "150.8B", "Forecast": "", "Previous": "266.8B", "Impact": ""},
    {"Date": "2025-08-16", "Time": "Tentative", "Currency": "USD", "Event": "President Trump Speaks", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-17", "Time": "23:30", "Currency": "NZD", "Event": "BusinessNZ Services Index", "Actual": "47.3", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-1.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "05:30", "Currency": "JPY", "Event": "Tertiary Industry Activity m/m", "Actual": "0.1%", "Forecast": "0.6%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "10:00", "Currency": "EUR", "Event": "Trade Balance", "Actual": "18.1B", "Forecast": "16.2B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "13:15", "Currency": "CAD", "Event": "Housing Starts", "Actual": "270K", "Forecast": "284K", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "13:30", "Currency": "CAD", "Event": "Foreign Securities Purchases", "Actual": "-4.75B", "Forecast": "-2.79B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "15:00", "Currency": "USD", "Event": "NAHB Housing Market Index", "Actual": "34", "Forecast": "33", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Input q/q", "Actual": "1.4%", "Forecast": "2.9%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-18", "Time": "23:45", "Currency": "NZD", "Event": "PPI Output q/q", "Actual": "1.0%", "Forecast": "2.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "01:30", "Currency": "AUD", "Event": "Westpac Consumer Sentiment", "Actual": "0.6%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "Tentative", "Currency": "CNY", "Event": "Foreign Direct Investment ytd/y", "Actual": "-15.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "09:00", "Currency": "EUR", "Event": "Current Account", "Actual": "33.4B", "Forecast": "32.3B", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "CPI m/m", "Actual": "0.4%", "Forecast": "0.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "Median CPI y/y", "Actual": "3.1%", "Forecast": "3.1%", "Previous": "", "Impact": ""},
    {"Date": "2025-08-19", "Time": "13:30", "Currency": "CAD", "Event": "BoC Business Outlook Survey", "Actual": "", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-20", "Time": "00:01", "Currency": "GBP", "Event": "Rightmove HPI m/m", "Actual": "-0.2%", "Forecast": "", "Previous": "", "Impact": ""},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "CPI y/y", "Actual": "2.5%", "Forecast": "2.6%", "Previous": "2.7%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "02:30", "Currency": "CNY", "Event": "PPI y/y", "Actual": "-3.3%", "Forecast": "-3.0%", "Previous": "-3.1%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "08:00", "Currency": "EUR", "Event": "German PPI m/m", "Actual": "0.2%", "Forecast": "0.1%", "Previous": "0.1%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "13:30", "Currency": "CAD", "Event": "Manufacturing Sales m/m", "Actual": "0.3%", "Forecast": "0.5%", "Previous": "-1.2%", "Impact": ""},
    {"Date": "2025-08-20", "Time": "14:30", "Currency": "USD", "Event": "Crude Oil Inventories", "Actual": "-5.3M", "Forecast": "-1.2M", "Previous": "-0.6M", "Impact": ""},
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Employment Change", "Actual": "36.1K", "Forecast": "30.0K", "Previous": "-10.0K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "00:30", "Currency": "AUD", "Event": "Unemployment Rate", "Actual": "3.6%", "Forecast": "3.7%", "Previous": "3.8%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI y/y", "Actual": "3.2%", "Forecast": "3.3%", "Previous": "3.0%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "08:30", "Currency": "EUR", "Event": "French Flash CPI m/m", "Actual": "0.3%", "Forecast": "0.4%", "Previous": "0.1%", "Impact": ""},
    {"Date": "2025-08-21", "Time": "14:00", "Currency": "EUR", "Event": "ECB Interest Rate Decision", "Actual": "0.50%", "Forecast": "0.50%", "Previous": "0.25%", "Impact": "High"},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Initial Jobless Claims", "Actual": "218K", "Forecast": "220K", "Previous": "217K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "14:30", "Currency": "USD", "Event": "Continuing Claims", "Actual": "1445K", "Forecast": "1450K", "Previous": "1440K", "Impact": ""},
    {"Date": "2025-08-21", "Time": "15:00", "Currency": "USD", "Event": "Existing Home Sales", "Actual": "4.25M", "Forecast": "4.23M", "Previous": "4.19M", "Impact": ""},
    {"Date": "2025-08-22", "Time": "09:30", "Currency": "GBP", "Event": "Retail Sales m/m", "Actual": "0.5%", "Forecast": "0.3%", "Previous": "0.2%", "Impact": "Medium"},
]
econ_df = pd.DataFrame(econ_calendar_data)
econ_df["DateTime"] = pd.to_datetime(econ_df["Date"] + " " + econ_df["Time"].replace("All Day","00:00"))
econ_df = econ_df.sort_values(["DateTime","Currency"])

# ===================== HEADER =====================
def render_header():
    user = get_user()
    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown(
            f"""
            <div class="header">
              <div class="app-title">💹 {APP_NAME}</div>
              <div class="badge">Live Macro • Sentiment • TA • Journals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        with st.container(border=True):
            now = datetime.utcnow().strftime("%a, %d %b %Y • %H:%M UTC")
            st.write("**Status**")
            st.caption(f"Updated: {now}")
            if user:
                st.write(f"Logged in as **{user}**")
            else:
                st.write("Not logged in")

render_header()

# ===================== NAVIGATION =====================
tabs = [
    "Dashboard",
    "Technical Analysis",
    "Tools",
    "MT5 Pro",
    "Learn",
    "My Account",
]
selected_tab = st.tabs(tabs)

# ===================== DASHBOARD (Fundamentals) =====================
with selected_tab[0]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("🗓️ Upcoming Economic Events")
        currencies = sorted(econ_df["Currency"].unique().tolist())
        c1, c2, c3 = st.columns(3)
        with c1:
            primary = st.selectbox("Primary currency", ["None"] + currencies, index=0, key="cal_primary")
        with c2:
            secondary = st.selectbox("Secondary currency", ["None"] + currencies, index=0, key="cal_secondary")
        with c3:
            days_ahead = st.slider("Days ahead", 1, 14, 7, key="cal_days")
        time_limit = pd.Timestamp.utcnow() + pd.Timedelta(days=days_ahead)
        view_df = econ_df[econ_df["DateTime"] <= time_limit].copy()

        def highlight_currency_row(row):
            style = [''] * len(row)
            if primary != "None" and row['Currency'] == primary:
                style = ['background-color:#171447; color:white' if col == 'Currency' else 'background-color:#171447' for col in row.index]
            if secondary != "None" and row['Currency'] == secondary:
                style = ['background-color:#471414; color:white' if col == 'Currency' else 'background-color:#471414' for col in row.index]
            return style

        st.dataframe(
            view_df[["Date","Time","Currency","Event","Actual","Forecast","Previous","Impact"]]
            .style.apply(highlight_currency_row, axis=1),
            use_container_width=True, height=360
        )

    with right:
        st.subheader("📰 Sentiment by Currency (last 72h)")
        df_news = get_fxstreet_forex_news()
        if df_news.empty:
            st.info("No recent headlines available.")
        else:
            agg = (
                df_news[df_news["Currency"] != "Unknown"]
                .groupby("Currency")["Polarity"]
                .agg(["count","mean"])
                .reset_index()
                .rename(columns={"count":"Headlines","mean":"AvgPolarity"})
                .sort_values("AvgPolarity", ascending=False)
            )
            k1, k2, k3 = st.columns(3)
            top = agg.head(1)
            bottom = agg.tail(1)
            with k1:
                st.markdown('<div class="kpi"><div class="label">Most Bullish</div><div class="value">'
                            f'{top["Currency"].iloc[0] if not top.empty else "—"}</div></div>', unsafe_allow_html=True)
            with k2:
                st.markdown('<div class="kpi"><div class="label">Most Bearish</div><div class="value">'
                            f'{bottom["Currency"].iloc[0] if len(bottom)>0 else "—"}</div></div>', unsafe_allow_html=True)
            with k3:
                st.markdown('<div class="kpi"><div class="label">Total Headlines</div><div class="value">'
                            f'{len(df_news)}</div></div>', unsafe_allow_html=True)
            fig_sent = px.bar(
                agg, x="Currency", y="AvgPolarity",
                hover_data=["Headlines"], title="Average Sentiment by Currency"
            )
            fig_sent.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_sent, use_container_width=True)

    # ---- Next high-impact window ----
    st.markdown("### ⏱ Next High-Impact Window")
    hi = econ_df[econ_df["Impact"].str.contains("High", case=False, na=False)].copy()
    if hi.empty:
        st.info("No high-impact events flagged in the static calendar provided.")
    else:
        hi["DateTime"] = pd.to_datetime(hi["Date"] + " " + hi["Time"].replace("All Day","00:00"))
        upcoming = hi[hi["DateTime"] >= pd.Timestamp.utcnow()].sort_values("DateTime").head(5)
        st.dataframe(upcoming[["Date","Time","Currency","Event","Impact"]], use_container_width=True)

    # ---- Interest rate tiles ----
    st.markdown("### 💹 Major Central Bank Policy Rates")
    rate_tiles = [
        {"Currency": "USD", "Current": "4.50%", "Previous": "4.75%", "Changed": "2024-12-18"},
        {"Currency": "GBP", "Current": "4.00%", "Previous": "4.25%", "Changed": "2025-08-07"},
        {"Currency": "EUR", "Current": "2.15%", "Previous": "2.40%", "Changed": "2025-06-05"},
        {"Currency": "JPY", "Current": "0.50%", "Previous": "0.25%", "Changed": "2025-01-24"},
        {"Currency": "AUD", "Current": "3.60%", "Previous": "3.85%", "Changed": "2025-08-12"},
        {"Currency": "CAD", "Current": "2.75%", "Previous": "3.00%", "Changed": "2025-03-12"},
        {"Currency": "NZD", "Current": "3.25%", "Previous": "3.50%", "Changed": "2025-05-28"},
        {"Currency": "CHF", "Current": "0.00%", "Previous": "0.25%", "Changed": "2025-06-19"},
    ]
    per_row = 4
    for i in range(0, len(rate_tiles), per_row):
        cols = st.columns(per_row)
        for j, r in enumerate(rate_tiles[i:i+per_row]):
            with cols[j]:
                st.markdown(
                    f"""
                    <div class="card">
                      <div style="background-color:{'#171447' if j%2==0 else '#471414'};border-radius:12px;padding:14px;color:white;text-align:center;">
                        <h3 style="margin:4px 0 8px 0;">{r['Currency']}</h3>
                        <div><b>Current:</b> {r['Current']}</div>
                        <div><b>Previous:</b> {r['Previous']}</div>
                        <div><b>Changed:</b> {r['Changed']}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ===================== TECHNICAL ANALYSIS =====================
with selected_tab[1]:
    st.subheader("📊 Technical Analysis")
    st.caption("TradingView live chart, watchlist, and pair-specific news.")

    pairs_map = {
        "EUR/USD": "FX:EURUSD",
        "USD/JPY": "FX:USDJPY",
        "GBP/USD": "FX:GBPUSD",
        "USD/CHF": "OANDA:USDCHF",
        "AUD/USD": "FX:AUDUSD",
        "NZD/USD": "OANDA:NZDUSD",
        "USD/CAD": "CMCMARKETS:USDCAD",
        "EUR/GBP": "FX:EURGBP",
    }
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
        watchlist = list(pairs_map.values())
        tv_symbol = pairs_map[pair]
        tv_html = f"""
        <div class="tradingview-widget-container" style="height:{TV_HEIGHT}px; width:100%">
          <div id="tradingview_chart" class="tradingview-widget-container__widget" style="height:{TV_HEIGHT}px; width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "60",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "hide_top_toolbar": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "save_image": true,
            "calendar": false,
            "studies": [],
            "watchlist": {watchlist}
          }}
          </script>
        </div>
        """
        components.html(tv_html, height=TV_HEIGHT+20, scrolling=False)

    with right:
        st.markdown("### 📰 Pair News & Sentiment")
        df_news = get_fxstreet_forex_news()
        if df_news.empty:
            st.info("News feed unavailable right now.")
        else:
            base, quote = pair.split("/")
            filtered_df = df_news[df_news["Currency"].isin([base, quote])].copy()
            try:
                filtered_df["HighProb"] = filtered_df.apply(
                    lambda row: "🔥" if (row["Impact"] in ["Significantly Bullish","Significantly Bearish"])
                                         and (row["Date"] >= pd.Timestamp.utcnow() - pd.Timedelta(days=1))
                    else "", axis=1
                )
            except Exception:
                filtered_df["HighProb"] = ""
            filtered_df["HeadlineDisplay"] = filtered_df["HighProb"] + " " + filtered_df["Headline"]
            if filtered_df.empty:
                st.info("No pair-specific headlines in the last 72h.")
            else:
                pick = st.selectbox("Select a headline", filtered_df["HeadlineDisplay"].tolist())
                row = filtered_df[filtered_df["HeadlineDisplay"] == pick].iloc[0]
                st.markdown(f"**[{row['Headline']}]({row['Link']})**")
                ts = row["Date"].strftime("%Y-%m-%d %H:%M UTC") if not pd.isna(row["Date"]) else "—"
                st.write(f"**Published:** {ts}")
                st.write(f"**Detected currency:** {row['Currency']} • **Impact:** {row['Impact']}")
                with st.expander("Summary"):
                    st.write(row["Summary"])

# ===================== TOOLS =====================
with selected_tab[2]:
    st.subheader("🛠 Tools")
    sub = st.tabs(["Position Sizing / Risk", "Quick P/L", "Backtesting Journal"])

    # ---- Position Sizing / Risk ----
    with sub[0]:
        st.markdown("**Compute optimal lot size given account, risk %, and stop distance.**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            account_ccy = st.selectbox("Account Currency", ["USD","EUR","GBP","JPY"], index=0)
        with c2:
            account_balance = st.number_input("Account Balance", min_value=0.0, value=10000.0, step=100.0)
        with c3:
            risk_percent = st.number_input("Risk % per Trade", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        with c4:
            pair_rr = st.selectbox("Pair", ["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD","EUR/GBP"], index=0, key="risk_pair")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            stop_pips = st.number_input("Stop Distance (pips)", min_value=1.0, value=20.0, step=0.1)
        with c6:
            tp_pips = st.number_input("Target (pips)", min_value=1.0, value=40.0, step=0.1)
        with c7:
            entry_price = st.number_input("Entry Price", min_value=0.0001, value=1.1000, step=0.0001, format="%.5f")
        with c8:
            direction = st.radio("Direction", ["Long","Short"], horizontal=True)

        risk_amount = account_balance * (risk_percent / 100.0)

        # Simplified pip value calculation:
        # For majors quoted in USD (xxx/USD): pip per lot = $10 per pip (non-JPY), for JPY: ~¥1000 per pip -> convert
        def pip_value_per_lot(pair: str, price: float, account_ccy: str) -> float:
            base, quote = pair.split("/")
            is_jpy = quote == "JPY"
            if quote == "USD":  # e.g., EUR/USD
                usd_per_pip_per_lot = 10.0 if not is_jpy else 1000.0 / price
                # If account is USD -> return; else naive FX to account ccy (demo uses 1.0)
                conv = 1.0
                return usd_per_pip_per_lot * conv
            elif base == "USD":  # USD/JPY etc.
                # pip value in USD per lot
                if is_jpy:
                    usd_per_pip_per_lot = 1000.0 / price
                else:
                    usd_per_pip_per_lot = 10.0
                conv = 1.0
                return usd_per_pip_per_lot * conv
            else:
                # Crosses (rough): treat like 10 quote-ccy units per pip per lot; no live conversion
                return 10.0

        pv = pip_value_per_lot(pair_rr, entry_price, account_ccy)
        lot_size = risk_amount / (stop_pips * pv) if stop_pips > 0 and pv > 0 else 0.0
        rr = (tp_pips / stop_pips) if stop_pips > 0 else 0.0
        exp_pl = (tp_pips * pv * lot_size)  # expected gross if TP hit
        max_loss = (stop_pips * pv * lot_size)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Risk Amount", f"{risk_amount:,.2f} {account_ccy}")
        k2.metric("Lot Size", f"{lot_size:.3f} lots")
        k3.metric("R:R", f"{rr:.2f}")
        k4.metric("TP P/L", f"{exp_pl:,.2f} {account_ccy}")

        st.caption("Note: pip value conversions for crosses are approximate without live FX rates.")

    # ---- Quick P/L ----
    with sub[1]:
        st.markdown("**Back-of-the-napkin P/L estimator.**")
        cp = st.selectbox("Currency Pair", ["EUR/USD","GBP/USD","USD/JPY"], key="pl_pair")
        acc = st.selectbox("Account Currency", ["USD","EUR","GBP","JPY"], key="pl_acc")
        lots = st.number_input("Position Size (lots)", min_value=0.01, value=0.10, step=0.01)
        opx = st.number_input("Open Price", value=1.1000, step=0.0001, format="%.5f")
        cpx = st.number_input("Close Price", value=1.1050, step=0.0001, format="%.5f")
        side = st.radio("Trade Direction", ["Long","Short"], horizontal=True)
        pip_mult = 100 if "JPY" in cp else 10000
        pip_move = (cpx - opx) * pip_mult
        if side == "Short":
            pip_move = -pip_move
        exch = 1.0
        pv = (0.01 if "JPY" in cp else 0.0001) / max(exch, 1e-9) * lots * 100000
        pl_val = pip_move * pv
        st.write(f"**Pip Movement**: {pip_move:.2f} pips")
        st.write(f"**Pip Value**: {pv:.2f} {acc}")
        st.write(f"**Potential P/L**: {pl_val:.2f} {acc}")

    # ---- Journal ----
    with sub[2]:
        st.markdown("**Lightweight strategy backtesting journal (editable, saved to your account).**")
        journal_cols = ["Date", "Symbol", "Direction", "Entry", "Exit", "Lots", "Notes"]
        user = get_user()

        # Load from session or account
        if "journal_df" not in st.session_state:
            if user:
                saved = load_user_blob(user, "journal", [])
                st.session_state.journal_df = pd.DataFrame(saved, columns=journal_cols) if saved else pd.DataFrame(columns=journal_cols)
            else:
                st.session_state.journal_df = pd.DataFrame(columns=journal_cols)

        updated = st.data_editor(
            data=st.session_state.journal_df,
            num_rows="dynamic",
            use_container_width=True,
            key="journal_editor"
        )
        st.session_state.journal_df = updated

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save to My Account"):
                if user:
                    save_user_blob(user, "journal", st.session_state.journal_df.to_dict(orient="records"))
                    st.success("Journal saved.")
                else:
                    st.info("Login required to save.")
        with c2:
            if st.button("📂 Reload from Account"):
                if user:
                    saved = load_user_blob(user, "journal", [])
                    st.session_state.journal_df = pd.DataFrame(saved, columns=journal_cols) if saved else pd.DataFrame(columns=journal_cols)
                    st.success("Journal loaded.")
                else:
                    st.info("Login required to load.")

# ===================== MT5 PRO =====================
with selected_tab[3]:
    st.subheader("📊 MT5 Pro Stats Dashboard")
    st.markdown("Upload your MetaTrader 5 trade history CSV file to get advanced trading insights.")
    up = st.file_uploader("Upload MT5 Trade History CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        if "profit" not in df.columns or "time" not in df.columns:
            st.error("CSV must contain at least 'time' and 'profit' columns.")
        else:
            df["Profit"] = df["profit"]
            df["Symbol"] = df["symbol"] if "symbol" in df.columns else "N/A"
            df["Type"] = df["type"] if "type" in df.columns else "N/A"
            df["Date"] = pd.to_datetime(df["time"], errors="coerce")
            df["Hour"] = df["Date"].dt.hour
            df["Day"] = df["Date"].dt.day_name()

            st.markdown("### 📈 Key Metrics")
            total = len(df)
            wins = df[df["Profit"] > 0]
            losses = df[df["Profit"] < 0]
            win_rate = (len(wins)/total*100) if total else 0
            avg_trade = df["Profit"].mean() if total else 0
            largest_win = df["Profit"].max() if total else 0
            largest_loss = df["Profit"].min() if total else 0
            profit_factor = (wins["Profit"].sum() / abs(losses["Profit"].sum())) if len(losses) else np.nan

            # streaks
            df_sorted = df.sort_values("Date")
            streak = 0; max_win_streak = 0; max_loss_streak = 0; last = None
            for p in df_sorted["Profit"]:
                if p > 0:
                    streak = streak + 1 if (last is True or last is None) else 1
                    max_win_streak = max(max_win_streak, streak); last = True
                elif p < 0:
                    streak = streak + 1 if (last is False or last is None) else 1
                    max_loss_streak = max(max_loss_streak, streak); last = False

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Trades", f"{total}")
            m2.metric("Win Rate", f"{win_rate:.1f}%")
            m3.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "—")
            m4.metric("Avg Trade P/L", f"{avg_trade:.2f}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Total Profit", f"{df['Profit'].sum():.2f}")
            m6.metric("Largest Win", f"{largest_win:.2f}")
            m7.metric("Largest Loss", f"{largest_loss:.2f}")
            m8.metric("Max Win Streak", max_win_streak)

            # Cumulative & drawdown
            st.markdown("### 📉 Cumulative P/L & Drawdowns")
            df_sorted["CumPL"] = df_sorted["Profit"].cumsum()
            df_sorted["DD"] = df_sorted["CumPL"].cummax() - df_sorted["CumPL"]
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=df_sorted["Date"], y=df_sorted["CumPL"], mode="lines", name="Cumulative P/L"))
            fig_cum.add_trace(go.Scatter(x=df_sorted["Date"], y=df_sorted["DD"], mode="lines", name="Drawdown", line=dict(dash="dot")))
            fig_cum.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_cum, use_container_width=True)

            # Distribution
            st.markdown("### 📊 Profit Distribution")
            fig_dist = px.histogram(df, x="Profit", nbins=40, title="Profit Distribution")
            fig_dist.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_dist, use_container_width=True)

            # Symbol / Type
            st.markdown("### 🧩 Trades by Symbol / Type")
            sym_counts = df["Symbol"].value_counts().reset_index()
            sym_counts.columns = ["Symbol","Trades"]
            fig_sym = px.bar(sym_counts, x="Symbol", y="Trades", text="Trades", title="Trades per Symbol")
            st.plotly_chart(fig_sym, use_container_width=True, height=300)

            type_counts = df["Type"].value_counts().reset_index()
            type_counts.columns = ["Type","Count"]
            fig_type = px.pie(type_counts, names="Type", values="Count", title="Buy vs Sell")
            st.plotly_chart(fig_type, use_container_width=True, height=300)

            # Time-based
            st.markdown("### ⏱ Profit by Hour & Day")
            fig_hour = px.bar(df.groupby("Hour")["Profit"].sum().reset_index(), x="Hour", y="Profit", title="Profit by Hour")
            st.plotly_chart(fig_hour, use_container_width=True, height=300)
            fig_day = px.bar(
                df.groupby("Day")["Profit"].sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index(),
                x="Day", y="Profit", title="Profit by Day"
            )
            st.plotly_chart(fig_day, use_container_width=True, height=300)

            st.markdown("### 📥 Download Processed CSV")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "mt5_trades_processed.csv", "text/csv")

# ===================== LEARN (Understanding Forex Fundamentals) =====================
with selected_tab[4]:
    st.subheader("📖 Understanding Forex Fundamentals")
    st.caption("Core drivers of currencies, explained simply.")

    with st.expander("Interest Rates & Central Banks", expanded=True):
        st.write("""
- Central banks adjust rates to control inflation and growth.
- Higher nominal/real rates tend to attract capital → stronger currency.
- Watch: FOMC (USD), ECB (EUR), BoE (GBP), BoJ (JPY), RBA (AUD), BoC (CAD), SNB (CHF), RBNZ (NZD).
        """)

    with st.expander("Inflation & Growth"):
        st.write("""
- Inflation (CPI/PPI) steers policy via real yields and expectations.
- Growth (GDP, PMIs, employment) shifts risk appetite and future rate paths.
        """)

    with st.expander("Risk Sentiment & Commodities"):
        st.write("""
- Risk-on often lifts AUD/NZD; risk-off supports USD/JPY/CHF.
- Oil impacts CAD; gold sometimes aligns with AUD risk cycles.
        """)

    with st.expander("How to Use the Economic Calendar"):
        st.write("""
1) Filter to currencies you trade.
2) Track forecast vs. actual for surprises.
3) Expect volatility around high-impact events; reduce size or widen stops.
        """)

# ===================== ACCOUNT =====================
with selected_tab[5]:
    st.subheader("👤 My Account")
    ensure_accounts_file()

    # Auth widgets
    login, signup = st.columns(2)
    with login:
        st.markdown("**Login**")
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Login"):
            acc = read_accounts()
            if u in acc and acc[u].get("password_hash") == sha256(p):
                st.session_state.logged_in_user = u
                st.success(f"Welcome back, {u}!")
            else:
                st.error("Invalid credentials.")

    with signup:
        st.markdown("**Sign Up**")
        nu = st.text_input("New Username", key="signup_u")
        npw = st.text_input("New Password", type="password", key="signup_p")
        if st.button("Create Account"):
            acc = read_accounts()
            if nu in acc:
                st.error("Username already exists.")
            elif not nu or not npw:
                st.error("Please enter a username and password.")
            else:
                acc[nu] = {"password_hash": sha256(npw), "data": {}}
                write_accounts(acc)
                st.success(f"Account created for {nu}.")

    # Preferences
    if get_user():
        st.markdown("---")
        st.subheader("Profile Preferences")
        pref = load_user_blob(get_user(), "prefs", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            name = st.text_input("Name", value=pref.get("name",""))
        with c2:
            base_ccy = st.selectbox("Base Currency", ["USD","EUR","GBP","JPY","AUD","CAD","NZD","CHF"], index=0)
        with c3:
            email = st.text_input("Email", value=pref.get("email",""))
        with c4:
            alerts = st.checkbox("Email me before high-impact events", value=pref.get("alerts", True))
        if st.button("Save Preferences"):
            save_user_blob(get_user(), "prefs", {"name": name, "email": email, "base": base_ccy, "alerts": alerts})
            st.success("Preferences saved.")

# ===================== FOOTER =====================
st.caption("© 2025 Forex Pro — For educational purposes only. This is not financial advice.")
