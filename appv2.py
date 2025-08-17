import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os, json

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="My Forex AI", layout="wide")

# Theme
THEME_PRIMARY = "#00aaff"
BACKGROUND_OPACITY = 0.12

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown(
    f"""
<style>
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
}}
@keyframes moveGrid {{
  0% {{ transform: translateY(0px); }}
  100% {{ transform: translateY(42px); }}
}}
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
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# NAVIGATION
# =========================================================
tabs = [
    "Forex Fundamentals",
    "Technical Analysis",
    "Tools",
    "My Account",
    "MT5 Stats Dashboard"
]
selected_tab = st.tabs(tabs)

# =========================================================
# TAB 1: FOREX FUNDAMENTALS
# =========================================================
with selected_tab[0]:
    st.title("🌍 Forex Fundamentals")
    st.caption("Stay ahead with core market drivers and daily updates.")

    st.markdown("### 📌 Economic Calendar")
    try:
        econ_df = pd.read_csv("economic_calendar.csv")
        if "Date" in econ_df.columns and "Time" in econ_df.columns:
            econ_df["Time"] = econ_df["Time"].fillna("00:00")
            econ_df["DateTime"] = pd.to_datetime(
                econ_df["Date"].astype(str) + " " + econ_df["Time"].astype(str),
                errors="coerce",
                dayfirst=True
            )
            econ_df = econ_df.dropna(subset=["DateTime"])
            st.dataframe(
                econ_df[["DateTime", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("⚠️ Economic calendar file is missing required columns.")
    except FileNotFoundError:
        st.error("❌ No economic calendar data found (economic_calendar.csv missing).")

    st.markdown("---")

    # Button to open "Understanding Forex Fundamentals" section
    if st.button("📖 Learn: Understanding Forex Fundamentals"):
        st.session_state.show_understanding = True
    else:
        st.session_state.show_understanding = st.session_state.get("show_understanding", False)

    if st.session_state.show_understanding:
        st.subheader("📖 Understanding Forex Fundamentals")
        with st.expander("Interest Rates & Central Banks"):
            st.write("""
- Central banks adjust rates to control inflation and growth.
- Higher rates tend to attract capital → stronger currency.
- Watch: FOMC (USD), ECB (EUR), BoE (GBP), BoJ (JPY), RBA (AUD), BoC (CAD), SNB (CHF), RBNZ (NZD).
            """)

        with st.expander("Inflation & Growth"):
            st.write("""
- Inflation (CPI/PPI) impacts real yields and policy expectations.
- Growth indicators (GDP, PMIs, employment) shift risk appetite and rate paths.
            """)

        with st.expander("Risk Sentiment & Commodities"):
            st.write("""
- Risk-on often lifts AUD/NZD; risk-off supports USD/JPY/CHF.
- Oil impacts CAD; gold sometimes correlates with AUD.
            """)

        with st.expander("How to Use the Economic Calendar"):
            st.write("""
1) Filter by the currency you trade.
2) Note forecast vs. actual.
3) Expect volatility around high-impact events; widen stops or reduce size.
            """)

# =========================================================
# TAB 2: TECHNICAL ANALYSIS
# =========================================================
with selected_tab[1]:
    st.title("📊 Technical Analysis")
    st.caption("Analyze charts, indicators, and correlations.")

    # Sub-tabs inside Technical Analysis
    ta_tabs = st.tabs(["Charts", "Indicators", "Currency Correlation"])

    # -------- Charts --------
    with ta_tabs[0]:
        st.subheader("Live Charts (placeholder)")
        st.write("You can integrate Plotly, mplfinance, or TradingView widgets here.")
        # Example: placeholder
        st.line_chart(pd.DataFrame({
            "EUR/USD": np.random.randn(50).cumsum(),
            "USD/JPY": np.random.randn(50).cumsum()
        }))

    # -------- Indicators --------
    with ta_tabs[1]:
        st.subheader("Common Indicators")
        st.write("""
- Moving Averages (SMA, EMA)
- RSI / MACD / Bollinger Bands
- Fibonacci Retracement
- Pivot Points
        """)
        st.info("You can add interactive selection and chart overlays later.")

    # -------- Currency Correlation --------
    with ta_tabs[2]:
        st.subheader("💹 Currency Correlation")
        try:
            corr_df = pd.read_csv("currency_correlation.csv", index_col=0)
            st.dataframe(corr_df, use_container_width=True)
        except FileNotFoundError:
            st.error("❌ currency_correlation.csv not found. Upload your correlation data.")

# =========================================================
# TAB 3: TOOLS
# =========================================================
with selected_tab[2]:
    st.title("🛠 Trading Tools")
    st.write("Quick access to calculators, risk management, and MT5 utilities.")

    st.markdown("### 📈 Risk Management Calculator")
    st.write("Placeholder for lot size, risk %, stop-loss calculation")

    st.markdown("### 🔄 MT5 Utilities")
    st.write("Connect to MT5 API, fetch live account data, automate trade signals")

# =========================================================
# TAB 4: MY ACCOUNT
# =========================================================
with selected_tab[3]:
    st.title("👤 My Account")
    
    # Session State Initialization
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
    
    if not st.session_state.logged_in:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Dummy authentication for demo
            if username == "demo" and password == "demo123":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Logged in as {username}")
            else:
                st.error("❌ Invalid credentials")
    else:
        st.subheader(f"Welcome, {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.success("Logged out successfully")

        # Account info placeholders
        st.markdown("### Account Details")
        st.write("""
        - Account Balance: $10,000  
        - Open Positions: 2  
        - Margin Level: 120%
        """)

        # MT5 Connection Status
        st.markdown("### MT5 Connection")
        mt5_connected = True  # Replace with actual MT5 API check
        if mt5_connected:
            st.success("✅ Connected to MT5")
        else:
            st.error("❌ Not connected to MT5")

        # Quick Stats
        st.markdown("### Trading Stats")
        stats = {
            "Total Trades": 50,
            "Winning Trades": 30,
            "Losing Trades": 20,
            "Win Rate": "60%",
            "Profit/Loss": "$1,500"
        }
        st.json(stats)

# =========================================================
# Session Info / Debug (optional)
# =========================================================
with st.expander("Session Info"):
    st.write(st.session_state)

# =========================================================
# TAB 5: DATA UPLOAD / DOWNLOAD
# =========================================================
with selected_tab[4]:
    st.title("📂 Data Upload / Download")

    # Upload CSV
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.success("✅ File loaded successfully")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    # Download CSV
    st.subheader("Download Sample CSV")
    import io
    import pandas as pd
    sample_data = pd.DataFrame({
        "Symbol": ["EURUSD", "GBPUSD", "USDJPY"],
        "Price": [1.08, 1.23, 109.5],
        "Volume": [1000, 500, 1200]
    })
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv_buffer.getvalue(),
        file_name="sample_data.csv",
        mime="text/csv"
    )

# =========================================================
# GLOBAL ERROR HANDLING
# =========================================================
try:
    pass  # Your main app logic already covered
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("© 2025 MyTradingApp. All rights reserved.")
