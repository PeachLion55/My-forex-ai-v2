import streamlit as st

# Set page config
st.set_page_config(
    page_title="Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the sidebar
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide default streamlit elements */
    .css-1rs6os, .css-17ziqus {
        visibility: hidden;
    }
    
    .css-1v0mbdj {
        margin-top: -75px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2b2b2b 0%, #1a1a1a 100%) !important;
        width: 280px !important;
        padding-top: 0rem !important;
    }
    
    .css-1cypcdb {
        background: linear-gradient(180deg, #2b2b2b 0%, #1a1a1a 100%) !important;
        padding: 1rem 1.5rem !important;
    }
    
    /* Logo section */
    .logo-section {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
        border-bottom: 1px solid #333;
        margin-bottom: 1.5rem;
    }
    
    .logo-text {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ff6b35;
        text-decoration: none;
        display: inline-block;
    }
    
    /* Navigation buttons */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #ff6b35 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        text-align: left !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        height: 50px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff7a45 0%, #ff8c2e 50%, #ff7a45 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(255, 107, 53, 0.3) !important;
        color: white !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Add Trade button (gray) */
    .add-trade-btn > button {
        background: linear-gradient(135deg, #666 0%, #555 50%, #666 100%) !important;
        color: white !important;
    }
    
    .add-trade-btn > button:hover {
        background: linear-gradient(135deg, #777 0%, #666 50%, #777 100%) !important;
        color: white !important;
    }
    
    /* Strategy section */
    .strategy-section {
        margin: 2rem 0 1.5rem 0;
        padding-top: 1.5rem;
        border-top: 1px solid #333;
    }
    
    .strategy-label {
        color: #ccc;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox > div > div > div {
        background: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
        border-radius: 5px !important;
    }
    
    .stSelectbox > div > div > div > div {
        color: white !important;
    }
    
    /* Bottom section */
    .bottom-section {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #333;
    }
    
    /* Main content area */
    .main-content {
        background: #f8f9fa;
        min-height: 100vh;
        padding: 2rem;
    }
    
    /* Hide streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling for selectbox dropdown */
    div[data-baseweb="select"] > div {
        background-color: #333;
        border: 1px solid #555;
    }
    
    /* Page title styling */
    .page-title {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #333;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #ff6b35;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    .metric-change {
        font-size: 0.8rem;
        font-weight: 600;
        color: #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'DASHBOARD'

# Sidebar content
with st.sidebar:
    # Logo section
    st.markdown("""
        <div class="logo-section">
            <span class="logo-text">TD</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    if st.button("📊 DASHBOARD", key="dashboard_btn"):
        st.session_state.active_page = 'DASHBOARD'
    
    if st.button("💹 MARKETS", key="markets_btn"):
        st.session_state.active_page = 'MARKETS'
    
    if st.button("📅 CALENDAR", key="calendar_btn"):
        st.session_state.active_page = 'CALENDAR'
    
    if st.button("📈 ANALYTICS", key="analytics_btn"):
        st.session_state.active_page = 'ANALYTICS'
    
    if st.button("🧮 CALCULATOR", key="calculator_btn"):
        st.session_state.active_page = 'CALCULATOR'
    
    if st.button("🤖 MENTAI", key="mentai_btn"):
        st.session_state.active_page = 'MENTAI'
    
    if st.button("⚡ BACKTEST", key="backtest_btn"):
        st.session_state.active_page = 'BACKTEST'
    
    if st.button("🔄 TRADES", key="trades_btn"):
        st.session_state.active_page = 'TRADES'
    
    # Add Trade button (with special styling)
    st.markdown('<div class="add-trade-btn">', unsafe_allow_html=True)
    if st.button("➕ ADD TRADE", key="add_trade_btn"):
        st.session_state.active_page = 'ADD_TRADE'
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy section
    st.markdown("""
        <div class="strategy-section">
            <div class="strategy-label">Strategy 1</div>
        </div>
    """, unsafe_allow_html=True)
    
    strategy_options = ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4"]
    selected_strategy = st.selectbox("", strategy_options, key="strategy_select", label_visibility="collapsed")
    
    # Bottom section
    st.markdown('<div class="bottom-section">', unsafe_allow_html=True)
    
    if st.button("⚙️ SETTINGS", key="settings_btn"):
        st.session_state.active_page = 'SETTINGS'
    
    if st.button("🚪 LOGOUT", key="logout_btn"):
        st.session_state.active_page = 'LOGOUT'
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Display content based on active page
if st.session_state.active_page == 'DASHBOARD':
    st.markdown('<h1 class="page-title">📊 Dashboard</h1>', unsafe_allow_html=True)
    
    # Dashboard metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">$12,345</div>
                <div class="metric-label">Total Profit</div>
                <div class="metric-change">+5.2% this month</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">68%</div>
                <div class="metric-label">Win Rate</div>
                <div class="metric-change">+2.1% this week</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">7</div>
                <div class="metric-label">Active Trades</div>
                <div class="metric-change">-1 from yesterday</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("Welcome to your trading dashboard! Here you can monitor your portfolio performance, track active positions, and analyze your trading statistics.")

elif st.session_state.active_page == 'MARKETS':
    st.markdown('<h1 class="page-title">💹 Markets</h1>', unsafe_allow_html=True)
    st.write("Market analysis and real-time data")
    st.info("This section will display market data, charts, and analysis tools.")

elif st.session_state.active_page == 'CALENDAR':
    st.markdown('<h1 class="page-title">📅 Calendar</h1>', unsafe_allow_html=True)
    st.write("Economic calendar and important events")
    st.info("View upcoming economic events, earnings releases, and market-moving announcements.")

elif st.session_state.active_page == 'ANALYTICS':
    st.markdown('<h1 class="page-title">📈 Analytics</h1>', unsafe_allow_html=True)
    st.write("Trading analytics and performance metrics")
    st.info("Detailed performance analysis, risk metrics, and trading statistics.")

elif st.session_state.active_page == 'CALCULATOR':
    st.markdown('<h1 class="page-title">🧮 Calculator</h1>', unsafe_allow_html=True)
    st.write("Position size and risk calculators")
    st.info("Calculate position sizes, risk management, and profit/loss scenarios.")

elif st.session_state.active_page == 'MENTAI':
    st.markdown('<h1 class="page-title">🤖 MentAI</h1>', unsafe_allow_html=True)
    st.write("AI-powered trading insights and analysis")
    st.info("Get AI-generated market insights, trade suggestions, and sentiment analysis.")

elif st.session_state.active_page == 'BACKTEST':
    st.markdown('<h1 class="page-title">⚡ Backtest</h1>', unsafe_allow_html=True)
    st.write("Strategy backtesting tools")
    st.info("Test your trading strategies against historical data.")

elif st.session_state.active_page == 'TRADES':
    st.markdown('<h1 class="page-title">🔄 Trades</h1>', unsafe_allow_html=True)
    st.write("Your trading history and active positions")
    st.info("View all your trades, both active and closed positions.")

elif st.session_state.active_page == 'ADD_TRADE':
    st.markdown('<h1 class="page-title">➕ Add Trade</h1>', unsafe_allow_html=True)
    st.write("Add a new trade position")
    
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
        st.selectbox("Trade Type", ["Buy", "Sell"])
        st.number_input("Lot Size", min_value=0.01, value=0.1, step=0.01)
    
    with col2:
        st.number_input("Entry Price", min_value=0.0, value=1.0000, step=0.0001, format="%.4f")
        st.number_input("Stop Loss", min_value=0.0, value=0.9950, step=0.0001, format="%.4f")
        st.number_input("Take Profit", min_value=0.0, value=1.0050, step=0.0001, format="%.4f")
    
    if st.button("Execute Trade", type="primary"):
        st.success("Trade executed successfully!")

elif st.session_state.active_page == 'SETTINGS':
    st.markdown('<h1 class="page-title">⚙️ Settings</h1>', unsafe_allow_html=True)
    st.write("Application settings and preferences")
    
    st.subheader("Trading Settings")
    st.checkbox("Enable notifications")
    st.checkbox("Auto-close positions")
    st.selectbox("Default timeframe", ["1M", "5M", "15M", "1H", "4H", "1D"])

elif st.session_state.active_page == 'LOGOUT':
    st.markdown('<h1 class="page-title">🚪 Logout</h1>', unsafe_allow_html=True)
    st.write("Are you sure you want to logout?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Yes, Logout", type="primary"):
            st.success("Logged out successfully!")
    with col2:
        if st.button("Cancel"):
            st.session_state.active_page = 'DASHBOARD'

st.markdown('</div>', unsafe_allow_html=True)
