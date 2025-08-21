import streamlit as st
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Hide default streamlit sidebar */
    .css-1d391kg {
        display: none;
    }
    
    /* Main container */
    .main-container {
        display: flex;
        height: 100vh;
    }
    
    /* Custom sidebar */
    .custom-sidebar {
        width: 250px;
        background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 20px 15px;
        display: flex;
        flex-direction: column;
        height: 100vh;
        position: fixed;
        left: 0;
        top: 0;
        z-index: 1000;
    }
    
    /* Logo */
    .logo {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .logo-text {
        font-size: 32px;
        font-weight: bold;
        color: #ff6b35;
        text-decoration: none;
    }
    
    /* Navigation buttons */
    .nav-button {
        display: block;
        width: 100%;
        padding: 12px 15px;
        margin: 5px 0;
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #ff6b35 100%);
        color: white;
        text-decoration: none;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 107, 53, 0.3);
        background: linear-gradient(135deg, #ff7a45 0%, #ff8c2e 50%, #ff7a45 100%);
        color: white;
        text-decoration: none;
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #ff8c2e 0%, #ff6b35 50%, #ff8c2e 100%);
        transform: translateY(-1px);
    }
    
    /* Add trade button */
    .add-trade-button {
        display: block;
        width: 100%;
        padding: 12px 15px;
        margin: 5px 0;
        background: linear-gradient(135deg, #666 0%, #555 50%, #666 100%);
        color: white;
        text-decoration: none;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .add-trade-button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #777 0%, #666 50%, #777 100%);
        color: white;
        text-decoration: none;
    }
    
    /* Strategy dropdown */
    .strategy-section {
        margin-top: auto;
        margin-bottom: 20px;
    }
    
    .strategy-dropdown {
        width: 100%;
        padding: 10px;
        background: #333;
        color: white;
        border: 1px solid #555;
        border-radius: 5px;
        font-size: 14px;
    }
    
    .strategy-label {
        color: #ccc;
        font-size: 14px;
        margin-bottom: 8px;
        display: block;
    }
    
    /* Bottom buttons */
    .bottom-buttons {
        margin-top: 10px;
    }
    
    /* Icons */
    .nav-icon {
        margin-right: 10px;
        font-size: 16px;
    }
    
    /* Main content area */
    .main-content {
        margin-left: 250px;
        padding: 20px;
        background: #f5f5f5;
        min-height: 100vh;
        width: calc(100% - 250px);
    }
    
    /* Hamburger menu */
    .hamburger {
        display: none;
        background: transparent;
        border: none;
        color: #ff6b35;
        font-size: 20px;
        padding: 10px;
        cursor: pointer;
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 1001;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .hamburger {
            display: block;
        }
        
        .custom-sidebar {
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }
        
        .custom-sidebar.open {
            transform: translateX(0);
        }
        
        .main-content {
            margin-left: 0;
            width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'DASHBOARD'

if 'sidebar_open' not in st.session_state:
    st.session_state.sidebar_open = False

# Navigation function
def navigate_to(page):
    st.session_state.active_page = page
    st.rerun()

# Create the sidebar HTML
sidebar_html = f"""
<div class="custom-sidebar" id="sidebar">
    <div class="logo">
        <span class="logo-text">TD</span>
    </div>
    
    <div class="nav-menu">
        <button class="nav-button {'active' if st.session_state.active_page == 'DASHBOARD' else ''}" onclick="setActivePage('DASHBOARD')">
            <span class="nav-icon">📊</span>DASHBOARD
        </button>
        
        <button class="nav-button {'active' if st.session_state.active_page == 'MARKETS' else ''}" onclick="setActivePage('MARKETS')">
            <span class="nav-icon">💹</span>MARKETS
        </button>
        
        <button class="nav-button {'active' if st.session_state.active_page == 'CALENDAR' else ''}" onclick="setActivePage('CALENDAR')">
            <span class="nav-icon">📅</span>CALENDAR
        </button>
        
        <button class="nav-button {'active' if st.session_state.active_page == 'ANALYTICS' else ''}" onclick="setActivePage('ANALYTICS')">
            <span class="nav-icon">📈</span>ANALYTICS
        </button>
        
        <button class="nav-button {'active' if st.session_state.active_page == 'CALCULATOR' else ''}" onclick="setActivePage('CALCULATOR')">
            <span class="nav-icon">🧮</span>CALCULATOR
        </button>
        
        <button class="nav-button {'active' if st.session_state.active_page == 'MENTAI' else ''}" onclick="setActivePage('MENTAI')">
            <span class="nav-icon">🤖</span>MENTAI
        </button>
        
        <button class="nav-button {'active' if st.session_state.active_page == 'BACKTEST' else ''}" onclick="setActivePage('BACKTEST')">
            <span class="nav-icon">⚡</span>BACKTEST
        </button>
        
        <button class="nav-button {'active' if st.session_state.active_page == 'TRADES' else ''}" onclick="setActivePage('TRADES')">
            <span class="nav-icon">🔄</span>TRADES
        </button>
        
        <button class="add-trade-button" onclick="setActivePage('ADD_TRADE')">
            <span class="nav-icon">➕</span>ADD TRADE
        </button>
    </div>
    
    <div class="strategy-section">
        <label class="strategy-label">Strategy 1</label>
        <select class="strategy-dropdown">
            <option>Strategy 1</option>
            <option>Strategy 2</option>
            <option>Strategy 3</option>
        </select>
    </div>
    
    <div class="bottom-buttons">
        <button class="nav-button" onclick="setActivePage('SETTINGS')">
            <span class="nav-icon">⚙️</span>SETTINGS
        </button>
        
        <button class="nav-button" onclick="setActivePage('LOGOUT')">
            <span class="nav-icon">🚪</span>LOGOUT
        </button>
    </div>
</div>

<button class="hamburger" onclick="toggleSidebar()">
    ☰
</button>

<script>
    function setActivePage(page) {{
        // Send the page selection back to Streamlit
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: page
        }}, '*');
    }}
    
    function toggleSidebar() {{
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('open');
    }}
</script>
"""

# Create columns for layout
col1, col2 = st.columns([1, 4])

with col1:
    # Render the custom sidebar
    selected_page = components.html(sidebar_html, height=600, key="sidebar")
    
    # Handle page selection
    if selected_page:
        st.session_state.active_page = selected_page

with col2:
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Display content based on active page
    if st.session_state.active_page == 'DASHBOARD':
        st.title("📊 Dashboard")
        st.write("Welcome to your trading dashboard!")
        
        # Add some sample dashboard content
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Profit", "$12,345", "5.2%")
        with col2:
            st.metric("Win Rate", "68%", "2.1%")
        with col3:
            st.metric("Active Trades", "7", "-1")
            
    elif st.session_state.active_page == 'MARKETS':
        st.title("💹 Markets")
        st.write("Market analysis and data")
        
    elif st.session_state.active_page == 'CALENDAR':
        st.title("📅 Calendar")
        st.write("Economic calendar and events")
        
    elif st.session_state.active_page == 'ANALYTICS':
        st.title("📈 Analytics")
        st.write("Trading analytics and performance metrics")
        
    elif st.session_state.active_page == 'CALCULATOR':
        st.title("🧮 Calculator")
        st.write("Position size and risk calculators")
        
    elif st.session_state.active_page == 'MENTAI':
        st.title("🤖 MentAI")
        st.write("AI-powered trading insights")
        
    elif st.session_state.active_page == 'BACKTEST':
        st.title("⚡ Backtest")
        st.write("Strategy backtesting tools")
        
    elif st.session_state.active_page == 'TRADES':
        st.title("🔄 Trades")
        st.write("Your trading history and active positions")
        
    elif st.session_state.active_page == 'ADD_TRADE':
        st.title("➕ Add Trade")
        st.write("Add a new trade position")
        
    elif st.session_state.active_page == 'SETTINGS':
        st.title("⚙️ Settings")
        st.write("Application settings and preferences")
        
    elif st.session_state.active_page == 'LOGOUT':
        st.title("🚪 Logout")
        st.write("Are you sure you want to logout?")
        if st.button("Confirm Logout"):
            st.success("Logged out successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add some additional styling for the main content
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: none;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
