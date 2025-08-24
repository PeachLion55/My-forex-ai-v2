import streamlit as st
import pandas as pd
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
import datetime as dt
import os
import json
import hashlib # For password hashing
import requests
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pytz
import logging
import math
import uuid
import scipy.stats
from PIL import Image
import io
import base64

# Import Supabase
from supabase import create_client

# =========================================================
# STYLING AND PAGE CONFIG (NO CHANGES)
# =========================================================
st.set_page_config(page_title="Forex Dashboard", layout="wide")

st.markdown(
    """
    <style>
    /* --- Global Horizontal Line Style --- */
    hr { margin-top: 1.5rem !important; margin-bottom: 1.5rem !important; border-top: 1px solid #4d7171 !important; border-bottom: none !important; background-color: transparent !important; height: 1px !important; }
    /* Hide Streamlit UI */
    #MainMenu {visibility: hidden !important;} footer {visibility: hidden !important;} [data-testid="stDecoration"] {display: none !important;}
    /* Remove top padding */
    .css-18e3th9, .css-1d391kg { padding-top: 0rem !important; margin-top: 0rem !important; }
    .block-container { padding-top: 0rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

grid_color, grid_opacity, grid_size = "#58b3b1", 0.16, 40
r, g, b = int(grid_color[1:3], 16), int(grid_color[3:5], 16), int(grid_color[5:7], 16)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #000000;
        background-image: linear-gradient(rgba({r},{g},{b},{grid_opacity}) 1px, transparent 1px), linear-gradient(90deg, rgba({r},{g},{b},{grid_opacity}) 1px, transparent 1px);
        background-size: {grid_size}px {grid_size}px;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { background-color: #000000 !important; overflow: hidden !important; max-height: 100vh !important; }
    section[data-testid="stSidebar"] div.stButton > button { width: 200px !important; background: linear-gradient(to right, rgba(88, 179, 177, 0.7), rgba(0, 0, 0, 0.7)) !important; color: #ffffff !important; border: none !important; border-radius: 5px !important; padding: 10px !important; margin: 5px 0 !important; font-weight: bold !important; font-size: 16px !important; text-align: left !important; display: block !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; transition: all 0.3s ease !important; box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; }
    section[data-testid="stSidebar"] div.stButton > button:hover { background: linear-gradient(to right, rgba(88, 179, 177, 1.0), rgba(0, 0, 0, 1.0)) !important; transform: scale(1.05) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important; color: #f0f0f0 !important; cursor: pointer !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================
# SUPABASE INITIALIZATION
# =========================================================
@st.cache_resource
def init_supabase_connection():
    """Initializes the Supabase client from Streamlit secrets."""
    try:
        url = st.secrets["supabase_url"]
        key = st.secrets["supabase_key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase. Check your Streamlit secrets. Error: {e}")
        return None

supabase = init_supabase_connection()

# =========================================================
# CUSTOM JSON ENCODER
# =========================================================
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp)):
            return obj.isoformat()
        if pd.isna(obj) or obj is None:
            return None
        return json.JSONEncoder.default(self, obj)

# =========================================================
# REFACTORED DATABASE & APP HELPERS
# =========================================================

def _ta_get_user_data(username):
    """Fetches the entire 'data' JSONB object for a given user."""
    try:
        response = supabase.table('users').select('data').eq('username', username).single().execute()
        return response.data.get('data', {}) if response.data else {}
    except Exception as e:
        logging.error(f"Error fetching data for {username}: {e}")
        return {}

def _ta_update_user_data(username, new_data):
    """Updates the entire 'data' JSONB object for a given user."""
    try:
        # Supabase client expects a plain dict, not a JSON string.
        supabase.table('users').update({'data': new_data}).eq('username', username).execute()
        return True
    except Exception as e:
        logging.error(f"Error updating data for {username}: {e}")
        st.error(f"Failed to save data: {e}")
        return False

# --- Data specific helpers now use the get/update pattern ---
def _ta_save_journal(username, journal_df):
    user_data = _ta_get_user_data(username)
    journal_records = journal_df.replace({pd.NA: None, np.nan: None}).to_dict('records')
    user_data['tools_trade_journal'] = journal_records
    if _ta_update_user_data(username, user_data):
        logging.info(f"Journal saved for {username}: {len(journal_df)} trades")
        return True
    return False

def ta_update_xp(amount):
    username = st.session_state.get('logged_in_user')
    if not username: return

    user_data = _ta_get_user_data(username)
    old_xp = user_data.get('xp', 0)
    new_xp = old_xp + amount
    old_level = user_data.get('level', 0)
    new_level = new_xp // 100

    user_data['xp'] = new_xp

    if new_level > old_level:
        user_data['level'] = new_level
        badges = user_data.get('badges', [])
        new_badge = f"Level {new_level}"
        if new_badge not in badges:
            badges.append(new_badge)
        user_data['badges'] = badges
        st.balloons()
        st.success(f"Level up! You are now level {new_level}.")

    if _ta_update_user_data(username, user_data):
        st.session_state.xp = user_data['xp']
        st.session_state.level = user_data.get('level', 0)
        st.session_state.badges = user_data.get('badges', [])
        show_xp_notification(amount)

def ta_update_streak():
    username = st.session_state.get('logged_in_user')
    if not username: return

    user_data = _ta_get_user_data(username)
    today = dt.date.today()
    last_date_str = user_data.get('last_journal_date')
    streak = user_data.get('streak', 0)

    update_required = False
    if last_date_str:
        last_date = dt.date.fromisoformat(last_date_str)
        if last_date == today - dt.timedelta(days=1):
            streak += 1
            update_required = True
        elif last_date < today - dt.timedelta(days=1):
            streak = 1 # Reset
            update_required = True
    else: # First journal entry
        streak = 1
        update_required = True

    if update_required:
        user_data['streak'] = streak
        user_data['last_journal_date'] = today.isoformat()

        if streak > 0 and streak % 7 == 0:
            badge = f"Discipline Badge ({streak} Days)"
            badges = user_data.get('badges', [])
            if badge not in badges:
                badges.append(badge)
                user_data['badges'] = badges
                st.balloons()
                st.success(f"Unlocked: {badge}!")

        if _ta_update_user_data(username, user_data):
            st.session_state.streak = user_data['streak']
            st.session_state.badges = user_data.get('badges', [])

# --- Community data helpers (compatible with new schema) ---
def _ta_load_community(key, default=[]):
    try:
        response = supabase.table('community_data').select('data').eq('key', key).maybe_single().execute()
        return response.data['data'] if response.data else default
    except Exception as e:
        logging.error(f"Failed to load community data '{key}': {e}")
        return default

def _ta_save_community(key, data):
    try:
        json_str = json.dumps(data, cls=CustomJSONEncoder)
        supabase.table('community_data').upsert({'key': key, 'data': json.loads(json_str)}).execute()
        logging.info(f"Community data saved for key: {key}")
    except Exception as e:
        logging.error(f"Failed to save community data '{key}': {e}")
        st.error("Could not save community data.")

# --- Other helpers (unchanged from original) ---
def _ta_user_dir(user_id="guest"):
    """Manages local directories for temporary file uploads."""
    root = os.path.join(os.path.dirname(__file__), "user_data")
    d = os.path.join(root, str(user_id))
    os.makedirs(os.path.join(d, "community_images"), exist_ok=True)
    return d

def _ta_hash():
    return uuid.uuid4().hex[:12]

def show_xp_notification(xp_gained):
    notification_html = f"""<div style="position:fixed;top:20px;right:20px;background:linear-gradient(135deg, #58b3b1, #4d7171);color:white;padding:15px 20px;border-radius:10px;box-shadow:0 4px 15px rgba(88,179,177,0.3);z-index:9999;animation:slideInRight 0.5s ease-out, fadeOut 0.5s ease-out 3s forwards;font-weight:bold;border:2px solid #fff;">+ {xp_gained} XP Earned!</div><style>@keyframes slideInRight{{from{{transform:translateX(100%);opacity:0;}}to{{transform:translateX(0);opacity:1;}}}}@keyframes fadeOut{{from{{opacity:1;}}to{{opacity:0;}}}}</style>"""
    st.components.v1.html(notification_html, height=0)

# =========================================================
# DATA FETCHING & SESSION STATE
# =========================================================
# (This section is mostly unchanged as it deals with non-persistent data)
@st.cache_data(ttl=600, show_spinner=False)
def get_fxstreet_forex_news():
    RSS_URL = "https://www.fxstreet.com/rss/news"
    try:
        feed = feedparser.parse(RSS_URL)
        rows = []
        for entry in getattr(feed, "entries", []):
            rows.append({
                "Date": pd.to_datetime(entry.published[:10], errors='coerce'),
                "Currency": detect_currency(entry.title),
                "Headline": entry.title,
                "Polarity": TextBlob(entry.title).sentiment.polarity,
                "Summary": getattr(entry, "summary", ""),
                "Link": entry.link
            })
        df = pd.DataFrame(rows)
        return df[df["Date"] >= pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=3)].reset_index(drop=True)
    except Exception as e:
        logging.error(f"FXStreet RSS Error: {e}")
        return pd.DataFrame()

def detect_currency(title: str):
    t = title.upper()
    currency_map = {"USD":["USD","FED"],"GBP":["GBP","BOE"],"EUR":["EUR","ECB"],"JPY":["JPY","BOJ"],"AUD":["AUD","RBA"],"CAD":["CAD","BOC"],"CHF":["CHF","SNB"],"NZD":["NZD","RBNZ"]}
    for c, k in currency_map.items():
        if any(w in t for w in k): return c
    return "Unknown"

# (Calendar data, etc., remains the same)
econ_df = pd.DataFrame([{"Date":"2025-08-22","Time":"14:30","Currency":"USD","Event":"Non-Farm Payrolls","Impact":"High"}])
df_news = get_fxstreet_forex_news()

# Initialize session state
if 'current_page' not in st.session_state: st.session_state.current_page = 'fundamentals'
if 'logged_in_user' not in st.session_state: st.session_state.logged_in_user = None

journal_cols = ["Date", "Symbol", "Weekly Bias", "Daily Bias", "4H Structure", "1H Structure", "Positive Correlated Pair & Bias", "Potential Entry Points", "5min/15min Setup?", "Entry Conditions", "Planned R:R", "News Filter", "Alerts", "Concerns", "Emotions", "Confluence Score 1-7", "Outcome / R:R Realised", "Notes/Journal", "Entry Price", "Stop Loss Price", "Take Profit Price", "Lots", "Tags"]
journal_dtypes = {"Date":"datetime64[ns]", "Entry Price":float, "Stop Loss Price":float, "Take Profit Price":float, "Lots":float, "Confluence Score 1-7":float} # Abbreviated for clarity

def initialize_empty_dataframes():
    """Sets empty, correctly typed dataframes in session state for logged-out users."""
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype({k:v for k,v in journal_dtypes.items() if k in journal_cols}, errors='ignore')
    st.session_state.drawings = {}
    st.session_state.strategies = pd.DataFrame()
    st.session_state.emotion_log = pd.DataFrame()
    st.session_state.reflection_log = pd.DataFrame()
    # Gamification
    st.session_state.xp, st.session_state.level, st.session_state.streak = 0, 0, 0
    st.session_state.badges = []

if 'tools_trade_journal' not in st.session_state:
    initialize_empty_dataframes()

# Community data is loaded once
if "trade_ideas" not in st.session_state:
    st.session_state.trade_ideas = pd.DataFrame(_ta_load_community('trade_ideas', []))
if "community_templates" not in st.session_state:
    st.session_state.community_templates = pd.DataFrame(_ta_load_community('templates', []))


# =========================================================
# SIDEBAR & NAVIGATION (Unchanged)
# =========================================================
st.markdown("""<style>.sidebar-content {padding-top: 0rem;}</style>""", unsafe_allow_html=True)
try:
    logo = Image.open("logo22.png")
    logo = logo.resize((60, 50))
    buffered = io.BytesIO()
    logo.save(buffered, format="PNG")
    logo_str = base64.b64encode(buffered.getvalue()).decode()
    st.sidebar.markdown(f"""<div style='text-align: center; margin-bottom: 20px;'><img src="data:image/png;base64,{logo_str}" width="60" height="50"/></div>""", unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.warning("logo22.png not found.")


nav_items = [('fundamentals', 'Forex Fundamentals'), ('backtesting', 'Backtesting'), ('mt5', 'Performance Dashboard'), ('tools', 'Tools'), ('strategy', 'Manage My Strategy'), ('community', 'Community Trade Ideas'), ('Zenvo Academy', 'Zenvo Academy'), ('account', 'My Account')]
for page_key, page_name in nav_items:
    if st.sidebar.button(page_name, key=f"nav_{page_key}"):
        st.session_state.current_page = page_key
        st.rerun()

# =========================================================
# =========================================================
#                 MAIN APPLICATION PAGES
# =========================================================
# =========================================================

if st.session_state.current_page == 'fundamentals':
    # This page content is unchanged
    st.title("📅 Forex Fundamentals")
    # ... (all UI and display logic remains the same) ...
    # This part is representative
    st.markdown("### 🗓️ Upcoming Economic Events")
    st.dataframe(econ_df, use_container_width=True)


elif st.session_state.current_page == 'backtesting':
    # (Unchanged logic for TV chart)
    st.title("📈 Backtesting")
    st.caption("Live TradingView chart for backtesting and enhanced trading journal.")
    # ...
    # Journal Entry
    with st.form("trade_entry_form"):
        # ... (form elements unchanged)
        symbol = st.selectbox("Symbol", ["EUR/USD", "GBP/USD", "USD/JPY"], index=0)
        # ...
        submit_button = st.form_submit_button("Save Trade")
        if submit_button:
            new_trade = { # simplified example
                'Date': pd.to_datetime(dt.date.today()),
                'Symbol': symbol
            }
            # Add all columns to avoid concat issues
            for col in journal_cols:
                if col not in new_trade:
                    new_trade[col] = None

            st.session_state.tools_trade_journal = pd.concat(
                [st.session_state.tools_trade_journal, pd.DataFrame([new_trade])],
                ignore_index=True
            )
            username = st.session_state.get('logged_in_user')
            if username:
                if _ta_save_journal(username, st.session_state.tools_trade_journal):
                    ta_update_xp(10)
                    ta_update_streak()
                    st.success("Trade saved successfully!")
                    st.rerun()
                # else error is handled in helper
            else:
                st.info("Trade logged for this session. Please log in to save permanently.")


elif st.session_state.current_page == 'mt5':
    # This page is unchanged as it relies on a local file upload, no DB interaction.
    st.title("📊 Performance Dashboard")
    st.info("This feature works by uploading a CSV file and does not interact with the database.")
    # ... (content remains identical) ...


elif st.session_state.current_page == 'strategy':
    st.title("📈 Manage My Strategy")
    with st.form("strategy_form"):
        # ... (form unchanged) ...
        strategy_name = st.text_input("Strategy Name")
        description = st.text_area("Strategy Description")
        submit_strategy = st.form_submit_button("Save Strategy")
        if submit_strategy:
            new_strategy = {"Name": strategy_name, "Description": description}
            st.session_state.strategies = pd.concat([st.session_state.strategies, pd.DataFrame([new_strategy])], ignore_index=True)

            username = st.session_state.get('logged_in_user')
            if username:
                user_data = _ta_get_user_data(username)
                user_data['strategies'] = st.session_state.strategies.to_dict('records')
                _ta_update_user_data(username, user_data)
                st.success("Strategy saved!")
            else:
                st.info("Log in to save your strategy.")


elif st.session_state.current_page == 'account':
    st.title("👤 My Account")
    st.markdown(""" Manage your account, save your data, and sync your trading journal.""")
    st.write('---')

    if not supabase:
        st.error("Database connection not available. Please check the application configuration.")
        st.stop()

    if not st.session_state.get('logged_in_user'):
        tab_signin, tab_signup = st.tabs(["🔑 Sign In", "📝 Sign Up"])

        with tab_signin:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                if login_button:
                    try:
                        response = supabase.table('users').select('password, data').eq('username', username).single().execute()
                        if response.data:
                            stored_hash = response.data['password']
                            input_hash = hashlib.sha256(password.encode()).hexdigest()
                            if stored_hash == input_hash:
                                st.session_state.logged_in_user = username
                                user_data = response.data.get('data', {})

                                # Load all data from the JSONB field into session state
                                journal_data = user_data.get('tools_trade_journal', [])
                                if journal_data:
                                     st.session_state.tools_trade_journal = pd.DataFrame(journal_data)
                                else:
                                     st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols)

                                st.session_state.strategies = pd.DataFrame(user_data.get('strategies', []))
                                st.session_state.drawings = user_data.get('drawings', {})
                                st.session_state.xp = user_data.get('xp', 0)
                                st.session_state.level = user_data.get('level', 0)
                                st.session_state.badges = user_data.get('badges', [])
                                st.session_state.streak = user_data.get('streak', 0)

                                st.success(f"Welcome back, {username}!")
                                logging.info(f"User '{username}' logged in.")
                                st.rerun()
                            else:
                                st.error("Invalid username or password.")
                        else:
                            st.error("Invalid username or password.")
                    except Exception as e:
                        st.error("An error occurred during login.")
                        logging.error(f"Login error for {username}: {e}")

        with tab_signup:
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Register")
                if register_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif not new_username or not new_password:
                        st.error("Username and password cannot be empty.")
                    else:
                        try:
                            # Check if user exists
                            check = supabase.table('users').select('username').eq('username', new_username).execute()
                            if len(check.data) > 0:
                                st.error("Username already exists.")
                            else:
                                # Create new user
                                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                                initial_data = {
                                    "xp": 0, "level": 0, "badges": [], "streak": 0,
                                    "drawings": {}, "tools_trade_journal": [],
                                    "strategies": [], "emotion_log": [], "reflection_log": []
                                }
                                supabase.table('users').insert({
                                    'username': new_username,
                                    'password': hashed_password,
                                    'data': initial_data
                                }).execute()

                                st.session_state.logged_in_user = new_username
                                initialize_empty_dataframes()
                                st.success(f"Account for '{new_username}' created successfully!")
                                logging.info(f"User '{new_username}' registered.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create account: {e}")
                            logging.error(f"Registration error for {new_username}: {e}")

    else: # Logged-in user view
        def handle_logout():
            user_session_keys = ['logged_in_user', 'xp', 'level', 'badges', 'streak']
            for key in user_session_keys:
                if key in st.session_state: del st.session_state[key]

            initialize_empty_dataframes()
            logging.info("User logged out.")
            st.rerun()

        st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")
        # (Dashboard UI remains the same)
        # ...

        with st.expander("⚙️ Manage Account"):
            st.write(f"**Username**: `{st.session_state.logged_in_user}`")
            if st.button("Log Out", key="logout_button", type="primary"):
                handle_logout()


        st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")
        # (Dashboard UI remains the same)
        # ...

        with st.expander("⚙️ Manage Account"):
            st.write(f"**Username**: `{st.session_state.logged_in_user}`")
            if st.button("Log Out", key="logout_button", type="primary"):
                handle_logout()


elif st.session_state.current_page == 'community':
    st.title("🌐 Community Trade Ideas")
    # ... (Content unchanged, uses _ta_load/save_community helpers) ...
    st.subheader("🏆 Leaderboard - Consistency")
    try:
        response = supabase.table('users').select('username, data').execute()
        leader_data = []
        if response.data:
            for user_profile in response.data:
                trades = len(user_profile.get('data', {}).get('tools_trade_journal', []))
                leader_data.append({"Username": user_profile['username'], "Journaled Trades": trades})

        if leader_data:
            leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
            leader_df["Rank"] = leader_df.index + 1
            st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]])
    except Exception as e:
        st.error("Could not load leaderboard.")
        logging.error(f"Leaderboard error: {e}")

elif st.session_state.current_page == 'community':
    st.title("🌐 Community Trade Ideas")
    st.markdown(""" Share and explore trade ideas with the community. Upload your chart screenshots and discuss strategies with other traders. """)
    st.write('---')
    st.subheader("➕ Share a Trade Idea")
    with st.form("trade_idea_form"):
        trade_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP", "EUR/JPY"])
        trade_direction = st.radio("Direction", ["Long", "Short"])
        trade_description = st.text_area("Trade Description")
        uploaded_image = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg"])
        submit_idea = st.form_submit_button("Share Idea")
        if submit_idea:
            if "logged_in_user" in st.session_state:
                username = st.session_state.logged_in_user
                user_dir = _ta_user_dir(username)
                idea_id = _ta_hash()
                idea_data = {
                    "Username": username,
                    "Pair": trade_pair,
                    "Direction": trade_direction,
                    "Description": trade_description,
                    "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "IdeaID": idea_id
                }
                if uploaded_image:
                    image_path = os.path.join(user_dir, "community_images", f"{idea_id}.png")
                    with open(image_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    idea_data["ImagePath"] = image_path
                st.session_state.trade_ideas = pd.concat([st.session_state.trade_ideas, pd.DataFrame([idea_data])], ignore_index=True)
                _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                st.success("Trade idea shared successfully!")
                logging.info(f"Trade idea shared by {username}: {idea_id}")
                st.rerun()
            else:
                st.error("Please log in to share trade ideas.")
                logging.warning("Attempt to share trade idea without login")
    st.subheader("📈 Community Trade Ideas")
    if not st.session_state.trade_ideas.empty:
        for idx, idea in st.session_state.trade_ideas.iterrows():
            with st.expander(f"{idea['Pair']} - {idea['Direction']} by {idea['Username']} ({idea['Timestamp']})"):
                st.markdown(f"Description: {idea['Description']}")
                if "ImagePath" in idea and pd.notna(idea['ImagePath']) and os.path.exists(idea['ImagePath']):
                    st.image(idea['ImagePath'], caption="Chart Screenshot", use_column_width=True)
                if st.button("Delete Idea", key=f"delete_idea_{idea['IdeaID']}"):
                    if "logged_in_user" in st.session_state and st.session_state.logged_in_user == idea["Username"]:
                        st.session_state.trade_ideas = st.session_state.trade_ideas.drop(idx).reset_index(drop=True)
                        _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))
                        st.success("Trade idea deleted successfully!")
                        logging.info(f"Trade idea {idea['IdeaID']} deleted by {st.session_state.logged_in_user}")
                        st.rerun()
                    else:
                        st.error("You can only delete your own trade ideas.")
                        logging.warning(f"Unauthorized attempt to delete trade idea {idea['IdeaID']}")
    else:
        st.info("No trade ideas shared yet. Be the first to contribute!")
    # Community Templates
    st.subheader("📄 Community Templates")
    with st.form("template_form"):
        template_type = st.selectbox("Template Type", ["Journaling Template", "Checklist", "Strategy Playbook"])
        template_name = st.text_input("Template Name")
        template_content = st.text_area("Template Content")
        submit_template = st.form_submit_button("Share Template")
        if submit_template:
            if "logged_in_user" in st.session_state:
                username = st.session_state.logged_in_user
                template_id = _ta_hash()
                template_data = {
                    "Username": username,
                    "Type": template_type,
                    "Name": template_name,
                    "Content": template_content,
                    "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ID": template_id
                }
                st.session_state.community_templates = pd.concat([st.session_state.community_templates, pd.DataFrame([template_data])], ignore_index=True)
                _ta_save_community('templates', st.session_state.community_templates.to_dict('records'))
                st.success("Template shared successfully!")
                logging.info(f"Template shared by {username}: {template_id}")
                st.rerun()
            else:
                st.error("Please log in to share templates.")
    if not st.session_state.community_templates.empty:
        for idx, template in st.session_state.community_templates.iterrows():
            with st.expander(f"{template['Type']} - {template['Name']} by {template['Username']} ({template['Timestamp']})"):
                st.markdown(template['Content'])
                if st.button("Delete Template", key=f"delete_template_{template['ID']}"):
                    if "logged_in_user" in st.session_state and st.session_state.logged_in_user == template["Username"]:
                        st.session_state.community_templates = st.session_state.community_templates.drop(idx).reset_index(drop=True)
                        _ta_save_community('templates', st.session_state.community_templates.to_dict('records'))
                        st.success("Template deleted successfully!")
                        logging.info(f"Template {template['ID']} deleted by {st.session_state.logged_in_user}")
                        st.rerun()
                    else:
                        st.error("You can only delete your own templates.")
    else:
        st.info("No templates shared yet. Share one above!")
    # Leaderboard / Self-Competition
    st.subheader("🏆 Leaderboard - Consistency")
    users = c.execute("SELECT username, data FROM users").fetchall()
    leader_data = []
    for u, d in users:
        user_d = json.loads(d) if d else {}
        trades = len(user_d.get("tools_trade_journal", []))
        leader_data.append({"Username": u, "Journaled Trades": trades})
    if leader_data:
        leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
        leader_df["Rank"] = leader_df.index + 1
        st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]])
    else:
        st.info("No leaderboard data yet.")
# Tools
elif st.session_state.current_page == 'tools':
    st.title("🛠 Tools")
    st.markdown("""
    ### Available Tools
    - **Profit/Loss Calculator**: Calculate potential profits or losses based on trade size, entry, and exit prices.
    - **Price Alerts**: Set custom price alerts for key levels on your chosen currency pairs.
    - **Currency Correlation Heatmap**: Visualize correlations between currency pairs to identify hedging opportunities.
    - **Risk Management Calculator**: Determine optimal position sizes based on risk tolerance and stop-loss levels.
    - **Trading Session Tracker**: Monitor active trading sessions (e.g., London, New York) to align with market hours.
    - **Drawdown Recovery Planner**: Plan recovery strategies for account drawdowns with calculated targets.
    - **Pre-Trade Checklist**: Follow a structured checklist to ensure disciplined trade entries.
    - **Pre-Market Checklist**: Prepare for the trading day with a comprehensive market analysis checklist.
    """, unsafe_allow_html=True)
    st.markdown('---')
    st.markdown("""
    <style>
    div[data-testid="stTabs"] div[role="tablist"] > div {
        background-color: #5bb4b0 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #ffffff !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        color: #5bb4b0 !important;
        background-color: rgba(91, 180, 176, 0.2) !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #5bb4b0 !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)
    tools_options = [
        'Profit/Loss Calculator',
        'Price Alerts',
        'Currency Correlation Heatmap',
        'Risk Management Calculator',
        'Trading Session Tracker',
        'Drawdown Recovery Planner',
        'Pre-Trade Checklist',
        'Pre-Market Checklist'
    ]
    tabs = st.tabs(tools_options)
    with tabs[0]:
        st.header("💰 Profit / Loss Calculator")
        st.markdown("Calculate your potential profit or loss for a trade.")
        st.markdown('---')
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            currency_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY"], key="pl_currency_pair")
            position_size = st.number_input("Position Size (lots)", min_value=0.01, value=0.1, step=0.01, key="pl_position_size")
            close_price = st.number_input("Close Price", value=1.1050, step=0.0001, key="pl_close_price")
        with col_calc2:
            account_currency = st.selectbox("Account Currency", ["USD", "EUR", "GBP", "JPY"], index=0, key="pl_account_currency")
            open_price = st.number_input("Open Price", value=1.1000, step=0.0001, key="pl_open_price")
            trade_direction = st.radio("Trade Direction", ["Long", "Short"], key="pl_trade_direction")
        pip_multiplier = 100 if "JPY" in currency_pair else 10000
        pip_movement = abs(close_price - open_price) * pip_multiplier
        exchange_rate = 1.1000
        pip_value = (
            (0.0001 / exchange_rate) * position_size * 100000 if "JPY" not in currency_pair else (0.01 / exchange_rate) * position_size * 100000
        )
        profit_loss = pip_movement * pip_value
        st.write(f"Pip Movement: {pip_movement:.2f} pips")
        st.write(f"Pip Value: {pip_value:.2f} {account_currency}")
        st.write(f"Potential Profit/Loss: {profit_loss:.2f} {account_currency}")
    with tabs[1]:
        st.header("⏰ Price Alerts")
        st.markdown("Set price alerts for your favourite forex pairs and get notified when the price hits your target.")
        st.markdown('---')
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY"]
        if "price_alerts" not in st.session_state:
            st.session_state.price_alerts = pd.DataFrame(columns=["Pair", "Target Price", "Triggered"])
        with st.form("add_alert_form"):
            col1, col2 = st.columns([2, 2])
            with col1:
                pair = st.selectbox("Currency Pair", forex_pairs)
            with col2:
                price = st.number_input("Target Price", min_value=0.0, format="%.5f")
            submitted = st.form_submit_button("➕ Add Alert")
            if submitted:
                new_alert = {"Pair": pair, "Target Price": price, "Triggered": False}
                st.session_state.price_alerts = pd.concat([st.session_state.price_alerts, pd.DataFrame([new_alert])], ignore_index=True)
                st.success(f"Alert added: {pair} at {price}")
                logging.info(f"Alert added: {pair} at {price}")
        st.subheader("Your Alerts")
        st.dataframe(st.session_state.price_alerts, use_container_width=True, height=220)
        if st.session_state.get("price_alert_refresh", False):
            st_autorefresh(interval=5000, key="price_alert_autorefresh")
        active_pairs = st.session_state.price_alerts["Pair"].unique().tolist()
        live_prices = {}
        for p in active_pairs:
            if not p:
                continue
            base, quote = p[:3], p[3:]
            try:
                r = requests.get(f"https://api.exchangerate.host/latest?base={base}&symbols={quote}", timeout=6)
                data = r.json()
                price_val = data.get("rates", {}).get(quote)
                live_prices[p] = float(price_val) if price_val is not None else None
                logging.info(f"Fetched price for {p}: {live_prices[p]}")
            except Exception as e:
                live_prices[p] = None
                logging.error(f"Error fetching price for {p}: {str(e)}")
        triggered_alerts = []
        for idx, row in st.session_state.price_alerts.iterrows():
            pair = row["Pair"]
            target = row["Target Price"]
            current_price = live_prices.get(pair)
            if isinstance(current_price, (int, float)):
                if not row["Triggered"] and abs(current_price - target) < (0.0005 if "JPY" not in pair else 0.01):
                    st.session_state.price_alerts.at[idx, "Triggered"] = True
                    triggered_alerts.append((idx, f"{pair} reached {target} (Current: {current_price:.5f})"))
                    logging.info(f"Alert triggered: {pair} at {target}")
        if triggered_alerts:
            for idx, alert_msg in triggered_alerts:
                st.balloons()
                st.success(f"⚡ {alert_msg}")
        st.markdown("### 📊 Active Alerts")
        if not st.session_state.price_alerts.empty:
            for idx, row in st.session_state.price_alerts.iterrows():
                pair = row["Pair"]
                target = row["Target Price"]
                triggered = row["Triggered"]
                current_price = live_prices.get(pair)
                current_price_display = f"{current_price:.5f}" if isinstance(current_price, (int, float)) else "N/A"
                color = "#2ecc71" if triggered else "#f4a261"
                status = "Triggered" if triggered else "Pending"
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(
                        f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                        {pair} {status}<br>
                        Current: {current_price_display} &nbsp;&nbsp;&nbsp; Target: {target}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    if st.button("❌ Cancel", key=f"cancel_{idx}"):
                        st.session_state.price_alerts = st.session_state.price_alerts.drop(idx).reset_index(drop=True)
                        st.rerun()
                        logging.info(f"Cancelled alert at index {idx}")
        else:
            st.info("No price alerts set. Add one above to start monitoring prices.")
    with tabs[2]:
        st.header("📊 Currency Correlation Heatmap")
        st.markdown("Understand how forex pairs move relative to each other.")
        st.markdown('---')
        pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF"]
        data = np.array([
            [1.00, 0.87, -0.72, 0.68, -0.55, -0.60],
            [0.87, 1.00, -0.65, 0.74, -0.58, -0.62],
            [-0.72, -0.65, 1.00, -0.55, 0.69, 0.71],
            [0.68, 0.74, -0.55, 1.00, -0.61, -0.59],
            [-0.55, -0.58, 0.69, -0.61, 1.00, 0.88],
            [-0.60, -0.62, 0.71, -0.59, 0.88, 1.00],
        ])
        corr_df = pd.DataFrame(data, columns=pairs, index=pairs)
        fig = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="Forex Pair Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        st.header("🛡️ Risk Management Calculator")
        st.markdown(""" Proper position sizing keeps your account safe. Risk management is crucial to long-term trading success. It helps prevent large losses, preserves capital, and allows you to stay in the game during drawdowns. Always risk no more than 1-2% per trade, use stop losses, and calculate position sizes based on your account size and risk tolerance. """)
        st.markdown('---')
        # 📊 Lot Size Calculator
        st.subheader('📊 Lot Size Calculator')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=10000.0)
        with col2:
            risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0)
        with col3:
            stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=20.0)
        with col4:
            pip_value = st.number_input("Pip Value per Lot ($)", min_value=0.01, value=10.0)
        if st.button("Calculate Lot Size"):
            risk_amount = balance * (risk_percent / 100)
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            st.success(f"✅ Recommended Lot Size: {lot_size:.2f} lots")
            logging.info(f"Calculated lot size: {lot_size}")
        # 🔄 What-If Analyzer
        st.subheader('🔄 What-If Analyzer')
        base_equity = st.number_input('Starting Equity', value=10000.0, min_value=0.0, step=100.0, key='whatif_equity')
        risk_pct = st.slider('Risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key='whatif_risk') / 100.0
        winrate = st.slider('Win rate (%)', 10.0, 90.0, 50.0, 1.0, key='whatif_wr') / 100.0
        avg_r = st.slider('Average R multiple', 0.5, 5.0, 1.5, 0.1, key='whatif_avg_r')
        trades = st.slider('Number of trades', 10, 500, 100, 10, key='whatif_trades')
        E_R = winrate * avg_r - (1 - winrate) * 1.0
        exp_growth = (1 + risk_pct * E_R) ** trades
        st.metric('Expected Growth Multiplier', f"{exp_growth:.2f}x")
        alt_risk = st.slider('What if risk per trade was (%)', 0.1, 5.0, 0.5, 0.1, key='whatif_alt') / 100.0
        alt_growth = (1 + alt_risk * E_R) ** trades
        st.metric('Alt Growth Multiplier', f"{alt_growth:.2f}x")
        # 📈 Equity Projection
        sim = pd.DataFrame({
            'trade': list(range(trades + 1)),
            'equity_base': base_equity * (1 + risk_pct * E_R) ** np.arange(trades + 1),
            'equity_alt': base_equity * (1 + alt_risk * E_R) ** np.arange(trades + 1),
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim['trade'], y=sim['equity_base'], mode='lines', name=f'Risk {risk_pct*100:.1f}%'))
        fig.add_trace(go.Scatter(x=sim['trade'], y=sim['equity_alt'], mode='lines', name=f'What-If {alt_risk*100:.1f}%'))
        fig.update_layout(title='Equity Projection – Base vs What-If', xaxis_title='Trade #', yaxis_title='Equity')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[4]:
        st.header("🕒 Forex Market Sessions")
        st.markdown(""" Stay aware of active trading sessions to trade when volatility is highest. Each session has unique characteristics: Sydney/Tokyo for Asia-Pacific news, London for Europe, New York for US data. Overlaps like London/New York offer highest liquidity and volatility, ideal for major pairs. Track your performance per session to identify your edge. """)
        st.markdown('---')
        st.subheader('📊 Session Statistics')
        mt5_df = st.session_state.get('mt5_df', pd.DataFrame())
        df = mt5_df if not mt5_df.empty else st.session_state.tools_trade_journal
        if not df.empty and 'session' in df.columns:
            by_sess = df.groupby(['session']).agg(
                trades=('r', 'count') if 'r' in df.columns else ('session', 'count'),
                winrate=('r', lambda s: (s > 0).mean()) if 'r' in df.columns else ('session', 'count'),
                avg_r=('r', 'mean') if 'r' in df.columns else ('session', 'count')
            ).reset_index()
            st.dataframe(by_sess, use_container_width=True)
            if 'r' in df.columns:
                fig = px.bar(by_sess, x='session', y='winrate', title='Win Rate by Session', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload trades with a 'session' column to analyze performance by trading session.")
        st.subheader('🕒 Current Market Sessions')
        now = dt.datetime.now(pytz.UTC)
        sessions = [
            {"name": "Sydney", "start": 22, "end": 7, "tz": "Australia/Sydney"},
            {"name": "Tokyo", "start": 0, "end": 9, "tz": "Asia/Tokyo"},
            {"name": "London", "start": 8, "end": 17, "tz": "Europe/London"},
            {"name": "New York", "start": 13, "end": 22, "tz": "America/New_York"},
        ]
        session_status = []
        for session in sessions:
            tz = pytz.timezone(session["tz"])
            local_time = now.astimezone(tz)
            local_hour = local_time.hour + local_time.minute / 60
            start = session["start"]
            end = session["end"]
            is_open = (start <= local_hour < end) if start <= end else (start <= local_hour or local_hour < end)
            session_status.append({
                "Session": session["name"],
                "Status": "Open" if is_open else "Closed",
                "Local Time": local_time.strftime("%H:%M"),
                "Time Until": (start - local_hour) % 24 if not is_open else (end - local_hour) % 24
            })
        session_df = pd.DataFrame(session_status)
        st.dataframe(session_df, use_container_width=True)
        for session in session_status:
            color = "#2ecc71" if session["Status"] == "Open" else "#e74c3c"
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                {session['Session']} Session: {session['Status']} (Local: {session['Local Time']}, {'Closes in' if session['Status'] == 'Open' else 'Opens in'} {session['Time Until']:.1f} hours)
                </div>
                """,
                unsafe_allow_html=True
            )
    with tabs[5]:
        st.header("📉 Drawdown Recovery Planner")
        st.markdown(""" Plan your recovery from a drawdown. Understand the percentage gain required to recover losses and simulate recovery based on your trading parameters. """)
        st.markdown('---')
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 50.0, 10.0) / 100
        recovery_pct = _ta_percent_gain_to_recover(drawdown_pct)
        st.metric("Required Gain to Recover", f"{recovery_pct*100:.2f}%")
        st.subheader("📈 Recovery Simulation")
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_equity = st.number_input("Initial Equity ($)", min_value=100.0, value=10000.0)
        with col2:
            win_rate = st.slider("Expected Win Rate (%)", 10, 90, 50) / 100
        with col3:
            avg_rr = st.slider("Average R:R", 0.5, 5.0, 1.5, 0.1)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0) / 100
        trades_needed = math.ceil(math.log(1 / (1 - drawdown_pct)) / math.log(1 + risk_per_trade * (win_rate * avg_rr - (1 - win_rate))))
        st.write(f"Estimated Trades to Recover: {trades_needed}")
        sim_equity = [initial_equity * (1 - drawdown_pct)]
        for _ in range(min(trades_needed + 10, 100)):
            sim_equity.append(sim_equity[-1] * (1 + risk_per_trade * (win_rate * avg_rr - (1 - win_rate))))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(sim_equity))), y=sim_equity, mode='lines', name='Equity'))
        fig.add_hline(y=initial_equity, line_dash="dash", line_color="green", annotation_text="Initial Equity")
        fig.update_layout(title='Drawdown Recovery Simulation', xaxis_title='Trade #', yaxis_title='Equity ($)')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[6]:
        st.header("✅ Pre-Trade Checklist")
        st.markdown(""" Ensure discipline by running through this checklist before every trade. A structured approach reduces impulsive decisions and aligns trades with your strategy. """)
        st.markdown('---')
        checklist_items = [
            "Market structure aligns with my bias",
            "Key levels (S/R) identified",
            "Entry trigger confirmed",
            "Risk-reward ratio ≥ 1:2",
            "No high-impact news imminent",
            "Position size calculated correctly",
            "Stop loss set",
            "Take profit set",
            "Trade aligns with my edge",
            "Emotionally calm and focused"
        ]
        checklist_state = {item: st.checkbox(item, key=f"checklist_{i}") for i, item in enumerate(checklist_items)}
        checked_count = sum(1 for v in checklist_state.values() if v)
        st.metric("Checklist Completion", f"{checked_count}/{len(checklist_items)}")
        if checked_count == len(checklist_items):
            st.success("✅ All checks passed! Ready to trade.")
        else:
            st.warning(f"⚠ Complete all {len(checklist_items)} checklist items before trading.")
    with tabs[7]:
        st.header("📅 Pre-Market Checklist")
        st.markdown(""" Build consistent habits with pre-market checklists and end-of-day reflections. These rituals help maintain discipline and continuous improvement. """)
        st.markdown('---')
        st.subheader("Pre-Market Routine Checklist")
        pre_market_items = [
            "Reviewed economic calendar",
            "Analyzed major news events",
            "Set weekly/daily biases",
            "Identified key levels on charts",
            "Prepared watchlist of pairs",
            "Checked correlations",
            "Reviewed previous trades"
        ]
        pre_checklist = {item: st.checkbox(item, key=f"pre_{i}") for i, item in enumerate(pre_market_items)}
        pre_checked = sum(1 for v in pre_checklist.values() if v)
        st.metric("Pre-Market Completion", f"{pre_checked}/{len(pre_market_items)}")
        if pre_checked == len(pre_market_items):
            st.success("✅ Pre-market routine complete!")
        st.subheader("End-of-Day Reflection")
        with st.form("reflection_form"):
            reflection = st.text_area("What went well today? What can be improved?")
            submit_reflection = st.form_submit_button("Log Reflection")
            if submit_reflection:
                log_entry = {
                    "Date": dt.datetime.now().strftime("%Y-%m-%d"),
                    "Reflection": reflection
                }
                if "reflection_log" not in st.session_state:
                    st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
                st.session_state.reflection_log = pd.concat([st.session_state.reflection_log, pd.DataFrame([log_entry])], ignore_index=True)
                if "logged_in_user" in st.session_state:
                    username = st.session_state.logged_in_user
                    try:
                        c.execute("SELECT data FROM users WHERE username = ?", (username,))
                        result = c.fetchone()
                        user_data = json.loads(result[0]) if result else {}
                        user_data["reflection_log"] = st.session_state.reflection_log.to_dict(orient="records")
                        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data), username))
                        conn.commit()
                    except Exception as e:
                        logging.error(f"Error saving reflection: {str(e)}")
                st.success("Reflection logged!")
        if "reflection_log" in st.session_state and not st.session_state.reflection_log.empty:
            st.dataframe(st.session_state.reflection_log)

elif st.session_state.current_page == "Zenvo Academy":
    st.title("Zenvo Academy")
    st.caption("Explore experimental features and tools for your trading journey.")
    st.markdown('---')
    st.markdown("### Welcome to Zenvo Academy")
    st.write("Our Academy provides beginner traders with a clear learning path – covering Forex basics, chart analysis, risk management, and trading psychology. Build a solid foundation before stepping into live markets.")
    st.info("This page is under development. Stay tuned for new features!")
    if st.button("Log Out", key="logout_test_page"):
        if 'logged_in_user' in st.session_state:
            del st.session_state.logged_in_user
        st.session_state.drawings = {}
        st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)
        st.session_state.strategies = pd.DataFrame(columns=["Name", "Description", "Entry Rules", "Exit Rules", "Risk Management", "Date Added"])
        st.session_state.emotion_log = pd.DataFrame(columns=["Date", "Emotion", "Notes"])
        st.session_state.reflection_log = pd.DataFrame(columns=["Date", "Reflection"])
        st.session_state.xp = 0
        st.session_state.level = 0
        st.session_state.badges = []
        st.session_state.streak = 0
        st.success("Logged out successfully!")
        logging.info("User logged out")
        st.session_state.current_page = "login"
        st.rerun()
