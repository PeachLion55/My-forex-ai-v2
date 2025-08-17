# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import os
import json
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import time
import pytz
from forex_python.converter import CurrencyRates

# =========================================================
# CONSTANTS & CONFIGURATION
# =========================================================
# Path to your accounts JSON file
ACCOUNTS_FILE = "accounts.json"  # or a full path if needed

# API Keys (would normally be stored securely)
FOREX_API_KEY = "demo"  # Replace with actual API key in production
NEWS_API_KEY = "demo"   # Replace with actual API key in production

# Trading pairs configuration
TRADING_PAIRS = {
    "EUR/USD": {"symbol": "FX:EURUSD", "color": "#00BFFF"},
    "USD/JPY": {"symbol": "FX:USDJPY", "color": "#FF6347"},
    "GBP/USD": {"symbol": "FX:GBPUSD", "color": "#32CD32"},
    "USD/CHF": {"symbol": "OANDA:USDCHF", "color": "#9370DB"},
    "AUD/USD": {"symbol": "FX:AUDUSD", "color": "#FFD700"},
    "NZD/USD": {"symbol": "OANDA:NZDUSD", "color": "#20B2AA"},
    "USD/CAD": {"symbol": "CMCMARKETS:USDCAD", "color": "#FF4500"},
    "EUR/GBP": {"symbol": "FX:EURGBP", "color": "#BA55D3"},
}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Forex Pro Dashboard",
    layout="wide",
    page_icon="💹",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS & STYLING
# =========================================================
def load_custom_css():
    st.markdown(
        f"""
        <style>
        /* Futuristic dark background with animated grid */
        .stApp {{
            background:
                radial-gradient(circle at 15% 20%, rgba(255,215,0,0.09) 0%, transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(0,170,255,0.06) 0%, transparent 25%),
                linear-gradient(135deg, #0b0b0b 0%, #0a0a0a 100%);
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px),
                linear-gradient(0deg, rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 42px 42px, 42px 42px;
            animation: moveGrid 38s linear infinite;
            pointer-events: none;
            z-index: 0;
            opacity: 1;
        }}
        @keyframes moveGrid {{
            0%   {{ transform: translateY(0px); }}
            100% {{ transform: translateY(42px); }}
        }}
        
        /* Main content styling */
        .main, .block-container, .stTabs, .stMarkdown, .css-ffhzg2, .css-1d391kg {{ 
            position: relative; 
            z-index: 1; 
        }}
        
        /* Tab styling */
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: #FFD700 !important;
            color: black !important;
            font-weight: 700;
            padding: 14px 26px !important;
            border-radius: 10px;
            margin-right: 10px !important;
            box-shadow: 0 4px 8px rgba(255, 215, 0, 0.3);
            transition: all 0.3s ease;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="false"] {{
            background-color: #1b1b1b !important;
            color: #bbb !important;
            padding: 14px 26px !important;
            border-radius: 10px;
            margin-right: 10px !important;
            border: 1px solid #242424 !important;
            transition: all 0.3s ease;
        }}
        div[data-baseweb="tab-list"] button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(255, 215, 0, 0.2);
        }}
        
        /* Card styling */
        .card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.25);
            transition: all 0.3s ease;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.3);
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255,215,0,0.3);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255,215,0,0.5);
        }}
        
        /* Custom input styling */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input, 
        .stSelectbox>div>div>select {{
            background-color: rgba(255,255,255,0.05) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
        }}
        
        /* Custom button styling */
        .stButton>button {{
            border: 1px solid #FFD700 !important;
            color: #FFD700 !important;
            background-color: rgba(255, 215, 0, 0.1) !important;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: rgba(255, 215, 0, 0.2) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(255, 215, 0, 0.2);
        }}
        
        /* Custom metric styling */
        [data-testid="metric-container"] {{
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.25);
        }}
        
        /* Custom dataframe styling */
        .dataframe {{
            background-color: rgba(0,0,0,0.5) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

load_custom_css()

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def get_current_exchange_rate(base_currency, quote_currency):
    """Get current exchange rate using forex-python"""
    try:
        c = CurrencyRates()
        rate = c.get_rate(base_currency, quote_currency)
        return rate
    except Exception as e:
        st.warning(f"Couldn't fetch live rate: {str(e)}")
        return None

def format_currency(value, currency):
    """Format currency value with appropriate symbol"""
    currency_symbols = {
        "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
        "AUD": "A$", "CAD": "C$", "NZD": "NZ$", "CHF": "CHF"
    }
    symbol = currency_symbols.get(currency, currency)
    if currency == "JPY":
        return f"{symbol}{value:,.0f}"
    return f"{symbol}{value:,.2f}"

def get_crypto_price(crypto_symbol):
    """Get current crypto price (placeholder - would use API in production)"""
    crypto_prices = {
        "BTC": 65000, "ETH": 3500, "XRP": 0.60, 
        "SOL": 180, "ADA": 0.45, "DOGE": 0.12
    }
    return crypto_prices.get(crypto_symbol.upper(), 0)

# =========================================================
# NAVIGATION
# =========================================================
def create_navigation():
    tabs = [
        "📊 Market Overview", 
        "📅 Economic Calendar", 
        "💹 Trading Tools",
        "📈 Technical Analysis",
        "📚 Education",
        "👤 My Account"
    ]
    return st.tabs(tabs)

# Initialize navigation
selected_tab = create_navigation()
