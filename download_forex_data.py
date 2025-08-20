# =========================================================
# Pre-download all Forex pair data for all timeframes
# =========================================================
import yfinance as yf
import pandas as pd
import os

# Map of pairs to Yahoo tickers
pairs_map = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "CAD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "AUD/NZD": "AUDNZD=X",
    "AUD/CAD": "AUDCAD=X",
    "AUD/CHF": "AUDCHF=X",
    "CAD/JPY": "CADJPY=X",
    "CHF/JPY": "CHFJPY=X",
    "EUR/AUD": "EURAUD=X",
    "EUR/CAD": "EURCAD=X",
    "EUR/CHF": "EURCHF=X",
    "GBP/AUD": "GBPAUD=X",
    "GBP/CAD": "GBPCAD=X",
    "GBP/CHF": "GBPCHF=X",
    "NZD/JPY": "NZDJPY=X",
    "NZD/CAD": "NZDCAD=X",
    "NZD/CHF": "NZDCHF=X",
    "CAD/CHF": "CADCHF=X",
}

timeframes = ["1m","5m","15m","1h","4h","1d","1wk"]

# Create folder if not exists
os.makedirs("data", exist_ok=True)

for pair, symbol in pairs_map.items():
    for tf in timeframes:
        print(f"Downloading {pair} ({symbol}) at {tf}")
        try:
            data = yf.download(symbol, period="30d", interval=tf)
            data.reset_index(inplace=True)
            data.rename(columns={"Datetime":"time","Date":"time","Open":"open","High":"high",
                                 "Low":"low","Close":"close","Volume":"volume"}, inplace=True)

            # Convert time to UNIX timestamp
            def convert_time(x):
                if isinstance(x, pd.Timestamp):
                    return int(x.timestamp())
                elif isinstance(x, pd.Period):
                    return int(x.start_time.timestamp())
                return None
            data["time"] = data["time"].apply(convert_time)

            # Ensure numeric types
            for col in ["open","high","low","close"]:
                data[col] = data[col].astype(float)

            # Save CSV
            filename = f"data/{symbol}_{tf}.csv"
            data.to_csv(filename, index=False)
            print(f"Saved to {filename}")
        except Exception as e:
            print(f"Failed {pair} {tf}: {e}")
