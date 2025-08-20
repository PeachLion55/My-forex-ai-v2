import streamlit as st
import sqlite3
import json
import pandas as pd
from string import Template

# ---------- Helper to make JSON serializable ----------
def _to_jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

# ---------- SQLite setup ----------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    data TEXT
)
""")
conn.commit()

# ---------- Save user data ----------
def save_user_data(username, user_data):
    safe_data = _to_jsonable(user_data)
    c.execute("UPDATE users SET data = ? WHERE username = ?", 
              (json.dumps(safe_data), username))
    conn.commit()

# ---------- Prepare candles for Lightweight Charts ----------
def prepare_lightweight_candles(df: pd.DataFrame):
    if not {"timestamp", "open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame missing required columns")

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": int(pd.to_datetime(row["timestamp"]).timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"])
        })
    return candles

# ---------- Streamlit App ----------
st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("📊 MT5 Performance Dashboard")

# Example demo DataFrame for chart
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=50, freq="H"),
    "open": [1.1 + i*0.001 for i in range(50)],
    "high": [1.2 + i*0.001 for i in range(50)],
    "low": [1.0 + i*0.001 for i in range(50)],
    "close": [1.15 + i*0.001 for i in range(50)],
})

candles = prepare_lightweight_candles(df)
chart_json = json.dumps(candles)

# ---------- Inject Lightweight Charts ----------
chart_template = Template("""
<div id="chart-container" style="width:100%; height:500px;"></div>
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {
    const container = document.getElementById("chart-container");
    if (!container) return;

    const chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 500,
        layout: {
            background: { type: 'Solid', color: '#ffffff' },
            textColor: '#333'
        },
        grid: {
            vertLines: { color: '#f0f3fa' },
            horzLines: { color: '#f0f3fa' }
        }
    });

    const candleSeries = chart.addCandlestickSeries();
    const data = $candles;
    candleSeries.setData(data);

    // Resize listener
    window.addEventListener("resize", () => {
        chart.applyOptions({ width: container.clientWidth });
    });
})();
</script>
""")

st.components.v1.html(chart_template.substitute(candles=chart_json), height=520)
