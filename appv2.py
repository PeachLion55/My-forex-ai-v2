import json
import math
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from string import Template

# ---------- Helper: build candles in the exact format Lightweight Charts needs ----------
def to_lw_candles(df: pd.DataFrame) -> list[dict]:
    """
    Accepts a DataFrame with columns:
      - time / datetime / date (any datetime-like or epoch) OR 'timestamp'
      - open, high, low, close
    Returns a list of dicts: [{time: <unix seconds>, open:..., high:..., low:..., close:...}, ...]
    """
    if df is None or df.empty:
        return []

    # Try to find a time-like column
    time_col = None
    for candidate in ["time", "timestamp", "datetime", "date"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        # Try index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "time"})
            time_col = "time"
        else:
            raise ValueError("No time/datetime column found (expected 'time', 'timestamp', 'datetime', or 'date').")

    # Normalize to pandas datetime
    if pd.api.types.is_datetime64_any_dtype(df[time_col]) is False:
        try:
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        except Exception:
            # Could be epoch seconds
            try:
                df[time_col] = pd.to_datetime(df[time_col].astype(int), unit="s", utc=True, errors="coerce")
            except Exception as e:
                raise ValueError(f"Could not parse time column: {e}")

    # Drop rows with invalid time
    df = df.dropna(subset=[time_col])

    # Ensure OHLC present
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    # Build list for JS (time as UNIX seconds)
    out = []
    for _, r in df.iterrows():
        t = int(r[time_col].to_pydatetime().replace(tzinfo=timezone.utc).timestamp())
        out.append({
            "time": t,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        })
    # Sort by time ascending (required)
    out.sort(key=lambda x: x["time"])
    return out

# ---------- Fallback demo data (renders chart even if no DataFrame) ----------
def demo_candles(n: int = 200, seed: int = 42) -> list[dict]:
    rng = pd.date_range(datetime.now(timezone.utc) - timedelta(days=n), periods=n, freq="D")
    price = 1.1000
    out = []
    for i, dt in enumerate(rng):
        # simple pseudo walk
        drift = math.sin(i / 12) * 0.0015
        price = max(0.5, price + drift + (0.0005 if i % 5 == 0 else -0.0003))
        o = price
        h = o + abs(drift) * 1.5 + 0.0006
        l = o - abs(drift) * 1.5 - 0.0006
        c = o + (0.0004 if i % 3 == 0 else -0.0002)
        out.append({
            "time": int(dt.timestamp()),
            "open": round(o, 5),
            "high": round(h, 5),
            "low":  round(l, 5),
            "close": round(c, 5),
        })
    return out

# ---------- Use your DataFrame if available; otherwise demo ----------
# Replace `your_df` with the DataFrame you actually use (ensure it has time/open/high/low/close)
your_df = None
try:
    # Example: if you already have a df called `ohlc_df` for selected pair
    # your_df = ohlc_df  # uncomment and set properly in your app
    pass
except Exception:
    pass

candles = to_lw_candles(your_df) if isinstance(your_df, pd.DataFrame) and not your_df.empty else demo_candles()

# ---------- Inject chart with Lightweight Charts ----------
chart_json = json.dumps(candles, ensure_ascii=False)

html_tpl = Template(r"""
<div id="chart-container" style="width: 100%; height: ${height}px;"></div>

<!-- Lightweight Charts -->
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>

<script>
(function(){
    const container = document.getElementById('chart-container');
    if (!container) { console.error('Chart container not found'); return; }

    const chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: ${height},
        layout: {
            background: { type: 'Solid', color: '#ffffff' },
            textColor: '#1f2937'
        },
        grid: {
            vertLines: { color: '#e5e7eb' },
            horzLines: { color: '#e5e7eb' }
        },
        crosshair: { mode: 1 },
        rightPriceScale: { borderVisible: false },
        timeScale: { borderVisible: false, rightOffset: 6, barSpacing: 8 }
    });

    const series = chart.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
    });

    const data = ${data_json};
    series.setData(data);

    // Handle responsive width
    function resize() {
        chart.applyOptions({ width: container.clientWidth, height: ${height} });
    }
    window.addEventListener('resize', resize);

    // Optional: fit content
    chart.timeScale().fitContent();

})();
</script>
""")

st.components.v1.html(
    html_tpl.substitute(
        height=500,           # match your previous widget height
        data_json=chart_json  # already valid JS array
    ),
    height=520
)
