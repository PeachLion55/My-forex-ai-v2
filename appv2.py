# zenvodash_app.py
# Streamlit dashboard with:
# - Lightweight Charts component (drawing tools + replay + localStorage persistence)
# - Myfxbook connect UI that talks to a FastAPI backend
# - Revamped Journaling tab with per-trade tabs, screenshots, and notes
# Run with:  streamlit run zenvodash_app.py

import json
import time
import base64
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import requests
import numpy as np

st.set_page_config(page_title="ZenvoDash", page_icon="📊", layout="wide")

# ---------------------------
# SQLite setup
# ---------------------------
def get_conn():
    conn = sqlite3.connect("zenvodash.db", check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS myfxbook_tokens (username TEXT PRIMARY KEY, token TEXT, created_at TEXT)")
    return conn

conn = get_conn()
c = conn.cursor()

# ---------------------------
# Session state init
# ---------------------------
for key, default in [
    ("logged_in_user", None),
    ("drawings", {}),
    ("journal_trades", []),
    ("pair", "EUR/USD"),
    ("candles_cache", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------
# Utilities
# ---------------------------
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def get_user_data(username: str) -> Dict[str, Any]:
    r = c.execute("SELECT data FROM users WHERE username = ?", (username,)).fetchone()
    return json.loads(r[0]) if r and r[0] else {}

def save_user_data(username: str, data: Dict[str, Any]):
    c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(data), username))
    conn.commit()

def demo_candles() -> List[Dict[str, Any]]:
    # Generate simple OHLC series (hourly) for demo
    np.random.seed(0)
    t0 = int(time.time()) - 60*60*24*100  # ~100 days ago
    o = 1.10
    candles = []
    for i in range(1500):
        t = t0 + i*60*60  # hourly candles
        change = (np.random.randn() * 0.002)
        cpx = max(0.5, o + change)
        h = max(o, cpx) + abs(np.random.randn())*0.001
        l = min(o, cpx) - abs(np.random.randn())*0.001
        candles.append({"time": t, "open": round(o,5), "high": round(h,5), "low": round(l,5), "close": round(cpx,5)})
        o = cpx
    return candles

def get_pair_candles(pair: str) -> List[Dict[str, Any]]:
    if pair in st.session_state.candles_cache:
        return st.session_state.candles_cache[pair]
    data = demo_candles()
    st.session_state.candles_cache[pair] = data
    return data

# ---------------------------
# Lightweight Charts HTML component (tools + replay + localStorage persistence)
# ---------------------------
def render_lightweight_chart(pair: str, height: int = 520, width: int = 0):
    """Renders a Lightweight Charts canvas with:
      - Tools: cursor/select, trendline, hline, vline, rectangle, fibonacci
      - Draggable/resizable drawings
      - Save/Load to localStorage (per pair key)
      - Replay mode (choose start index, play/pause/speed/reset)
    Drawings are persisted in localStorage under key f"lw_drawings_{pair}".
    If st.session_state.drawings has JSON for this pair, it will be loaded.
    """
    import streamlit.components.v1 as components

    candles = get_pair_candles(pair)
    initial_drawings = st.session_state.drawings.get(pair, "")
    candles_json = json.dumps(candles)
    init_drawings_json = json.dumps(initial_drawings) if isinstance(initial_drawings, str) else json.dumps(initial_drawings or "")

    width_js = "parentDiv.clientWidth" if width == 0 else str(width)

    html = f"""
    <div id="lw-wrapper" style="width:100%; position:relative;">
      <style>
        .zw-toolbar {{
          display:flex; gap:8px; flex-wrap:wrap; align-items:center;
          font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          margin-bottom: 8px;
        }}
        .zw-toolbar button {{
          padding:6px 10px; border:1px solid #333; background:#1f2937; color:#fff; border-radius:8px; cursor:pointer;
        }}
        .zw-toolbar button.active {{ background:#0ea5e9; }}
        .zw-panel {{ background:#0b1220; border-radius:12px; padding:8px; }}
        .zw-status {{ color:#cbd5e1; font-size:12px; margin-left:6px; }}
        .zw-legend {{ color:#cbd5e1; font-size:12px; margin-top:4px; }}
        .zw-controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
        .zw-divider {{ width:1px; height:26px; background:#334155; margin:0 6px; }}
      </style>

      <div class="zw-panel">
        <div class="zw-toolbar">
          <div class="zw-controls">
            <button id="tool-select">Select</button>
            <button id="tool-trend">Trendline</button>
            <button id="tool-hline">H-Line</button>
            <button id="tool-vline">V-Line</button>
            <button id="tool-rect">Rectangle</button>
            <button id="tool-fib">Fibonacci</button>
            <span class="zw-divider"></span>
            <button id="btn-save">Save</button>
            <button id="btn-load">Load</button>
            <button id="btn-clear">Clear</button>
            <span class="zw-divider"></span>
            <label style="color:#cbd5e1;">Replay start idx:</label>
            <input id="replay-start" type="number" min="0" step="1" value="300" style="width:90px;">
            <button id="replay-play">Play</button>
            <button id="replay-pause">Pause</button>
            <button id="replay-fast">Fast</button>
            <button id="replay-reset">Reset</button>
          </div>
          <span class="zw-status" id="status">Ready</span>
        </div>
        <div id="chart" style="width:100%; height:{height}px; position:relative;"></div>
        <div class="zw-legend">Tip: Click once or twice depending on tool. Drag endpoints to adjust. Data + drawings saved per pair.</div>
      </div>
    </div>

    <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        const pair = {json.dumps(pair)};
        const parentDiv = document.getElementById('lw-wrapper');
        const chartDiv = document.getElementById('chart');
        chartDiv.style.width = {width_js} + 'px';

        const data = {candles_json};
        let drawings = [];
        try {{
            const injected = {init_drawings_json};
            if (injected && typeof injected === 'string' && injected.trim().length>0) {{
                drawings = JSON.parse(injected);
            }} else if (injected && typeof injected === 'object') {{
                drawings = injected;
            }}
        }} catch(e) {{ /* ignore */ }}

        const chart = LightweightCharts.createChart(chartDiv, {{
            layout: {{
                background: {{ type: 'solid', color: '#0b1220' }},
                textColor: '#D1D5DB',
            }},
            grid: {{
                vertLines: {{ color: '#1F2937' }},
                horzLines: {{ color: '#1F2937' }},
            }},
            rightPriceScale: {{ borderColor: '#334155' }},
            timeScale: {{ borderColor: '#334155' }},
            crosshair: {{ mode: 0 }},
            width: chartDiv.clientWidth,
            height: {height},
        }});

        const series = chart.addCandlestickSeries({{
            upColor: '#22c55e', downColor: '#ef4444', borderVisible: false,
            wickUpColor: '#22c55e', wickDownColor: '#ef4444'
        }});
        series.setData(data);

        // Replay state
        let replayIndex = parseInt(document.getElementById('replay-start').value, 10) || 300;
        let playing = false;
        let speedMs = 800;

        function setStatus(msg) {{
            const s = document.getElementById('status');
            s.textContent = msg;
        }}

        // Overlay canvas for drawings
        const overlay = document.createElement('canvas');
        overlay.width = chartDiv.clientWidth;
        overlay.height = {height};
        overlay.style.position = 'absolute';
        overlay.style.left = '0';
        overlay.style.top = '0';
        overlay.style.pointerEvents = 'none';
        chartDiv.appendChild(overlay);
        const ctx = overlay.getContext('2d');

        // Handle resize
        const resizeObserver = new ResizeObserver((entries) => {{
           const w = chartDiv.clientWidth;
           chart.applyOptions({{ width: w }});
           overlay.width = w;
           overlay.height = {height};
           redraw();
        }});
        resizeObserver.observe(chartDiv);

        // Tool handling
        let currentTool = 'select';
        let tempPoints = [];
        let dragging = null; // {{id, handleIndex}}
        let hover = null;

        function setTool(t) {{
            currentTool = t;
            document.querySelectorAll('.zw-toolbar button').forEach(b=>b.classList.remove('active'));
            const idMap = {{
                'select':'tool-select','trend':'tool-trend','hline':'tool-hline','vline':'tool-vline','rect':'tool-rect','fib':'tool-fib'
            }};
            const el = document.getElementById(idMap[t]);
            if (el) el.classList.add('active');
            setStatus('Tool: ' + t);
            tempPoints = [];
        }}

        document.getElementById('tool-select').onclick = () => setTool('select');
        document.getElementById('tool-trend').onclick  = () => setTool('trend');
        document.getElementById('tool-hline').onclick  = () => setTool('hline');
        document.getElementById('tool-vline').onclick  = () => setTool('vline');
        document.getElementById('tool-rect').onclick   = () => setTool('rect');
        document.getElementById('tool-fib').onclick    = () => setTool('fib');
        setTool('select'); // default

        // Save/Load/Clear
        document.getElementById('btn-save').onclick = () => {{
            try {{
                localStorage.setItem('lw_drawings_' + pair, JSON.stringify(drawings));
                setStatus('Saved drawings to localStorage');
            }} catch(e) {{ setStatus('Save failed'); }}
        }};
        document.getElementById('btn-load').onclick = () => {{
            try {{
                const j = localStorage.getItem('lw_drawings_' + pair);
                if (j) {{
                   drawings = JSON.parse(j);
                   redraw();
                   setStatus('Loaded drawings from localStorage');
                }}
            }} catch(e) {{ setStatus('Load failed'); }}
        }};
        document.getElementById('btn-clear').onclick = () => {{
            drawings = [];
            redraw();
        }};

        // Replay controls
        function applyReplay() {{
            const slice = data.slice(0, Math.min(replayIndex, data.length));
            if (slice.length > 0) {{ series.setData(slice); }}
        }}
        document.getElementById('replay-reset').onclick = () => {{
            playing = false;
            replayIndex = parseInt(document.getElementById('replay-start').value, 10) || 300;
            series.setData(data.slice(0, replayIndex));
            setStatus('Replay reset');
        }};
        document.getElementById('replay-pause').onclick = () => {{ playing = false; }};
        document.getElementById('replay-fast').onclick = () => {{
            speedMs = Math.max(50, Math.floor(speedMs/2));
            setStatus('Speed: ' + speedMs + 'ms');
        }};
        document.getElementById('replay-start').onchange = () => {{
            replayIndex = parseInt(document.getElementById('replay-start').value, 10) || 300;
            applyReplay();
        }};

        function stepReplay() {{
            if (!playing) return;
            if (replayIndex < data.length) {{
                replayIndex += 1;
                series.update(data[replayIndex-1]);
                setTimeout(stepReplay, speedMs);
            }} else {{
                playing = false;
                setStatus('Replay finished');
            }}
        }}
        document.getElementById('replay-play').onclick = () => {{ playing = true; stepReplay(); }};

        // Coordinate helpers
        function pxToPrice(y) {{ return series.priceScale().coordinateToPrice(y); }}
        function priceToPx(p) {{ return series.priceScale().priceToCoordinate(p); }}
        function pxToTime(x)  {{ return chart.timeScale().coordinateToTime(x); }}
        function timeToPx(t)  {{ return chart.timeScale().timeToCoordinate(t); }}

        // Drawing primitives
        function newId() {{ return 'd' + Math.random().toString(36).slice(2); }}

        function drawAll() {{
            ctx.clearRect(0,0,overlay.width, overlay.height);
            drawings.forEach(d => drawOne(d));
            if (tempPoints.length > 0 && currentTool !== 'select') {{ drawPreview(); }}
        }}

        function drawOne(d) {{
            ctx.save();
            ctx.lineWidth = 2;
            ctx.strokeStyle = d.color || '#0ea5e9';
            ctx.fillStyle = (d.fill || 'rgba(14,165,233,0.15)');
            if (d.type === 'trend') {{
                const p1 = toXY(d.points[0]);
                const p2 = toXY(d.points[1]);
                strokeLineExtended(p1, p2);
                drawHandle(p1); drawHandle(p2);
            }} else if (d.type === 'hline') {{
                const y = priceToPx(d.price);
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(overlay.width, y); ctx.stroke();
                drawHandle({{x: 30, y}});
            }} else if (d.type === 'vline') {{
                const x = timeToPx(d.time);
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, overlay.height); ctx.stroke();
                drawHandle({{x, y: 30}});
            }} else if (d.type === 'rect') {{
                const a = toXY(d.points[0]), b = toXY(d.points[1]);
                const x = Math.min(a.x,b.x), y = Math.min(a.y,b.y);
                const w = Math.abs(a.x-b.x), h = Math.abs(a.y-b.y);
                ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
                drawHandle(a); drawHandle(b);
            }} else if (d.type === 'fib') {{
                const a = toXY(d.points[0]), b = toXY(d.points[1]);
                const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
                const left = Math.min(a.x,b.x), right = Math.max(a.x,b.x);
                levels.forEach(l => {{
                    const py = a.y + (b.y - a.y) * l;
                    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
                }});
                drawHandle(a); drawHandle(b);
            }}
            ctx.restore();
        }}

        function drawPreview() {{
            ctx.save();
            ctx.setLineDash([6,6]);
            ctx.strokeStyle = '#94a3b8';
            if (currentTool === 'trend' && tempPoints.length === 1) {{
                const p1 = toXY(tempPoints[0]);
                const p2 = {{x: lastMouse.x, y: lastMouse.y}};
                strokeLineExtended(p1, p2);
            }} else if (currentTool === 'rect' && tempPoints.length === 1) {{
                const a = toXY(tempPoints[0]), b = {{x:lastMouse.x, y:lastMouse.y}};
                const x = Math.min(a.x,b.x), y = Math.min(a.y,b.y);
                const w = Math.abs(a.x-b.x), h = Math.abs(a.y-b.y);
                ctx.strokeRect(x,y,w,h);
            }}
            ctx.restore();
        }}

        function drawHandle(p) {{
            ctx.save();
            ctx.fillStyle = '#f59e0b';
            ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI*2); ctx.fill();
            ctx.restore();
        }}

        function toXY(pt) {{ return {{x: timeToPx(pt.time), y: priceToPx(pt.price)}}; }}
        function fromXY(x,y) {{ return {{time: pxToTime(x), price: pxToPrice(y)}}; }}

        function strokeLineExtended(p1, p2) {{
            const dx = p2.x - p1.x; const dy = p2.y - p1.y;
            if (Math.abs(dx) < 1) {{
                ctx.beginPath(); ctx.moveTo(p1.x, 0); ctx.lineTo(p1.x, overlay.height); ctx.stroke(); return;
            }}
            const m = dy/dx;
            const b = p1.y - m * p1.x;
            const x0 = 0; const y0 = m*x0 + b;
            const x1 = overlay.width; const y1 = m*x1 + b;
            ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        }}

        // Mouse interaction
        let lastMouse = {{x:0,y:0}};
        chartDiv.addEventListener('mousemove', (e)=>{{
            const rect = chartDiv.getBoundingClientRect();
            lastMouse = {{x: e.clientX - rect.left, y: e.clientY - rect.top}};
            if (dragging) {{
                const d = drawings.find(x=>x.id===dragging.id);
                if (d) {{
                    if (d.type === 'trend' || d.type === 'rect' || d.type === 'fib') {{
                        d.points[dragging.handleIndex] = fromXY(lastMouse.x, lastMouse.y);
                    }} else if (d.type === 'hline') {{
                        d.price = pxToPrice(lastMouse.y);
                    }} else if (d.type === 'vline') {{
                        d.time = pxToTime(lastMouse.x);
                    }}
                }}
                redraw();
            }} else {{
                hover = hitTestHandles(lastMouse.x, lastMouse.y);
                chartDiv.style.cursor = hover ? 'grab' : (currentTool==='select' ? 'default':'crosshair');
            }}
        }});

        chartDiv.addEventListener('mousedown', (e)=>{{
            if (hover) {{ dragging = hover; return; }}
            if (currentTool === 'select') return;

            const pt = fromXY(lastMouse.x, lastMouse.y);
            if (currentTool === 'trend') {{
                tempPoints.push(pt);
                if (tempPoints.length === 2) {{
                    drawings.push({{id:newId(), type:'trend', points:[tempPoints[0], tempPoints[1]], color:'#0ea5e9'}});
                    tempPoints = [];
                }}
            }} else if (currentTool === 'hline') {{
                drawings.push({{id:newId(), type:'hline', price: pt.price, color:'#0ea5e9'}});
            }} else if (currentTool === 'vline') {{
                drawings.push({{id:newId(), type:'vline', time: pt.time, color:'#0ea5e9'}});
            }} else if (currentTool === 'rect') {{
                tempPoints.push(pt);
                if (tempPoints.length === 2) {{
                    drawings.push({{id:newId(), type:'rect', points:[tempPoints[0], tempPoints[1]], color:'#0ea5e9', fill:'rgba(14,165,233,0.12)'}});
                    tempPoints = [];
                }}
            }} else if (currentTool === 'fib') {{
                tempPoints.push(pt);
                if (tempPoints.length === 2) {{
                    drawings.push({{id:newId(), type:'fib', points:[tempPoints[0], tempPoints[1]], color:'#0ea5e9'}});
                    tempPoints = [];
                }}
            }}
            redraw();
        }});

        window.addEventListener('mouseup', ()=>{{ dragging = null; }});

        function hitTestHandles(x,y) {{
            const r = 8;
            for (const d of drawings) {{
                if (d.type==='trend' || d.type==='rect' || d.type==='fib') {{
                    for (let i=0;i<2;i++) {{
                        const p = toXY(d.points[i]);
                        const dx = p.x-x, dy=p.y-y;
                        if (dx*dx+dy*dy <= r*r) return {{id:d.id, handleIndex:i}};
                    }}
                }} else if (d.type==='hline') {{
                    const yy = priceToPx(d.price);
                    if (Math.abs(yy - y) < r) return {{id:d.id, handleIndex:0}};
                }} else if (d.type==='vline') {{
                    const xx = timeToPx(d.time);
                    if (Math.abs(xx - x) < r) return {{id:d.id, handleIndex:0}};
                }}
            }}
            return null;
        }}

        function redraw() {{ drawAll(); }}
        redraw();

        // Keep drawings aligned on time scale changes
        chart.timeScale().subscribeVisibleTimeRangeChange(()=> redraw());
    })();
    </script>
    """
    # Render
    import streamlit.components.v1 as components
    components.html(html, height=height+120, scrolling=False)

# ---------------------------
# Myfxbook Connect (Streamlit UI calling our FastAPI backend)
# ---------------------------
BACKEND_URL = st.secrets.get("MYFXBOOK_BACKEND_URL", "http://127.0.0.1:8000")

def myfxbook_connect_ui():
    st.markdown("### 🔗 Connect Myfxbook")
    if st.session_state.logged_in_user is None:
        st.info("Login to your ZenvoDash account to connect Myfxbook.")
        return
    with st.expander("Connect Myfxbook", expanded=True):
        email = st.text_input("Myfxbook Email")
        pw = st.text_input("Myfxbook Password", type="password")
        if st.button("Connect Myfxbook"):
            try:
                r = requests.post(f"{BACKEND_URL}/connect-myfxbook", json={
                    "username": st.session_state.logged_in_user,
                    "email": email,
                    "password": pw
                }, timeout=30)
                if r.status_code == 200 and r.json().get("ok"):
                    st.success("Myfxbook connected successfully.")
                else:
                    st.error(f"Failed: {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("#### Fetch Accounts")
    if st.button("Get My Accounts"):
        try:
            r = requests.get(f"{BACKEND_URL}/myfxbook/accounts", params={"username": st.session_state.logged_in_user}, timeout=30)
            if r.status_code == 200:
                st.json(r.json())
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

    account_id = st.text_input("Account ID for history/open orders/daily gain")
    cols = st.columns(3)
    with cols[0]:
        if st.button("Get History"):
            if account_id:
                r = requests.get(f"{BACKEND_URL}/myfxbook/history/{account_id}", params={"username": st.session_state.logged_in_user}, timeout=60)
                st.json(r.json() if r.status_code == 200 else {"error": r.text})
            else:
                st.warning("Enter Account ID")
    with cols[1]:
        if st.button("Get Open Orders"):
            if account_id:
                r = requests.get(f"{BACKEND_URL}/myfxbook/open-orders/{account_id}", params={"username": st.session_state.logged_in_user}, timeout=60)
                st.json(r.json() if r.status_code == 200 else {"error": r.text})
            else:
                st.warning("Enter Account ID")
    with cols[2]:
        if st.button("Get Daily Gain"):
            if account_id:
                r = requests.get(f"{BACKEND_URL}/myfxbook/daily-gain/{account_id}", params={"username": st.session_state.logged_in_user}, timeout=60)
                st.json(r.json() if r.status_code == 200 else {"error": r.text})
            else:
                st.warning("Enter Account ID")

# ---------------------------
# Journaling (tabs per trade + screenshots)
# ---------------------------
def journaling_ui():
    st.markdown("### 📝 Trade Journal")
    st.caption("Create a tab per trade. Add screenshots and reflections. Everything saves to your account.")

    with st.expander("➕ Add New Trade", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA:
            symbol = st.text_input("Symbol", value=st.session_state.pair)
        with colB:
            direction = st.selectbox("Direction", ["Long", "Short"])
        with colC:
            date = st.date_input("Date", value=datetime.utcnow().date())

        entry = st.number_input("Entry Price", value=1.0, format="%.5f")
        exitp = st.number_input("Exit Price", value=1.0, format="%.5f")
        qty = st.number_input("Quantity", value=1.0, min_value=0.0, step=0.1)
        rr = st.text_input("R Multiple (optional)", value="")

        ww = st.text_area("What went well?")
        wi = st.text_area("What can be improved?")
        notes = st.text_area("Notes")
        screenshots = st.file_uploader("Upload screenshots", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)

        if st.button("Add Trade"):
            tid = "T" + hashlib.md5(f"{symbol}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8]
            imgs = []
            for f in screenshots or []:
                b64 = base64.b64encode(f.read()).decode()
                imgs.append({"name": f.name, "b64": b64})
            trade = {
                "id": tid, "symbol": symbol, "direction": direction, "date": str(date),
                "entry": entry, "exit": exitp, "qty": qty, "r": rr, "went_well": ww,
                "improve": wi, "notes": notes, "images": imgs
            }
            st.session_state.journal_trades.append(trade)
            if st.session_state.logged_in_user:
                data = get_user_data(st.session_state.logged_in_user)
                data.setdefault("journal_trades", [])
                data["journal_trades"] = st.session_state.journal_trades
                save_user_data(st.session_state.logged_in_user, data)
            st.success(f"Trade {tid} added.")

    if st.session_state.logged_in_user:
        data = get_user_data(st.session_state.logged_in_user)
        if data.get("journal_trades") and not st.session_state.journal_trades:
            st.session_state.journal_trades = data["journal_trades"]

    trades = st.session_state.journal_trades
    if not trades:
        st.info("No trades yet. Add one above.")
        return

    tab_labels = [f"{t['id']} | {t['symbol']} | {t['direction']} | {t['date']}" for t in trades]
    tabs = st.tabs(tab_labels)
    for i, t in enumerate(trades):
        with tabs[i]:
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader(f"Trade {t['id']} — {t['symbol']} ({t['direction']})")
                render_lightweight_chart(pair=t['symbol'] if "/" in t['symbol'] else st.session_state.pair, height=320)
            with col2:
                st.markdown("#### Trade Details")
                st.write(f"**Date:** {t['date']}")
                st.write(f"**Entry:** {t['entry']:.5f}")
                st.write(f"**Exit:** {t['exit']:.5f}")
                st.write(f"**Qty:** {t['qty']}")
                if t.get("r"):
                    st.write(f"**R Multiple:** {t['r']}")
                st.markdown("#### Reflections")
                st.write(f"**What went well:** {t.get('went_well','')}")
                st.write(f"**What can be improved:** {t.get('improve','')}")
                st.write(f"**Notes:** {t.get('notes','')}")
                st.markdown("#### Screenshots")
                if t.get("images"):
                    for im in t["images"]:
                        st.image(base64.b64decode(im["b64"]), caption=im["name"], use_column_width=True)
                else:
                    st.caption("No screenshots attached.")

            with st.expander("✏️ Edit Trade"):
                t["went_well"] = st.text_area("What went well?", value=t.get("went_well",""), key=f"ww_{t['id']}")
                t["improve"] = st.text_area("What can be improved?", value=t.get("improve",""), key=f"wi_{t['id']}")
                t["notes"] = st.text_area("Notes", value=t.get("notes",""), key=f"nt_{t['id']}")
                if st.button("Save Changes", key=f"save_{t['id']}"):
                    if st.session_state.logged_in_user:
                        data = get_user_data(st.session_state.logged_in_user)
                        data["journal_trades"] = st.session_state.journal_trades
                        save_user_data(st.session_state.logged_in_user, data)
                    st.success("Saved.")

# ---------------------------
# Account UI (Login/Sign up tabs + benefits section)
# ---------------------------
def account_ui():
    st.title("👤 My Account")
    st.caption("Sign in to sync drawings, journal, and Myfxbook. Gain insights with gamified metrics and benchmarks.")
    tab_login, tab_signup, tab_benefits = st.tabs(["Sign In", "Sign Up", "Benefits"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            if submitted:
                row = c.execute("SELECT password FROM users WHERE username = ?", (username,)).fetchone()
                if row and row[0] == hash_pw(password):
                    st.session_state.logged_in_user = username
                    st.success(f"Welcome back, {username}!")
                else:
                    st.error("Invalid credentials.")

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account")
            if submitted:
                if not new_username or not new_password:
                    st.error("Please fill all fields.")
                elif new_password != confirm:
                    st.error("Passwords do not match.")
                elif c.execute("SELECT username FROM users WHERE username=?", (new_username,)).fetchone():
                    st.error("Username already exists.")
                else:
                    c.execute("INSERT INTO users (username, password, data) VALUES (?,?,?)",
                              (new_username, hash_pw(new_password), json.dumps({})))
                    conn.commit()
                    st.session_state.logged_in_user = new_username
                    st.success("Account created.")

    with tab_benefits:
        st.subheader("Why create a ZenvoDash account?")
        st.markdown("""
        - 🔒 **Secure Sync** — Your drawings, journal, and preferences sync across devices.
        - 🏆 **Gamified Progress** — Levels, streaks, and badges to reinforce consistency.
        - 📈 **Performance Insights** — Benchmarks vs. peers and trend analytics.
        - 🔗 **Myfxbook Integration** — Pull performance & history into one dashboard.
        - ☁️ **Backups** — Export & import your complete account data as JSON anytime.
        """)
        if st.session_state.logged_in_user:
            if st.button("Logout"):
                st.session_state.logged_in_user = None
                st.success("Logged out.")

# ---------------------------
# Sidebar & Navigation
# ---------------------------
st.sidebar.title("ZenvoDash")
st.sidebar.caption("Next-level Trading Dashboard")

tabs = st.tabs(["Backtesting", "MT5 Performance", "Journal", "Account"])

with tabs[0]:
    st.title("📊 Backtesting")
    pairs_map = {
        "EUR/USD":"EUR/USD",
        "GBP/USD":"GBP/USD",
        "USD/JPY":"USD/JPY",
        "USD/CHF":"USD/CHF",
        "AUD/USD":"AUD/USD",
        "NZD/USD":"NZD/USD",
        "USD/CAD":"USD/CAD",
        "EUR/GBP":"EUR/GBP",
        "EUR/JPY":"EUR/JPY",
        "GBP/JPY":"GBP/JPY",
        "AUD/JPY":"AUD/JPY",
        "AUD/NZD":"AUD/NZD",
        "AUD/CAD":"AUD/CAD",
        "AUD/CHF":"AUD/CHF",
        "CAD/JPY":"CAD/JPY",
        "CHF/JPY":"CHF/JPY",
        "EUR/AUD":"EUR/AUD",
        "EUR/CAD":"EUR/CAD",
        "EUR/CHF":"EUR/CHF",
        "GBP/AUD":"GBP/AUD",
        "GBP/CAD":"GBP/CAD",
        "GBP/CHF":"GBP/CHF",
        "NZD/JPY":"NZD/JPY",
        "NZD/CAD":"NZD/CAD",
        "NZD/CHF":"NZD/CHF",
        "CAD/CHF":"CAD/CHF",
    }
    st.session_state.pair = st.selectbox("Select Pair", options=list(pairs_map.keys()), index=0, key="pair_select")
    render_lightweight_chart(pair=st.session_state.pair, height=520)

    st.markdown("#### Drawings JSON (export/import via manual copy)")
    colA, colB = st.columns(2)
    with colA:
        drawings_text = st.text_area("Paste drawings JSON here to store in account:", value=st.session_state.drawings.get(st.session_state.pair, ""), height=120, key="drawings_textarea")
        if st.button("Save drawings to account"):
            st.session_state.drawings[st.session_state.pair] = drawings_text
            if st.session_state.logged_in_user:
                data = get_user_data(st.session_state.logged_in_user)
                data.setdefault("drawings", {})
                data["drawings"][st.session_state.pair] = drawings_text
                save_user_data(st.session_state.logged_in_user, data)
            st.success("Saved drawings JSON to app state / account.")
    with colB:
        st.caption("Tip: Use the chart toolbar Save/Load for browser localStorage, or persist JSON to your account here.")

with tabs[1]:
    st.title("📈 MT5 Performance Dashboard")
    st.caption("Connect your Myfxbook to pull accounts, history, open orders, and performance.")
    myfxbook_connect_ui()

with tabs[2]:
    journaling_ui()

with tabs[3]:
    account_ui()
