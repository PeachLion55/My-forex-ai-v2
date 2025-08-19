
# gpt_app.py
# Streamlit Trading Analytics App – Extended with requested features
# ---------------------------------------------------------------
# How to run locally:
#   streamlit run gpt_app.py
#
# Notes:
# - This app stores lightweight user data (playbooks, checklist configs, community posts)
#   as JSON files under a local ./user_data directory by user_id.
# - MT5/Backtesting trade history should be supplied as a CSV upload (see Data > Backtesting / MT5 Upload).
#   Expected columns (case-insensitive, we auto-map if names differ):
#     - trade_id (optional), date/time, symbol/pair, timeframe, session (e.g., London, NY, Asia), 
#       direction (buy/sell), qty (optional), entry, exit, pips (optional), r (R-multiple), 
#       pnl (base currency), emotions (comma-separated tags like "fear,greed").
# - If your columns differ, use the "Column Mapper" in the Data tab to map your field names once and save.
# - The "Edge Finder" and "Session Statistics" work best when you provide timeframe, symbol, session, r/pips.
# - The "Psychology" features use the "emotions" column from the same dataset.
#
# Dependencies (as provided):
#   streamlit, requests, pandas, feedparser, textblob, plotly, newspaper3k, transformers, torch,
#   beautifulsoup4, selenium, pytz, yfinance, streamlit-autorefresh, forex-python
#
# --------------------------------------------------------------

import os
import io
import json
import math
import uuid
import time
import glob
import shutil
import random
import datetime as dt
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional: used if you want currency conversion for PnL
try:
    from forex_python.converter import CurrencyRates
    C = CurrencyRates()
except Exception:
    C = None

APP_VERSION = "1.0.0"

# ------------------- Utilities -------------------

DATA_ROOT = os.path.join(os.path.dirname(__file__), "user_data")
os.makedirs(DATA_ROOT, exist_ok=True)

def get_user_dir(user_id: str) -> str:
    d = os.path.join(DATA_ROOT, user_id)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "community_images"), exist_ok=True)
    os.makedirs(os.path.join(d, "playbooks"), exist_ok=True)
    return d

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def kpi_card(label, value, help_text=None):
    st.metric(label, value)
    if help_text:
        st.caption(help_text)

def safe_lower(s):
    return str(s).strip().lower().replace(" ", "_")

def human_pct(x, nd=1):
    if pd.isna(x):
        return "—"
    return f"{x*100:.{nd}f}%"

def human_num(x, nd=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"

def hash_id():
    return uuid.uuid4().hex[:12]

# ------------------- State & Config -------------------

DEFAULT_COLMAP = {
    "datetime": ["date", "datetime", "time", "timestamp"],
    "symbol": ["symbol", "pair", "instrument"],
    "timeframe": ["timeframe", "tf"],
    "session": ["session", "market_session"],
    "direction": ["direction", "side"],
    "qty": ["qty", "size", "volume"],
    "entry": ["entry", "entry_price"],
    "exit": ["exit", "exit_price"],
    "pips": ["pips", "pip"],
    "r": ["r", "r_multiple", "rr", "r_multiple"],
    "pnl": ["pnl", "profit", "loss", "net"],
    "emotions": ["emotions", "emotion", "mood"],
    "setup": ["setup", "strategy", "pattern"]
}

def init_session():
    if "user_id" not in st.session_state:
        st.session_state.user_id = "guest"
    if "colmap" not in st.session_state:
        st.session_state.colmap = DEFAULT_COLMAP.copy()
    if "trades_df" not in st.session_state:
        st.session_state.trades_df = pd.DataFrame()

init_session()

# ------------------- Data Management -------------------

def user_paths():
    user_dir = get_user_dir(st.session_state.user_id)
    return {
        "dir": user_dir,
        "colmap": os.path.join(user_dir, "column_map.json"),
        "checklist": os.path.join(user_dir, "pretrade_checklist.json"),
        "playbooks": os.path.join(user_dir, "playbooks"),
        "community": os.path.join(user_dir, "community.json"),
        "badges": os.path.join(user_dir, "badges.json"),
    }

def load_colmap():
    p = user_paths()["colmap"]
    stored = load_json(p, {})
    # Merge defaults with stored
    final = DEFAULT_COLMAP.copy()
    for k, v in stored.items():
        final[k] = v
    st.session_state.colmap = final

def save_colmap(new_map):
    p = user_paths()["colmap"]
    save_json(p, new_map)
    st.session_state.colmap = new_map

def map_columns(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    # return df with normalized columns (expected keys)
    norm = {}
    lower_cols = {safe_lower(c): c for c in df.columns}
    for target, candidates in colmap.items():
        for cand in candidates:
            key = safe_lower(cand)
            if key in lower_cols:
                norm[target] = lower_cols[key]
                break
    # Apply rename
    ren = {v: k for k, v in norm.items()}
    renamed = df.rename(columns=ren)
    return renamed

def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        try:
            df["datetime"] = pd.to_datetime(df["datetime"])
        except Exception:
            pass
    for numcol in ["pips","r","pnl","entry","exit","qty"]:
        if numcol in df.columns:
            df[numcol] = pd.to_numeric(df[numcol], errors="coerce")
    if "emotions" in df.columns:
        df["emotions"] = df["emotions"].fillna("").astype(str)
    if "timeframe" in df.columns:
        df["timeframe"] = df["timeframe"].astype(str)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    return df

# ------------------- Gamification Engine -------------------

BADGE_RULES = [
    # (badge_id, name, description, condition_func)
    ("first_10_trades", "First 10 Trades", "Log your first 10 trades.", lambda df: len(df) >= 10),
    ("green_week", "Green Week", "7-day positive PnL streak.", 
        lambda df: daily_pnl(df).tail(7)["pnl"].sum() > 0 if not daily_pnl(df).empty else False),
    ("discipline_5", "Discipline x5", "Complete pre-trade checklist 5 times.", None), # counter-based
    ("no_overrisk", "Risk Guardian", "No position risked > 1% in last 20 trades.", None),  # require 'risk' column if available
    ("emotion_logged", "Mindful Trader", "Log emotions for 20 trades.", lambda df: (df["emotions"].str.len()>0).sum() >= 20 if "emotions" in df.columns else False),
]

def daily_pnl(df):
    if "datetime" in df.columns and "pnl" in df.columns:
        tmp = df.dropna(subset=["datetime"]).copy()
        tmp["date"] = tmp["datetime"].dt.date
        return tmp.groupby("date", as_index=False)["pnl"].sum()
    return pd.DataFrame(columns=["date","pnl"])

def compute_streaks(df):
    # simple green-day streak based on daily pnl
    d = daily_pnl(df)
    if d.empty: 
        return {"current": 0, "best": 0}
    streak = 0
    best = 0
    for pnl in d["pnl"]:
        if pnl > 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return {"current": streak, "best": best}

def load_badges():
    p = user_paths()["badges"]
    return load_json(p, {"counters": {"discipline_5": 0}, "earned": []})

def save_badges(data):
    p = user_paths()["badges"]
    save_json(p, data)

def update_badges(df, badges_state):
    # auto earn badge based on rules
    earned = set(badges_state.get("earned", []))
    for bid, name, desc, cond in BADGE_RULES:
        if cond is None:
            # counter-based or unavailable
            continue
        try:
            if cond(df) and bid not in earned:
                earned.add(bid)
        except Exception:
            pass
    badges_state["earned"] = sorted(list(earned))
    save_badges(badges_state)
    return badges_state

# ------------------- Calculators -------------------

def percent_gain_to_recover(drawdown_pct):
    """
    If you're down D%, the gain needed to recover is D / (1 - D).
    drawdown_pct is expressed as 0.2 for 20%.
    """
    if drawdown_pct <= 0: 
        return 0.0
    if drawdown_pct >= 0.99:
        return float("inf")
    return drawdown_pct / (1 - drawdown_pct)

def trades_to_recover(drawdown_pct, winrate, avg_r):
    """
    Approximate number of trades to recover a given drawdown given winrate and average R multiple.
    Expected R per trade E[R] = winrate*avg_win - (1-winrate)*avg_loss
    Here we assume avg_win = avg_r, avg_loss = 1 (i.e., R is defined relative to risk).
    Then E[R] = winrate*avg_r - (1-winrate)*1
    If risk per trade is 'risk_pct', expected percentage gain per trade ≈ risk_pct * E[R]. (Handled in UI)
    We'll solve number of trades n s.t. (1 + g)^n >= 1 + target_gain
    with g = risk_pct * E[R].
    """
    # This function returns E[R], g is computed in UI to include risk per trade
    E_R = winrate*avg_r - (1 - winrate)*1.0
    return E_R

def expectancy_by_group(df, group_cols):
    # Expects 'r' column for R-multiple; groups & computes expectancy
    g = df.dropna(subset=["r"]).groupby(group_cols)
    res = g["r"].agg(
        trades="count",
        winrate=lambda s: (s > 0).mean(),
        avg_win=lambda s: s[s>0].mean() if (s>0).any() else 0.0,
        avg_loss=lambda s: -s[s<0].mean() if (s<0).any() else 0.0,
        expectancy=lambda s: (s > 0).mean()*(s[s>0].mean() if (s>0).any() else 0.0) - (1-(s>0).mean())*(-s[s<0].mean() if (s<0).any() else 0.0)
    ).reset_index()
    return res

def profit_factor(df):
    if "pnl" not in df.columns: 
        return np.nan
    gross_profit = df.loc[df["pnl"]>0, "pnl"].sum()
    gross_loss = -df.loc[df["pnl"]<0, "pnl"].sum()
    if gross_loss == 0:
        return np.nan if gross_profit == 0 else float("inf")
    return gross_profit / gross_loss

def max_drawdown_from_pnl(df):
    if df is None or df.empty or "pnl" not in df.columns:
        return np.nan
    equity = df["pnl"].fillna(0).cumsum()
    peak = equity.cummax()
    dd = (equity - peak)
    return dd.min()  # negative value

# ------------------- UI -------------------

st.set_page_config(page_title="Trading Analytics Pro", layout="wide")

with st.sidebar:
    st.title("Trading Analytics Pro")
    st.caption(f"Version {APP_VERSION}")
    st.session_state.user_id = st.text_input("User ID", st.session_state.user_id, help="Used for saving your data locally.")
    st.write("---")
    st.subheader("Data")
    st.caption("Upload your MT5/Backtesting CSV. Use the Column Mapper if your headings differ.")
    uploaded = st.file_uploader("Upload trades CSV", type=["csv"], key="csv_upload")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # Apply mapping if saved
        load_colmap()
        df = map_columns(df, st.session_state.colmap)
        df = ensure_types(df)
        st.session_state.trades_df = df
        st.success(f"Loaded {len(df)} rows.")
    if st.button("Save Current Column Map"):
        save_colmap(st.session_state.colmap)
        st.success("Column map saved.")
    if st.button("Reload Saved Column Map"):
        load_colmap()
        st.info("Column map reloaded.")

    st.write("---")
    st.subheader("Gamification")
    badges_state = load_badges()
    if "trades_df" in st.session_state and not st.session_state.trades_df.empty:
        badges_state = update_badges(st.session_state.trades_df, badges_state)
    earned = set(badges_state.get("earned", []))
    st.caption("Badges Earned")
    for bid, name, desc, _ in BADGE_RULES:
        if bid in earned:
            st.success(f"🏅 {name}")
    st.caption("Discipline Checklist Completions")
    st.write(badges_state.get("counters", {}).get("discipline_5", 0))

tabs = st.tabs([
    "Home", 
    "MT5 Stats Dashboard", 
    "Tools", 
    "Psychology", 
    "Playbook Builder",
    "Community Trade Ideas",
    "Data (Column Mapper)"
])

# ------------------- HOME -------------------
with tabs[0]:
    st.header("Welcome 👋")
    st.write("This build includes: Drawdown Recovery Planner, Edge Finder, Customisable Dashboards, Community Trade Ideas, Gamification, Psychology Tracker, Playbook Builder, Pre-Trade Checklist, Session Statistics, and Risk What‑If Analyzer.")
    df = st.session_state.trades_df
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card("Trades", len(df))
        with col2:
            if "r" in df.columns:
                kpi_card("Win Rate", human_pct((df["r"]>0).mean()))
            else:
                kpi_card("Win Rate", "—")
        with col3:
            pf = profit_factor(df)
            kpi_card("Profit Factor", "∞" if pf==float("inf") else (human_num(pf) if not pd.isna(pf) else "—"))
        with col4:
            dd = max_drawdown_from_pnl(df)
            kpi_card("Max Drawdown (PnL)", human_num(dd) if not pd.isna(dd) else "—")

# ------------------- MT5 STATS DASHBOARD -------------------
with tabs[1]:
    st.header("MT5 Stats Dashboard")
    df = st.session_state.trades_df
    if df.empty:
        st.info("Upload your trades in the sidebar to unlock this dashboard.")
    else:
        subtab = st.tabs(["Metrics & Edge Finder", "Customisable Dashboard"])

        # ---- Metrics & Edge Finder ----
        with subtab[0]:
            st.subheader("Core Metrics")
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Total Trades", len(df))
            with colB:
                st.metric("Win Rate", human_pct((df["r"]>0).mean()) if "r" in df.columns else "—")
            with colC:
                st.metric("Avg R", human_num(df["r"].mean()) if "r" in df.columns else "—")
            with colD:
                st.metric("Profit Factor", human_num(profit_factor(df)) if not pd.isna(profit_factor(df)) else "—")

            st.write(" ")
            st.subheader("Edge Finder – Highest Expectancy Segments")
            group_cols = []
            if "timeframe" in df.columns:
                group_cols.append("timeframe")
            if "symbol" in df.columns:
                group_cols.append("symbol")
            if "setup" in df.columns:
                group_cols.append("setup")
            if not group_cols:
                st.warning("Edge Finder requires at least one of: timeframe, symbol, setup.")
            else:
                agg = expectancy_by_group(df, group_cols)
                agg = agg.sort_values("expectancy", ascending=False)
                st.dataframe(agg, use_container_width=True)
                top_n = st.slider("Show Top N", 5, 50, 15)
                st.write(px.bar(agg.head(top_n), x="expectancy", y=group_cols, orientation="h"))

        # ---- Customisable Dashboard ----
        with subtab[1]:
            st.subheader("Customisable KPIs")
            all_kpis = [
                "Total Trades", "Win Rate", "Avg R", "Profit Factor",
                "Max Drawdown (PnL)", "Best Symbol", "Worst Symbol",
                "Best Timeframe", "Worst Timeframe", "Sharpe (approx)"
            ]
            chosen = st.multiselect("Select KPIs to display", all_kpis, default=["Total Trades","Win Rate","Avg R","Profit Factor"])
            cols = st.columns(4)
            i = 0

            def sharpe_approx(series):
                if series.std() == 0:
                    return np.nan
                return series.mean() / series.std() * np.sqrt(252)

            best_sym = df.groupby("symbol")["r"].mean().sort_values(ascending=False).index[0] if "symbol" in df.columns and not df["r"].isna().all() else "—"
            worst_sym = df.groupby("symbol")["r"].mean().sort_values(ascending=True).index[0] if "symbol" in df.columns and not df["r"].isna().all() else "—"
            best_tf = df.groupby("timeframe")["r"].mean().sort_values(ascending=False).index[0] if "timeframe" in df.columns and not df["r"].isna().all() else "—"
            worst_tf = df.groupby("timeframe")["r"].mean().sort_values(ascending=True).index[0] if "timeframe" in df.columns and not df["r"].isna().all() else "—"
            metrics_map = {
                "Total Trades": len(df),
                "Win Rate": human_pct((df["r"]>0).mean()) if "r" in df.columns else "—",
                "Avg R": human_num(df["r"].mean()) if "r" in df.columns else "—",
                "Profit Factor": human_num(profit_factor(df)) if not pd.isna(profit_factor(df)) else "—",
                "Max Drawdown (PnL)": human_num(max_drawdown_from_pnl(df)),
                "Best Symbol": best_sym,
                "Worst Symbol": worst_sym,
                "Best Timeframe": best_tf,
                "Worst Timeframe": worst_tf,
                "Sharpe (approx)": human_num(sharpe_approx(df["r"])) if "r" in df.columns else "—",
            }
            for k in chosen:
                with cols[i % 4]:
                    st.metric(k, metrics_map.get(k, "—"))
                i += 1

            st.write("---")
            st.subheader("Equity Curve")
            if "pnl" in df.columns:
                df["cum_pnl"] = df["pnl"].cumsum()
                st.plotly_chart(px.line(df.sort_values("datetime"), x="datetime", y="cum_pnl", markers=True), use_container_width=True)
            elif "r" in df.columns:
                df["cum_r"] = df["r"].cumsum()
                st.plotly_chart(px.line(df.sort_values("datetime"), x="datetime", y="cum_r", markers=True), use_container_width=True)

# ------------------- TOOLS -------------------
with tabs[2]:
    st.header("Tools")
    st.caption("Includes Drawdown Recovery Planner, Pre-Trade Checklist, Risk Management Calculator (with What‑If), and Trading Session Tracker.")
    sub = st.tabs(["Drawdown Recovery Planner", "Pre-Trade Checklist", "Risk Mgmt Calculator + What‑If", "Trading Session Tracker"])

    # ---- Drawdown Recovery Planner ----
    with sub[0]:
        st.subheader("Drawdown Recovery Planner")
        col1, col2, col3 = st.columns(3)
        dd_pct = col1.slider("Current Drawdown (%)", 0.0, 90.0, 20.0, 0.5) / 100.0
        winrate = col2.slider("Win Rate (%)", 10.0, 90.0, 50.0, 1.0) / 100.0
        avg_r = col3.slider("Average Win (R multiple)", 0.5, 5.0, 1.5, 0.1)
        risk_pct = st.slider("Risk per trade (% of equity)", 0.1, 5.0, 1.0, 0.1) / 100.0

        needed_gain = percent_gain_to_recover(dd_pct)
        st.metric("Gain Required to Break Even", human_pct(needed_gain))

        E_R = trades_to_recover(dd_pct, winrate, avg_r)
        g = risk_pct * E_R  # expected percentage gain per trade
        if g <= 0:
            st.warning("Expected gain per trade ≤ 0. Increase win rate / avg R or reduce risk.")
        else:
            n = math.ceil(math.log(1 + needed_gain) / math.log(1 + g))
            st.metric("Approx. Trades Needed", f"{n}")

        st.write("Projected Recovery")
        horizon = 100
        equity = [1.0]
        for i in range(horizon):
            equity.append(equity[-1]*(1+g))
        proj = pd.DataFrame({"trade": list(range(horizon+1)), "equity": equity})
        target = 1 + needed_gain
        fig = px.line(proj, x="trade", y="equity", title="Projected Equity Under Expected Return")
        fig.add_hline(y=target, line_dash="dot", annotation_text="Break-even target")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Pre-Trade Checklist ----
    with sub[1]:
        st.subheader("Pre-Trade Checklist Enforcement")
        p = user_paths()
        checklist_path = p["checklist"]
        current = load_json(checklist_path, {"criteria": [{"enabled": True, "text": "Trend Direction in favor"},
                                                         {"enabled": True, "text": "Confluence present"},
                                                         {"enabled": True, "text": "RR ≥ 1:2"}]})
        max_rows = 20
        st.caption("Toggle and edit up to 20 criteria. This configuration is saved to your account.")
        data = pd.DataFrame(current["criteria"][:max_rows])
        data = st.data_editor(data, num_rows="dynamic", use_container_width=True)
        if st.button("Save to my account"):
            new_list = data.to_dict(orient="records")[:max_rows]
            save_json(checklist_path, {"criteria": new_list})
            # increment discipline counter for gamification
            badges_state = load_badges()
            cnt = badges_state.get("counters", {}).get("discipline_5", 0) + 1
            badges_state.setdefault("counters", {})["discipline_5"] = cnt
            if cnt >= 5 and "discipline_5" not in badges_state.get("earned", []):
                badges_state.setdefault("earned", []).append("discipline_5")
            save_badges(badges_state)
            st.success("Checklist saved. ✅")

        st.write("---")
        st.subheader("Pre-Trade Enforcement")
        st.caption("You must tick all enabled criteria to proceed.")
        enabled = [c for c in load_json(checklist_path, {"criteria": []})["criteria"] if c.get("enabled")]
        status = {}
        for i, c in enumerate(enabled):
            status[i] = st.checkbox(c.get("text",""), key=f"ck_{i}")
        all_ok = all(status.values()) if enabled else True
        st.button("Proceed to Log Trade", disabled=not all_ok)

    # ---- Risk Mgmt Calculator + What-If ----
    with sub[2]:
        st.subheader("Risk Management Calculator")
        base_equity = st.number_input("Starting Equity", value=10000.0, min_value=0.0, step=100.0)
        risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
        winrate = st.slider("Win rate (%)", 10.0, 90.0, 50.0, 1.0) / 100.0
        avg_r = st.slider("Average R multiple", 0.5, 5.0, 1.5, 0.1)
        trades = st.slider("Number of trades", 10, 500, 100, 10)
        E_R = winrate*avg_r - (1-winrate)*1.0
        exp_growth = (1 + risk_pct*E_R) ** trades
        st.metric("Expected Growth Multiplier", human_num(exp_growth))

        st.write("---")
        st.subheader("What‑If Analyzer")
        alt_risk = st.slider("What if risk per trade was (%)", 0.1, 5.0, 0.5, 0.1) / 100.0
        alt_growth = (1 + alt_risk*E_R) ** trades
        if exp_growth > 0:
            dd_ratio = alt_risk / risk_pct if risk_pct>0 else np.nan
        st.metric("Alt Growth Multiplier", human_num(alt_growth))
        st.caption("Example: If you had risked 0.5% instead of 1%, your drawdown could scale roughly with risk (approximation).")

        sim = pd.DataFrame({
            "trade": list(range(trades+1)),
            "equity_base": base_equity * (1 + risk_pct*E_R) ** np.arange(trades+1),
            "equity_alt": base_equity * (1 + alt_risk*E_R) ** np.arange(trades+1),
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim["trade"], y=sim["equity_base"], mode="lines", name=f"Risk {risk_pct*100:.1f}%"))
        fig.add_trace(go.Scatter(x=sim["trade"], y=sim["equity_alt"], mode="lines", name=f"What‑If {alt_risk*100:.1f}%"))
        fig.update_layout(title="Equity Projection – Base vs What‑If", xaxis_title="Trade #", yaxis_title="Equity")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Trading Session Tracker ----
    with sub[3]:
        st.subheader("Trading Session Statistics")
        df = st.session_state.trades_df
        if df.empty or "session" not in df.columns:
            st.info("Requires 'session' column in your dataset (e.g., London, New York, Asia).")
        else:
            st.caption("Example insight: “In London open, EURUSD averages 28 pips in first 2 hours, your setups work 15% better here.”")
            by_sess = df.groupby(["session"]).agg(
                trades=("r","count"),
                winrate=("r", lambda s: (s>0).mean() if s.notna().any() else np.nan),
                avg_r=("r","mean"),
                avg_pips=("pips","mean") if "pips" in df.columns else ("r", "mean")
            ).reset_index()
            st.dataframe(by_sess, use_container_width=True)
            st.plotly_chart(px.bar(by_sess, x="session", y="avg_r", title="Average R by Session"), use_container_width=True)

            if "symbol" in df.columns:
                sess_symbol = df.groupby(["session","symbol"]).agg(expectancy=("r", lambda s: (s>0).mean()*(s[s>0].mean() if (s>0).any() else 0) - (1-(s>0).mean())*(-s[s<0].mean() if (s<0).any() else 0))).reset_index()
                st.plotly_chart(px.density_heatmap(sess_symbol, x="session", y="symbol", z="expectancy", title="Expectancy Heatmap"), use_container_width=True)

# ------------------- PSYCHOLOGY -------------------
with tabs[3]:
    st.header("Trading Psychology Tracker")
    df = st.session_state.trades_df
    if df.empty or "emotions" not in df.columns:
        st.info("Requires 'emotions' column in your dataset (comma-separated values per trade).")
    else:
        # explode emotions
        tmp = df.copy()
        tmp["emotions"] = tmp["emotions"].fillna("").astype(str)
        tmp["emotion_list"] = tmp["emotions"].apply(lambda x: [e.strip().lower() for e in x.split(",") if e.strip()])
        exploded = tmp.explode("emotion_list")
        exploded = exploded[exploded["emotion_list"].notna() & (exploded["emotion_list"]!="")]
        by_emotion = exploded.groupby("emotion_list").agg(
            trades=("r","count"),
            winrate=("r", lambda s: (s>0).mean() if s.notna().any() else np.nan),
            avg_r=("r","mean")
        ).reset_index().sort_values("trades", ascending=False)
        st.subheader("Emotions Impact on Results")
        st.dataframe(by_emotion, use_container_width=True)
        st.plotly_chart(px.bar(by_emotion, x="emotion_list", y="avg_r", title="Average R by Emotion"), use_container_width=True)

        st.write("---")
        st.subheader("Emotion Timeline")
        if "datetime" in df.columns:
            emo_daily = exploded.copy()
            emo_daily["date"] = pd.to_datetime(emo_daily["datetime"]).dt.date
            emo_daily = emo_daily.groupby(["date","emotion_list"]).agg(avg_r=("r","mean")).reset_index()
            st.plotly_chart(px.line(emo_daily, x="date", y="avg_r", color="emotion_list", markers=True), use_container_width=True)

# ------------------- PLAYBOOK BUILDER -------------------
with tabs[4]:
    st.header("Playbook Builder")
    p = user_paths()
    playbooks_dir = p["playbooks"]
    os.makedirs(playbooks_dir, exist_ok=True)

    mode = st.radio("Mode", ["Create", "View / Edit"], horizontal=True)
    if mode == "Create":
        name = st.text_input("Strategy Name")
        tags = st.text_input("Tags (comma separated)")
        description = st.text_area("Description / Notes", height=150)
        rules = st.text_area("Entry/Exit Rules", height=130)
        uploads = st.file_uploader("Attach screenshots (optional)", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
        if st.button("Save Playbook", disabled=(not name)):
            pb_id = hash_id()
            pb_dir = os.path.join(playbooks_dir, pb_id)
            os.makedirs(pb_dir, exist_ok=True)
            images = []
            for f in uploads or []:
                out = os.path.join(pb_dir, f.name)
                with open(out, "wb") as wf:
                    wf.write(f.read())
                images.append(os.path.join(pb_id, f.name))
            meta = {
                "id": pb_id,
                "name": name,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "description": description,
                "rules": rules,
                "images": images,
                "created_at": time.time(),
                "updated_at": time.time(),
                "performance": {},
            }
            save_json(os.path.join(pb_dir, "meta.json"), meta)
            st.success("Playbook saved.")
    else:
        all_pb = []
        for d in glob.glob(os.path.join(playbooks_dir, "*")):
            if os.path.isdir(d):
                meta = load_json(os.path.join(d,"meta.json"), {})
                if meta:
                    all_pb.append(meta)
        if not all_pb:
            st.info("No playbooks yet.")
        else:
            sel = st.selectbox("Select Playbook", [f'{pb["name"]} ({pb["id"]})' for pb in all_pb])
            pb = next(pb for pb in all_pb if f'{pb["name"]} ({pb["id"]})' == sel)
            st.subheader(pb["name"])
            st.caption(", ".join(pb.get("tags",[])))
            st.write(pb.get("description",""))
            st.write("**Rules**")
            st.code(pb.get("rules",""))
            img_cols = st.columns(3)
            for i, img_rel in enumerate(pb.get("images", [])):
                img_path = os.path.join(playbooks_dir, img_rel)
                with img_cols[i % 3]:
                    st.image(img_path, use_column_width=True)
            st.write("---")
            st.subheader("Edit")
            new_desc = st.text_area("Description", pb.get("description",""))
            new_rules = st.text_area("Rules", pb.get("rules",""))
            if st.button("Save Changes"):
                pb["description"] = new_desc
                pb["rules"] = new_rules
                pb["updated_at"] = time.time()
                save_json(os.path.join(playbooks_dir, pb["id"], "meta.json"), pb)
                st.success("Updated.")

# ------------------- COMMUNITY TRADE IDEAS -------------------
with tabs[5]:
    st.header("Community Trade Ideas")
    st.caption("Upload screenshots of your trade ideas so others can view.")
    p = user_paths()
    community_path = p["community"]
    db = load_json(community_path, {"posts": []})

    st.subheader("Create Post")
    title = st.text_input("Title")
    desc = st.text_area("Description")
    img = st.file_uploader("Screenshot", type=["png","jpg","jpeg","webp"], key="cti_upl")
    if st.button("Publish", disabled=(not title or img is None)):
        img_id = hash_id()
        out = os.path.join(p["dir"], "community_images", f"{img_id}_{img.name}")
        with open(out, "wb") as wf:
            wf.write(img.read())
        post = {"id": hash_id(), "title": title, "desc": desc, "image": out, "user": st.session_state.user_id, "ts": time.time()}
        db["posts"].insert(0, post)
        save_json(community_path, db)
        st.success("Posted!")

    st.write("---")
    st.subheader("Feed")
    if not db["posts"]:
        st.info("No community posts yet. Be the first!")
    else:
        for post in db["posts"]:
            st.markdown(f"### {post['title']}  \nby {post['user']} • {dt.datetime.fromtimestamp(post['ts']).strftime('%Y-%m-%d %H:%M')}")
            st.write(post["desc"])
            st.image(post["image"], use_column_width=True)
            st.write("---")

# ------------------- DATA (COLUMN MAPPER) -------------------
with tabs[6]:
    st.header("Data – Column Mapper")
    st.caption("Map your CSV column names to the expected fields. Save once; future uploads will apply automatically.")
    current = st.session_state.colmap.copy()
    for key in list(current.keys()):
        editable = st.tags_input if hasattr(st, "tags_input") else None
        vals = current[key]
        new_vals = st.text_input(f"{key} aliases (comma separated)", ", ".join(vals))
        current[key] = [v.strip() for v in new_vals.split(",") if v.strip()]
    if st.button("Save Mapping"):
        save_colmap(current)
        st.success("Saved.")

    if not st.session_state.trades_df.empty:
        st.write("Preview of your loaded dataset:")
        st.dataframe(st.session_state.trades_df.head(50), use_container_width=True)

st.write(" ")
st.caption("© 2025 Trading Analytics Pro – All rights reserved.")
