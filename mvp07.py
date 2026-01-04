<<<<<<< HEAD

# app_polxium_mao_gemini.py
"""
Polxium — MAO-001 (fixed)
- Shows SVR predictions + small actual/predicted charts + portfolio rebalancer
- Gemini explanation button appears ONLY if GEMINI_API_KEY is present
- Robust error handling for missing data / environments
"""

import os
import math
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numbers
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

st.set_page_config(page_title="Polxium — MAO-001", layout="wide", initial_sidebar_state="expanded")

# ---------- Config ----------
ASSETS = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Reliance": "RELIANCE.NS",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Gold (GLD)": "GLD",
}
LOOKBACK_FEATURES = [5, 20, 60]
EVENT_WINDOWS = [
    ("2008-09-01", "2009-06-30", "Global Financial Crisis"),
    ("2010-04-01", "2011-01-31", "European Debt / Flash Crash"),
    ("2014-06-01", "2016-02-29", "Oil Crash & Global Slowdown"),
    ("2020-02-15", "2020-05-31", "COVID Crash"),
    ("2021-11-01", "2022-04-30", "Inflation Surge"),
    ("2022-03-01", "2023-09-30", "Interest-rate Hike Cycle"),
    ("2022-02-01", "2022-06-30", "Russia-Ukraine War"),
]

# ---------- Helpers ----------
@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()

def add_features(df):
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    # avoid log of zero or negative
    out["LogReturn"] = np.log(out["Close"].replace(0, np.nan)).diff()
    for w in LOOKBACK_FEATURES:
        out[f"MA{w}"] = out["Close"].rolling(w).mean()
        out[f"VOL{w}"] = out["Return"].rolling(w).std()
    out["Target_Close"] = out["Close"].shift(-1)
    out["Next_Return"] = out["Return"].shift(-1)
    out["Target_Dir"] = (out["Next_Return"] > 0).astype(int)
    out = out.dropna()
    return out

def prepare_xy(df_feat):
    feature_cols = ["Return", "LogReturn"] + [f for w in LOOKBACK_FEATURES for f in (f"MA{w}", f"VOL{w}")]
    # drop any columns with NaNs in features
    Xdf = df_feat[feature_cols].dropna()
    X = Xdf.values
    y_reg = df_feat.loc[Xdf.index, "Target_Close"].values
    y_cls = df_feat.loc[Xdf.index, "Target_Dir"].values
    dates = Xdf.index
    close = df_feat.loc[Xdf.index, "Close"]
    return X, y_reg, y_cls, dates, close

def train_svr_small(X_train, y_train, use_grid=True):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    svr = SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")
    # only run grid if enough rows and grid enabled
    if use_grid and len(X_train) > 200:
        param_grid = {"C":[10,100], "epsilon":[0.01,0.1]}
        splits = min(3, max(2, len(X_train)//200))
        try:
            tscv = TimeSeriesSplit(n_splits=splits)
            grid = GridSearchCV(SVR(kernel="rbf"), param_grid, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=1)
            grid.fit(Xs, y_train)
            svr = grid.best_estimator_
        except Exception:
            svr.fit(Xs, y_train)
    else:
        svr.fit(Xs, y_train)

    # simple logistic direction model (not critical)
    log = None
    scaler_cls = None
    try:
        scaler_cls = StandardScaler()
        Xs_cls = scaler_cls.fit_transform(X_train)
        # we need a target for logistic: use forward direction
        if len(y_train) > 2:
            cls_target = (y_train[1:] > y_train[:-1]).astype(int)
            # align Xs_cls to same length by dropping last row
            Xc = Xs_cls[:-1]
            log = LogisticRegression(max_iter=1000).fit(Xc, cls_target)
        else:
            log = None
    except Exception:
        log = None
        scaler_cls = None

    return svr, scaler, log, scaler_cls

def to_scalar_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            if getattr(x, "size", None) == 1:
                return float(x.iloc[0]) if hasattr(x, "iloc") else float(x.values.ravel()[0])
            return float(np.asarray(x).ravel()[0])
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.ravel()[0])
            return float(x.ravel()[0])
        if isinstance(x, numbers.Number):
            return float(x)
        return float(x)
    except Exception:
        return np.nan

def compute_suggested_holdings_from_returns(actual_returns_map, expected_returns_map):
    score = {}
    for k in actual_returns_map:
        ar = to_scalar_float(actual_returns_map.get(k, np.nan))
        er = to_scalar_float(expected_returns_map.get(k, np.nan))
        if math.isnan(ar): ar = 0.0
        if math.isnan(er): er = 0.0
        # heuristic scoring: expected_return more important, actual reduces/increases score
        s = max(er * 100.0 + ar * 10.0 + 0.001, 0.001)
        score[k] = s
    total = sum(score.values()) or 1.0
    suggested = {k: (v / total) * 100.0 for k, v in score.items()}
    return suggested

def format_pct_or_na(x, precision=2):
    try:
        if x is None:
            return "N/A"
        xv = float(x)
        if math.isnan(xv):
            return "N/A"
        return f"{xv*100:.{precision}f}%"
    except Exception:
        return "N/A"

# ---------- Gemini helpers ----------
def get_gemini_api_key():
    # check streamlit secrets first, then environment
    try:
        if isinstance(st.secrets, dict) and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

@st.cache_data(ttl=3600)
def cached_gemini_explain(cache_key: str, prompt_payload: str) -> str:
    # attempt to call Gemini; returns helpful error strings if not possible
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"Gemini SDK not installed or import failed. To enable, run: pip install google-generativeai. ({e})"

    api_key = get_gemini_api_key()
    if not api_key:
        return "GEMINI API KEY NOT FOUND. Set GEMINI_API_KEY in environment or Streamlit secrets."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        prompt = (
            "You are MAO-001 (Polxium). Given the numeric summary below for a single ticker "
            "and its performance over a specified period, produce 3 clear bullet points (1-2 sentences each):\n"
            "  1) A succinct description of performance (direction + magnitude). Mention any obvious crisis/period if it overlaps.\n"
            "  2) A short explanation of possible causes or market drivers.\n"
            "  3) One plain-language note for a retail investor.\n"
            "End with: 'This is experimental and not financial advice.'\n\n"
            "=== INPUT SUMMARY ===\n"
            f"{prompt_payload}\n\n"
            "Be concise and avoid jargon."
        )

        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            try:
                text = resp.output[0].content[0].text
            except Exception:
                text = str(resp)
        return text.strip()
    except Exception as e:
        return f"Gemini call failed: {e}"

# ---------- UI ----------
hcol1, hcol2 = st.columns([3,1])
with hcol1:
    st.markdown("<div style='font-size:36px; font-weight:700;'>Polxium</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px; color:gray; margin-top:-10px;'>MAO-001 — Market Analysis & Overview</div>", unsafe_allow_html=True)
with hcol2:
    st.write("")

st.write("---")

with st.sidebar:
    st.header("Controls")
    asset_select = st.selectbox("Choose asset", list(ASSETS.keys()), index=0)
    start_date = st.date_input("Start date", value=pd.to_datetime("2007-01-01").date(), key="start")
    end_date = st.date_input("End date", value=pd.Timestamp.today().date(), key="end")
    fast_mode = st.checkbox("Fast mode (no grid search)", value=True)
    st.markdown("## Portfolio settings")
    if 'weights' not in st.session_state:
        default_pct = round(100.0 / len(ASSETS), 2)
        st.session_state['weights'] = {k: default_pct for k in ASSETS.keys()}
    if '_normalizing' not in st.session_state:
        st.session_state['_normalizing'] = False

    def _normalize_all():
        if st.session_state.get('_normalizing', False):
            return
        st.session_state['_normalizing'] = True
        current = {}
        for name in ASSETS.keys():
            key = f"input_{name}"
            val = st.session_state.get(key, None)
            if val is None:
                val = st.session_state['weights'].get(name, 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            current[name] = max(val, 0.0)
        total = sum(current.values())
        if total <= 0:
            n = len(current)
            normalized = {k: round(100.0 / n, 2) for k in current.keys()}
        else:
            normalized = {}
            for k, v in current.items():
                normalized[k] = round((v / total) * 100.0, 2)
        for k, v in normalized.items():
            st.session_state[f"input_{k}"] = v
            st.session_state['weights'][k] = v
        st.session_state['_normalizing'] = False

    for name in ASSETS.keys():
        key = f"input_{name}"
        if key not in st.session_state:
            st.session_state[key] = st.session_state['weights'].get(name, round(100.0/len(ASSETS),2))
        st.number_input(f"{name} %", min_value=0.0, max_value=100.0, value=st.session_state[key],
                        step=0.5, key=key, on_change=_normalize_all)
    st.markdown("Values auto-scale to 100% after any edit.")
    run = st.button("Run analysis")

col_left, col_right = st.columns([3,1])
with col_left:
    st.markdown(f"### {asset_select}")
    short_map = {
        "Apple": "Large-cap Technology (consumer hardware & services).",
        "Tesla": "Large-cap Automotive & EV; high growth & volatility.",
        "Reliance": "Indian conglomerate: energy, retail, telecom.",
        "Microsoft": "Large-cap Technology (cloud, software & enterprise).",
        "Nvidia": "AI/semiconductor leader; high growth, cyclical.",
        "Gold (GLD)": "Gold ETF — safe-haven commodity exposure.",
    }
    st.markdown(f"**{short_map.get(asset_select,'')}**")
with col_right:
    st.write("")

st.write("---")

if run:
    # prepare weights used
    weights_used = {k: float(st.session_state.get(f"input_{k}", st.session_state['weights'].get(k, 0.0))) for k in ASSETS.keys()}
    df_main = download_data(ASSETS[asset_select], start_date.isoformat(), end_date.isoformat())
    if df_main.empty:
        st.error("No data returned for this asset & date range. Try different dates.")
        st.stop()

    # Features & modelling
    preds = np.array([])
    dates_test = np.array([])
    regime_labels = np.array([])
    df_feat = add_features(df_main) if len(df_main) > 10 else None
    if df_feat is not None and len(df_feat) > 30:
        try:
            X, y_reg, y_cls, dates_all, close_all = prepare_xy(df_feat)
            n = len(X)
            split = int(n * 0.8)
            if split < 5: split = int(n * 0.7)
            if split >= n: split = n-1
            X_train, X_test = X[:split], X[split:]
            y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
            dates_train, dates_test = dates_all[:split], dates_all[split:]
            svr_model, scaler_reg, log_model, scaler_cls = train_svr_small(X_train, y_reg_train, use_grid=(not fast_mode))
            if scaler_reg is not None and len(X_test)>0:
                X_test_s = scaler_reg.transform(X_test)
                preds = svr_model.predict(X_test_s)
            else:
                preds = np.array([])
            if log_model is not None and scaler_cls is not None and len(X_test)>1:
                try:
                    probs_up = log_model.predict_proba(scaler_cls.transform(X_test)[:-1])[:,1]
                    regime_labels = np.array(["GOOD" if p>0.6 else ("POOR" if p<0.4 else "STABLE") for p in probs_up])
                    # align preds length with regime_labels if needed (regression preds length may differ)
                    if len(regime_labels) < len(preds):
                        regime_labels = np.concatenate([regime_labels, np.array(["STABLE"]*(len(preds)-len(regime_labels)))])
                    elif len(regime_labels) > len(preds):
                        regime_labels = regime_labels[:len(preds)]
                except Exception:
                    regime_labels = np.array(["STABLE"] * len(preds))
            else:
                regime_labels = np.array(["STABLE"] * len(preds))
        except Exception:
            preds = np.array([]); regime_labels = np.array([])

    # Plotting main
    col_main, col_side = st.columns((3,1))
    with col_main:
        st.subheader(f"{asset_select} — Actual vs MAO-001 (SVR) predicted")
        fig, ax = plt.subplots(figsize=(11,5), constrained_layout=True)
        ax.grid(alpha=0.25, linestyle="--")
        ax.plot(df_main.index, df_main["Close"].values, color="black", linewidth=1.0, label="Actual Close")
        if preds.size > 0:
            ax.plot(dates_test, preds, color="#1f77b4", linewidth=1.6, alpha=0.95, label="SVR Predicted (test)")
            good_mask = regime_labels == "GOOD"
            poor_mask = regime_labels == "POOR"
            stable_mask = regime_labels == "STABLE"
            if good_mask.any():
                ax.scatter(np.array(dates_test)[good_mask], np.array(preds)[good_mask], color="green", s=26, label="GOOD")
            if poor_mask.any():
                ax.scatter(np.array(dates_test)[poor_mask], np.array(preds)[poor_mask], color="red", s=26, label="POOR")
            if stable_mask.any():
                ax.scatter(np.array(dates_test)[stable_mask], np.array(preds)[stable_mask], color="orange", s=20, label="STABLE")
        ymin, ymax = ax.get_ylim()
        label_positions = np.linspace(ymax - (ymax - ymin)*0.05, ymax - (ymax - ymin)*0.25, len(EVENT_WINDOWS))
        for i,(s,e,label) in enumerate(EVENT_WINDOWS):
            try:
                sdt = pd.to_datetime(s)
                edt = pd.to_datetime(e)
                # only draw event if intersects date range
                if edt >= df_main.index[0] and sdt <= df_main.index[-1]:
                    ax.axvspan(sdt, edt, color="gray", alpha=0.10)
                    ax.text(sdt, label_positions[i % len(label_positions)], label, fontsize=8, rotation=90, va="top", color="#444")
            except Exception:
                pass
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # ---------- Company summary (SAFE — no crashes if data missing) ----------
st.markdown("### Company summary (numeric)")

close_vals = df_main["Close"].values
num_days = len(close_vals)

if num_days == 0:
    st.warning("No price data available for this period.")
else:
    first_close = float(close_vals[0])
    last_close  = float(close_vals[-1])

    total_return = (last_close / (first_close + 1e-12)) - 1.0

    recent_30d_pct = None
    if num_days > 30:
        recent_30d_pct = (last_close - close_vals[-30]) / (close_vals[-30] + 1e-12)

    st.write(f"- Date range: {start_date} → {end_date} ({num_days} trading days)")
    st.write(f"- Start close: {first_close:.4f} → Last close: {last_close:.4f}")
    st.write(f"- Total return over period: {total_return*100:.2f}%")
    st.write(f"- Recent ~30-day change (approx): {format_pct_or_na(recent_30d_pct)}")


    with col_side:
        st.markdown("### Small charts")
        fig1, ax1 = plt.subplots(figsize=(3.6,2.8), constrained_layout=True)
        ax1.plot(df_main.index, df_main["Close"].values, linewidth=1); ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("Actual", fontsize=10)
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots(figsize=(3.6,2.8), constrained_layout=True)
        if preds.size > 0:
            ax2.plot(dates_test, preds, color="#1f77b4", linewidth=1)
        else:
            ax2.text(0.5,0.5,"No predictions", ha="center")
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("Predicted (test)", fontsize=10)
        st.pyplot(fig2)

    # ---------------- Gemini explanation UI (only if key present) ----------------
    st.markdown("---")
    st.subheader("MAO-001 AI explanation (Gemini)")
    gem_key = get_gemini_api_key()
    if not gem_key:
        st.info("AI explanation is disabled for this demo. Add a GEMINI_API_KEY to enable it.")
    else:
        st.caption("Press the button to generate a short natural-language explanation. Cached for 1 hour.")
        payload = {
            "ticker": ASSETS[asset_select],
            "start": str(start_date),
            "end": str(end_date),
            "days": int(num_days),
            "start_close": round(first_close, 6),
            "last_close": round(last_close, 6),
            "total_return_pct": round(total_return*100, 3),
            "recent_30d_pct": (round(float(recent_30d_pct*100), 3) if recent_30d_pct is not None and not math.isnan(float(recent_30d_pct)) else None),
            "pred_mean": (round(float(np.mean(preds)),4) if preds.size>0 and not math.isnan(np.mean(preds)) else None),
            "pred_len": int(len(preds)),
            "regimes": dict(pd.Series(regime_labels).value_counts().to_dict()) if regime_labels.size>0 else {},
        }
        payload_str = "\n".join([f"{k}: {v}" for k, v in payload.items()])
        cache_key = f"gemini_{ASSETS[asset_select]}_{start_date}_{end_date}"
        if st.button("Generate AI explanation (MAO-001)"):
            with st.spinner("Requesting Gemini (cached 1h)..."):
                explanation = cached_gemini_explain(cache_key, payload_str)
                st.markdown("#### Gemini explanation")
                st.write(explanation)

    # ---------------- Portfolio analyzer ----------------
    st.write("---")
    st.subheader("Portfolio analyzer (MAO-001 driven rebalancing)")
    with st.spinner("Analyzing portfolio across selected period..."):
        actual_returns = {}
        expected_returns = {}
        for name, ticker in ASSETS.items():
            df = download_data(ticker, start_date.isoformat(), end_date.isoformat())
            if df.empty or "Close" not in df.columns:
                actual_returns[name] = np.nan; expected_returns[name] = np.nan; continue
            try:
                actual_returns[name] = to_scalar_float((df["Close"].iloc[-1] / (df["Close"].iloc[0] + 1e-12)) - 1)
            except Exception:
                actual_returns[name] = np.nan
            try:
                df_feat2 = add_features(df)
                if df_feat2 is None or len(df_feat2) < 40:
                    expected_returns[name] = actual_returns[name]; continue
                X2, y2, _, _, _ = prepare_xy(df_feat2)
                n2 = len(X2)
                split2 = int(n2 * 0.8)
                if split2 < 10: split2 = int(n2 * 0.7)
                if split2 >= n2: split2 = n2-1
                X_train2, X_test2 = X2[:split2], X2[split2:]
                y_train2, y_test2 = y2[:split2], y2[split2:]
                svr_model2, scaler2, _, _ = train_svr_small(X_train2, y_train2, use_grid=(not fast_mode))
                if scaler2 is None or X_test2.size == 0:
                    expected_returns[name] = actual_returns[name]; continue
                X_test_s2 = scaler2.transform(X_test2)
                p2 = svr_model2.predict(X_test_s2)
                if p2.size>0:
                    # expected return based on mean predicted / last observed train close
                    baseline = float(y_train2[-1]) if len(y_train2)>0 else float(df["Close"].iloc[-1])
                    pred_rets = (p2 - baseline) / (abs(baseline) + 1e-12)
                    expected_returns[name] = to_scalar_float(np.nanmean(pred_rets))
                else:
                    expected_returns[name] = actual_returns[name]
            except Exception:
                expected_returns[name] = np.nan

        suggested = compute_suggested_holdings_from_returns(actual_returns, expected_returns)

        rows = []
        for name in ASSETS.keys():
            rows.append({
                "Asset": name,
                "Ticker": ASSETS[name],
                "CurrentHolding%": round(to_scalar_float(weights_used.get(name, 0.0)), 2),
                "ActualTotalReturn%": (round(to_scalar_float(actual_returns.get(name, np.nan))*100,2) if not math.isnan(to_scalar_float(actual_returns.get(name, np.nan))) else np.nan),
                "ExpectedReturn(SVR)%": (round(to_scalar_float(expected_returns.get(name, np.nan))*100,2) if not math.isnan(to_scalar_float(expected_returns.get(name, np.nan))) else np.nan),
                "SuggestedHolding%": round(to_scalar_float(suggested.get(name, np.nan)), 2)
            })
        df_port = pd.DataFrame(rows).sort_values("SuggestedHolding%", ascending=True).reset_index(drop=True)

    st.markdown("**Suggested holdings (ascending by suggested %)**")
    st.dataframe(df_port[["Asset","Ticker","CurrentHolding%","ActualTotalReturn%","ExpectedReturn(SVR)%","SuggestedHolding%"]].round(2), height=280)

    # textual summaries
    df_nonan = df_port.dropna(subset=["ActualTotalReturn%"])
    if not df_nonan.empty:
        top = df_nonan.sort_values("ActualTotalReturn%", ascending=False).head(2)
        bottom = df_nonan.sort_values("ActualTotalReturn%", ascending=True).head(2)
        desc1 = f"Top performers: {', '.join(top['Asset'].tolist())}. Largest declines: {', '.join(bottom['Asset'].tolist())}."
    else:
        desc1 = "No usable returns found in the selected range."
    rec_lines = []
    for _, r in df_port.iterrows():
        asset = r["Asset"]
        er = r["ExpectedReturn(SVR)%"]
        er_str = ("N/A" if pd.isna(er) else f"{er}%")
        if pd.isna(er):
            action = "HOLD"
        else:
            if er > 5:
                action = "BUY / increase"
            elif er < -2:
                action = "SELL / decrease"
            else:
                action = "HOLD"
        rec_lines.append(f"{asset}: {action} (expected {er_str}).")
    desc2 = " ".join(rec_lines)
if run:
    st.markdown("**Portfolio — current situation**"); st.write(desc1)
    st.markdown("**Portfolio — recommended actions (why change/hold)**"); st.write(desc2)
    st.info("MAO-001 predictions are experimental and intended for pattern analysis. Not financial advice.")

else:
    st.info("Adjust weights and press Run analysis.")
=======

# app_polxium_mao_gemini.py
"""
Polxium — MAO-001 (fixed)
- Shows SVR predictions + small actual/predicted charts + portfolio rebalancer
- Gemini explanation button appears ONLY if GEMINI_API_KEY is present
- Robust error handling for missing data / environments
"""

import os
import math
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numbers
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

st.set_page_config(page_title="Polxium — MAO-001", layout="wide", initial_sidebar_state="expanded")

# ---------- Config ----------
ASSETS = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Reliance": "RELIANCE.NS",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Gold (GLD)": "GLD",
}
LOOKBACK_FEATURES = [5, 20, 60]
EVENT_WINDOWS = [
    ("2008-09-01", "2009-06-30", "Global Financial Crisis"),
    ("2010-04-01", "2011-01-31", "European Debt / Flash Crash"),
    ("2014-06-01", "2016-02-29", "Oil Crash & Global Slowdown"),
    ("2020-02-15", "2020-05-31", "COVID Crash"),
    ("2021-11-01", "2022-04-30", "Inflation Surge"),
    ("2022-03-01", "2023-09-30", "Interest-rate Hike Cycle"),
    ("2022-02-01", "2022-06-30", "Russia-Ukraine War"),
]

# ---------- Helpers ----------
@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()

def add_features(df):
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    # avoid log of zero or negative
    out["LogReturn"] = np.log(out["Close"].replace(0, np.nan)).diff()
    for w in LOOKBACK_FEATURES:
        out[f"MA{w}"] = out["Close"].rolling(w).mean()
        out[f"VOL{w}"] = out["Return"].rolling(w).std()
    out["Target_Close"] = out["Close"].shift(-1)
    out["Next_Return"] = out["Return"].shift(-1)
    out["Target_Dir"] = (out["Next_Return"] > 0).astype(int)
    out = out.dropna()
    return out

def prepare_xy(df_feat):
    feature_cols = ["Return", "LogReturn"] + [f for w in LOOKBACK_FEATURES for f in (f"MA{w}", f"VOL{w}")]
    # drop any columns with NaNs in features
    Xdf = df_feat[feature_cols].dropna()
    X = Xdf.values
    y_reg = df_feat.loc[Xdf.index, "Target_Close"].values
    y_cls = df_feat.loc[Xdf.index, "Target_Dir"].values
    dates = Xdf.index
    close = df_feat.loc[Xdf.index, "Close"]
    return X, y_reg, y_cls, dates, close

def train_svr_small(X_train, y_train, use_grid=True):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    svr = SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")
    # only run grid if enough rows and grid enabled
    if use_grid and len(X_train) > 200:
        param_grid = {"C":[10,100], "epsilon":[0.01,0.1]}
        splits = min(3, max(2, len(X_train)//200))
        try:
            tscv = TimeSeriesSplit(n_splits=splits)
            grid = GridSearchCV(SVR(kernel="rbf"), param_grid, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=1)
            grid.fit(Xs, y_train)
            svr = grid.best_estimator_
        except Exception:
            svr.fit(Xs, y_train)
    else:
        svr.fit(Xs, y_train)

    # simple logistic direction model (not critical)
    log = None
    scaler_cls = None
    try:
        scaler_cls = StandardScaler()
        Xs_cls = scaler_cls.fit_transform(X_train)
        # we need a target for logistic: use forward direction
        if len(y_train) > 2:
            cls_target = (y_train[1:] > y_train[:-1]).astype(int)
            # align Xs_cls to same length by dropping last row
            Xc = Xs_cls[:-1]
            log = LogisticRegression(max_iter=1000).fit(Xc, cls_target)
        else:
            log = None
    except Exception:
        log = None
        scaler_cls = None

    return svr, scaler, log, scaler_cls

def to_scalar_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            if getattr(x, "size", None) == 1:
                return float(x.iloc[0]) if hasattr(x, "iloc") else float(x.values.ravel()[0])
            return float(np.asarray(x).ravel()[0])
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.ravel()[0])
            return float(x.ravel()[0])
        if isinstance(x, numbers.Number):
            return float(x)
        return float(x)
    except Exception:
        return np.nan

def compute_suggested_holdings_from_returns(actual_returns_map, expected_returns_map):
    score = {}
    for k in actual_returns_map:
        ar = to_scalar_float(actual_returns_map.get(k, np.nan))
        er = to_scalar_float(expected_returns_map.get(k, np.nan))
        if math.isnan(ar): ar = 0.0
        if math.isnan(er): er = 0.0
        # heuristic scoring: expected_return more important, actual reduces/increases score
        s = max(er * 100.0 + ar * 10.0 + 0.001, 0.001)
        score[k] = s
    total = sum(score.values()) or 1.0
    suggested = {k: (v / total) * 100.0 for k, v in score.items()}
    return suggested

def format_pct_or_na(x, precision=2):
    try:
        if x is None:
            return "N/A"
        xv = float(x)
        if math.isnan(xv):
            return "N/A"
        return f"{xv*100:.{precision}f}%"
    except Exception:
        return "N/A"

# ---------- Gemini helpers ----------
def get_gemini_api_key():
    # check streamlit secrets first, then environment
    try:
        if isinstance(st.secrets, dict) and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

@st.cache_data(ttl=3600)
def cached_gemini_explain(cache_key: str, prompt_payload: str) -> str:
    # attempt to call Gemini; returns helpful error strings if not possible
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"Gemini SDK not installed or import failed. To enable, run: pip install google-generativeai. ({e})"

    api_key = get_gemini_api_key()
    if not api_key:
        return "GEMINI API KEY NOT FOUND. Set GEMINI_API_KEY in environment or Streamlit secrets."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        prompt = (
            "You are MAO-001 (Polxium). Given the numeric summary below for a single ticker "
            "and its performance over a specified period, produce 3 clear bullet points (1-2 sentences each):\n"
            "  1) A succinct description of performance (direction + magnitude). Mention any obvious crisis/period if it overlaps.\n"
            "  2) A short explanation of possible causes or market drivers.\n"
            "  3) One plain-language note for a retail investor.\n"
            "End with: 'This is experimental and not financial advice.'\n\n"
            "=== INPUT SUMMARY ===\n"
            f"{prompt_payload}\n\n"
            "Be concise and avoid jargon."
        )

        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            try:
                text = resp.output[0].content[0].text
            except Exception:
                text = str(resp)
        return text.strip()
    except Exception as e:
        return f"Gemini call failed: {e}"

# ---------- UI ----------
hcol1, hcol2 = st.columns([3,1])
with hcol1:
    st.markdown("<div style='font-size:36px; font-weight:700;'>Polxium</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px; color:gray; margin-top:-10px;'>MAO-001 — Market Analysis & Overview</div>", unsafe_allow_html=True)
with hcol2:
    st.write("")

st.write("---")

with st.sidebar:
    st.header("Controls")
    asset_select = st.selectbox("Choose asset", list(ASSETS.keys()), index=0)
    start_date = st.date_input("Start date", value=pd.to_datetime("2007-01-01").date(), key="start")
    end_date = st.date_input("End date", value=pd.Timestamp.today().date(), key="end")
    fast_mode = st.checkbox("Fast mode (no grid search)", value=True)
    st.markdown("## Portfolio settings")
    if 'weights' not in st.session_state:
        default_pct = round(100.0 / len(ASSETS), 2)
        st.session_state['weights'] = {k: default_pct for k in ASSETS.keys()}
    if '_normalizing' not in st.session_state:
        st.session_state['_normalizing'] = False

    def _normalize_all():
        if st.session_state.get('_normalizing', False):
            return
        st.session_state['_normalizing'] = True
        current = {}
        for name in ASSETS.keys():
            key = f"input_{name}"
            val = st.session_state.get(key, None)
            if val is None:
                val = st.session_state['weights'].get(name, 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            current[name] = max(val, 0.0)
        total = sum(current.values())
        if total <= 0:
            n = len(current)
            normalized = {k: round(100.0 / n, 2) for k in current.keys()}
        else:
            normalized = {}
            for k, v in current.items():
                normalized[k] = round((v / total) * 100.0, 2)
        for k, v in normalized.items():
            st.session_state[f"input_{k}"] = v
            st.session_state['weights'][k] = v
        st.session_state['_normalizing'] = False

    for name in ASSETS.keys():
        key = f"input_{name}"
        if key not in st.session_state:
            st.session_state[key] = st.session_state['weights'].get(name, round(100.0/len(ASSETS),2))
        st.number_input(f"{name} %", min_value=0.0, max_value=100.0, value=st.session_state[key],
                        step=0.5, key=key, on_change=_normalize_all)
    st.markdown("Values auto-scale to 100% after any edit.")
    run = st.button("Run analysis")

col_left, col_right = st.columns([3,1])
with col_left:
    st.markdown(f"### {asset_select}")
    short_map = {
        "Apple": "Large-cap Technology (consumer hardware & services).",
        "Tesla": "Large-cap Automotive & EV; high growth & volatility.",
        "Reliance": "Indian conglomerate: energy, retail, telecom.",
        "Microsoft": "Large-cap Technology (cloud, software & enterprise).",
        "Nvidia": "AI/semiconductor leader; high growth, cyclical.",
        "Gold (GLD)": "Gold ETF — safe-haven commodity exposure.",
    }
    st.markdown(f"**{short_map.get(asset_select,'')}**")
with col_right:
    st.write("")

st.write("---")

if run:
    # prepare weights used
    weights_used = {k: float(st.session_state.get(f"input_{k}", st.session_state['weights'].get(k, 0.0))) for k in ASSETS.keys()}
    df_main = download_data(ASSETS[asset_select], start_date.isoformat(), end_date.isoformat())
    if df_main.empty:
        st.error("No data returned for this asset & date range. Try different dates.")
        st.stop()

    # Features & modelling
    preds = np.array([])
    dates_test = np.array([])
    regime_labels = np.array([])
    df_feat = add_features(df_main) if len(df_main) > 10 else None
    if df_feat is not None and len(df_feat) > 30:
        try:
            X, y_reg, y_cls, dates_all, close_all = prepare_xy(df_feat)
            n = len(X)
            split = int(n * 0.8)
            if split < 5: split = int(n * 0.7)
            if split >= n: split = n-1
            X_train, X_test = X[:split], X[split:]
            y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
            dates_train, dates_test = dates_all[:split], dates_all[split:]
            svr_model, scaler_reg, log_model, scaler_cls = train_svr_small(X_train, y_reg_train, use_grid=(not fast_mode))
            if scaler_reg is not None and len(X_test)>0:
                X_test_s = scaler_reg.transform(X_test)
                preds = svr_model.predict(X_test_s)
            else:
                preds = np.array([])
            if log_model is not None and scaler_cls is not None and len(X_test)>1:
                try:
                    probs_up = log_model.predict_proba(scaler_cls.transform(X_test)[:-1])[:,1]
                    regime_labels = np.array(["GOOD" if p>0.6 else ("POOR" if p<0.4 else "STABLE") for p in probs_up])
                    # align preds length with regime_labels if needed (regression preds length may differ)
                    if len(regime_labels) < len(preds):
                        regime_labels = np.concatenate([regime_labels, np.array(["STABLE"]*(len(preds)-len(regime_labels)))])
                    elif len(regime_labels) > len(preds):
                        regime_labels = regime_labels[:len(preds)]
                except Exception:
                    regime_labels = np.array(["STABLE"] * len(preds))
            else:
                regime_labels = np.array(["STABLE"] * len(preds))
        except Exception:
            preds = np.array([]); regime_labels = np.array([])

    # Plotting main
    col_main, col_side = st.columns((3,1))
    with col_main:
        st.subheader(f"{asset_select} — Actual vs MAO-001 (SVR) predicted")
        fig, ax = plt.subplots(figsize=(11,5), constrained_layout=True)
        ax.grid(alpha=0.25, linestyle="--")
        ax.plot(df_main.index, df_main["Close"].values, color="black", linewidth=1.0, label="Actual Close")
        if preds.size > 0:
            ax.plot(dates_test, preds, color="#1f77b4", linewidth=1.6, alpha=0.95, label="SVR Predicted (test)")
            good_mask = regime_labels == "GOOD"
            poor_mask = regime_labels == "POOR"
            stable_mask = regime_labels == "STABLE"
            if good_mask.any():
                ax.scatter(np.array(dates_test)[good_mask], np.array(preds)[good_mask], color="green", s=26, label="GOOD")
            if poor_mask.any():
                ax.scatter(np.array(dates_test)[poor_mask], np.array(preds)[poor_mask], color="red", s=26, label="POOR")
            if stable_mask.any():
                ax.scatter(np.array(dates_test)[stable_mask], np.array(preds)[stable_mask], color="orange", s=20, label="STABLE")
        ymin, ymax = ax.get_ylim()
        label_positions = np.linspace(ymax - (ymax - ymin)*0.05, ymax - (ymax - ymin)*0.25, len(EVENT_WINDOWS))
        for i,(s,e,label) in enumerate(EVENT_WINDOWS):
            try:
                sdt = pd.to_datetime(s)
                edt = pd.to_datetime(e)
                # only draw event if intersects date range
                if edt >= df_main.index[0] and sdt <= df_main.index[-1]:
                    ax.axvspan(sdt, edt, color="gray", alpha=0.10)
                    ax.text(sdt, label_positions[i % len(label_positions)], label, fontsize=8, rotation=90, va="top", color="#444")
            except Exception:
                pass
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # company summary - ALWAYS show summary (no blocking "not enough data")
        st.markdown("### Company summary (numeric)")
        close_vals = df_main["Close"].values
        num_days = len(close_vals)
        last_close = float(close_vals[-1])
        first_close = float(close_vals[0])
        total_return = (last_close / (first_close + 1e-12)) - 1.0
        recent_30d_pct = None
        if num_days >= 2:
            lookback_idx = max(0, num_days - 30)
            recent_30d_pct = (last_close - close_vals[lookback_idx]) / (close_vals[lookback_idx] + 1e-12)

        st.write(f"- Date range: {start_date} → {end_date} ({num_days} trading days)")
        st.write(f"- Start close: {first_close:.4f}  →  Last close: {last_close:.4f}")
        st.write(f"- Total return over period: {total_return*100:.2f}%")
        st.write(f"- Recent ~30-day change (approx): {format_pct_or_na(recent_30d_pct)}")

    with col_side:
        st.markdown("### Small charts")
        fig1, ax1 = plt.subplots(figsize=(3.6,2.8), constrained_layout=True)
        ax1.plot(df_main.index, df_main["Close"].values, linewidth=1); ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("Actual", fontsize=10)
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots(figsize=(3.6,2.8), constrained_layout=True)
        if preds.size > 0:
            ax2.plot(dates_test, preds, color="#1f77b4", linewidth=1)
        else:
            ax2.text(0.5,0.5,"No predictions", ha="center")
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("Predicted (test)", fontsize=10)
        st.pyplot(fig2)

    # ---------------- Gemini explanation UI (only if key present) ----------------
    st.markdown("---")
    st.subheader("MAO-001 AI explanation (Gemini)")
    gem_key = get_gemini_api_key()
    if not gem_key:
        st.info("AI explanation is disabled for this demo. Add a GEMINI_API_KEY to enable it.")
    else:
        st.caption("Press the button to generate a short natural-language explanation. Cached for 1 hour.")
        payload = {
            "ticker": ASSETS[asset_select],
            "start": str(start_date),
            "end": str(end_date),
            "days": int(num_days),
            "start_close": round(first_close, 6),
            "last_close": round(last_close, 6),
            "total_return_pct": round(total_return*100, 3),
            "recent_30d_pct": (round(float(recent_30d_pct*100), 3) if recent_30d_pct is not None and not math.isnan(float(recent_30d_pct)) else None),
            "pred_mean": (round(float(np.mean(preds)),4) if preds.size>0 and not math.isnan(np.mean(preds)) else None),
            "pred_len": int(len(preds)),
            "regimes": dict(pd.Series(regime_labels).value_counts().to_dict()) if regime_labels.size>0 else {},
        }
        payload_str = "\n".join([f"{k}: {v}" for k, v in payload.items()])
        cache_key = f"gemini_{ASSETS[asset_select]}_{start_date}_{end_date}"
        if st.button("Generate AI explanation (MAO-001)"):
            with st.spinner("Requesting Gemini (cached 1h)..."):
                explanation = cached_gemini_explain(cache_key, payload_str)
                st.markdown("#### Gemini explanation")
                st.write(explanation)

    # ---------------- Portfolio analyzer ----------------
    st.write("---")
    st.subheader("Portfolio analyzer (MAO-001 driven rebalancing)")
    with st.spinner("Analyzing portfolio across selected period..."):
        actual_returns = {}
        expected_returns = {}
        for name, ticker in ASSETS.items():
            df = download_data(ticker, start_date.isoformat(), end_date.isoformat())
            if df.empty or "Close" not in df.columns:
                actual_returns[name] = np.nan; expected_returns[name] = np.nan; continue
            try:
                actual_returns[name] = to_scalar_float((df["Close"].iloc[-1] / (df["Close"].iloc[0] + 1e-12)) - 1)
            except Exception:
                actual_returns[name] = np.nan
            try:
                df_feat2 = add_features(df)
                if df_feat2 is None or len(df_feat2) < 40:
                    expected_returns[name] = actual_returns[name]; continue
                X2, y2, _, _, _ = prepare_xy(df_feat2)
                n2 = len(X2)
                split2 = int(n2 * 0.8)
                if split2 < 10: split2 = int(n2 * 0.7)
                if split2 >= n2: split2 = n2-1
                X_train2, X_test2 = X2[:split2], X2[split2:]
                y_train2, y_test2 = y2[:split2], y2[split2:]
                svr_model2, scaler2, _, _ = train_svr_small(X_train2, y_train2, use_grid=(not fast_mode))
                if scaler2 is None or X_test2.size == 0:
                    expected_returns[name] = actual_returns[name]; continue
                X_test_s2 = scaler2.transform(X_test2)
                p2 = svr_model2.predict(X_test_s2)
                if p2.size>0:
                    # expected return based on mean predicted / last observed train close
                    baseline = float(y_train2[-1]) if len(y_train2)>0 else float(df["Close"].iloc[-1])
                    pred_rets = (p2 - baseline) / (abs(baseline) + 1e-12)
                    expected_returns[name] = to_scalar_float(np.nanmean(pred_rets))
                else:
                    expected_returns[name] = actual_returns[name]
            except Exception:
                expected_returns[name] = np.nan

        suggested = compute_suggested_holdings_from_returns(actual_returns, expected_returns)

        rows = []
        for name in ASSETS.keys():
            rows.append({
                "Asset": name,
                "Ticker": ASSETS[name],
                "CurrentHolding%": round(to_scalar_float(weights_used.get(name, 0.0)), 2),
                "ActualTotalReturn%": (round(to_scalar_float(actual_returns.get(name, np.nan))*100,2) if not math.isnan(to_scalar_float(actual_returns.get(name, np.nan))) else np.nan),
                "ExpectedReturn(SVR)%": (round(to_scalar_float(expected_returns.get(name, np.nan))*100,2) if not math.isnan(to_scalar_float(expected_returns.get(name, np.nan))) else np.nan),
                "SuggestedHolding%": round(to_scalar_float(suggested.get(name, np.nan)), 2)
            })
        df_port = pd.DataFrame(rows).sort_values("SuggestedHolding%", ascending=True).reset_index(drop=True)

    st.markdown("**Suggested holdings (ascending by suggested %)**")
    st.dataframe(df_port[["Asset","Ticker","CurrentHolding%","ActualTotalReturn%","ExpectedReturn(SVR)%","SuggestedHolding%"]].round(2), height=280)

    # textual summaries
    df_nonan = df_port.dropna(subset=["ActualTotalReturn%"])
    if not df_nonan.empty:
        top = df_nonan.sort_values("ActualTotalReturn%", ascending=False).head(2)
        bottom = df_nonan.sort_values("ActualTotalReturn%", ascending=True).head(2)
        desc1 = f"Top performers: {', '.join(top['Asset'].tolist())}. Largest declines: {', '.join(bottom['Asset'].tolist())}."
    else:
        desc1 = "No usable returns found in the selected range."
    rec_lines = []
    for _, r in df_port.iterrows():
        asset = r["Asset"]
        er = r["ExpectedReturn(SVR)%"]
        er_str = ("N/A" if pd.isna(er) else f"{er}%")
        if pd.isna(er):
            action = "HOLD"
        else:
            if er > 5:
                action = "BUY / increase"
            elif er < -2:
                action = "SELL / decrease"
            else:
                action = "HOLD"
        rec_lines.append(f"{asset}: {action} (expected {er_str}).")
    desc2 = " ".join(rec_lines)

    st.markdown("**Portfolio — current situation**"); st.write(desc1)
    st.markdown("**Portfolio — recommended actions (why change/hold)**"); st.write(desc2)
    st.info("MAO-001 predictions are experimental and intended for pattern analysis. Not financial advice.")

else:
    st.info("Adjust weights and press Run analysis.")
>>>>>>> c35ee141ca658101a4cbfa0ec0e720df439d0ff3
