# ============================================================
#  StockMind AI — Production V2.1 (Notebook-Faithful)
#  © 2025 Jacinth G. All Rights Reserved.
#  Paavai College of Engineering | B.Tech AI & Data Science
#  Project  : StockMind AI – Deep Learning Stock Price Predictor
#  Version  : 2.1.0
#  WARNING  : Unauthorized use, reproduction or distribution is
#             strictly prohibited. Copyright is non-removable.
# ============================================================
# KEY FIXES vs V2.0:
#   - Multi-feature input: close+high+low+open (4 features, like notebook)
#   - Scaler fits ALL 4 columns together (not just close)
#   - 80/20 row-split BEFORE windowing (like notebook)
#   - Model output = 4 features, then inverse_transform gives real prices
#   - LSTM1 = 2-layer (100,100), LSTM2 = 3-layer (150,100,100) — exact notebook arch
#   - Prediction uses last test_data window, not full dataset window
#   - Accuracy = 100 - MAPE (matches notebook formula)
# ============================================================

__author__    = "jacinth G"
__copyright__ = "© 2025 Jacinth G | Selladurai S | Thamesh Raj G . All Rights Reserved."
__college__   = "Paavai College of Engineering"
__degree__    = "B.Tech – Artificial Intelligence & Data Science"
__project__   = "StockMind AI – Deep Learning Stock Price Predictor"
__version__   = "2.1.0"
__license__   = "Proprietary – All Rights Reserved"

import os, math, hashlib, warnings, datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ── Dirs ──────────────────────────────────────────────────────
MODEL_DIR = "saved_models"
DATA_DIR  = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "stock_data.csv")

COPYRIGHT = "© 2025 Jacinth G · Paavai College of Engineering · All Rights Reserved"
WATERMARK = hashlib.sha256(__copyright__.encode()).hexdigest()[:16].upper()

FEATURES  = ["close", "high", "low", "open"]   # exact notebook column order
WINDOW    = 60                                   # notebook uses 60

CURRENCIES = {
    "₹ INR – Indian Rupee":  {"symbol": "₹",  "code": "INR"},
    "$ USD – US Dollar":      {"symbol": "$",  "code": "USD"},
    "€ EUR – Euro":           {"symbol": "€",  "code": "EUR"},
    "£ GBP – British Pound":  {"symbol": "£",  "code": "GBP"},
    "¥ JPY – Japanese Yen":   {"symbol": "¥",  "code": "JPY"},
}

# ════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════
st.set_page_config(page_title="StockMind AI v2.1", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
:root{{
    --bg:#07090f;--bg2:#0d1322;--bg3:#111b2d;--bg4:#16223a;
    --gold:#e8b84b;--green:#00e5a0;--red:#ff5370;--blue:#4d9fff;
    --purple:#9b79ff;--muted:#4a5c7a;--text:#d4dff0;--border:#192840;
}}
html,body,[class*="css"]{{font-family:'Plus Jakarta Sans',sans-serif;background:var(--bg);color:var(--text);}}
.stApp{{background:var(--bg);}}
[data-testid="stSidebar"]{{background:var(--bg2)!important;border-right:1px solid var(--border);}}
[data-testid="stSidebar"] *{{color:var(--text)!important;}}
header[data-testid="stHeader"]{{background:transparent!important;}}
.app-hdr{{background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid var(--border);
    border-radius:18px;padding:28px 36px;margin-bottom:22px;position:relative;overflow:hidden;}}
.app-hdr::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--gold) 40%,var(--green) 70%,transparent);}}
.app-hdr::before{{content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse at 80% 50%,rgba(232,184,75,.06),transparent 70%);pointer-events:none;}}
.app-title{{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;color:var(--gold);margin:0;letter-spacing:-.5px;}}
.app-sub{{color:var(--muted);font-size:.82rem;margin-top:4px;}}
.cp-badge{{position:absolute;top:14px;right:18px;font-family:'IBM Plex Mono',monospace;
    font-size:.6rem;color:var(--muted);background:rgba(232,184,75,.07);
    border:1px solid rgba(232,184,75,.18);border-radius:6px;padding:4px 10px;letter-spacing:.5px;}}
.kpi{{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:16px 20px;
    position:relative;overflow:hidden;transition:border-color .2s;}}
.kpi:hover{{border-color:rgba(232,184,75,.4);}}
.kpi-lbl{{font-size:.65rem;text-transform:uppercase;letter-spacing:2px;color:var(--muted);font-weight:600;}}
.kpi-val{{font-family:'IBM Plex Mono',monospace;font-size:1.35rem;font-weight:700;color:var(--text);margin:4px 0 2px;}}
.kpi-d{{font-size:.75rem;font-weight:500;}}
.kpi-bar{{position:absolute;bottom:0;left:0;right:0;height:3px;}}
.up{{color:var(--green);}} .down{{color:var(--red);}}
.sec{{font-family:'IBM Plex Mono',monospace;font-size:.62rem;text-transform:uppercase;
    letter-spacing:3px;color:var(--gold);border-bottom:1px solid var(--border);
    padding-bottom:7px;margin:22px 0 14px;}}
.card{{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:20px 24px;margin-bottom:16px;}}
.res-card{{background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid rgba(0,229,160,.25);
    border-radius:14px;padding:20px 24px;margin-bottom:16px;box-shadow:0 0 40px rgba(0,229,160,.05);}}
.live-card{{background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid rgba(77,159,255,.3);
    border-radius:14px;padding:20px 24px;margin-bottom:16px;}}
.acc-card{{background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid rgba(232,184,75,.35);
    border-radius:14px;padding:20px 24px;margin-bottom:16px;box-shadow:0 0 30px rgba(232,184,75,.06);}}
.b-ok{{display:inline-flex;align-items:center;gap:5px;padding:3px 11px;border-radius:20px;
    font-size:.7rem;font-weight:700;font-family:'IBM Plex Mono',monospace;
    background:rgba(0,229,160,.12);color:var(--green);border:1px solid rgba(0,229,160,.3);}}
.b-warn{{display:inline-flex;align-items:center;gap:5px;padding:3px 11px;border-radius:20px;
    font-size:.7rem;font-weight:700;font-family:'IBM Plex Mono',monospace;
    background:rgba(232,184,75,.12);color:var(--gold);border:1px solid rgba(232,184,75,.3);}}
.b-blue{{display:inline-flex;align-items:center;gap:5px;padding:3px 11px;border-radius:20px;
    font-size:.7rem;font-weight:700;font-family:'IBM Plex Mono',monospace;
    background:rgba(77,159,255,.12);color:var(--blue);border:1px solid rgba(77,159,255,.3);}}
.b-live{{display:inline-flex;align-items:center;gap:5px;padding:3px 11px;border-radius:20px;
    font-size:.7rem;font-weight:700;font-family:'IBM Plex Mono',monospace;
    background:rgba(0,229,160,.15);color:var(--green);border:1px solid rgba(0,229,160,.4);
    animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.6;}}}}
.sig-buy{{background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.35);border-radius:12px;
    padding:18px 22px;color:var(--green);font-weight:700;font-family:'IBM Plex Mono',monospace;text-align:center;}}
.sig-sell{{background:rgba(255,83,112,.1);border:1px solid rgba(255,83,112,.35);border-radius:12px;
    padding:18px 22px;color:var(--red);font-weight:700;font-family:'IBM Plex Mono',monospace;text-align:center;}}
.sig-hold{{background:rgba(232,184,75,.1);border:1px solid rgba(232,184,75,.35);border-radius:12px;
    padding:18px 22px;color:var(--gold);font-weight:700;font-family:'IBM Plex Mono',monospace;text-align:center;}}
.stButton>button{{background:linear-gradient(135deg,var(--gold),#c9960a)!important;color:#07090f!important;
    border:none!important;border-radius:10px!important;font-family:'IBM Plex Mono',monospace!important;
    font-weight:700!important;font-size:.8rem!important;letter-spacing:1px!important;
    transition:all .2s!important;width:100%!important;}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 10px 28px rgba(232,184,75,.35)!important;}}
div[data-testid="stMetricValue"]{{font-family:'IBM Plex Mono',monospace!important;color:var(--gold)!important;font-size:1.2rem!important;}}
div[data-testid="stMetricDelta"]{{font-size:.78rem!important;}}
.stProgress>div>div{{background:var(--gold)!important;}}
.stTabs [data-baseweb="tab-list"]{{background:var(--bg2);border-radius:12px;padding:4px;gap:4px;}}
.stTabs [data-baseweb="tab"]{{background:transparent;border-radius:8px;color:var(--muted);
    font-family:'IBM Plex Mono',monospace;font-size:.7rem;}}
.stTabs [aria-selected="true"]{{background:var(--bg4)!important;color:var(--gold)!important;}}
hr{{border-color:var(--border)!important;}}
.stSelectbox label,.stSlider label,.stRadio label{{font-size:.78rem!important;color:var(--muted)!important;}}
.footer{{text-align:center;padding:26px 0 10px;font-family:'IBM Plex Mono',monospace;
    font-size:.62rem;color:var(--muted);border-top:1px solid var(--border);margin-top:40px;letter-spacing:.4px;}}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  UTILITY
# ════════════════════════════════════════════════

def fmt_price(value, sym):
    if sym == "₹":
        s     = f"{abs(value):.2f}"
        i, d  = s.split(".")
        if len(i) > 3:
            last3 = i[-3:]
            rest  = i[:-3]
            grps  = []
            while len(rest) > 2:
                grps.append(rest[-2:])
                rest = rest[:-2]
            if rest: grps.append(rest)
            grps.reverse()
            i = ",".join(grps) + "," + last3
        return f"₹{i}.{d}"
    return f"{sym}{value:,.2f}"


def pt():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d4dff0", family="Plus Jakarta Sans", size=12),
        xaxis=dict(gridcolor="#192840", showgrid=True, zeroline=False,
                   tickfont=dict(size=11), showline=False),
        yaxis=dict(gridcolor="#192840", showgrid=True, zeroline=False,
                   tickfont=dict(size=11), showline=False),
        legend=dict(bgcolor="rgba(13,19,34,.85)", bordercolor="#192840",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=8, r=8, t=44, b=8),
        hoverlabel=dict(bgcolor="#0d1322", bordercolor="#192840",
                        font=dict(color="#d4dff0", size=12)),
    )


# ─────────────────────────────────────────────────
#  EXACT NOTEBOOK: prepare_time_series_data
# ─────────────────────────────────────────────────
def prepare_time_series_data(Data, window_size=60):
    """Exact replica of notebook function — works on DataFrame with DateIndex."""
    sequences, labels = [], []
    for i in range(len(Data) - window_size):
        sequence = Data.iloc[i:i + window_size].values   # shape (60, 4)
        label    = Data.iloc[i + window_size].values      # shape (4,)
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# ─────────────────────────────────────────────────
#  EXACT NOTEBOOK: predict_and_inverse_transform
# ─────────────────────────────────────────────────
def predict_and_inverse_transform(DF_scaled, X_test, model, scaler, test_index):
    """
    Mirrors the notebook's function exactly.
    DF_scaled : full scaled DataFrame (DateIndex, 4 cols)
    X_test    : (N, 60, 4)
    Returns   : DataFrame with actual + predicted columns, unscaled
    """
    predictions       = model.predict(X_test, verbose=0)               # (N, 4)
    inv_predictions   = scaler.inverse_transform(predictions)           # (N, 4)
    inv_pred_df       = pd.DataFrame(inv_predictions,
                                     columns=["Predicted Close","Predicted High",
                                              "Predicted Low","Predicted Open"],
                                     index=test_index)

    # Actual (scaled) slice aligned with predictions
    actual_scaled = DF_scaled.iloc[-len(X_test):].copy()
    actual_unscaled = scaler.inverse_transform(actual_scaled.values)
    actual_df = pd.DataFrame(actual_unscaled,
                              columns=["close","high","low","open"],
                              index=test_index)

    return pd.concat([actual_df, inv_pred_df], axis=1)


# ─────────────────────────────────────────────────
#  NOTEBOOK MODEL ARCHITECTURES
# ─────────────────────────────────────────────────
def build_lstm1(window, n_features):
    """LSTM1 — 2-layer, exactly like notebook Cell 16."""
    m = Sequential(name="LSTM1_StockMind")
    m.add(LSTM(100, return_sequences=True,  input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(LSTM(100, return_sequences=False, input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(Dense(n_features))
    m.compile(optimizer="adam", loss="mean_squared_error",
              metrics=["mean_absolute_error"])
    return m


def build_lstm2(window, n_features):
    """LSTM2 — 3-layer deeper, exactly like notebook Cell 24."""
    m = Sequential(name="LSTM2_StockMind")
    m.add(LSTM(150, return_sequences=True,  input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(LSTM(100, return_sequences=True,  input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(LSTM(100, return_sequences=False, input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(Dense(50))
    m.add(Dense(5))
    m.add(Dense(n_features))
    m.compile(optimizer="adam", loss="mean_squared_error",
              metrics=["mean_absolute_error"])
    return m


def build_gru(window, n_features):
    """GRU equivalent of LSTM2 — for GRU mode."""
    m = Sequential(name="GRU_StockMind")
    m.add(GRU(150, return_sequences=True,  input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(GRU(100, return_sequences=True,  input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(GRU(100, return_sequences=False, input_shape=(window, n_features)))
    m.add(Dropout(0.2))
    m.add(Dense(50))
    m.add(Dense(n_features))
    m.compile(optimizer="adam", loss="mean_squared_error",
              metrics=["mean_absolute_error"])
    return m


def calculate_accuracy(y_true, y_pred):
    """Notebook formula: accuracy = 100 - MAPE."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mape   = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape, 100 - mape


# ─────────────────────────────────────────────────
#  PREPROCESSING  (notebook-faithful)
# ─────────────────────────────────────────────────
def preprocess(sdf):
    """
    1. Select close/high/low/open (notebook order)
    2. Fit MinMaxScaler on all 4 cols together
    3. 80/20 row split
    4. Build sequences from each split separately
    Returns all needed objects.
    """
    avail = [c for c in FEATURES if c in sdf.columns]
    if len(avail) < 2:
        avail = ["close"]

    raw = sdf[avail].copy()
    raw.index = sdf["date"].values

    scaler = MinMaxScaler()
    scaled_vals = scaler.fit_transform(raw.values)
    DF = pd.DataFrame(scaled_vals, columns=avail, index=raw.index)

    training_size = round(len(DF) * 0.80)
    train_data = DF.iloc[:training_size]
    test_data  = DF.iloc[training_size:]

    X_train, y_train = prepare_time_series_data(train_data, WINDOW)
    X_test,  y_test  = prepare_time_series_data(test_data,  WINDOW)

    # Index for test predictions (needed for plotting)
    test_index = test_data.index[WINDOW:]

    return X_train, y_train, X_test, y_test, scaler, DF, avail, test_index, training_size


# ─────────────────────────────────────────────────
#  FUTURE FORECAST (multi-feature roll)
# ─────────────────────────────────────────────────
def run_forecast(model, scaler, DF_scaled, avail, fdays):
    last_seq = DF_scaled[avail].values[-WINDOW:].copy()   # (60, n_feat)
    seq      = last_seq.reshape(1, WINDOW, len(avail))
    preds    = []
    for _ in range(fdays):
        p = model.predict(seq, verbose=0)[0]              # (n_feat,)
        preds.append(p)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, :] = p
    inv    = scaler.inverse_transform(np.array(preds))    # (fdays, n_feat)
    prices = inv[:, 0]                                    # close is column 0
    return prices


def conf_band(preds, factor=0.072):
    p   = np.array(preds).flatten()
    std = np.std(p) * factor
    return p + std, p - std


def get_signal(prices, window=7):
    if len(prices) < window + 1:
        return "HOLD", 0.0
    recent   = np.array(prices[-window:])
    momentum = (recent[-1] - recent[0]) / recent[0] * 100
    if   momentum >  1.5: return "BUY",  momentum
    elif momentum < -1.5: return "SELL", momentum
    else:                 return "HOLD", momentum


@st.cache_data(show_spinner=False)
def load_csv(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live(ticker, period="6mo"):
    if not YF_AVAILABLE:
        return None, "yfinance not installed"
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            return None, f"No data returned for '{ticker}'"
        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adj close" in raw.columns:
            raw = raw.rename(columns={"adj close": "close"})
        raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
        raw = raw.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
        return raw, None
    except Exception as e:
        return None, str(e)


def mpath(sym, arch):
    return f"{MODEL_DIR}/{sym}_{arch}_v21.keras"

def spath(sym):
    return f"{MODEL_DIR}/{sym}_scaler_v21.pkl"


# ════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:14px 0 6px;'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:1.05rem;
                    color:#e8b84b;font-weight:700;'>📈 StockMind AI</div>
        <div style='color:#4a5c7a;font-size:.68rem;font-family:IBM Plex Mono,monospace;
                    margin-top:2px;'>v{__version__} · {__author__}</div>
    </div><hr>
    <div style='font-family:IBM Plex Mono,monospace;font-size:.58rem;
                text-transform:uppercase;letter-spacing:2px;color:#e8b84b;
                margin-bottom:10px;'>💱 Currency</div>
    """, unsafe_allow_html=True)

    cur_label = st.selectbox("Currency", list(CURRENCIES.keys()),
                             label_visibility="collapsed")
    CUR     = CURRENCIES[cur_label]
    CUR_SYM = CUR["symbol"]

    st.markdown("<hr><div style='font-family:IBM Plex Mono,monospace;font-size:.58rem;"
                "text-transform:uppercase;letter-spacing:2px;color:#e8b84b;"
                "margin-bottom:10px;'>⚙ Model Config</div>", unsafe_allow_html=True)

    arch = st.selectbox("Architecture",
                        ["LSTM1 (2-layer, fast)", "LSTM2 (3-layer, accurate)", "GRU (3-layer)"],
                        help="LSTM1 = notebook model 1  |  LSTM2 = notebook model 2 (deeper)  |  GRU = alternative")
    arch_key = arch.split(" ")[0]   # "LSTM1", "LSTM2", or "GRU"

    epochs     = st.slider("Max Epochs", 10, 80, 30, 5,
                           help="Notebook uses 30 epochs with EarlyStopping on loss")
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    fdays      = st.slider("Forecast Horizon (days)", 7, 60, 30)

    st.markdown("<hr>", unsafe_allow_html=True)
    show_live   = st.checkbox("🌐 Live validation (yfinance)",
                              value=YF_AVAILABLE, disabled=not YF_AVAILABLE)
    live_period = "6mo"
    if show_live:
        live_period = st.selectbox("Live window", ["3mo","6mo","1y","2y"], index=1)

    st.markdown(f"""<hr>
    <div style='font-family:IBM Plex Mono,monospace;font-size:.55rem;
                color:#1e2f48;line-height:1.7;text-align:center;'>
        WM: {WATERMARK}<br>{COPYRIGHT}
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════
st.markdown(f"""
<div class='app-hdr'>
    <div class='cp-badge'>⚿ {__author__} · {WATERMARK}</div>
    <div class='app-title'>StockMind AI ⚡</div>
    <div class='app-sub'>
        Production V2.1 · Notebook-Faithful · LSTM1 / LSTM2 / GRU ·
        Multi-Feature (close+high+low+open) · Live Validation · Multi-Currency
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  01 — DATASET
# ════════════════════════════════════════════════
st.markdown('<div class="sec">01 / Dataset</div>', unsafe_allow_html=True)

cu1, cu2 = st.columns([3, 2])
with cu1:
    uploaded = st.file_uploader(
        "Upload stock CSV  (auto-saved — no re-upload needed next time)", type=["csv"])
    if uploaded:
        with open(DATA_PATH, "wb") as f:
            f.write(uploaded.getbuffer())
        st.cache_data.clear()
        st.success("✅ Saved permanently — will auto-load on restart.")

df = None
if uploaded:
    df = load_csv(DATA_PATH); src = "Just uploaded"
elif os.path.exists(DATA_PATH):
    df = load_csv(DATA_PATH); src = "Auto-loaded from disk"

with cu2:
    if df is not None:
        st.markdown(f"""
        <div class='card' style='padding:14px 18px;'>
            <span class='b-ok'>● DATA READY</span>
            <div style='margin-top:8px;font-size:.78rem;color:#4a5c7a;line-height:1.8;'>
                Source: <b style='color:#d4dff0;'>{src}</b><br>
                Rows: <b style='color:#d4dff0;'>{len(df):,}</b> &nbsp;|&nbsp;
                Cols: <b style='color:#d4dff0;'>{len(df.columns)}</b><br>
                Symbols: <b style='color:#e8b84b;'>
                    {", ".join(sorted(df["symbol"].unique()))}
                </b>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:30px;'>
            <div style='font-size:2rem;'>📂</div>
            <div style='font-family:IBM Plex Mono,monospace;color:#e8b84b;font-size:.85rem;margin-top:8px;'>
                No dataset loaded
            </div>
            <div style='color:#4a5c7a;font-size:.78rem;margin-top:6px;'>
                Needs: symbol, date, close (+ high, low, open for best accuracy)
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

for req in ["symbol", "date", "close"]:
    if req not in df.columns:
        st.error(f"❌ Missing required column: `{req}`")
        st.stop()


# ════════════════════════════════════════════════
#  02 — SYMBOL
# ════════════════════════════════════════════════
st.markdown('<div class="sec">02 / Symbol & Summary</div>', unsafe_allow_html=True)

symbols = sorted(df["symbol"].unique().tolist())
sc1, sc2, sc3 = st.columns([2, 2, 3])
with sc1:
    sym = st.selectbox("Stock Symbol", symbols)
with sc2:
    avail_feats = [c for c in FEATURES if c in df.columns]
    st.markdown(f"""
    <div class='card' style='padding:10px 16px;'>
        <div class='kpi-lbl'>Features Used</div>
        <div style='color:#e8b84b;font-family:IBM Plex Mono,monospace;font-size:.8rem;margin-top:4px;'>
            {" + ".join(avail_feats)}
        </div>
        <div style='color:#4a5c7a;font-size:.7rem;margin-top:4px;'>
            {len(avail_feats)} feature(s) · window={WINDOW}d
        </div>
    </div>""", unsafe_allow_html=True)

sdf = df[df["symbol"] == sym].copy().sort_values("date").reset_index(drop=True)

with sc3:
    lc   = sdf["close"].iloc[-1]
    pc   = sdf["close"].iloc[-2]
    chg  = (lc - pc) / pc * 100
    dmin = sdf["date"].min().strftime("%d %b %Y")
    dmax = sdf["date"].max().strftime("%d %b %Y")
    st.markdown(f"""
    <div class='card' style='padding:10px 16px;'>
        <span class='b-ok'>● {sym}</span>
        <span style='margin-left:10px;color:#4a5c7a;font-size:.78rem;'>
            {dmin} → {dmax} · <b style='color:#d4dff0;'>{len(sdf)}</b> rows
        </span><br>
        <span style='font-family:IBM Plex Mono,monospace;font-size:.95rem;
                     color:#d4dff0;font-weight:600;'>{fmt_price(lc, CUR_SYM)}</span>
        <span style='font-size:.78rem;margin-left:8px;
                     color:{"#00e5a0" if chg>=0 else "#ff5370"};'>
            {"▲" if chg>=0 else "▼"} {abs(chg):.2f}%
        </span>
    </div>""", unsafe_allow_html=True)

# KPIs
h52  = sdf["high"].max() if "high" in sdf.columns else lc
l52  = sdf["low"].min()  if "low"  in sdf.columns else lc
avol = int(sdf["volume"].mean()) if "volume" in sdf.columns else 0
tr   = (lc - sdf["close"].iloc[0]) / sdf["close"].iloc[0] * 100

kpi_data = [
    ("Last Close",   fmt_price(lc, CUR_SYM),  f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}% today",
     "up" if chg>=0 else "down", "#e8b84b"),
    ("52W High",     fmt_price(h52, CUR_SYM), "Peak in dataset",    "", "#4d9fff"),
    ("52W Low",      fmt_price(l52, CUR_SYM), "Trough in dataset",  "", "#ff5370"),
    ("Total Return", f"{tr:+.1f}%",           "Across full dataset","up" if tr>=0 else "down","#00e5a0"),
    ("Avg Volume",   f"{avol:,}",             "Daily average",      "", "#9b79ff"),
]
kcols = st.columns(5)
for col, (lbl, val, dl, cls, clr) in zip(kcols, kpi_data):
    with col:
        st.markdown(f"""
        <div class='kpi'>
            <div class='kpi-lbl'>{lbl}</div>
            <div class='kpi-val'>{val}</div>
            <div class='kpi-d {cls}'>{dl}</div>
            <div class='kpi-bar' style='background:{clr};'></div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  03 — EDA
# ════════════════════════════════════════════════
st.markdown('<div class="sec">03 / Exploratory Analysis</div>', unsafe_allow_html=True)

t1,t2,t3,t4,t5,t6 = st.tabs([
    "📈 Price+MA","🕯 OHLC","📊 Volume","📉 Returns","🔥 Heatmap","🗂 Raw"])

with t1:
    fig = go.Figure()
    for ma_n, clr in [(20,"#4d9fff"),(50,"#e8b84b"),(200,"#ff5370")]:
        if len(sdf) > ma_n:
            sdf[f"MA{ma_n}"] = sdf["close"].rolling(ma_n).mean()
            fig.add_trace(go.Scatter(x=sdf["date"], y=sdf[f"MA{ma_n}"],
                mode="lines", name=f"MA{ma_n}",
                line=dict(color=clr, width=1.2, dash="dot"), opacity=.75))
    fig.add_trace(go.Scatter(x=sdf["date"], y=sdf["close"],
        mode="lines", name="Close Price",
        line=dict(color="#e8b84b", width=2),
        fill="tozeroy", fillcolor="rgba(232,184,75,.04)"))
    fig.update_layout(**pt(), title=f"{sym} — Close Price + Moving Averages",
        height=400, xaxis_rangeslider_visible=True, yaxis_tickprefix=CUR_SYM)
    st.plotly_chart(fig, width="stretch")
    st.caption("🟡 Close  ·  🔵 MA20 (dotted)  ·  🟡 MA50 (dotted)  ·  🔴 MA200 (dotted)")

with t2:
    if all(c in sdf.columns for c in ["open","high","low","close"]):
        fig = go.Figure(go.Candlestick(
            x=sdf["date"], open=sdf["open"], high=sdf["high"],
            low=sdf["low"], close=sdf["close"],
            increasing_line_color="#00e5a0", decreasing_line_color="#ff5370",
            increasing_fillcolor="rgba(0,229,160,.65)",
            decreasing_fillcolor="rgba(255,83,112,.65)"))
        fig.update_layout(**pt(), title=f"{sym} — OHLC Candlestick Chart",
            height=420, xaxis_rangeslider_visible=False, yaxis_tickprefix=CUR_SYM)
        st.plotly_chart(fig, width="stretch")
        st.caption("🟢 Green = Bullish (close ≥ open)  ·  🔴 Red = Bearish (close < open)")
    else:
        st.info("OHLC columns (open/high/low) not all present.")

with t3:
    if "volume" in sdf.columns:
        bclr = ["#00e5a0" if c >= o else "#ff5370"
                for c, o in zip(sdf["close"],
                                sdf["open"] if "open" in sdf.columns else sdf["close"])]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sdf["date"], y=sdf["volume"],
                             marker_color=bclr, name="Volume", opacity=.8))
        fig.add_trace(go.Scatter(x=sdf["date"],
                                 y=sdf["volume"].rolling(20).mean(),
                                 mode="lines", name="20D Avg Vol",
                                 line=dict(color="#e8b84b", width=1.8)))
        fig.update_layout(**pt(), title=f"{sym} — Trading Volume + 20D Average", height=360)
        st.plotly_chart(fig, width="stretch")
        st.caption("🟢 Up-day volume  ·  🔴 Down-day volume  ·  🟡 20-day moving average")
    else:
        st.info("No volume column in this dataset.")

with t4:
    sdf["ret"] = sdf["close"].pct_change() * 100
    fig = make_subplots(rows=2, cols=1, row_heights=[.6,.4], vertical_spacing=.08)
    fig.add_trace(go.Scatter(x=sdf["date"], y=sdf["ret"],
        mode="lines", name="Daily Return %",
        line=dict(color="#4d9fff", width=1),
        fill="tozeroy", fillcolor="rgba(77,159,255,.06)"), row=1, col=1)
    fig.add_trace(go.Histogram(x=sdf["ret"].dropna(),
        nbinsx=60, name="Distribution",
        marker_color="#e8b84b", opacity=.75), row=2, col=1)
    fig.update_layout(**pt(), title=f"{sym} — Daily Returns + Frequency Distribution",
        height=460)
    st.plotly_chart(fig, width="stretch")
    st.caption("🔵 Daily % change  ·  🟡 Return distribution histogram")

with t5:
    num_c = sdf.select_dtypes(include=np.number).columns.tolist()
    num_c = [c for c in num_c if "MA" not in c and "ret" not in c]
    if len(num_c) >= 2:
        corr = sdf[num_c].corr()
        fig  = px.imshow(corr, text_auto=".2f",
                         color_continuous_scale=[[0,"#ff5370"],[.5,"#162135"],[1,"#00e5a0"]],
                         title=f"{sym} — Feature Correlation Heatmap")
        fig.update_layout(**pt(), height=420)
        st.plotly_chart(fig, width="stretch")

with t6:
    show_c = [c for c in ["date","symbol","open","high","low","close","volume"]
              if c in sdf.columns]
    st.dataframe(sdf[show_c].head(200), use_container_width=True, height=320)
    st.caption(f"Showing first 200 of {len(sdf)} rows for {sym}")


# ════════════════════════════════════════════════
#  04 — TRAIN / LOAD
# ════════════════════════════════════════════════
st.markdown('<div class="sec">04 / Train or Load Model</div>', unsafe_allow_html=True)

if len(sdf) < WINDOW + 50:
    st.warning(f"Not enough rows for {sym}. Need ≥ {WINDOW + 50}.")
    st.stop()

mp      = mpath(sym, arch_key)
mexists = os.path.exists(mp)

b1, b2, b3 = st.columns([1.5, 1.5, 4])
with b1:
    train_btn = st.button(f"🚀 TRAIN {arch_key}")
with b2:
    load_btn  = st.button("⚡ LOAD SAVED", disabled=not mexists)
with b3:
    if mexists:
        mts = datetime.datetime.fromtimestamp(
            os.path.getmtime(mp)).strftime("%d %b %Y, %H:%M")
        st.markdown(f"""
        <div class='card' style='padding:10px 16px;'>
            <span class='b-ok'>● SAVED MODEL FOUND</span>
            <span style='margin-left:8px;color:#4a5c7a;font-size:.76rem;'>
                {arch_key} · {sym} · Saved: {mts}
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='card' style='padding:10px 16px;'>
            <span class='b-warn'>○ NO SAVED MODEL</span>
            <span style='margin-left:8px;color:#4a5c7a;font-size:.76rem;'>
                Click TRAIN {arch_key} to build and save for {sym}.
                Loads instantly next time.
            </span>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
model    = None
scaler   = None
hist_obj = None
trained  = False

if train_btn or (load_btn and mexists):

    # Preprocess (notebook-faithful)
    X_train, y_train, X_test, y_test, scaler, DF_scaled, avail, test_index, training_size = \
        preprocess(sdf)

    n_feat = X_train.shape[2]

    if train_btn:
        st.markdown('<div class="sec">── Training Progress</div>', unsafe_allow_html=True)
        pb = st.progress(0, text="Initialising model…")

        if   arch_key == "LSTM1": model = build_lstm1(WINDOW, n_feat)
        elif arch_key == "LSTM2": model = build_lstm2(WINDOW, n_feat)
        else:                     model = build_gru(WINDOW, n_feat)

        pb.progress(8, text=f"{arch_key} built ({model.count_params():,} params). Training…")

        class StCB(tf.keras.callbacks.Callback):
            def __init__(self, total, pb):
                self.total = total; self.pb = pb
            def on_epoch_end(self, epoch, logs=None):
                pct = min(int(8 + (epoch+1)/self.total*85), 94)
                self.pb.progress(pct,
                    text=f"Epoch {epoch+1}/{self.total} · "
                         f"loss={logs['loss']:.5f} · val_loss={logs['val_loss']:.5f}")

        # Notebook uses EarlyStopping on 'loss' (not val_loss) with patience=5
        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=0)

        hist_obj = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[es, StCB(epochs, pb)],
            verbose=0,
            shuffle=False
        )
        model.save(mp)
        joblib.dump((scaler, avail, DF_scaled), spath(sym))
        pb.progress(100, text="✅ Training complete — model saved to disk!")
        trained = True

    else:  # LOAD
        model = load_model(mp)
        saved = joblib.load(spath(sym))
        scaler, avail, DF_scaled = saved
        # Re-run preprocess to get split indices
        _, _, _, _, _, _, _, test_index, training_size = preprocess(sdf)
        st.success("⚡ Model loaded from disk instantly!")
        trained = True

    # ════════════════════════════════════════════
    #  05 — RESULTS (notebook style)
    # ════════════════════════════════════════════
    if trained:
        test_df  = predict_and_inverse_transform(DF_scaled, X_test,  model, scaler, test_index)
        train_df = predict_and_inverse_transform(
            DF_scaled,
            X_train,
            model,
            scaler,
            DF_scaled.index[WINDOW:WINDOW + len(X_train)]
        )

        mape1,  acc1  = calculate_accuracy(test_df["close"], test_df["Predicted Close"])
        rmse_te       = math.sqrt(mean_squared_error(test_df["close"], test_df["Predicted Close"]))
        mae_te        = mean_absolute_error(test_df["close"], test_df["Predicted Close"])
        r2_te         = r2_score(test_df["close"], test_df["Predicted Close"])
        mape_tr, acc_tr = calculate_accuracy(train_df["close"], train_df["Predicted Close"])

        st.markdown('<div class="sec">05 / Model Performance</div>', unsafe_allow_html=True)

        mc = st.columns(5)
        mc[0].metric("Accuracy",     f"{acc1:.2f}%",  help="100 − MAPE (notebook formula)")
        mc[1].metric("Test RMSE",    fmt_price(rmse_te, CUR_SYM))
        mc[2].metric("Test MAE",     fmt_price(mae_te,  CUR_SYM))
        mc[3].metric("R² Score",     f"{r2_te:.4f}")
        mc[4].metric("MAPE",         f"{mape1:.2f}%")

        qual = ("Excellent" if acc1 > 97 else
                "Good"      if acc1 > 94 else
                "Fair"      if acc1 > 90 else "Needs Tuning")
        bcls = "b-ok" if acc1 > 94 else "b-warn" if acc1 > 90 else "b-blue"

        st.markdown(f"""
        <div class='acc-card'>
            <span class='{bcls}'>● {qual} — {arch_key}</span>
            <span style='margin-left:10px;color:#4a5c7a;font-size:.78rem;'>
                Accuracy={acc1:.2f}% · MAPE={mape1:.2f}% ·
                RMSE={fmt_price(rmse_te,CUR_SYM)} · R²={r2_te:.4f} ·
                Features: {"+".join(avail)} · Window=60d
            </span>
        </div>""", unsafe_allow_html=True)

        # ────────────────────────────────────────
        #  06 — ACTUAL vs PREDICTED (notebook style)
        # ────────────────────────────────────────
        st.markdown('<div class="sec">06 / Actual vs Predicted (Test Set)</div>',
                    unsafe_allow_html=True)

        # Error column
        test_df["Error"] = test_df["close"] - test_df["Predicted Close"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_df.index, y=test_df["close"],
            mode="lines", name="Actual Close Price",
            line=dict(color="#e8b84b", width=2)))
        # Confidence band
        tu, tl = conf_band(test_df["Predicted Close"].values)
        fig.add_trace(go.Scatter(
            x=list(test_df.index) + list(test_df.index[::-1]),
            y=list(tu) + list(tl[::-1]),
            fill="toself", fillcolor="rgba(0,229,160,.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band"))
        fig.add_trace(go.Scatter(
            x=test_df.index, y=test_df["Predicted Close"],
            mode="lines", name="Predicted Close Price",
            line=dict(color="#00e5a0", width=2)))
        fig.update_layout(**pt(),
            title=f"{sym} — Comparison of Actual and Predicted Close Prices ({arch_key})",
            height=460, xaxis_title="Date",
            yaxis_title=f"Price ({CUR_SYM})", yaxis_tickprefix=CUR_SYM)
        st.plotly_chart(fig, width="stretch")
        st.caption("🟡 Actual Close Price  ·  🟢 Predicted Close Price  ·  🟢shaded Confidence Band")

        # Error chart
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=test_df.index, y=test_df["Error"],
            marker_color=["#00e5a0" if v >= 0 else "#ff5370" for v in test_df["Error"]],
            name="Prediction Error", opacity=.8))
        fig_err.add_hline(y=0, line_color="#4a5c7a", line_width=1)
        fig_err.update_layout(**pt(),
            title=f"{sym} — Close Price Prediction Error (Actual − Predicted)",
            height=280, xaxis_title="Date",
            yaxis_title=f"Error ({CUR_SYM})", yaxis_tickprefix=CUR_SYM)
        st.plotly_chart(fig_err, width="stretch")
        st.caption("🟢 Over-predicted (actual > pred)  ·  🔴 Under-predicted (actual < pred)")

        # Error table (notebook Cell 31 style)
        with st.expander("📋 Prediction Error Table (first 20 rows)"):
            err_show = test_df[["close","Predicted Close","Error"]].copy()
            err_show.columns = [f"Actual Close ({CUR_SYM})",
                                 f"Predicted Close ({CUR_SYM})",
                                 f"Error ({CUR_SYM})"]
            err_show = err_show.round(2).head(20)
            err_show.index = err_show.index.astype(str)
            st.dataframe(err_show, use_container_width=True)

        # Train vs Test prediction overlay (full picture)
        st.markdown('<div class="sec">── Full Dataset Overlay</div>', unsafe_allow_html=True)
        fig_full = go.Figure()
        fig_full.add_trace(go.Scatter(
            x=sdf["date"], y=sdf["close"], mode="lines",
            name="Full Actual Close",
            line=dict(color="#4a5c7a", width=1.5)))
        fig_full.add_trace(go.Scatter(
            x=train_df.index, y=train_df["Predicted Close"], mode="lines",
            name="Train Prediction",
            line=dict(color="#4d9fff", width=1.4, dash="dot"), opacity=.8))
        fig_full.add_trace(go.Scatter(
            x=test_df.index, y=test_df["Predicted Close"], mode="lines",
            name="Test Prediction",
            line=dict(color="#00e5a0", width=2.2)))
        fig_full.update_layout(**pt(),
            title=f"{sym} — Full Dataset: Actual + Train + Test Predictions",
            height=420, xaxis_title="Date",
            yaxis_title=f"Price ({CUR_SYM})", yaxis_tickprefix=CUR_SYM)
        st.plotly_chart(fig_full, width="stretch")
        st.caption("⬛ Full actual  ·  🔵 Train prediction (dotted)  ·  🟢 Test prediction")

        # ────────────────────────────────────────
        #  07 — LIVE VALIDATION
        # ────────────────────────────────────────
        if show_live:
            st.markdown('<div class="sec">07 / Live Market Validation (yfinance)</div>',
                        unsafe_allow_html=True)
            live_df, err_msg = fetch_live(sym, period=live_period)

            if err_msg:
                st.warning(f"⚠ yfinance failed for **{sym}**: {err_msg}  "
                           f"→ Falling back to uploaded data.")
                live_df  = sdf.copy()
                live_src = "Uploaded CSV (fallback)"
            else:
                live_src = f"Yahoo Finance · {live_period}"

            if live_df is not None and "close" in live_df.columns and len(live_df) > WINDOW + 10:
                # Build mini DF of available features, scale with saved scaler
                lv_avail = [c for c in avail if c in live_df.columns]
                lv_raw   = live_df[lv_avail].copy()
                lv_raw.index = live_df["date"].values

                # Refit scaler on live close to match scale (same approach as notebook)
                lv_sc = MinMaxScaler()
                lv_sc_vals = lv_sc.fit_transform(lv_raw.values)
                lv_df_sc   = pd.DataFrame(lv_sc_vals, columns=lv_avail, index=lv_raw.index)

                lv_X, lv_y = prepare_time_series_data(lv_df_sc, WINDOW)
                if len(lv_X) > 0:
                    lv_n   = lv_X.shape[2]
                    if lv_n == n_feat:
                        lv_pred_s  = model.predict(lv_X, verbose=0)
                        lv_pred_i  = lv_sc.inverse_transform(lv_pred_s)[:, 0]
                        lv_actual  = lv_sc.inverse_transform(lv_y)[:, 0]
                        lv_dates   = lv_raw.index[WINDOW:]

                        lv_rmse = math.sqrt(mean_squared_error(lv_actual, lv_pred_i))
                        lv_r2   = r2_score(lv_actual, lv_pred_i)
                        lv_mape, lv_acc = calculate_accuracy(lv_actual, lv_pred_i)
                        lv_u, lv_l = conf_band(lv_pred_i)

                        lc1, lc2, lc3, lc4 = st.columns(4)
                        lc1.metric("Live Accuracy", f"{lv_acc:.2f}%")
                        lc2.metric("Live RMSE",     fmt_price(lv_rmse, CUR_SYM))
                        lc3.metric("Live R²",       f"{lv_r2:.4f}")
                        lc4.metric("Live MAPE",     f"{lv_mape:.2f}%")

                        st.markdown(f"""
                        <div class='live-card'>
                            <span class='b-live'>● LIVE VALIDATION</span>
                            <span style='margin-left:10px;color:#4a5c7a;font-size:.76rem;'>
                                {live_src} · {len(lv_actual)} points ·
                                Accuracy={lv_acc:.2f}% · MAPE={lv_mape:.2f}%
                            </span>
                        </div>""", unsafe_allow_html=True)

                        fig_lv = go.Figure()
                        fig_lv.add_trace(go.Scatter(x=lv_dates, y=lv_actual,
                            mode="lines", name="Real Market Price",
                            line=dict(color="#d4dff0", width=2)))
                        fig_lv.add_trace(go.Scatter(
                            x=list(lv_dates) + list(lv_dates[::-1]),
                            y=list(lv_u) + list(lv_l[::-1]),
                            fill="toself", fillcolor="rgba(77,159,255,.08)",
                            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))
                        fig_lv.add_trace(go.Scatter(x=lv_dates, y=lv_pred_i,
                            mode="lines", name="Model Prediction",
                            line=dict(color="#4d9fff", width=2.2)))
                        fig_lv.update_layout(**pt(),
                            title=f"{sym} — Real Market vs Model [{live_src}]",
                            height=420, yaxis_tickprefix=CUR_SYM,
                            xaxis_title="Date", yaxis_title=f"Close ({CUR_SYM})")
                        st.plotly_chart(fig_lv, width="stretch")
                        st.caption("⬜ Real market data  ·  🔵 Model prediction  ·  🔵shaded Confidence band")
                    else:
                        st.info(f"Live data has {lv_n} features but model expects {n_feat}. "
                                "Skipping live validation.")

        # ────────────────────────────────────────
        #  08 — FUTURE FORECAST
        # ────────────────────────────────────────
        sn = "08" if show_live else "07"
        st.markdown(f'<div class="sec">{sn} / Future Forecast</div>',
                    unsafe_allow_html=True)

        fp    = run_forecast(model, scaler, DF_scaled, avail, fdays)
        fd    = pd.date_range(
            start=sdf["date"].iloc[-1] + pd.Timedelta(days=1),
            periods=fdays, freq="B")
        fu, fl = conf_band(fp)

        fig2 = go.Figure()
        ctx  = min(120, len(sdf))
        fig2.add_trace(go.Scatter(
            x=sdf["date"].iloc[-ctx:], y=sdf["close"].iloc[-ctx:],
            mode="lines", name="Historical Close",
            line=dict(color="#4a5c7a", width=1.8)))
        fig2.add_trace(go.Scatter(
            x=list(fd) + list(fd[::-1]),
            y=list(fu) + list(fl[::-1]),
            fill="toself", fillcolor="rgba(232,184,75,.09)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))
        fig2.add_trace(go.Scatter(
            x=fd, y=fp, mode="lines+markers",
            name=f"{fdays}D Forecast",
            line=dict(color="#e8b84b", width=2.5),
            marker=dict(size=4, color="#e8b84b")))
        fig2.update_layout(**pt(),
            title=f"{sym} — {fdays}-Day Future Price Forecast ({arch_key})",
            height=440, xaxis_title="Date",
            yaxis_title=f"Price ({CUR_SYM})", yaxis_tickprefix=CUR_SYM)
        st.plotly_chart(fig2, width="stretch")
        st.caption("⬛ Historical (last 120 days)  ·  🟡 Forecast  ·  🟡shaded Confidence band")

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Forecast Start", fmt_price(fp[0],    CUR_SYM))
        fc2.metric("Forecast End",   fmt_price(fp[-1],   CUR_SYM),
                   delta=f"{(fp[-1]-fp[0])/fp[0]*100:+.2f}%")
        fc3.metric("Forecast High",  fmt_price(fp.max(), CUR_SYM))
        fc4.metric("Forecast Low",   fmt_price(fp.min(), CUR_SYM))

        with st.expander("📋 Full Forecast Table"):
            fdf = pd.DataFrame({
                "Date":                  fd.strftime("%Y-%m-%d"),
                f"Predicted ({CUR_SYM})":np.round(fp, 2),
                f"Upper ({CUR_SYM})":    np.round(fu, 2),
                f"Lower ({CUR_SYM})":    np.round(fl, 2),
                "Change %":             np.round(
                    pd.Series(fp).pct_change().fillna(0).values * 100, 3)
            })
            st.dataframe(fdf, use_container_width=True)

        # ────────────────────────────────────────
        #  09 — SIGNAL
        # ────────────────────────────────────────
        sn2 = "09" if show_live else "08"
        st.markdown(f'<div class="sec">{sn2} / Trading Signal</div>',
                    unsafe_allow_html=True)

        signal, mom = get_signal(fp, window=min(7, fdays))
        scls = {"BUY":"sig-buy","SELL":"sig-sell","HOLD":"sig-hold"}[signal]
        sico = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}[signal]
        sdsc = {
            "BUY":  f"Forecasted upward momentum of {abs(mom):.2f}% — model indicates potential price appreciation.",
            "SELL": f"Forecasted downward momentum of {abs(mom):.2f}% — model indicates potential price decline.",
            "HOLD": f"Flat momentum ({abs(mom):.2f}%) — no strong directional trend detected.",
        }[signal]

        sg1, sg2 = st.columns([1, 3])
        with sg1:
            st.markdown(f"""
            <div class='{scls}' style='font-size:1.3rem;padding:20px;'>
                {sico} {signal} SIGNAL
                <div style='font-size:.72rem;margin-top:6px;opacity:.8;'>
                    Momentum: {mom:+.2f}%
                </div>
            </div>""", unsafe_allow_html=True)
        with sg2:
            st.markdown(f"""
            <div class='card' style='padding:16px 20px;'>
                <div style='font-size:.83rem;color:#d4dff0;line-height:1.7;'>
                    {sdsc}<br><br>
                    <span style='color:#4a5c7a;font-size:.73rem;'>
                        ⚠ Signal is generated by a deep learning model for
                        <b>educational purposes only</b>. Does not constitute
                        financial advice. Consult a SEBI-registered advisor
                        before making any trading decisions.
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

        sig_clr  = {"BUY":"#e8b84b","SELL":"#ff5370","HOLD":"#4d9fff"}[signal]
        sig_fill = {"BUY":"rgba(232,184,75,.06)","SELL":"rgba(255,83,112,.06)",
                    "HOLD":"rgba(77,159,255,.06)"}[signal]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=fd, y=fp, mode="lines+markers",
            line=dict(color=sig_clr, width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor=sig_fill, name=f"{signal} Signal"))
        fig3.update_layout(**pt(),
            title=f"Signal Chart — {signal} · {fdays}D Forecast",
            height=270, xaxis_title="Date",
            yaxis_title=f"Price ({CUR_SYM})", yaxis_tickprefix=CUR_SYM)
        st.plotly_chart(fig3, width="stretch")

        # ────────────────────────────────────────
        #  10 — LOSS CURVE
        # ────────────────────────────────────────
        if train_btn and hist_obj:
            st.markdown('<div class="sec">10 / Training Loss Curve</div>',
                        unsafe_allow_html=True)
            losses = pd.DataFrame(hist_obj.history)
            fig4   = go.Figure()
            fig4.add_trace(go.Scatter(y=losses["loss"], mode="lines",
                name="Train Loss", line=dict(color="#4d9fff", width=2)))
            if "val_loss" in losses.columns:
                fig4.add_trace(go.Scatter(y=losses["val_loss"], mode="lines",
                    name="Val Loss", line=dict(color="#ff5370", width=2)))
            if "mean_absolute_error" in losses.columns:
                fig4.add_trace(go.Scatter(y=losses["mean_absolute_error"], mode="lines",
                    name="Train MAE", line=dict(color="#e8b84b", width=1.5, dash="dot")))
            fig4.update_layout(**pt(),
                title=f"Loss & MAE vs Epochs — {arch_key}",
                height=300, xaxis_title="Epoch", yaxis_title="Loss / MAE")
            st.plotly_chart(fig4, width="stretch")
            st.caption("🔵 Train loss  ·  🔴 Val loss  ·  🟡 Train MAE (dotted)  —  lower & converging = good model")

            # Accuracy bar chart (like notebook Cell 33)
            st.markdown('<div class="sec">── Model Accuracy Summary</div>',
                        unsafe_allow_html=True)
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=["Train Accuracy", "Test Accuracy"],
                y=[round(acc_tr, 2), round(acc1, 2)],
                marker_color=["#4d9fff", "#00e5a0"],
                text=[f"{acc_tr:.2f}%", f"{acc1:.2f}%"],
                textposition="outside",
                width=0.4
            ))
            fig_acc.update_layout(**pt(),
                title=f"Model Accuracy Comparison — {arch_key} ({sym})",
                height=300, yaxis_range=[0, 105],
                xaxis_title="Split", yaxis_title="Accuracy (%)")
            st.plotly_chart(fig_acc, width="stretch")
            st.caption("🔵 Train accuracy  ·  🟢 Test accuracy  —  (Accuracy = 100 − MAPE, notebook formula)")

# Idle
elif not train_btn and not load_btn:
    st.markdown(f"""
    <div class='card' style='text-align:center;padding:36px;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>🤖</div>
        <div style='font-family:IBM Plex Mono,monospace;color:#e8b84b;font-size:.88rem;'>
            Ready to train or load
        </div>
        <div style='color:#4a5c7a;font-size:.8rem;line-height:1.8;margin-top:8px;'>
            Select <b style='color:#e8b84b;'>Architecture</b> in sidebar →
            click <b style='color:#e8b84b;'>🚀 TRAIN {arch_key}</b> to build and save.<br>
            Once trained, click <b style='color:#00e5a0;'>⚡ LOAD SAVED</b> on all future visits — instant results.<br><br>
            <b style='color:#d4dff0;'>Developer workflow:</b> Train on your machine →
            commit <code style='color:#e8b84b;'>saved_models/</code> →
            deploy → users get results without any training.
        </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  FOOTER — NON-REMOVABLE
# ════════════════════════════════════════════════
st.markdown(f"""
<div class='footer'>
    {COPYRIGHT}<br>
    {__project__} · {__degree__} · {__college__}<br>
    Watermark ID: {WATERMARK} · Version {__version__}
</div>""", unsafe_allow_html=True)
