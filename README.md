# StockMind AI — Production V2
### © 2025 Selladurai S · Paavai College of Engineering · All Rights Reserved

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at: **http://localhost:8501**

---

## 🧑‍💻 Developer Workflow (Recommended)

```
1. Train the model on your server/machine:
   → Open app → Upload data.csv → Click TRAIN LSTM → done

2. Commit saved_models/ folder to your repo or deployment

3. Deploy app → users just upload CSV → see results instantly
   (No training required on user side)
```

---

## ✅ V2 Feature List

| Feature | Details |
|---------|---------|
| 💱 **Multi-Currency** | ₹ INR · $ USD · € EUR · £ GBP · ¥ JPY — Indian-style grouping for ₹ |
| 💾 **Persistent Data** | CSV saved to `data/stock_data.csv` — auto-loads on restart |
| 🧠 **Persistent Model** | Saved to `saved_models/` — loads in <2s, no retraining |
| 🌐 **Live Validation** | yfinance fetches real market data → overlay vs model prediction |
| 📡 **Fallback** | If yfinance fails → falls back to uploaded CSV silently |
| 🔮 **Future Forecast** | 7–60 day ahead forecast on business days only |
| 📊 **Confidence Bands** | Upper/lower bands on test predictions AND future forecast |
| 🟢 **Buy/Sell/Hold** | Momentum signal from forecast trend with SEBI disclaimer |
| 📈 **Full EDA** | Price+MA(20/50/200), OHLC, Volume, Returns, Heatmap, Raw data |
| 🏷️ **Copyright** | Non-removable watermark: sidebar + header badge + footer |
| ⚠️ **Zero Warnings** | `use_container_width` fully replaced, TF silenced completely |
| 🎯 **Any Dataset** | Works for NSE/BSE/US/Crypto — any CSV with symbol+date+close |

---

## 📁 Folder Structure

```
your-project/
├── app.py
├── requirements.txt
├── data/
│   └── stock_data.csv          ← auto-saved on first upload
└── saved_models/
    ├── GOOG_LSTM_lb60_u128.keras   ← trained model
    └── GOOG_close_scaler.pkl       ← fitted scaler
```

---

## 📋 CSV Column Reference

| Column | Required | Notes |
|--------|----------|-------|
| `symbol` | ✅ | Ticker: GOOG, RELIANCE.NS, TCS.NS, etc. |
| `date` | ✅ | Any standard date format |
| `close` | ✅ | Closing price (model trains on this by default) |
| `open`, `high`, `low` | Optional | Enables OHLC candlestick chart |
| `volume` | Optional | Enables volume bar chart |
| `adjClose` | Optional | Can be selected as prediction target |

---

## 🌐 Supported Live Tickers (yfinance)

| Exchange | Example Symbols |
|----------|----------------|
| NSE India | RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS |
| BSE India | RELIANCE.BO, TCS.BO |
| US | AAPL, GOOG, MSFT, TSLA |
| Crypto | BTC-USD, ETH-USD |

---

## 📊 Graph Legend Reference

| Chart | Lines |
|-------|-------|
| Price+MA | 🟡 Close · 🔵 MA20 · 🟡 MA50 · 🔴 MA200 |
| OHLC | 🟢 Bullish · 🔴 Bearish |
| Volume | 🟢 Up-day · 🔴 Down-day · 🟡 20D Avg |
| Returns | 🔵 Daily % · 🟡 Distribution histogram |
| Actual vs Predicted | ⬛ Actual · 🔵 Train (dotted) · 🟢 Test · 🟢shaded Band |
| Live Validation | ⬜ Real market · 🔵 Model · 🔵shaded Band |
| Forecast | ⬛ History · 🟡 Forecast · 🟡shaded Band |
| Signal Chart | Signal-colored line based on BUY/SELL/HOLD |

---

## ⚖️ Legal Notice

**© 2025 Selladurai S. All Rights Reserved.**
Paavai College of Engineering · B.Tech – Artificial Intelligence & Data Science
StockMind AI – Deep Learning Stock Price Predictor

Unauthorized copying, modification, or redistribution of this software is
strictly prohibited. The copyright watermark is non-removable and embedded
in both source code and UI rendering.

**For educational use only. Does not constitute financial advice.**
