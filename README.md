# Institutional Portfolio Optimizer

An institutional-grade Indian equity portfolio optimizer powered by **FinBERT**, **Black-Litterman**, **LSTM Volatility Classification**, and a strict walk-forward backtesting engine.

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-optimizer-m29d.onrender.com/)

## 🤖 GitHub Actions: Daily Inference
Every weekday at **5:30 PM IST** (after NSE close), a GitHub Actions workflow automatically:
1. Fetches latest closing prices.
2. Runs **FinBERT** on live financial RSS headlines.
3. Maps sentiment + confidence to Black-Litterman investor views.
4. Computes optimal portfolio weights and saves a timestamped artifact.

Prediction artifacts are versioned by date (`predictions/<YYYY-MM-DD>/daily_report.json`) and never overwritten, providing a complete, auditable historical record.

## Architecture

```
Historical Market Data  +  Live RSS News
          │                      │
   Calendar-Aware            FinBERT NLP
   Data Loader (no bfill)         │
          │              Sentiment + Confidence
          └──────────┬──────────┘
                     ↓
             Black-Litterman
       (Market-implied returns + AI Views)
                     ↓
           Portfolio Optimizer
       (Max weight, sector caps enforced)
                     ↓
       Walk-Forward Backtester
    (Invariants checked at every step)
                     ↓
      Ablation: EqualWeight vs Markowitz vs BL vs BL+LSTM
```

## Project Structure

```
Portfolio-Optimizer/
│
├── app.py                        # Streamlit dashboard
├── requirements.txt
│
├── data/
│   ├── nifty50.py                # Universe & sector mappings
│   └── loader.py                 # Calendar-aware, no-bfill loader
│
├── engine/
│   └── backtester.py             # Walk-forward backtester with invariants
│
├── models/
│   ├── black_litterman.py        # BL optimizer + FinBERT view mapping
│   └── lstm_volatility.py        # LSTM volatility regime classifier
│
├── news/
│   ├── fetcher.py                # RSS aggregator
│   ├── sentiment.py              # FinBERT inference engine
│   └── macro.py                  # Macro event detection
│
├── scripts/
│   ├── daily_inference.py        # GitHub Actions daily pipeline script
│   └── artifact_reader.py        # Reads versioned prediction artifacts
│
├── predictions/
│   └── <YYYY-MM-DD>/
│       └── daily_report.json     # Timestamped daily output (never overwritten)
│
└── .github/workflows/
    └── daily_pipeline.yml        # Weekday cron job
```

## How to Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → Select your repo → set `app.py` as the main file.
4. Click **Deploy**. You will get a public link instantly.

> **Disclaimer:** This tool is for educational/research purposes only. It is NOT financial advice. Past performance does not guarantee future results.
