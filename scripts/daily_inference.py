"""
Daily Inference Script — runs inside GitHub Actions.

Responsibilities:
  1. Fetch today's closing prices (point-in-time).
  2. Run FinBERT on the latest RSS headlines.
  3. Map sentiment → Black-Litterman views.
  4. Compute optimal portfolio weights.
  5. Save a timestamped JSON artifact to predictions/<date>/daily_report.json
     so historical records are never overwritten.

This script is intentionally lightweight — no LSTM training,
only inference using the pre-trained weights saved in models/saved/.
"""

import json
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── Make project root importable ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nifty50 import DEFAULT_STOCKS, NIFTY50
from data.loader import load_historical_data
from news.fetcher import fetch_headlines, filter_headlines_for_stocks
from news.sentiment import analyze_headlines, compute_stock_sentiment
from models.black_litterman import (
    compute_implied_returns,
    generate_black_litterman_views,
    optimize_black_litterman,
)
from scipy.optimize import minimize

# ── Config ────────────────────────────────────────────────────────────────
TICKERS        = DEFAULT_STOCKS
RISK_FREE_RATE = 0.07
RISK_AVERSION  = 2.5
MAX_WEIGHT     = max(0.30, 1.0 / len(TICKERS))  # ensures feasibility
SECTOR_CAP     = 0.50
WINDOW         = 252
LOOKBACK_START = "2022-01-01"

RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()
RUN_DATE      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
OUTPUT_DIR    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "predictions", RUN_DATE,
)


def optimize_portfolio(expected_returns, cov_matrix, n, max_weight=0.30, risk_free=0.07):
    """Maximize Sharpe subject to constraints."""
    init = np.ones(n) / n
    bounds = tuple((0.0, max_weight) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        ret = np.dot(w, expected_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(ret - risk_free) / vol if vol > 0 else 0.0

    res = minimize(neg_sharpe, init, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    return res.x if res.success else init


def main():
    report = {
        "generated_at": RUN_TIMESTAMP,
        "run_date": RUN_DATE,
        "pipeline_version": "v2.0",
        "universe": TICKERS,
        "status": "ok",
        "errors": [],
    }

    # ── 1. Market data ────────────────────────────────────────────────────
    print(f"[1/4] Fetching market data from {LOOKBACK_START} …")
    data, failed = load_historical_data(TICKERS, LOOKBACK_START)
    if failed:
        report["errors"].append(f"Failed tickers: {failed}")
    if data.empty or len(data.columns) < 2:
        report["status"] = "failed"
        report["errors"].append("Insufficient market data.")
        _save(report)
        return

    active = list(data.columns)
    n = len(active)
    returns = data.pct_change().dropna(how="all").fillna(0)
    hist = returns.iloc[-WINDOW:]
    mean_rets = hist.mean() * 252
    cov_matrix = hist.cov() * 252

    # ── 2. Sentiment ──────────────────────────────────────────────────────
    print("[2/4] Fetching news and running FinBERT …")
    sentiment_views = {}
    try:
        headlines = fetch_headlines(max_per_feed=5)
        stock_headlines = filter_headlines_for_stocks(headlines, active)
        for ticker in active:
            hl = stock_headlines.get(ticker, [])
            if hl:
                analyzed = analyze_headlines(hl)
                sent = compute_stock_sentiment(analyzed)
            else:
                sent = {
                    "overall_label": "Neutral",
                    "overall_score": 0.0,
                    "confidence": 0.0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "risk_flag": False,
                }
            sentiment_views[ticker] = sent
    except Exception as e:
        report["errors"].append(f"Sentiment error: {e}")
        sentiment_views = {t: {"overall_score": 0.0, "confidence": 0.0} for t in active}

    report["sentiment_views"] = sentiment_views

    # ── 3. Black-Litterman weights ────────────────────────────────────────
    print("[3/4] Running Black-Litterman optimisation …")
    try:
        market_weights = np.ones(n) / n
        implied_rets = compute_implied_returns(cov_matrix, market_weights, RISK_AVERSION)
        P, Q, Omega = generate_black_litterman_views(active, implied_rets, cov_matrix, sentiment_views)
        post_rets, post_cov = optimize_black_litterman(implied_rets, cov_matrix, P, Q, Omega)
        weights = optimize_portfolio(post_rets.values, post_cov.values, n, MAX_WEIGHT, RISK_FREE_RATE)
    except Exception as e:
        report["errors"].append(f"BL error: {e}\n{traceback.format_exc()}")
        weights = np.ones(n) / n

    report["optimal_weights"] = {t: round(float(w), 6) for t, w in zip(active, weights)}

    # ── 4. Portfolio analytics ────────────────────────────────────────────
    print("[4/4] Computing portfolio analytics …")
    try:
        w = weights
        port_ret = float(np.dot(w, mean_rets))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
        sharpe = (port_ret - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0.0
        report["portfolio_analytics"] = {
            "expected_annual_return": round(port_ret, 6),
            "annual_volatility": round(port_vol, 6),
            "sharpe_ratio": round(sharpe, 4),
        }
    except Exception as e:
        report["errors"].append(f"Analytics error: {e}")

    _save(report)
    print(f"\n✅  Report saved → {OUTPUT_DIR}/daily_report.json")


def _save(report):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "daily_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
