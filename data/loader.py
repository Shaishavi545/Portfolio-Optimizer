import yfinance as yf
import pandas as pd
import time

def load_historical_data(tickers, start_date_str, end_date_str=None, max_retries=3):
    """
    Fetch adjusted close prices in a single batch request to avoid Yahoo Finance rate limits.
    Aligns prices strictly to a trading calendar.
    Forward-fills missing values where economically appropriate, but DOES NOT backward-fill.
    """
    data = pd.DataFrame()
    failed = []
    
    # ── 1. Batch download benchmark and all tickers together in ONE call ──────
    all_symbols = list(set(["^NSEI"] + list(tickers)))
    
    batch_df = None
    for attempt in range(max_retries):
        try:
            batch_df = yf.download(
                all_symbols,
                start=start_date_str,
                end=end_date_str,
                progress=False,
                threads=True,
                group_by="ticker"
            )
            if batch_df is not None and not batch_df.empty:
                break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
            
    if batch_df is None or batch_df.empty:
        # Fallback to business days calendar if batch download failed completely
        trading_calendar = pd.date_range(start=start_date_str, end=end_date_str or pd.Timestamp.today(), freq='B')
    else:
        # Check if ^NSEI benchmark is present for calendar
        if "^NSEI" in batch_df.columns.get_level_values(0):
            bench_close = batch_df["^NSEI"]["Close"].dropna()
            trading_calendar = bench_close.index if not bench_close.empty else batch_df.index
        else:
            trading_calendar = batch_df.index

    # ── 2. Extract Close prices for requested tickers ──────────────────────────
    for ticker in tickers:
        try:
            if batch_df is not None and ticker in batch_df.columns.get_level_values(0):
                close = batch_df[ticker]["Close"].dropna()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                if not close.empty:
                    data[ticker] = close
                else:
                    failed.append(ticker)
            else:
                failed.append(ticker)
        except Exception:
            failed.append(ticker)
            
    if data.empty:
        return data, failed
        
    # Reindex to strict trading calendar & forward-fill missing days (no bfill)
    data = data.reindex(trading_calendar).ffill().dropna(how='all')
    
    return data, failed

def load_benchmark(start_date_str, end_date_str=None, max_retries=3):
    """Fetch Nifty 50 index data for benchmarking with retries."""
    for attempt in range(max_retries):
        try:
            bench = yf.download("^NSEI", start=start_date_str, end=end_date_str, progress=False)
            if bench is not None and not bench.empty:
                close = bench["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None
