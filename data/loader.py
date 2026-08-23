import yfinance as yf
import pandas as pd
import time

def load_historical_data(tickers, start_date_str, end_date_str=None):
    """
    Fetch adjusted close prices and align them strictly to a trading calendar.
    Forward-fills missing values where economically appropriate, but DOES NOT backward-fill.
    """
    data = pd.DataFrame()
    failed = []
    
    # We use the benchmark index (Nifty 50) to define the true trading calendar
    benchmark_ticker = "^NSEI"
    try:
        benchmark = yf.download(benchmark_ticker, start=start_date_str, end=end_date_str, progress=False)
        if benchmark.empty:
            trading_calendar = pd.date_range(start=start_date_str, end=end_date_str or pd.Timestamp.today(), freq='B')
        else:
            trading_calendar = benchmark.index
    except Exception:
        # Fallback to business days if benchmark fails
        trading_calendar = pd.date_range(start=start_date_str, end=end_date_str or pd.Timestamp.today(), freq='B')

    for ticker in tickers:
        try:
            temp = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
            if temp.empty:
                failed.append(ticker)
                continue
            close = temp["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            data[ticker] = close
            time.sleep(0.1)
        except Exception:
            failed.append(ticker)
            
    if data.empty:
        return data, failed
        
    # Reindex to the strict trading calendar
    data = data.reindex(trading_calendar)
    
    # Forward fill to handle random missing days, but do not bfill
    # This ensures no future information leaks backwards.
    data = data.ffill()
    
    # Drop rows where all elements are NaN (e.g. at the very beginning)
    data = data.dropna(how='all')
    
    return data, failed

def load_benchmark(start_date_str, end_date_str=None):
    """Fetch Nifty 50 index data for benchmarking."""
    try:
        bench = yf.download("^NSEI", start=start_date_str, end=end_date_str, progress=False)
        if bench.empty:
            return None
        close = bench["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close
    except Exception:
        return None
