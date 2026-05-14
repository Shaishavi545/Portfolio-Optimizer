"""
Nifty 50 stock universe with sector classifications and Gold ETF options.
"""

# Full Nifty 50 constituents with NSE tickers and GICS-style sector tags
# Updated as of May 2026
NIFTY50 = {
    # BFSI — Banking, Financial Services & Insurance
    "HDFCBANK.NS":   "BFSI",
    "ICICIBANK.NS":  "BFSI",
    "KOTAKBANK.NS":  "BFSI",
    "SBIN.NS":       "BFSI",
    "AXISBANK.NS":   "BFSI",
    "INDUSINDBK.NS": "BFSI",
    "BAJFINANCE.NS": "BFSI",
    "BAJAJFINSV.NS": "BFSI",
    "HDFCLIFE.NS":   "BFSI",
    "SBILIFE.NS":    "BFSI",

    # IT — Information Technology
    "TCS.NS":        "IT",
    "INFY.NS":       "IT",
    "HCLTECH.NS":    "IT",
    "WIPRO.NS":      "IT",
    "TECHM.NS":      "IT",
    "LTIMindtree.NS":"IT",

    # Energy & Oil
    "RELIANCE.NS":   "Energy",
    "ONGC.NS":       "Energy",
    "NTPC.NS":       "Energy",
    "POWERGRID.NS":  "Energy",
    "ADANIENT.NS":   "Energy",
    "ADANIPORTS.NS": "Energy",
    "BPCL.NS":       "Energy",
    "COALINDIA.NS":  "Energy",

    # FMCG — Fast Moving Consumer Goods
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS":        "FMCG",
    "NESTLEIND.NS":  "FMCG",
    "TATACONSUM.NS": "FMCG",
    "BRITANNIA.NS":  "FMCG",

    # Auto — Automobile & Ancillaries
    "MARUTI.NS":     "Auto",
    "TATAMOTORS.NS": "Auto",
    "M&M.NS":        "Auto",
    "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS":  "Auto",
    "HEROMOTOCO.NS": "Auto",

    # Pharma & Healthcare
    "SUNPHARMA.NS":  "Pharma",
    "DRREDDY.NS":    "Pharma",
    "CIPLA.NS":      "Pharma",
    "APOLLOHOSP.NS": "Pharma",
    "DIVISLAB.NS":   "Pharma",

    # Metals & Mining
    "TATASTEEL.NS":  "Metals",
    "JSWSTEEL.NS":   "Metals",
    "HINDALCO.NS":   "Metals",

    # Telecom
    "BHARTIARTL.NS": "Telecom",

    # Cement & Construction
    "ULTRACEMCO.NS": "Cement",
    "GRASIM.NS":     "Cement",

    # Others
    "TITAN.NS":      "Consumer Durables",
    "ASIANPAINT.NS": "Consumer Durables",
    "LTIM.NS":       "IT",
}

# Gold ETFs available on NSE
GOLD_ETFS = {
    "GOLDBEES.NS": "Nippon Gold ETF",
}

# Nifty 50 benchmark index
BENCHMARK_INDEX = "^NSEI"


def get_all_tickers():
    """Return sorted list of all Nifty 50 tickers."""
    return sorted(NIFTY50.keys())


def get_sectors(tickers):
    """
    Return a dict mapping sector → list of tickers for the given tickers.

    Parameters
    ----------
    tickers : list[str]
        List of NSE ticker symbols.

    Returns
    -------
    dict[str, list[str]]
        Sector name → list of tickers in that sector.
    """
    sectors = {}
    for t in tickers:
        sector = NIFTY50.get(t, "Other")
        sectors.setdefault(sector, []).append(t)
    return sectors


def get_sector_weights(tickers, weights):
    """
    Compute portfolio weight per sector.

    Parameters
    ----------
    tickers : list[str]
    weights : array-like
        Portfolio weights (same order as tickers).

    Returns
    -------
    dict[str, float]
        Sector name → total weight in that sector.
    """
    sector_weights = {}
    for t, w in zip(tickers, weights):
        sector = NIFTY50.get(t, "Other")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w
    return sector_weights


# Default stocks for first-time users
DEFAULT_STOCKS = [
    "HDFCBANK.NS", "INFY.NS", "RELIANCE.NS",
    "HINDUNILVR.NS", "TCS.NS"
]
