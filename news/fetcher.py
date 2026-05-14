

import feedparser
import re
from datetime import datetime, timedelta


# RSS feed sources — Indian financial news
RSS_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Economic Times Stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "Moneycontrol Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Moneycontrol Stocks": "https://www.moneycontrol.com/rss/latestnews.xml",
    "LiveMint Markets": "https://www.livemint.com/rss/markets",
}


def _clean_html(text):
    """Remove HTML tags from text."""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


def _extract_ticker_name(ticker):
    """
    Convert NSE ticker to a searchable company name.
    e.g., 'HDFCBANK.NS' → 'HDFC Bank', 'RELIANCE.NS' → 'Reliance'
    """
    # Remove .NS suffix
    name = ticker.replace(".NS", "")

    # Common mappings for better search matching
    name_map = {
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank",
        "KOTAKBANK": "Kotak",
        "SBIN": "SBI",
        "AXISBANK": "Axis Bank",
        "INDUSINDBK": "IndusInd Bank",
        "BAJFINANCE": "Bajaj Finance",
        "BAJAJFINSV": "Bajaj Finserv",
        "HDFCLIFE": "HDFC Life",
        "SBILIFE": "SBI Life",
        "TCS": "TCS",
        "INFY": "Infosys",
        "HCLTECH": "HCL Tech",
        "WIPRO": "Wipro",
        "TECHM": "Tech Mahindra",
        "RELIANCE": "Reliance",
        "ONGC": "ONGC",
        "NTPC": "NTPC",
        "POWERGRID": "Power Grid",
        "ADANIENT": "Adani Enterprises",
        "ADANIPORTS": "Adani Ports",
        "BPCL": "BPCL",
        "COALINDIA": "Coal India",
        "HINDUNILVR": "Hindustan Unilever",
        "ITC": "ITC",
        "NESTLEIND": "Nestle India",
        "TATACONSUM": "Tata Consumer",
        "BRITANNIA": "Britannia",
        "MARUTI": "Maruti",
        "TATAMOTORS": "Tata Motors",
        "M&M": "Mahindra",
        "BAJAJ-AUTO": "Bajaj Auto",
        "EICHERMOT": "Eicher Motors",
        "HEROMOTOCO": "Hero MotoCorp",
        "SUNPHARMA": "Sun Pharma",
        "DRREDDY": "Dr Reddy",
        "CIPLA": "Cipla",
        "APOLLOHOSP": "Apollo Hospitals",
        "DIVISLAB": "Divi's Lab",
        "TATASTEEL": "Tata Steel",
        "JSWSTEEL": "JSW Steel",
        "HINDALCO": "Hindalco",
        "BHARTIARTL": "Bharti Airtel",
        "ULTRACEMCO": "UltraTech Cement",
        "GRASIM": "Grasim",
        "TITAN": "Titan",
        "ASIANPAINT": "Asian Paints",
        "GOLDBEES": "Gold ETF",
    }

    return name_map.get(name, name)


def fetch_headlines(max_per_feed=10):
    """
    Fetch latest financial news headlines from all RSS feeds.

    Parameters
    ----------
    max_per_feed : int
        Maximum headlines to pull per feed source.

    Returns
    -------
    list[dict]
        List of headline dicts with keys:
        - title: str
        - summary: str
        - source: str
        - published: str
        - link: str
    """
    all_headlines = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                headline = {
                    "title": _clean_html(entry.get("title", "")),
                    "summary": _clean_html(entry.get("summary", ""))[:300],
                    "source": source_name,
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                }
                all_headlines.append(headline)
        except Exception:
            # Silently skip feeds that fail — RSS sources can be flaky
            continue

    return all_headlines


def filter_headlines_for_stocks(headlines, tickers):
    """
    Filter headlines relevant to specific stocks.

    Parameters
    ----------
    headlines : list[dict]
        All fetched headlines.
    tickers : list[str]
        NSE ticker symbols (e.g., ['HDFCBANK.NS', 'INFY.NS']).

    Returns
    -------
    dict[str, list[dict]]
        Mapping of ticker → list of relevant headlines.
    """
    stock_headlines = {t: [] for t in tickers}

    # Build search terms for each ticker
    search_terms = {}
    for ticker in tickers:
        name = _extract_ticker_name(ticker)
        raw = ticker.replace(".NS", "")
        # Search for both the company name and raw ticker
        search_terms[ticker] = [name.lower(), raw.lower()]

    for headline in headlines:
        text = (headline["title"] + " " + headline["summary"]).lower()
        for ticker in tickers:
            for term in search_terms[ticker]:
                if term in text:
                    stock_headlines[ticker].append(headline)
                    break  # Don't double-add

    return stock_headlines


def get_general_market_headlines(headlines, limit=10):
    """
    Return top general market headlines (not stock-specific).

    Parameters
    ----------
    headlines : list[dict]
    limit : int

    Returns
    -------
    list[dict]
    """
    market_keywords = [
        "market", "nifty", "sensex", "rally", "crash", "bull",
        "bear", "trade", "invest", "portfolio", "index", "stock market"
    ]

    general = []
    for h in headlines:
        text = (h["title"] + " " + h["summary"]).lower()
        if any(kw in text for kw in market_keywords):
            general.append(h)
        if len(general) >= limit:
            break

    return general
