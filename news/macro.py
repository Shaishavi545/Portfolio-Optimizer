"""
Macro event detector for portfolio-wide risk alerts.

Scans news headlines for keywords indicating macroeconomic events
that could affect the entire portfolio, not just individual stocks.
"""

# Macro event keyword groups with descriptions
MACRO_KEYWORDS = {
    "RBI Policy": [
        "rbi rate", "rbi policy", "repo rate", "reverse repo",
        "monetary policy", "rbi governor", "rate cut", "rate hike",
        "interest rate", "rbi mpc"
    ],
    "Budget & Fiscal": [
        "union budget", "fiscal deficit", "budget 2026", "budget 2025",
        "finance minister", "tax reform", "gst rate", "fiscal policy"
    ],
    "FII / FPI Flows": [
        "fii outflow", "fii inflow", "fpi outflow", "fpi inflow",
        "foreign investor", "foreign institutional", "fii sell",
        "fii buy", "fpi sell", "fpi buy"
    ],
    "Crude Oil": [
        "crude oil", "oil price", "brent crude", "opec",
        "oil surge", "oil crash", "petroleum"
    ],
    "Currency": [
        "rupee fall", "rupee rise", "dollar rupee", "inr usd",
        "rupee depreci", "rupee appreci", "forex reserve"
    ],
    "Inflation": [
        "cpi inflation", "wpi inflation", "inflation data",
        "consumer price", "wholesale price", "inflation surge"
    ],
    "Global Events": [
        "us fed", "federal reserve", "recession", "global slowdown",
        "trade war", "tariff", "geopolitical", "china economy"
    ],
}


def detect_macro_events(headlines):
    """
    Scan headlines for macroeconomic event keywords.

    Parameters
    ----------
    headlines : list[dict]
        List of headline dicts with 'title' and 'summary' keys.

    Returns
    -------
    list[dict]
        List of detected macro events with keys:
        - category: str (e.g., "RBI Policy")
        - headline: dict (the matching headline)
        - matched_keyword: str
    """
    events = []
    seen_titles = set()

    for headline in headlines:
        text = (headline["title"] + " " + headline.get("summary", "")).lower()

        for category, keywords in MACRO_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text and headline["title"] not in seen_titles:
                    events.append({
                        "category": category,
                        "headline": headline,
                        "matched_keyword": keyword,
                    })
                    seen_titles.add(headline["title"])
                    break  # One match per headline per category

    return events


def get_alert_level(events):
    """
    Determine overall alert level based on number of macro events.

    Parameters
    ----------
    events : list[dict]

    Returns
    -------
    tuple[str, str]
        (level, message) where level is 'low', 'medium', or 'high'.
    """
    n = len(events)
    categories = set(e["category"] for e in events)

    if n == 0:
        return "low", "No significant macro events detected."
    elif n <= 2 and len(categories) <= 1:
        return "medium", f"{n} macro event(s) detected in {', '.join(categories)}. Monitor closely."
    else:
        return "high", f"{n} macro events across {len(categories)} categories detected! Review portfolio risk."
