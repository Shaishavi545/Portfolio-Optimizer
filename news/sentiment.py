"""
Sentiment analysis engine using FinBERT.

Uses ProsusAI/finbert — a BERT model fine-tuned on financial text.
Runs locally, no API key needed. ~250MB model download on first run.
"""

import streamlit as st

# Lazy-load the model to avoid importing torch at module level
_classifier = None


@st.cache_resource(show_spinner="Loading FinBERT sentiment model...")
def _load_model():
    """Load the FinBERT model. Cached so it only loads once per session."""
    try:
        from transformers import pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,  # Return all scores
        )
        return classifier
    except Exception as e:
        st.warning(f"Could not load FinBERT model: {e}. Sentiment analysis disabled.")
        return None


def analyze_sentiment(text):
    """
    Analyze sentiment of a single text string.

    Parameters
    ----------
    text : str
        Financial news headline or text.

    Returns
    -------
    dict
        {
            "label": "Positive" | "Negative" | "Neutral",
            "score": float (0-1 confidence),
            "all_scores": dict of all label scores
        }
        Returns None if model is unavailable.
    """
    classifier = _load_model()
    if classifier is None:
        return None

    try:
        # FinBERT max length is 512 tokens — truncate long text
        text = text[:512]
        results = classifier(text)

        if results and len(results) > 0:
            scores = results[0]  # list of {label, score} dicts
            # Find the label with highest confidence
            best = max(scores, key=lambda x: x["score"])
            all_scores = {s["label"].capitalize(): round(s["score"], 4) for s in scores}

            return {
                "label": best["label"].capitalize(),
                "score": round(best["score"], 4),
                "all_scores": all_scores,
            }
    except Exception:
        pass

    return None


def analyze_headlines(headlines):
    """
    Analyze sentiment of multiple headlines.

    Parameters
    ----------
    headlines : list[dict]
        List of headline dicts with 'title' key.

    Returns
    -------
    list[dict]
        Same headlines with added 'sentiment' key containing
        sentiment analysis results.
    """
    analyzed = []
    for headline in headlines:
        h = headline.copy()
        sentiment = analyze_sentiment(h["title"])
        h["sentiment"] = sentiment
        analyzed.append(h)
    return analyzed


def compute_stock_sentiment(stock_headlines):
    """
    Compute aggregate sentiment score for a stock.

    Parameters
    ----------
    stock_headlines : list[dict]
        List of headline dicts with 'sentiment' key
        (output of analyze_headlines).

    Returns
    -------
    dict
        {
            "overall_label": str,
            "overall_score": float (-1 to 1),
            "positive_count": int,
            "negative_count": int,
            "neutral_count": int,
            "risk_flag": bool (True if 2+ negative signals)
        }
    """
    if not stock_headlines:
        return {
            "overall_label": "Neutral",
            "overall_score": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "risk_flag": False,
        }

    pos_count = 0
    neg_count = 0
    neu_count = 0
    score_sum = 0.0
    valid_count = 0

    for h in stock_headlines:
        s = h.get("sentiment")
        if s is None:
            continue

        valid_count += 1
        label = s["label"].lower()
        confidence = s["score"]

        if label == "positive":
            pos_count += 1
            score_sum += confidence
        elif label == "negative":
            neg_count += 1
            score_sum -= confidence
        else:
            neu_count += 1

    avg_score = score_sum / valid_count if valid_count > 0 else 0.0

    if avg_score > 0.1:
        overall_label = "Positive"
    elif avg_score < -0.1:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    return {
        "overall_label": overall_label,
        "overall_score": round(avg_score, 4),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "risk_flag": neg_count >= 2,
    }


def get_rebalancing_suggestions(stock_sentiments, tickers, weights):
    """
    Generate rebalancing suggestions based on sentiment analysis.

    If a stock has 3+ negative headlines, suggest reducing its weight.
    Purely advisory — never auto-trades.

    Parameters
    ----------
    stock_sentiments : dict[str, dict]
        Ticker → sentiment summary (output of compute_stock_sentiment).
    tickers : list[str]
    weights : array-like

    Returns
    -------
    list[dict]
        List of suggestion dicts with keys:
        - ticker: str
        - current_weight: float
        - reason: str
        - severity: 'warning' | 'danger'
    """
    suggestions = []

    for ticker, w in zip(tickers, weights):
        sentiment = stock_sentiments.get(ticker)
        if sentiment is None:
            continue

        neg_count = sentiment["negative_count"]

        if neg_count >= 3:
            suggestions.append({
                "ticker": ticker,
                "current_weight": round(w * 100, 2),
                "reason": f"{neg_count} negative news signals detected. Consider reducing exposure.",
                "severity": "danger",
            })
        elif neg_count >= 2:
            suggestions.append({
                "ticker": ticker,
                "current_weight": round(w * 100, 2),
                "reason": f"{neg_count} negative news signals detected. Monitor closely.",
                "severity": "warning",
            })

    return suggestions
