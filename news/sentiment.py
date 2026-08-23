"""
Sentiment analysis engine using FinBERT (ProsusAI/finbert).

Replaces the previous NLTK VADER implementation with true financial
NLP, producing sentiment and confidence probabilities to be consumed
mathematically by the Black-Litterman optimizer.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner="Loading FinBERT model...")
def _load_model():
    """Load the FinBERT model from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def analyze_sentiment(text):
    """
    Analyze sentiment of a single text string using FinBERT.

    Returns
    -------
    dict
        {
            "label": "Positive" | "Negative" | "Neutral",
            "score": float (confidence of the predicted label, 0-1),
            "all_scores": {
                "Positive": float,
                "Negative": float,
                "Neutral": float
            }
        }
    """
    try:
        tokenizer, model = _load_model()
    except Exception:
        return None

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        
        # ProsusAI/finbert labels: 0 -> positive, 1 -> negative, 2 -> neutral
        scores = {
            "Positive": probs[0],
            "Negative": probs[1],
            "Neutral": probs[2]
        }
        
        # Get label with highest probability
        max_label = max(scores, key=scores.get)
        confidence = scores[max_label]

        return {
            "label": max_label,
            "score": round(confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in scores.items()}
        }
    except Exception:
        return None

def analyze_headlines(headlines):
    """Analyze sentiment of multiple headlines."""
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
    
    Averages the probabilities of positive, negative, and neutral,
    and returns the net sentiment and confidence to be used by Black-Litterman.
    """
    if not stock_headlines:
        return {
            "overall_label": "Neutral",
            "overall_score": 0.0,
            "confidence": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "risk_flag": False,
        }

    pos_prob_sum = 0.0
    neg_prob_sum = 0.0
    neu_prob_sum = 0.0
    valid_count = 0

    pos_count = 0
    neg_count = 0
    neu_count = 0

    for h in stock_headlines:
        s = h.get("sentiment")
        if s is None:
            continue

        valid_count += 1
        pos_prob_sum += s["all_scores"]["Positive"]
        neg_prob_sum += s["all_scores"]["Negative"]
        neu_prob_sum += s["all_scores"]["Neutral"]

        label = s["label"]
        if label == "Positive": pos_count += 1
        elif label == "Negative": neg_count += 1
        else: neu_count += 1

    if valid_count == 0:
        return {
            "overall_label": "Neutral",
            "overall_score": 0.0,
            "confidence": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "risk_flag": False,
        }

    avg_pos = pos_prob_sum / valid_count
    avg_neg = neg_prob_sum / valid_count
    avg_neu = neu_prob_sum / valid_count
    
    # Net sentiment score: range roughly [-1, 1]
    # If 100% positive -> +1
    # If 100% negative -> -1
    net_sentiment = avg_pos - avg_neg
    
    # Confidence: how strong is the dominant view over neutral?
    # Max is 1.0 (if pos or neg is 1.0). Min is 0.0 (if neu is 1.0).
    confidence = 1.0 - avg_neu

    if net_sentiment > 0.1:
        overall_label = "Positive"
    elif net_sentiment < -0.1:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    return {
        "overall_label": overall_label,
        "overall_score": round(net_sentiment, 4),
        "confidence": round(confidence, 4),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "risk_flag": neg_count >= 2,
    }

def get_rebalancing_suggestions(stock_sentiments, tickers, weights):
    """
    Generate rebalancing suggestions based on sentiment analysis.
    Purely advisory — never auto-trades.
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
