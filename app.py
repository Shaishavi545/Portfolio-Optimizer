"""
📊 Indian Portfolio Optimizer — Markowitz + AI News Intelligence
Built with Streamlit, yfinance, Plotly, and FinBERT.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import time

from data.nifty50 import (
    NIFTY50, GOLD_ETFS, BENCHMARK_INDEX, DEFAULT_STOCKS,
    get_all_tickers, get_sector_weights,
)

# ───────────────────────────── Page Config ─────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────── Custom CSS ──────────────────────────────
st.markdown("""
<style>
    /* Galaxy Palette */
    .stApp { background: linear-gradient(135deg, #1d161e, #2b1d5a, #282b75); color: #ffffff; }
    [data-testid="stSidebar"] { min-width: 400px !important; max-width: 400px !important; }
    .metric-card {
        background: rgba(162, 150, 202, 0.1);
        border: 1px solid #4c1c46;
        border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .sentiment-pos { color: #a296ca; font-weight: 700; }
    .sentiment-neg { color: #ff7b89; font-weight: 700; }
    .sentiment-neu { color: #ffffff; font-weight: 700; }
    .macro-banner {
        background: #4c1c46; color: #ffffff; padding: 12px 20px; border-radius: 8px;
        font-weight: 600; margin-bottom: 16px; border: 1px solid #a296ca;
    }
    .macro-banner-medium {
        background: #282b75; color: #ffffff; padding: 12px 20px; border-radius: 8px;
        font-weight: 600; margin-bottom: 16px; border: 1px solid #2b1d5a;
    }
    /* Increase tab size for easier navigation */
    button[data-baseweb="tab"] {
        font-size: 1.15rem !important;
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════ DATA LOADING ══════════════════════════════
@st.cache_data(show_spinner="Fetching stock data from Yahoo Finance...")
def load_data(tickers, start_date_str):
    """Fetch adjusted close prices for given tickers from yfinance."""
    data = pd.DataFrame()
    failed = []

    for ticker in tickers:
        try:
            temp = yf.download(ticker, start=start_date_str, progress=False)
            if temp.empty:
                failed.append(ticker)
                continue
            close = temp["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            data[ticker] = close
            time.sleep(0.5)  # Anti rate-limit sleep
        except Exception:
            failed.append(ticker)
    return data.dropna(), failed


@st.cache_data(show_spinner="Fetching Nifty 50 benchmark...")
def load_benchmark(start_date_str):
    """Fetch Nifty 50 index data for benchmarking."""
    try:
        bench = yf.download(BENCHMARK_INDEX, start=start_date_str, progress=False)
        if bench.empty:
            return None
        close = bench["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close
    except Exception:
        return None


# ═══════════════════════════ COMPUTE METRICS ═══════════════════════════
def compute_metrics(data, risk_free_rate):
    """Compute annualized returns, covariance, and correlation."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    return returns, mean_returns, cov_matrix, corr_matrix


# ═══════════════════════════ OPTIMIZATION ══════════════════════════════
def optimize(mean_returns, cov_matrix, risk_free_rate):
    """Run Markowitz optimization to find max-Sharpe portfolio."""
    num_assets = len(mean_returns)
    init_guess = [1.0 / num_assets] * num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sharpe(w):
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(ret - risk_free_rate) / vol if vol > 0 else 0

    result = minimize(neg_sharpe, init_guess, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
    """Compute return, volatility, Sharpe for a given weight vector."""
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
    return ret, vol, sharpe


def monte_carlo_portfolios(mean_returns, cov_matrix, risk_free_rate, n=10000):
    """Generate n random portfolios for the efficient frontier."""
    num_assets = len(mean_returns)
    results = np.zeros((n, 3))  # return, vol, sharpe
    all_weights = np.zeros((n, num_assets))

    for i in range(n):
        w = np.random.random(num_assets)
        w /= w.sum()
        all_weights[i] = w
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix, risk_free_rate)
        results[i] = [ret, vol, sharpe]

    return results, all_weights


# ═══════════════════════════ RENDER UI ═════════════════════════════════
def render_ui():
    """Main UI rendering function."""

    # ──── Header ────
    st.markdown("# Indian Portfolio Optimizer")
    st.markdown("---")

    # ──── Sidebar ────
    st.sidebar.markdown("## Configuration")

    all_tickers = get_all_tickers()
    selected_stocks = st.sidebar.multiselect(
        "Select Nifty 50 Stocks (2–15)",
        options=all_tickers,
        default=DEFAULT_STOCKS,
        help="Pick 2 to 15 stocks from the Nifty 50 universe.",
    )

    include_gold = st.sidebar.checkbox("Include Gold ETF (GOLDBEES.NS)", value=False)

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (annualized)", 0.0, 0.15, 0.07, 0.01,
        help="Typically the Indian 10-year govt bond yield (~7%)."
    )

    # Build ticker list
    tickers = list(selected_stocks)
    if include_gold:
        tickers += list(GOLD_ETFS.keys())

    # Validation
    if len(tickers) < 2:
        st.error("Please select at least 2 stocks to optimize a portfolio.")
        return
    if len(tickers) > 15:
        st.warning("Maximum 15 assets supported. Only the first 15 will be used.")
        tickers = tickers[:15]

    # ──── Load Data ────
    start_str = start_date.strftime("%Y-%m-%d")
    data, failed_tickers = load_data(tuple(tickers), start_str)

    if failed_tickers:
        st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")
    if data.empty or len(data.columns) < 2:
        st.error("Not enough valid data to optimize. Please adjust your stock selection or date range.")
        return

    active_tickers = list(data.columns)

    # ──── Compute ────
    returns, mean_returns, cov_matrix, corr_matrix = compute_metrics(data, risk_free_rate)
    optimal_weights = optimize(mean_returns, cov_matrix, risk_free_rate)
    opt_ret, opt_vol, opt_sharpe = portfolio_stats(
        optimal_weights, mean_returns, cov_matrix, risk_free_rate
    )

    # ──── Metrics Row ────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Return", f"{opt_ret:.2%}")
    with col2:
        st.metric("Volatility", f"{opt_vol:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{opt_sharpe:.3f}")

    # ──── Tabs ────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Allocation", "Efficient Frontier", "Analysis", "AI News"
    ])

    # ═══════════ TAB 1: Allocation ═══════════
    with tab1:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Optimized Asset Allocation")
            st.caption("These weights are mathematically calculated by the Markowitz Mean-Variance algorithm to maximize your expected return for the given level of risk.")
            weight_df = pd.DataFrame({
                "Stock": active_tickers,
                "Weight (%)": [round(w * 100, 2) for w in optimal_weights],
            }).sort_values("Weight (%)", ascending=False)
            weight_df = weight_df[weight_df["Weight (%)"] > 0]
            st.dataframe(weight_df, width="stretch", hide_index=True)

        with c2:
            st.subheader("Allocation Breakdown")
            fig_alloc = px.pie(
                weight_df[weight_df["Weight (%)"] > 0.1],
                values="Weight (%)", names="Stock",
                color_discrete_sequence=["#a296ca", "#282b75", "#4c1c46", "#2b1d5a", "#ffffff"],
                hole=0.4,
            )
            fig_alloc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
            )
            st.plotly_chart(fig_alloc, width="stretch")

        # ─── Sector Diversity ───
        st.subheader("Sector Diversity")
        sector_w = get_sector_weights(active_tickers, optimal_weights)
        sector_df = pd.DataFrame([
            {"Sector": s, "Weight (%)": round(w * 100, 2)}
            for s, w in sorted(sector_w.items(), key=lambda x: -x[1])
        ])

        sc1, sc2 = st.columns([1, 1])
        with sc1:
            fig_sector = px.pie(
                sector_df, values="Weight (%)", names="Sector",
                color_discrete_sequence=["#a296ca", "#282b75", "#4c1c46", "#2b1d5a", "#ffffff"],
                hole=0.35,
            )
            fig_sector.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
            )
            st.plotly_chart(fig_sector, width="stretch")

        with sc2:
            st.dataframe(sector_df, width="stretch", hide_index=True)
            # Concentration warning
            for sector, weight in sector_w.items():
                if weight > 0.40:
                    st.warning(
                        f"⚠️ **Concentration Risk:** {sector} has {weight:.1%} weight "
                        f"(>40%). Consider diversifying across sectors."
                    )

    # ═══════════ TAB 2: Efficient Frontier ═══════════
    with tab2:
        st.subheader("Efficient Frontier — 10,000 Random Portfolios")
        st.markdown(
            "The Efficient Frontier represents a set of optimal portfolios that offer the highest "
            "expected return for a defined level of risk. Each dot is a randomly simulated portfolio, "
            "and the optimal portfolio (marked with a star) provides the best risk-adjusted return."
        )

        with st.spinner("Simulating 10,000 portfolios..."):
            mc_results, mc_weights = monte_carlo_portfolios(
                mean_returns, cov_matrix, risk_free_rate, n=10000
            )

        fig_ef = go.Figure()

        # Scatter — random portfolios
        fig_ef.add_trace(go.Scatter(
            x=mc_results[:, 1], y=mc_results[:, 0],
            mode="markers",
            marker=dict(
                size=3, color=mc_results[:, 2],
                colorscale="Sunset", colorbar=dict(title="Sharpe"),
                opacity=0.6,
            ),
            text=[f"Sharpe: {s:.3f}" for s in mc_results[:, 2]],
            hovertemplate="Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>%{text}",
            name="Random Portfolios",
        ))

        # Star — optimal portfolio
        fig_ef.add_trace(go.Scatter(
            x=[opt_vol], y=[opt_ret],
            mode="markers+text",
            marker=dict(size=18, color="#FFD700", symbol="star"),
            text=["Optimal"],
            textposition="top center",
            textfont=dict(color="#FF8C00", size=14),
            name=f"Optimal (Sharpe={opt_sharpe:.3f})",
        ))

        fig_ef.update_layout(
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            xaxis_tickformat=".0%", yaxis_tickformat=".0%",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=True,
            height=550,
        )
        st.plotly_chart(fig_ef, width="stretch")

    # ═══════════ TAB 3: Analysis ═══════════
    with tab3:
        a1, a2 = st.columns([1, 1])

        with a1:
            st.subheader("Correlation Heatmap")
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f",
                color_continuous_scale="Purpor",
                aspect="auto",
            )
            fig_corr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
                height=450,
            )
            st.plotly_chart(fig_corr, width="stretch")

        with a2:
            st.subheader("Cumulative Returns vs Nifty 50")
            # Portfolio cumulative returns
            port_returns = returns.dot(optimal_weights)
            cum_port = (1 + port_returns).cumprod()

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port.values,
                mode="lines", name="Optimized Portfolio",
                line=dict(color="#00e676", width=2),
            ))

            # Benchmark
            bench_data = load_benchmark(start_str)
            if bench_data is not None:
                bench_returns = bench_data.pct_change().dropna()
                cum_bench = (1 + bench_returns).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=cum_bench.index, y=cum_bench.values,
                    mode="lines", name="Nifty 50 Index",
                    line=dict(color="#ff9100", width=2, dash="dash"),
                ))

            fig_cum.update_layout(
                xaxis_title="Date", yaxis_title="Growth of ₹1",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
                height=450,
                legend=dict(x=0.02, y=0.98),
            )
            st.plotly_chart(fig_cum, width="stretch")

        # Individual stock performance
        st.subheader("Individual Stock Returns")
        fig_stocks = go.Figure()
        for col in data.columns:
            norm = data[col] / data[col].iloc[0]
            fig_stocks.add_trace(go.Scatter(
                x=data.index, y=norm, mode="lines", name=col,
            ))
        fig_stocks.update_layout(
            xaxis_title="Date", yaxis_title="Normalized Price",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            height=400,
        )
        st.plotly_chart(fig_stocks, width="stretch")

    # ═══════════ TAB 4: AI News Intelligence ═══════════
    with tab4:
        render_news_tab(active_tickers, optimal_weights)

    # ──── Footer ────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.6; font-size:0.85rem;'>"
        "<b>Disclaimer:</b> This tool is for educational purposes only. "
        "It is NOT financial advice. Past performance does not guarantee future results. "
        "Always consult a qualified financial advisor before making investment decisions."
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════ NEWS TAB ══════════════════════════════════
def render_news_tab(tickers, weights):
    """Render the AI News Intelligence tab."""
    from news.fetcher import fetch_headlines, filter_headlines_for_stocks, get_general_market_headlines
    from news.sentiment import analyze_headlines, compute_stock_sentiment, get_rebalancing_suggestions
    from news.macro import detect_macro_events, get_alert_level

    st.subheader("AI News Intelligence")
    st.markdown("*Live financial news with FinBERT sentiment analysis*")

    if st.button("Fetch Latest News", type="primary"):
        st.session_state["news_fetched"] = True

    if not st.session_state.get("news_fetched", False):
        st.info("Click the button above to fetch live financial news and run AI sentiment analysis.")
        return

    with st.spinner("Fetching headlines from RSS feeds..."):
        headlines = fetch_headlines(max_per_feed=8)

    if not headlines:
        st.warning("Could not fetch any news headlines. Check your internet connection.")
        return

    st.success(f"Fetched {len(headlines)} headlines from financial news feeds.")

    # ─── Macro Event Alerts ───
    macro_events = detect_macro_events(headlines)
    alert_level, alert_msg = get_alert_level(macro_events)

    if alert_level == "high":
        st.markdown(f'<div class="macro-banner">{alert_msg}</div>', unsafe_allow_html=True)
        with st.expander("View Macro Event Details", expanded=True):
            for evt in macro_events:
                st.markdown(
                    f"**{evt['category']}** — [{evt['headline']['title']}]({evt['headline']['link']})  \n"
                    f"*Source: {evt['headline']['source']}* | "
                    f"Keyword: `{evt['matched_keyword']}`"
                )
    elif alert_level == "medium":
        st.markdown(f'<div class="macro-banner-medium">{alert_msg}</div>', unsafe_allow_html=True)

    # ─── Stock-Level Sentiment ───
    st.markdown("### Stock-Level Sentiment Analysis")

    with st.spinner("Running FinBERT sentiment analysis..."):
        stock_headlines = filter_headlines_for_stocks(headlines, tickers)
        stock_sentiments = {}

        for ticker in tickers:
            if stock_headlines[ticker]:
                analyzed = analyze_headlines(stock_headlines[ticker])
                stock_headlines[ticker] = analyzed
                stock_sentiments[ticker] = compute_stock_sentiment(analyzed)
            else:
                stock_sentiments[ticker] = compute_stock_sentiment([])

    # Render sentiment cards — 3 columns
    cols = st.columns(3)
    for i, ticker in enumerate(tickers):
        sentiment = stock_sentiments[ticker]
        with cols[i % 3]:
            label = sentiment["overall_label"]
            css_class = {"Positive": "sentiment-pos", "Negative": "sentiment-neg"}.get(label, "sentiment-neu")
            flag = " (Risk Flag)" if sentiment["risk_flag"] else ""

            st.markdown(f"""
<div class="metric-card">
    <h4>{ticker}{flag}</h4>
    <p>Sentiment: <span class="{css_class}">{label}</span> ({sentiment['overall_score']:+.2f})</p>
    <p style="font-size:0.85rem; opacity:0.7;">
        Pos: {sentiment['positive_count']} · Neg: {sentiment['negative_count']} · Neu: {sentiment['neutral_count']}
    </p>
</div>
""", unsafe_allow_html=True)

            # Top headlines for this stock
            for h in stock_headlines[ticker][:3]:
                s = h.get("sentiment")
                badge = ""
                if s:
                    colors = {"Positive": "(Pos)", "Negative": "(Neg)", "Neutral": "(Neu)"}
                    badge = colors.get(s["label"], "")
                st.markdown(f"{badge} [{h['title']}]({h['link']})")

    # ─── Rebalancing Suggestions ───
    suggestions = get_rebalancing_suggestions(stock_sentiments, tickers, weights)
    if suggestions:
        st.markdown("### AI Rebalancing Suggestions")
        st.caption("*Purely advisory — based on news sentiment, not a trading recommendation.*")
        for sug in suggestions:
            st.warning(f"**{sug['ticker']}** (current weight: {sug['current_weight']}%) — {sug['reason']}")

    # ─── General Market News ───
    with st.expander("General Market Headlines"):
        general = get_general_market_headlines(headlines, limit=10)
        if general:
            for h in general:
                st.markdown(f"**{h['title']}**  \n*{h['source']}*  \n[Read more]({h['link']})")
                st.markdown("---")
        else:
            st.info("No general market headlines found.")


# ═══════════════════════════ MAIN ══════════════════════════════════════
if __name__ == "__main__":
    render_ui()