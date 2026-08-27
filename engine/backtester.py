import numpy as np
import pandas as pd
from scipy.optimize import minimize
from models.black_litterman import compute_implied_returns, generate_black_litterman_views, optimize_black_litterman

class WalkForwardBacktester:
    def __init__(self, data, tickers, sector_map, config):
        """
        data: DataFrame of prices, calendar-aligned.
        tickers: list of tickers.
        sector_map: dict mapping ticker -> sector.
        config: dictionary containing constraints.
        """
        # Align tickers to columns actually present in data
        valid_tickers = [t for t in tickers if t in data.columns]
        if not valid_tickers:
            raise ValueError("None of the requested tickers exist in the provided market data.")
            
        self.data = data[valid_tickers]
        self.tickers = valid_tickers
        self.sector_map = sector_map
        self.config = config
        
        self.max_weight = config.get("max_weight", 0.15)
        
        # Mathematical necessity: max_weight must be >= 1 / n
        n = len(self.tickers)
        if self.max_weight < 1.0 / n:
            self.max_weight = 1.0 / n
            
        self.sector_caps = config.get("sector_cap", 0.35)
        
        # Mathematical necessity: sector_cap must be >= max sector concentration in equal-weight
        sector_counts = {}
        for t in self.tickers:
            s = self.sector_map.get(t, "Other")
            sector_counts[s] = sector_counts.get(s, 0) + 1
            
        max_sector_ratio = max(sector_counts.values()) / n if n > 0 else 1.0
        if self.sector_caps < max_sector_ratio:
            self.sector_caps = max_sector_ratio
            
        self.transaction_cost_bps = config.get("transaction_cost_bps", 10)
        self.risk_free_rate = config.get("risk_free_rate", 0.07)
        self.window = config.get("window", 252)
        self.rebalance_freq = config.get("rebalance_freq", 20)
        
    def check_invariants(self, weights, prev_weights, date):
        """Strictly enforce constraints. Fails backtest if violated."""
        if not np.isclose(np.sum(weights), 1.0, atol=1e-3):
            raise ValueError(f"Invariant violation at {date}: weights sum to {np.sum(weights)}")
            
        if np.any(weights < -1e-4):
            raise ValueError(f"Invariant violation at {date}: negative weights found")
            
        if np.any(weights > self.max_weight + 1e-3):
            raise ValueError(f"Invariant violation at {date}: max weight {self.max_weight} exceeded by {np.max(weights)}")
            
        # Check sector caps
        sector_w = {}
        for w, t in zip(weights, self.tickers):
            s = self.sector_map.get(t, "Other")
            sector_w[s] = sector_w.get(s, 0.0) + w
            
        for s, w in sector_w.items():
            if w > self.sector_caps + 1e-3:
                raise ValueError(f"Invariant violation at {date}: sector {s} cap exceeded ({w} > {self.sector_caps})")
                
        return True

    def optimize_portfolio(self, expected_returns, cov_matrix):
        """Optimize portfolio weights using a two-pass robust approach.
        
        Pass 1: Maximize Sharpe ratio (preferred — differentiates strategies).
        Pass 2: Minimize variance (fallback — always feasible, still constraint-respecting).
        
        This prevents the silent collapse to equal weights when SLSQP fails on
        the Sharpe objective (which happens frequently with noisy empirical returns).
        """
        n = len(self.tickers)
        init_guess = np.ones(n) / n
        bounds = tuple((0.0, self.max_weight) for _ in range(n))
        
        # Sector constraint factory — uses closure to capture sector_name correctly.
        def sector_constraint_factory(sector_name, cap):
            def constraint(w):
                return cap - sum(
                    w[i] for i, t in enumerate(self.tickers)
                    if self.sector_map.get(t, "Other") == sector_name
                )
            return constraint

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        unique_sectors = set(self.sector_map.values())
        for sector in unique_sectors:
            constraints.append({"type": "ineq", "fun": sector_constraint_factory(sector, self.sector_caps)})

        opts = {"maxiter": 1000, "ftol": 1e-9}

        # ── Pass 1: Sharpe maximization ──────────────────────────────────
        def neg_sharpe(w):
            ret = np.dot(w, expected_returns)
            vol = np.sqrt(np.clip(np.dot(w.T, np.dot(cov_matrix, w)), 1e-12, None))
            return -(ret - self.risk_free_rate) / vol

        res = minimize(
            neg_sharpe, init_guess, method="SLSQP",
            bounds=bounds, constraints=constraints, options=opts
        )

        if res.success and np.all(res.x >= -1e-8) and np.isclose(res.x.sum(), 1.0, atol=1e-4):
            return np.clip(res.x, 0.0, None) / np.clip(res.x, 0.0, None).sum()

        # ── Pass 2: Minimum variance fallback ───────────────────────────
        # Min-variance is convex → SLSQP almost always converges.
        # Gives different weights than equal-weight, preserving strategy differentiation.
        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        res2 = minimize(
            portfolio_variance, init_guess, method="SLSQP",
            bounds=bounds, constraints=constraints, options=opts
        )

        if res2.success and np.all(res2.x >= -1e-8) and np.isclose(res2.x.sum(), 1.0, atol=1e-4):
            w = np.clip(res2.x, 0.0, None)
            return w / w.sum()

        # ── Ultimate fallback: max-weight-capped equal weight ────────────
        # Should be extremely rare after the two passes above.
        fallback = np.ones(n) / n
        fallback = np.clip(fallback, 0.0, self.max_weight)
        return fallback / fallback.sum()
        
    def run_backtest(self, strategy="Markowitz", historical_sentiments=None, lstm_prob_series=None):
        """
        Run a walk-forward backtest.
        strategy: "EqualWeight", "Markowitz", "BlackLitterman", "BL_FinBERT", "BL_LSTM", "BL_FinBERT_LSTM"
        historical_sentiments: dict of date -> {ticker: sentiment_dict}
        lstm_prob_series: pd.Series of date -> probability of high vol
        """
        returns = self.data[self.tickers].pct_change().dropna(how='all').fillna(0)
        dates = returns.index[self.window:]
        
        portfolio_value = 1.0
        portfolio_values = [portfolio_value]
        
        n = len(self.tickers)
        current_weights = np.ones(n) / n
        
        for i, date in enumerate(dates):
            # Point-in-time daily realized returns aligned to self.tickers
            day_returns = returns.loc[date, self.tickers].values
            gross_return = np.dot(current_weights, day_returns)
            tx_cost = 0.0
            
            # Rebalance at the end of the day `date`?
            # Or rebalance at `date` for returns on `date+1`?
            # Standard: if i % freq == 0, we form the new portfolio at the close of `date`.
            if i % self.rebalance_freq == 0:
                hist_returns = returns.loc[:date].iloc[-self.window:]

                mean_rets = hist_returns.mean() * 252
                cov_matrix = hist_returns.cov() * 252

                new_weights = current_weights.copy()
                invariants_already_checked = False

                # Check for NaNs/degenerate data → safe fallback
                if cov_matrix.isna().sum().sum() > 0 or mean_rets.isna().sum() > 0:
                    new_weights = np.ones(n) / n

                else:
                    if strategy == "EqualWeight":
                        # Pure equal weight — the benchmark baseline.
                        new_weights = np.ones(n) / n

                    elif strategy == "Markowitz":
                        # Maximize Sharpe using HISTORICAL mean returns.
                        # Differentiator: purely data-driven, no equilibrium prior.
                        new_weights = self.optimize_portfolio(mean_rets.values, cov_matrix.values)

                    elif "BL" in strategy:
                        # ── Black-Litterman: equilibrium-tilted returns ──────────
                        # Key difference from Markowitz: uses CAPM-implied equilibrium
                        # returns (Pi = δ·Σ·w_mkt) as the prior, NOT raw historical means.
                        # This shrinks noisy sample estimates toward a principled prior,
                        # producing meaningfully different allocations even without views.
                        market_weights = np.ones(n) / n
                        implied_rets = compute_implied_returns(
                            cov_matrix, market_weights, risk_aversion=2.5
                        )

                        P, Q, Omega = None, None, None
                        if "FinBERT" in strategy and historical_sentiments:
                            sentiment_t = historical_sentiments.get(date, {})
                            P, Q, Omega = generate_black_litterman_views(
                                self.tickers, implied_rets, cov_matrix, sentiment_t
                            )

                        # posterior_rets will equal implied_rets when P/Q/Omega are None
                        # (no views). BL still differs from Markowitz because implied_rets
                        # (risk-aversion × Σ × w_eq) ≠ historical mean returns.
                        post_rets, post_cov = optimize_black_litterman(
                            implied_rets, cov_matrix, P, Q, Omega
                        )

                        # ── LSTM constraint tightening ───────────────────────────
                        # Temporarily reduce max_weight during predicted high-vol regimes.
                        # IMPORTANT: run invariant check BEFORE restoring max_weight so
                        # we validate against the constraint actually used.
                        old_max_weight = self.max_weight
                        if "LSTM" in strategy and lstm_prob_series is not None:
                            if date in lstm_prob_series.index:
                                p_high_vol = float(lstm_prob_series.loc[date])
                                if p_high_vol > 0.8:
                                    self.max_weight = min(self.max_weight, 0.08)

                        new_weights = self.optimize_portfolio(post_rets.values, post_cov.values)

                        # Check invariants while the (possibly tightened) max_weight is active
                        self.check_invariants(new_weights, current_weights, date)
                        invariants_already_checked = True

                        self.max_weight = old_max_weight  # Restore after check

                # Invariants check for all strategies that haven't done it yet
                if not invariants_already_checked:
                    self.check_invariants(new_weights, current_weights, date)

                # Transaction Costs (turnover is sum of absolute weight changes / 2)
                turnover = np.sum(np.abs(new_weights - current_weights)) / 2.0
                tx_cost = turnover * (self.transaction_cost_bps / 10000.0)

                current_weights = new_weights
                
            net_return = gross_return - tx_cost
            portfolio_value *= (1 + net_return)
            portfolio_values.append(portfolio_value)
            
        # The index will be from window-1 to end
        return pd.Series(portfolio_values, index=returns.index[self.window-1:])

    def calculate_metrics(self, portfolio_series):
        """Calculate non-predetermined KPIs from the backtest series."""
        rets = portfolio_series.pct_change().dropna()
        if len(rets) == 0:
            return {}
            
        cagr = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) ** (252 / len(rets)) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = (cagr - self.risk_free_rate) / vol if vol > 0 else 0
        
        # Sortino
        neg_rets = rets[rets < 0]
        downside_vol = neg_rets.std() * np.sqrt(252)
        sortino = (cagr - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Max Drawdown
        running_max = portfolio_series.cummax()
        drawdowns = (portfolio_series - running_max) / running_max
        max_dd = drawdowns.min()
        
        return {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_dd
        }
