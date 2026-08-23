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
        self.data = data
        self.tickers = tickers
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
        """Optimize using constraints."""
        n = len(self.tickers)
        init_guess = np.ones(n) / n
        bounds = tuple((0.0, self.max_weight) for _ in range(n))
        
        # Constraints: sum to 1, and sector limits
        def sector_constraint_factory(sector_name, cap):
            def constraint(w):
                sector_weight = sum(w[i] for i, t in enumerate(self.tickers) if self.sector_map.get(t, "Other") == sector_name)
                return cap - sector_weight
            return constraint

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        
        unique_sectors = set(self.sector_map.values())
        for sector in unique_sectors:
            constraints.append({"type": "ineq", "fun": sector_constraint_factory(sector, self.sector_caps)})

        def neg_sharpe(w):
            ret = np.dot(w, expected_returns)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if vol == 0: return 0.0
            return -(ret - self.risk_free_rate) / vol

        res = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if not res.success:
            return np.ones(n) / n
            
        return res.x
        
    def run_backtest(self, strategy="Markowitz", historical_sentiments=None, lstm_prob_series=None):
        """
        Run a walk-forward backtest.
        strategy: "EqualWeight", "Markowitz", "BlackLitterman", "BL_FinBERT", "BL_LSTM", "BL_FinBERT_LSTM"
        historical_sentiments: dict of date -> {ticker: sentiment_dict}
        lstm_prob_series: pd.Series of date -> probability of high vol
        """
        returns = self.data.pct_change().dropna(how='all').fillna(0)
        dates = returns.index[self.window:]
        
        portfolio_value = 1.0
        portfolio_values = [portfolio_value]
        
        n = len(self.tickers)
        current_weights = np.ones(n) / n
        
        for i, date in enumerate(dates):
            # The realized return is computed from the close of `date-1` to the close of `date`.
            # Our decision was made at `date-1` close, so it's strictly point-in-time.
            day_returns = returns.loc[date].values
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
                
                # Check for NaNs
                if cov_matrix.isna().sum().sum() > 0 or mean_rets.isna().sum() > 0:
                    new_weights = np.ones(n) / n
                else:
                    if strategy == "EqualWeight":
                        new_weights = np.ones(n) / n
                    elif strategy == "Markowitz":
                        new_weights = self.optimize_portfolio(mean_rets.values, cov_matrix.values)
                    elif "BL" in strategy:
                        # Assume market cap weights are roughly equal for simplicity if we don't have true caps
                        market_weights = np.ones(n) / n 
                        implied_rets = compute_implied_returns(cov_matrix, market_weights, risk_aversion=2.5)
                        
                        P, Q, Omega = None, None, None
                        
                        if "FinBERT" in strategy and historical_sentiments:
                            # We can only use sentiments published BEFORE or ON `date`
                            # We assume the dict already filters out future data.
                            sentiment_t = historical_sentiments.get(date, {})
                            P, Q, Omega = generate_black_litterman_views(self.tickers, implied_rets, cov_matrix, sentiment_t)
                            
                        post_rets, post_cov = optimize_black_litterman(implied_rets, cov_matrix, P, Q, Omega)
                        
                        # Dynamically tighten constraints if LSTM detects high risk
                        old_max_weight = self.max_weight
                        if "LSTM" in strategy and lstm_prob_series is not None:
                            if date in lstm_prob_series:
                                p_high_vol = lstm_prob_series.loc[date]
                                if p_high_vol > 0.8:
                                    # Defensive mode: force more diversification
                                    self.max_weight = min(self.max_weight, 0.08)
                                    
                        new_weights = self.optimize_portfolio(post_rets.values, post_cov.values)
                        self.max_weight = old_max_weight # Restore
                        
                # Invariants check
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
