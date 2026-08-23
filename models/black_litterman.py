import numpy as np
import pandas as pd

def compute_implied_returns(cov_matrix, market_weights, risk_aversion=2.5):
    """
    Compute market-implied equilibrium returns (Pi).
    Pi = delta * Sigma * w_mkt
    """
    pi = risk_aversion * cov_matrix.dot(market_weights)
    return pi

def generate_black_litterman_views(tickers, implied_returns, cov_matrix, stock_sentiments, tau=0.05):
    """
    Deterministic mapping from FinBERT sentiment/confidence to BL Q and Omega.
    
    - Strong positive sentiment (S) -> Positive return view (Q).
    - Higher confidence (C) -> Lower uncertainty (Omega) -> BL trusts the view more.
    """
    n_assets = len(tickers)
    view_indices = []
    Q_list = []
    omega_diag = []
    
    for i, ticker in enumerate(tickers):
        sentiment = stock_sentiments.get(ticker)
        if sentiment is None or sentiment.get("confidence", 0) == 0:
            continue
            
        view_indices.append(i)
        
        S = sentiment["overall_score"] # [-1, 1]
        C = sentiment["confidence"] # [0, 1]
        
        # Q: Absolute view on return
        # Baseline is implied return. We shift it based on sentiment.
        # A full +1 sentiment equates to a +5% annualized return boost over equilibrium.
        view_return = implied_returns.iloc[i] + (S * 0.05)
        Q_list.append(view_return)
        
        # Omega: Uncertainty of the view
        # Baseline uncertainty is tau * variance
        base_variance = tau * cov_matrix.iloc[i, i]
        
        # Confidence scales uncertainty inversely.
        # If C is high (e.g., 0.9), uncertainty is lower.
        # If C is low (e.g., 0.1), uncertainty is higher.
        # We add 0.1 to avoid division by zero and cap extreme trust.
        omega = base_variance / (C + 0.1)
        omega_diag.append(omega)
        
    K = len(view_indices)
    
    if K == 0:
        return None, None, None
        
    P = np.zeros((K, n_assets))
    for k, idx in enumerate(view_indices):
        P[k, idx] = 1.0
        
    Q = np.array(Q_list)
    Omega = np.diag(omega_diag)
    
    return P, Q, Omega

def optimize_black_litterman(implied_returns, cov_matrix, P, Q, Omega, tau=0.05):
    """
    Compute Black-Litterman posterior returns and covariance.
    """
    if P is None or Q is None or Omega is None:
        return implied_returns, cov_matrix
        
    # Equation for posterior returns E[R]
    # E[R] = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1 [ (tau*Sigma)^-1 Pi + P^T Omega^-1 Q ]
    
    tau_cov = tau * cov_matrix
    tau_cov_inv = np.linalg.inv(tau_cov.values)
    
    Omega_inv = np.linalg.inv(Omega)
    
    term1 = np.linalg.inv(tau_cov_inv + P.T.dot(Omega_inv).dot(P))
    term2 = tau_cov_inv.dot(implied_returns.values) + P.T.dot(Omega_inv).dot(Q)
    
    posterior_returns = term1.dot(term2)
    
    # Posterior covariance
    # Sigma_p = Sigma + term1
    posterior_cov = cov_matrix.values + term1
    
    return pd.Series(posterior_returns, index=implied_returns.index), pd.DataFrame(posterior_cov, index=cov_matrix.index, columns=cov_matrix.columns)
