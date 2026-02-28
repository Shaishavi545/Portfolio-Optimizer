import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.title("📊 Indian Portfolio Optimization (Markowitz)")

stocks = ['HDFCBANK.NS', 'INFY.NS', 'RELIANCE.NS', 
          'HINDUNILVR.NS', 'TCS.NS']

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
risk_free_rate = st.sidebar.slider("Risk Free Rate", 0.0, 0.15, 0.07)

@st.cache_data
def load_data():
    data = pd.DataFrame()
    for stock in stocks:
        temp = yf.download(stock, start=start_date)
        data[stock] = temp['Close']
    return data.dropna()

data = load_data()

returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

def portfolio_return(weights):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def sharpe_ratio(weights):
    return (portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)

num_assets = len(stocks)
init_guess = num_assets * [1./num_assets]
bounds = tuple((0,1) for _ in range(num_assets))
constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})

opt = minimize(lambda w: -sharpe_ratio(w),
               init_guess,
               method='SLSQP',
               bounds=bounds,
               constraints=constraints)

optimal_weights = opt.x

st.subheader("Optimal Portfolio Allocation")
for stock, weight in zip(stocks, optimal_weights):
    st.write(stock, ":", round(weight*100,2), "%")

st.subheader("Portfolio Metrics")
st.write("Expected Return:", round(portfolio_return(optimal_weights),4))
st.write("Volatility:", round(portfolio_volatility(optimal_weights),4))
st.write("Sharpe Ratio:", round(sharpe_ratio(optimal_weights),4))

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)