# regression.py

import yfinance as yf
import numpy as np
import pymc as pm
import arviz as az

def fetch_data(symbol):
    """
    Fetches 1 year of daily closing price data for the given stock symbol.
    Returns a pandas DataFrame with a 'Close' column.
    Raises ValueError if no data is found.
    """
    data = yf.download(symbol, period="1y", interval="1d")
    if data.empty or 'Close' not in data:
        raise ValueError(f"No data found for symbol: {symbol}")
    return data[['Close']]

def run_bayesian_regression(data, draws=1000, tune=1000, random_seed=42):
    """
    Performs Bayesian linear regression on the 'Close' price time series using PyMC.
    Returns the posterior samples and summary statistics.
    """
    y = data['Close'].values
    X = np.arange(len(y)).reshape(-1, 1)
    X_ = (X - X.mean()) / X.std()  # Standardize for better sampling

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = alpha + beta * X_.flatten()
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(draws=draws, tune=tune, random_seed=random_seed, progressbar=False)

    summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])
    return trace, summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python regression.py STOCK1 [STOCK2 ...]")
        sys.exit(1)
    symbols = [s.upper() for s in sys.argv[1:]]
    for symbol in symbols:
        try:
            data = fetch_data(symbol)
            trace, summary = run_bayesian_regression(data)
            print(f"\nBayesian Linear Regression for {symbol}")
            print(summary)
        except Exception as e:
            print(f"Error for {symbol}: {e}")
