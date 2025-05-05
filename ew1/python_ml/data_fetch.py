import yfinance as yf
import pandas as pd

def fetch_data(ticker, period='1y', interval='1d'):
    """
    Fetch historical stock data using yfinance.
    :param ticker: Stock symbol (e.g., 'AAPL', 'GOOG')
    :param period: Time period for data (e.g., '1y', '6mo', '1d')
    :param interval: Data interval (e.g., '1d', '1wk')
    :return: DataFrame with stock data
    """
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Close']]  # Only using 'Close' price for regression
    df.dropna(inplace=True)
    return df
