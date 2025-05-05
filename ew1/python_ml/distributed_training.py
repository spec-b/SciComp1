from multiprocessing import Manager, Process
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python_ml.distributed_worker import worker  # Adjust import as needed

import yfinance as yf


def run_distributed_training(num_workers=4, staleness=1, num_iters=10, tickers=None):
    if tickers is None or len(tickers) == 0:
        # Prompt for tickers if not provided
        tickers_str = input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOG): ")
        tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    
    # Fetch and concatenate close prices for all tickers
    all_X = []
    all_y = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1y", interval="1d")
            if data.empty or 'Close' not in data:
                print(f"Warning: No data for {ticker}")
                continue
            y = data['Close'].values
            X = np.arange(len(y))
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    if not all_y:
        raise ValueError("No valid stock data fetched.")

    # Flatten and concatenate all data
    X_full = np.concatenate(all_X)
    y_full = np.concatenate(all_y)

    # Partition data among workers
    X_splits = np.array_split(X_full, num_workers)
    y_splits = np.array_split(y_full, num_workers)
    for i, (X, y) in enumerate(zip(X_splits, y_splits)):
        if len(X) == 0 or len(y) == 0:
            print(f"Warning: Worker {i} has no data!")

    manager = Manager()
    params = manager.dict({'w': 0.0, 'b': 0.0})
    iterations = manager.dict({wid: 0 for wid in range(num_workers)})

    processes = []
    for wid in range(num_workers):
        p = Process(target=worker, args=(wid, params, iterations, staleness, X_splits[wid], y_splits[wid], num_iters))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Return the final parameters for display in the GUI
    return dict(params)
