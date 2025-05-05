import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import time
import os
from data_fetch import fetch_data
from bayes_model import GPModel
# This script runs a distributed Bayesian linear regression model on stock data.
# It fetches the data, runs a distributed MPI process, fits the model, and plots the results.
# It also writes progress to a file.
# Ensure the required directories exist

def write_progress(p):
    with open("progress.txt", "w") as f:
        f.write(f"{p:.2f}")

def run_distributed_blr(mode='batch', ticker='AAPL'):
    write_progress(0.0)

    # Step 1: Fetch data
    df = fetch_data(ticker, period='1y', interval='1d')
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values

    np.save('X.npy', X)
    np.save('y.npy', y)
    write_progress(0.2)

    # Step 2: Run distributed MPI process
    subprocess.run(['mpirun', '-n', '4', './bin/coordinator'])
    write_progress(0.5)

    # Step 3: Fit Bayesian model
    model = GPModel()
    model.fit(X, y)
    write_progress(0.7)

    # Step 4: Predict and plot
    X_pred = np.linspace(0, len(df)-1, len(df)).reshape(-1, 1)
    y_pred = model.predict(X_pred)

    plt.figure(figsize=(10, 5))
    plt.plot(X, y, label='Observed', color='black')
    plt.plot(X_pred, y_pred, label='Prediction', color='blue')
    plt.title(f"{ticker} Forecast")
    plt.xlabel("Time Index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast.png")
    plt.close()
    write_progress(1.0)

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    run_distributed_blr(mode='batch', ticker=ticker)
