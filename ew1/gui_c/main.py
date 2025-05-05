import sys
import subprocess

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from python_ml.regression import fetch_data, run_bayesian_regression

from python_ml.distributed_training import run_distributed_training

import yfinance as yf


# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of dependencies to check
required_packages = ['yfinance', 'pyqt5', 'scikit-learn', 'matplotlib', 'numpy', 'pymc', 'arviz']

# Check if each required package is installed, and install if not
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        install(package)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QLineEdit, QDialog, QTextEdit
)
import numpy as np
import matplotlib.pyplot as plt


class MultiStockDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multiple Stock Bayesian Regression")
        self.layout = QVBoxLayout()
        self.label = QLabel("Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOG):")
        self.layout.addWidget(self.label)
        self.symbol_input = QLineEdit()
        self.layout.addWidget(self.symbol_input)
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.layout.addWidget(self.result_area)
        self.run_button = QPushButton("Run Bayesian Regression")
        self.layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.run_multi_regression)
        self.setLayout(self.layout)


    def run_multi_regression(self):
        symbols = [s.strip().upper() for s in self.symbol_input.text().split(',') if s.strip()]
        if not symbols:
            self.result_area.setText("Please enter at least one stock symbol!")
            return

        results = []
        plt.figure(figsize=(10, 6))  # Create a new figure for all stocks

        for symbol in symbols:
            try:
                data = fetch_data(symbol)
                y = data['Close'].values
                X = np.arange(len(y))
                plt.plot(X, y, label=symbol)  # Plot each stock's price series

                _, summary = run_bayesian_regression(data, draws=500, tune=500)
                alpha = summary.loc['alpha']
                beta = summary.loc['beta']
                results.append(
                    f"{symbol}:\n"
                    f"  Intercept (alpha): {alpha['mean']:.2f} [{alpha['hdi_3%']:.2f}, {alpha['hdi_97%']:.2f}]\n"
                    f"  Slope (beta): {beta['mean']:.4f} [{beta['hdi_3%']:.4f}, {beta['hdi_97%']:.4f}]\n"
                )
            except Exception as e:
                results.append(f"{symbol}: Error: {str(e)}")

        plt.title("Stock Prices for Selected Symbols")
        plt.xlabel("Time (Days)")
        plt.ylabel("Close Price")
        plt.legend()
        plt.tight_layout()
        plt.show()  # Show the plot after all stocks are added

        self.result_area.setText('\n'.join(results))


class BayesianRegressionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bayesian Linear Regression for Finance")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.label = QLabel("Choose analysis mode:", self)
        self.layout.addWidget(self.label)

        self.ticker_input = QLineEdit(self)
        self.ticker_input.setPlaceholderText("Enter tickers (e.g., AAPL, MSFT, GOOG)")
        self.layout.addWidget(self.ticker_input)

        # Buttons for analysis modes
        self.single_button = QPushButton("Single Stock Analysis", self)
        self.layout.addWidget(self.single_button)
        self.single_button.clicked.connect(self.show_single_stock)

        self.multi_button = QPushButton("Multiple Stock Analysis", self)
        self.layout.addWidget(self.multi_button)
        self.multi_button.clicked.connect(self.show_multi_stock)

        # Widgets for single stock analysis (hidden by default)
        self.symbol_input = QLineEdit(self)
        self.symbol_input.setPlaceholderText("Enter Stock Symbol (e.g., AAPL, GOOG)")
        self.layout.addWidget(self.symbol_input)
        self.symbol_input.hide()

        self.table = QTableWidget(self)
        self.layout.addWidget(self.table)
        self.table.hide()

        self.load_button = QPushButton("Fetch Stock Data", self)
        self.layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.fetch_data)
        self.load_button.hide()

        self.run_button = QPushButton("Run Bayesian Regression", self)
        self.layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.run_regression)
        self.run_button.hide()

        self.dist_train_button = QPushButton("Run Distributed Training (SSP Demo)", self)
        self.layout.addWidget(self.dist_train_button)
        self.dist_train_button.clicked.connect(self.run_distributed_training_gui)


        self.result_label = QLabel("", self)
        self.layout.addWidget(self.result_label)
        self.result_label.hide()

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.current_data = None
        self.ticker_input = QLineEdit(self)



    def show_single_stock(self):
        self.symbol_input.show()
        self.table.show()
        self.load_button.show()
        self.run_button.show()
        self.result_label.show()

    def show_multi_stock(self):
        dialog = MultiStockDialog()
        dialog.exec_()

    def fetch_data(self):
        symbol = self.symbol_input.text().upper()
        if not symbol:
            self.result_label.setText("Please enter a stock symbol!")
            return

        try:
            data = fetch_data(symbol)
            self.current_data = data
            rows, cols = data.shape

            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            self.table.setHorizontalHeaderLabels([str(col) for col in data.columns])

            for i in range(rows):
                for j in range(cols):
                    self.table.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))

            self.result_label.setText(f"Fetched data for {symbol}")

        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")
            self.current_data = None

    def run_regression(self):
        if self.current_data is None:
            self.result_label.setText("Please fetch stock data first!")
            return

        try:
            _, summary = run_bayesian_regression(self.current_data, draws=500, tune=500)
            alpha = summary.loc['alpha']
            beta = summary.loc['beta']
            result_text = (
                f"Intercept (alpha): {alpha['mean']:.2f} [{alpha['hdi_3%']:.2f}, {alpha['hdi_97%']:.2f}]\n"
                f"Slope (beta): {beta['mean']:.4f} [{beta['hdi_3%']:.4f}, {beta['hdi_97%']:.4f}]"
            )
            self.result_label.setText(result_text)
            # Plot actual data
            y = self.current_data['Close'].values
            X = np.arange(len(y))
            plt.plot(X, y, label="Actual")
            plt.title(f"Stock Price for {self.symbol_input.text().upper()}")
            plt.legend()
            plt.show()
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")

    def run_distributed_training_gui(self):
        self.result_label.setText("Running distributed training...")
        self.repaint()  # Update the GUI immediately

        tickers_str = self.ticker_input.text()
        tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]

        try:
            final_params = run_distributed_training(tickers=tickers)
            result_text = (
                f"Distributed Training Finished!\n"
                f"Final parameters:\n"
                f"w = {final_params['w']:.4f}\n"
                f"b = {final_params['b']:.4f}"
            )
        except Exception as e:
            result_text = f"Error: {e}"

        self.result_label.setText(result_text)
        self.result_label.show()

        # Only plot actual data if self.current_data is available
        if self.current_data is not None:
            y = self.current_data['Close'].values
            X = np.arange(len(y))
            plt.plot(X, y, label="Actual")
            plt.title(f"Stock Price for {self.symbol_input.text().upper()}")
            plt.legend()
            plt.show()
            # Optionally update result_label to show both results
            self.result_label.setText(result_text + "\nPlotted actual stock data.")
        else:
            # No stock data loaded, so just show the result
            self.result_label.setText(result_text)
        
        self.result_label.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BayesianRegressionGUI()
    window.show()
    sys.exit(app.exec_())
