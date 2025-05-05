# SciComp1
Final Project for 21-765. Introduction to Parallel Computing and Scientific Computation

This is a Python-based scientific computation tool aimed at simplifying regression analysis and data visualization. The tool is built with a PyQt5 GUI to provide an interactive experience for users who wish to conduct regression analysis on numerical datasets. The application allows users to input tickers which sources the data from the internet, select regression models, and view the results of their analysis in an intuitive manner.
Key Features:
1. Regression Analysis: Performs linear regression, Bayesian Linear Regression, and other statistical analyses on input data.
2. Data Table: Users can enter data through a GUI interface, there's a little box.
3. Visualization: Basic visualization of regression results using plots and statistical metrics.
4. PyQt5 GUI: Interactive graphical user interface to make the tool accessible to non-programmers.

SciComp1/
│
├── gui_c/                     # GUI-related files
│   ├── main.py                # Main program entry point
│   ├── regression.py          # Regression models and computations
│   └── ui/                    # UI files (e.g., form layouts)
│
├── tests/                     # Unit tests for validation
│   ├── test_regression.py     # Tests for regression analysis
│   └── test_gui.py            # Tests for GUI components
│
├── requirements.txt           # List of dependencies
├── LICENSE                    # License file
├── README.md                  # This readme file
└── CHANGELOG.md               # History of changes


Prerequisites are Python 3.6 or higher and pip package manager. 

Instructions:
1. Clone the repository to your local machine using Git.
2. Create a virtual environment and install the required dependencies using pip:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Windows:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

3. The requirements.txt file includes all the necessary dependencies, including:

    PyQt5: For building the GUI

    numpy: For numerical operations and data handling

    matplotlib: For plotting and visualizing regression results

    yfinance: For downloading financial data

Running the application:
Run the program by executing the main.py file: python gui_c/main.py

The Graphical User Interface:
Data Input:
The main window contains a table where you can input numerical data for regression analysis.
Data should be entered in rows, where each row represents a data point and each column represents a feature or variable.

Selecting Regression Models:
After entering the data, you can select which regression mode to apply, if you want to look at a single stock or compare multiple ones, and there is an option to use stale synchronous processing which uses distributed computing techniques. 

Running the Regression:
Click the "Run Regression" button to perform the regression analysis.
The program will calculate the model's coefficients, statistical metrics, and display a plot of the results.

Viewing Results:
The results will include the regression coefficients, R-squared values, and other statistical metrics.
A plot will show the regression line (if valid data exists) along with the original data points.


Licensing
This project is licensed under the MIT License. See the LICENSE file for more details.

Changelog
All changes to the project are documented in the CHANGELOG.md. This includes bug fixes, new features, and any breaking changes.
Security Considerations
This repo does not directly handle any sensitive data, only uses open source APIs. 

For further discussion on security issues, please email. 


