from sklearn.linear_model import LinearRegression
import numpy as np

class GPModel:
    def __init__(self):
        # For now, we're using a simple Linear Regression model
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
