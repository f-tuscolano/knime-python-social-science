"""
Module for Linear Regression Node
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

class LinearRegressionNode:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)
