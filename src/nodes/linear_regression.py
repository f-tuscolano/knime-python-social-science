import logging
import warnings

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Set up the logger
LOGGER = logging.getLogger(__name__)


# Linear model learner class
class LinearModelLearner:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_coefficients(self):
        coef_columns = [
            "coef1",
            "coef2",
            "coef3",
        ]
        return self.model.coef_, coef_columns
