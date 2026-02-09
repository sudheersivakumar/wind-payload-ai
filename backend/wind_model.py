import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class WindModel:
    def __init__(self, csv_path):
        self.data = pd.read_csv("data/wind_sample.csv")
        self.altitudes = self.data["altitude_km"].values.reshape(-1, 1)
        self.u_wind = self.data["u_wind"].values
        self.v_wind = self.data["v_wind"].values

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(self.altitudes)

        self.u_model = LinearRegression().fit(X_poly, self.u_wind)
        self.v_model = LinearRegression().fit(X_poly, self.v_wind)

        self.poly = poly

    def predict(self, altitude_km):
        X = self.poly.transform(np.array([[altitude_km]]))
        u_pred = self.u_model.predict(X)
        v_pred = self.v_model.predict(X)
        
        # Safely extract scalar values from NumPy arrays
        u = float(u_pred[0]) if u_pred.size > 0 else 0.0
        v = float(v_pred[0]) if v_pred.size > 0 else 0.0
        return u, v
