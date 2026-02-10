import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class WindModel:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)  # ← NO spaces in filename
        X = self.data["altitude_km"].values.reshape(-1, 1)  # ← NO spaces in column names
        self.u_wind = self.data["u_wind"].values
        self.v_wind = self.data["v_wind"].values
        
        # Physics-informed kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(15.0, (5.0, 30.0)) + WhiteKernel(noise_level=0.5)
        
        self.gpr_u = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        self.gpr_v = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        self.gpr_u.fit(X, self.u_wind)
        self.gpr_v.fit(X, self.v_wind)
    
    def predict(self, altitude_km):
        X = np.array([[altitude_km]])
        u_mean, u_std = self.gpr_u.predict(X, return_std=True)
        v_mean, v_std = self.gpr_v.predict(X, return_std=True)
        return float(u_mean), float(v_mean), float(u_std), float(v_std)
