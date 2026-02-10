import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class WindModel:
    def __init__(self, csv_path):
        """
        Gaussian Process Regression wind model with uncertainty quantification
        Trained on synthetic dataset inspired by ERA5/MERRA-2 stratospheric patterns
        """
        # Remove trailing spaces in column names
        self.data = pd.read_csv(csv_path)
        X = self.data["altitude_km"].values.reshape(-1, 1)
        self.u_wind = self.data["u_wind"].values
        self.v_wind = self.data["v_wind"].values
        
        # Physics-informed kernel: smooth RBF + noise modeling
        kernel = C(1.0, (1e-3, 1e3)) * RBF(15.0, (5.0, 30.0)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-2, 10))
        
        # Initialize GPR models for zonal (u) and meridional (v) components
        self.gpr_u = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        self.gpr_v = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        
        # Train models
        self.gpr_u.fit(X, self.u_wind)
        self.gpr_v.fit(X, self.v_wind)
    
    def predict(self, altitude_km):
        """
        Predict wind vectors with uncertainty at given altitude
        Returns:
            u_mean, v_mean: Predicted wind components (m/s)
            u_std, v_std: Uncertainty estimates (standard deviation)
        """
        X = np.array([[altitude_km]])
        u_mean, u_std = self.gpr_u.predict(X, return_std=True)
        v_mean, v_std = self.gpr_v.predict(X, return_std=True)
        
        # Extract scalar values from arrays (FIX FOR TypeError)
        return float(u_mean[0]), float(v_mean[0]), float(u_std[0]), float(v_std[0])
