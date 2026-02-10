from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.wind_model import WindModel
from backend.simulator import simulate_payload_drop
import os

app = FastAPI(title="HAPS Wind & Payload Drop Prediction API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize wind model
try:
    wind_model = WindModel("data/wind_sample.csv")
except Exception as e:
    raise RuntimeError(f"Failed to load wind model: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "HAPS Wind & Payload Drop Prediction API",
        "status": "operational",
        "model": "Gaussian Process Regression (GPR) with Monte Carlo Uncertainty"
    }

@app.get("/wind-profile")
def get_wind_profile(altitude_km: float):
    if not (20 <= altitude_km <= 80):
        raise HTTPException(status_code=400, detail="Altitude must be between 20 and 80 km")
    
    u_mean, v_mean, u_std, v_std = wind_model.predict(altitude_km)
    wind_speed = (u_mean**2 + v_mean**2)**0.5
    
    return {
        "altitude_km": round(altitude_km, 2),
        "u_wind": round(u_mean, 2),        # ← CRITICAL: frontend expects this exact field name
        "v_wind": round(v_mean, 2),        # ← CRITICAL: frontend expects this exact field name
        "u_uncertainty": round(u_std, 2),
        "v_uncertainty": round(v_std, 2),
        "wind_speed": round(wind_speed, 2)
    }

@app.post("/simulate-drop")
def simulate_drop(
    drop_altitude_km: float = 30.0,
    payload_mass: float = 5.0,
    descent_rate: float = 5.0,
    monte_carlo_runs: int = 50
):
    if not (20 <= drop_altitude_km <= 50):
        raise HTTPException(status_code=400, detail="Drop altitude must be between 20 and 50 km")
    if descent_rate <= 0:
        raise HTTPException(status_code=400, detail="Descent rate must be positive")
    if not (10 <= monte_carlo_runs <= 200):
        raise HTTPException(status_code=400, detail="Monte Carlo runs must be between 10 and 200")
    
    return simulate_payload_drop(
        drop_altitude_km,
        payload_mass,
        descent_rate,
        wind_model,
        monte_carlo_runs
    )
