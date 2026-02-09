from fastapi import FastAPI
from backend.wind_model import WindModel
from backend.simulator import simulate_payload_drop

app = FastAPI(title="HAPS Wind & Payload Drop AI")

# Initialize wind model
wind_model = WindModel("data/wind_sample.csv")

@app.get("/")
def root():
    return {"message": "HAPS Wind & Payload Drop Prediction API"}

@app.get("/wind-profile")
def get_wind_profile(altitude_km: float):
    u, v = wind_model.predict(altitude_km)
    return {
        "altitude_km": altitude_km,
        "u_wind": u,
        "v_wind": v
    }

@app.post("/simulate-drop")
def simulate_drop(
    drop_altitude_km: float,
    payload_mass: float = 5.0,
    descent_rate: float = 5.0
):
    result = simulate_payload_drop(
        drop_altitude_km,
        payload_mass,
        descent_rate,
        wind_model
    )
    return result
