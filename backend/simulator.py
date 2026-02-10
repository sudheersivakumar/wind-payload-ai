import numpy as np
from backend.physics import update_position

def simulate_payload_drop(
    drop_altitude_km,
    payload_mass,
    descent_rate,
    wind_model,
    monte_carlo_runs=50  # ← Add Monte Carlo parameter
):
    dt = 1.0
    landing_points = []
    
    for _ in range(monte_carlo_runs):
        z = drop_altitude_km * 1000
        x, y = 0.0, 0.0
        trajectory = []
        
        while z > 0:
            current_alt_km = z / 1000
            
            # Get wind + uncertainty from GPR
            u_mean, v_mean, u_std, v_std = wind_model.predict(current_alt_km)
            
            # Monte Carlo sampling: perturb wind using uncertainty
            u_sample = np.random.normal(u_mean, max(u_std, 0.5))
            v_sample = np.random.normal(v_mean, max(v_std, 0.3))
            
            x, y, z = update_position(
                x, y, z,
                u_wind=u_sample,
                v_wind=v_sample,
                descent_rate=descent_rate,
                dt=dt
            )
            
            trajectory.append({
                "altitude_km": round(current_alt_km, 2),
                "x_drift_m": round(x, 2),
                "y_drift_m": round(y, 2)
            })
        
        landing_points.append({
            "x_drift_m": round(x, 2),
            "y_drift_m": round(y, 2)
        })
    
    # Calculate statistics
    landing_x = [p["x_drift_m"] for p in landing_points]
    landing_y = [p["y_drift_m"] for p in landing_points]
    
    return {
        "monte_carlo_runs": monte_carlo_runs,
        "landing_points": landing_points,
        "landing_statistics": {
            "mean_x_drift_m": round(np.mean(landing_x), 2),
            "mean_y_drift_m": round(np.mean(landing_y), 2),
            "std_x_drift_m": round(np.std(landing_x), 2),
            "std_y_drift_m": round(np.std(landing_y), 2)
        },
        "representative_trajectory": trajectory
    }
