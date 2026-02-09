from backend.physics import update_position

def simulate_payload_drop(
    drop_altitude_km,
    payload_mass,
    descent_rate,
    wind_model
):
    dt = 1.0  # seconds
    altitude_m = drop_altitude_km * 1000
    x, y, z = 0.0, 0.0, altitude_m
    trajectory = []

    while z > 0:
        current_alt_km = z / 1000
        u, v = wind_model.predict(current_alt_km)

        x, y, z = update_position(
            x, y, z,
            u_wind=u,
            v_wind=v,
            descent_rate=descent_rate,
            dt=dt
        )

        trajectory.append({
            "altitude_km": current_alt_km,
            "x_drift_m": x,
            "y_drift_m": y
        })

    return {
        "landing_point": {
            "x_drift_m": x,
            "y_drift_m": y
        },
        "trajectory": trajectory
    }
