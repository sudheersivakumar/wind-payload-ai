import numpy as np

def update_position(x, y, z, u_wind, v_wind, descent_rate, dt):
    """
    Simple physics-based update:
    - Horizontal drift from wind
    - Vertical descent at constant rate
    """
    x += u_wind * dt
    y += v_wind * dt
    z -= descent_rate * dt

    return x, y, z
