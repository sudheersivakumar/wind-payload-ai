def update_position(x, y, z, u_wind, v_wind, descent_rate, dt):
    """
    Physics-based position update for payload descent
    - Horizontal drift from wind exposure
    - Vertical descent at constant rate (parachute approximation)
    """
    x += u_wind * dt
    y += v_wind * dt
    z -= descent_rate * dt
    return x, y, z
