import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
BACKEND_URL = "https://wind-payload-ai.onrender.com/"

st.set_page_config(
    page_title="HAPS Wind & Payload Drop AI",
    layout="wide"
)

st.title("🌬️ AI-Based Wind Profiling & Payload Drop Prediction")
st.markdown(
    """
    This system models **stratospheric wind conditions (20–80 km)** and predicts
    **payload landing locations** for **High-Altitude Platform Stations (HAPS)**.
    """
)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("📌 Payload Drop Parameters")

drop_altitude = st.sidebar.slider(
    "Drop Altitude (km)",
    min_value=20,
    max_value=50,
    value=30
)

payload_mass = st.sidebar.number_input(
    "Payload Mass (kg)",
    min_value=1.0,
    max_value=50.0,
    value=5.0
)

descent_rate = st.sidebar.number_input(
    "Descent Rate (m/s)",
    min_value=1.0,
    max_value=20.0,
    value=5.0
)

run_simulation = st.sidebar.button("🚀 Run Simulation")

# -----------------------------
# WIND PROFILE SECTION
# -----------------------------
st.subheader("📊 Wind Profile Analysis (20–80 km)")

altitudes = list(range(20, 81))
u_winds = []
v_winds = []

for alt in altitudes:
    response = requests.get(
        f"{BACKEND_URL}/wind-profile",
        params={"altitude_km": alt}
    )
    data = response.json()
    u_winds.append(data["u_wind"])
    v_winds.append(data["v_wind"])

wind_df = pd.DataFrame({
    "Altitude (km)": altitudes,
    "U Wind (m/s)": u_winds,
    "V Wind (m/s)": v_winds
})

col1, col2 = st.columns(2)

with col1:
    fig_u = px.line(
        wind_df,
        x="U Wind (m/s)",
        y="Altitude (km)",
        title="Zonal Wind vs Altitude",
        markers=True
    )
    fig_u.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_u, use_container_width=True)

with col2:
    fig_v = px.line(
        wind_df,
        x="V Wind (m/s)",
        y="Altitude (km)",
        title="Meridional Wind vs Altitude",
        markers=True
    )
    fig_v.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_v, use_container_width=True)

# -----------------------------
# PAYLOAD DROP SIMULATION
# -----------------------------
if run_simulation:
    st.subheader("🎯 Payload Drop Trajectory Simulation")

    payload = {
        "drop_altitude_km": drop_altitude,
        "payload_mass": payload_mass,
        "descent_rate": descent_rate
    }

    response = requests.post(
        f"{BACKEND_URL}/simulate-drop",
        params=payload
    )

    result = response.json()
    trajectory = pd.DataFrame(result["trajectory"])

    landing_x = result["landing_point"]["x_drift_m"]
    landing_y = result["landing_point"]["y_drift_m"]

    col3, col4 = st.columns(2)

    # Trajectory plot
    with col3:
        fig_traj = go.Figure()

        fig_traj.add_trace(
            go.Scatter(
                x=trajectory["x_drift_m"],
                y=trajectory["y_drift_m"],
                mode="lines+markers",
                name="Payload Path"
            )
        )

        fig_traj.add_trace(
            go.Scatter(
                x=[landing_x],
                y=[landing_y],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="Landing Point"
            )
        )

        fig_traj.update_layout(
            title="Payload Drift Trajectory",
            xaxis_title="X Drift (m)",
            yaxis_title="Y Drift (m)",
            height=500
        )

        st.plotly_chart(fig_traj, use_container_width=True)

    # Landing summary
    with col4:
        st.metric("Landing X Drift (m)", f"{landing_x:.2f}")
        st.metric("Landing Y Drift (m)", f"{landing_y:.2f}")

        st.markdown(
            """
            **Interpretation**
            - Horizontal drift is caused by altitude-dependent wind layers
            - AI-assisted wind interpolation improves trajectory realism
            - This prediction represents the *expected landing point*
            """
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "🚀 **Built as a rapid AI + Physics proof-of-concept for HAPS payload operations**"
)

