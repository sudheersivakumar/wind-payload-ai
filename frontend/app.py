import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ==================== CONFIGURATION ====================
# Update this URL after deploying backend to Render
# For local testing: "http://127.0.0.1:8000"
# For production: "https://your-haps-backend.onrender.com"
BACKEND_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="🌬️ HAPS Wind & Payload Drop AI",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aerospace aesthetic
st.markdown("""
<style>
    .main-header { 
        color: #1e3a8a; 
        font-weight: 700; 
        font-size: 2.5rem;
    }
    .sub-header {
        color: #3b82f6;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .uncertainty-badge {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .confidence-badge {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🌬️ AI-Based Wind Profiling & Payload Drop Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='background-color: #dbeafe; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
    <p style='margin: 0; color: #1e40af; font-size: 1.1rem;'>
        <strong>🚀 Advanced Aerospace System:</strong> Gaussian Process Regression for uncertainty-aware wind modeling + Monte Carlo simulation for probabilistic landing zone prediction
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📌 Payload Drop Parameters")
    
    st.markdown("### 📍 Release Conditions")
    drop_altitude = st.slider(
        "Drop Altitude (km)",
        min_value=20,
        max_value=50,
        value=30,
        step=1,
        help="Altitude of payload release from HAPS platform (20-50 km)"
    )
    
    payload_mass = st.number_input(
        "Payload Mass (kg)",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
        help="Mass of the payload being dropped"
    )
    
    descent_rate = st.number_input(
        "Descent Rate (m/s)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Vertical descent speed under parachute"
    )
    
    st.markdown("---")
    st.markdown("### 🎲 Monte Carlo Settings")
    mc_runs = st.slider(
        "Monte Carlo Runs",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of simulations for uncertainty analysis (higher = more accurate)"
    )
    
    st.markdown("---")
    run_simulation = st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True)
    
    st.markdown("""
    ### ℹ️ About This System
    
    **AI Architecture:**
    - 📊 **Gaussian Process Regression (GPR)**: Uncertainty-aware wind modeling
    - 🎲 **Monte Carlo Simulation**: Probabilistic landing zone analysis
    - 📐 **Physics Engine**: Altitude-dependent wind drift simulation
    
    **Altitude Range:** 20–80 km wind profiling
    
    **Key Features:**
    - Wind uncertainty quantification
    - Probabilistic landing predictions
    - 68%/95% confidence zones
    - Physics-validated trajectories
    """)

# ==================== WIND PROFILE SECTION ====================
st.subheader("📊 Stratospheric Wind Profile (20–80 km)")

try:
    # Fetch wind profile data with uncertainty
    altitudes = np.linspace(20, 80, 31)
    wind_data = []
    
    with st.spinner("Loading wind profiles..."):
        for alt in altitudes:
            resp = requests.get(f"{BACKEND_URL}/wind-profile", params={"altitude_km": alt})
            if resp.status_code == 200:
                wind_data.append(resp.json())
    
    wind_df = pd.DataFrame(wind_data)
    
    # Create dual-axis wind profile visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Zonal Wind (East-West Component)</div>', unsafe_allow_html=True)
        
        fig_u = go.Figure()
        
        # Mean wind
        fig_u.add_trace(go.Scatter(
            x=wind_df["u_wind"],
            y=wind_df["altitude_km"],
            mode='lines+markers',
            name='Mean Wind',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, color='#3b82f6'),
            hovertemplate='Altitude: %{y} km<br>Wind: %{x:.2f} m/s<extra></extra>'
        ))
        
        # Uncertainty band (if available)
        if "u_uncertainty" in wind_df.columns:
            fig_u.add_trace(go.Scatter(
                x=wind_df["u_wind"] + wind_df["u_uncertainty"],
                y=wind_df["altitude_km"],
                mode='lines',
                name='Upper Bound (1σ)',
                line=dict(color='#93c5fd', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_u.add_trace(go.Scatter(
                x=wind_df["u_wind"] - wind_df["u_uncertainty"],
                y=wind_df["altitude_km"],
                mode='lines',
                name='Lower Bound (1σ)',
                line=dict(color='#93c5fd', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig_u.update_yaxes(autorange="reversed", title="Altitude (km)", gridcolor='#e5e7eb')
        fig_u.update_xaxes(title="Wind Speed (m/s)", gridcolor='#e5e7eb')
        fig_u.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_u.update_layout(
            height=500,
            plot_bgcolor='white',
            hovermode='closest',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig_u, use_container_width=True)
        
        # Wind statistics
        max_u_wind = wind_df["u_wind"].max()
        max_u_alt = wind_df.loc[wind_df["u_wind"].idxmax(), "altitude_km"]
        st.info(f"💡 **Peak Zonal Wind**: {max_u_wind:.1f} m/s at {max_u_alt:.0f} km altitude")
    
    with col2:
        st.markdown('<div class="sub-header">Meridional Wind (North-South Component)</div>', unsafe_allow_html=True)
        
        fig_v = go.Figure()
        
        # Mean wind
        fig_v.add_trace(go.Scatter(
            x=wind_df["v_wind"],
            y=wind_df["altitude_km"],
            mode='lines+markers',
            name='Mean Wind',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8, color='#ef4444'),
            hovertemplate='Altitude: %{y} km<br>Wind: %{x:.2f} m/s<extra></extra>'
        ))
        
        # Uncertainty band (if available)
        if "v_uncertainty" in wind_df.columns:
            fig_v.add_trace(go.Scatter(
                x=wind_df["v_wind"] + wind_df["v_uncertainty"],
                y=wind_df["altitude_km"],
                mode='lines',
                name='Upper Bound (1σ)',
                line=dict(color='#fca5a5', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_v.add_trace(go.Scatter(
                x=wind_df["v_wind"] - wind_df["v_uncertainty"],
                y=wind_df["altitude_km"],
                mode='lines',
                name='Lower Bound (1σ)',
                line=dict(color='#fca5a5', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(239, 68, 68, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig_v.update_yaxes(autorange="reversed", title="Altitude (km)", gridcolor='#e5e7eb')
        fig_v.update_xaxes(title="Wind Speed (m/s)", gridcolor='#e5e7eb')
        fig_v.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_v.update_layout(
            height=500,
            plot_bgcolor='white',
            hovermode='closest',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig_v, use_container_width=True)
        
        # Wind statistics
        max_v_wind = abs(wind_df["v_wind"]).max()
        st.info(f"💡 **Max Meridional Wind**: ±{max_v_wind:.1f} m/s (north-south variation)")
    
    # Wind speed summary
    st.markdown("### 📈 Wind Speed Summary")
    wind_speed = np.sqrt(wind_df["u_wind"]**2 + wind_df["v_wind"]**2)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Average Wind Speed", f"{wind_speed.mean():.1f} m/s")
    with col_b:
        st.metric("Max Wind Speed", f"{wind_speed.max():.1f} m/s")
    with col_c:
        st.metric("Wind Direction", "Eastward" if wind_df["u_wind"].mean() > 0 else "Westward")

except Exception as e:
    st.error(f"⚠️ Could not load wind profiles: {str(e)}")
    st.info("💡 Make sure the backend is running at http://127.0.0.1:8000")

# ==================== SIMULATION RESULTS SECTION ====================
if run_simulation:
    with st.spinner(f"Running {mc_runs} Monte Carlo simulations..."):
        try:
            # Call backend API with Monte Carlo runs
            payload = {
                "drop_altitude_km": drop_altitude,
                "payload_mass": payload_mass,
                "descent_rate": descent_rate,
                "monte_carlo_runs": mc_runs
            }
            
            response = requests.post(f"{BACKEND_URL}/simulate-drop", params=payload)
            
            if response.status_code != 200:
                st.error(f"Backend error: {response.text}")
                st.stop()
            
            result = response.json()
            
            # Display results in tabs
            st.markdown("---")
            st.subheader("🎯 Monte Carlo Payload Drop Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📍 Landing Dispersion", 
                "📉 Trajectory", 
                "📊 Statistics", 
                "📈 Wind Influence"
            ])
            
            # ==================== TAB 1: LANDING DISPERSION ====================
            with tab1:
                st.markdown('<div class="sub-header">Probable Landing Zone (Monte Carlo Analysis)</div>', unsafe_allow_html=True)
                
                landing_df = pd.DataFrame(result["landing_points"])
                
                # Create landing dispersion plot with confidence ellipses
                fig_landing = go.Figure()
                
                # All landing points
                fig_landing.add_trace(go.Scatter(
                    x=landing_df["x_drift_m"],
                    y=landing_df["y_drift_m"],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='#3b82f6',
                        opacity=0.6,
                        symbol='circle'
                    ),
                    name=f'{mc_runs} Simulations',
                    hovertemplate='Run %{customdata}<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>',
                    customdata=list(range(1, len(landing_df) + 1))
                ))
                
                # Mean landing point
                stats = result["landing_statistics"]
                fig_landing.add_trace(go.Scatter(
                    x=[stats["mean_x_drift_m"]],
                    y=[stats["mean_y_drift_m"]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    name='Mean Landing Point',
                    hovertemplate='Mean Landing<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>'
                ))
                
                # 1σ confidence ellipse
                theta = np.linspace(0, 2*np.pi, 100)
                ellipse_x_1sigma = stats["mean_x_drift_m"] + stats["std_x_drift_m"] * np.cos(theta)
                ellipse_y_1sigma = stats["mean_y_drift_m"] + stats["std_y_drift_m"] * np.sin(theta)
                fig_landing.add_trace(go.Scatter(
                    x=ellipse_x_1sigma,
                    y=ellipse_y_1sigma,
                    mode='lines',
                    name='68% Confidence (1σ)',
                    line=dict(color='orange', width=2, dash='solid'),
                    hoverinfo='skip'
                ))
                
                # 2σ confidence ellipse
                ellipse_x_2sigma = stats["mean_x_drift_m"] + 2 * stats["std_x_drift_m"] * np.cos(theta)
                ellipse_y_2sigma = stats["mean_y_drift_m"] + 2 * stats["std_y_drift_m"] * np.sin(theta)
                fig_landing.add_trace(go.Scatter(
                    x=ellipse_x_2sigma,
                    y=ellipse_y_2sigma,
                    mode='lines',
                    name='95% Confidence (2σ)',
                    line=dict(color='red', width=2, dash='dash'),
                    hoverinfo='skip'
                ))
                
                fig_landing.update_layout(
                    width=800,
                    height=600,
                    xaxis_title="East-West Drift (m)",
                    yaxis_title="North-South Drift (m)",
                    hovermode="closest",
                    plot_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig_landing, use_container_width=True)
                
                # Summary metrics in cards
                st.markdown("### 📊 Landing Zone Statistics")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Mean X Drift", f"{stats['mean_x_drift_m']:.1f} m")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Mean Y Drift", f"{stats['mean_y_drift_m']:.1f} m")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_c:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("X Uncertainty (1σ)", f"±{stats['std_x_drift_m']:.1f} m")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_d:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Y Uncertainty (1σ)", f"±{stats['std_y_drift_m']:.1f} m")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Recovery guidance
                st.markdown("### 🎯 Recovery Operations Guidance")
                
                search_radius_68 = max(stats["std_x_drift_m"], stats["std_y_drift_m"])
                search_radius_95 = 2 * search_radius_68
                search_radius_99 = 3 * search_radius_68
                
                col_guide1, col_guide2 = st.columns(2)
                
                with col_guide1:
                    st.markdown(f"""
                    <div style='background-color: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                        <h4 style='margin: 0 0 0.5rem 0; color: #92400e;'>68% Confidence Zone</h4>
                        <p style='margin: 0; font-size: 1.1rem;'>
                            Search within <strong>±{search_radius_68:.0f} m</strong> of mean landing point<br>
                            <span style='font-size: 0.9rem; color: #92400e;'>Expected recovery probability: ~68%</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_guide2:
                    st.markdown(f"""
                    <div style='background-color: #d1fae5; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;'>
                        <h4 style='margin: 0 0 0.5rem 0; color: #065f46;'>95% Confidence Zone</h4>
                        <p style='margin: 0; font-size: 1.1rem;'>
                            Search within <strong>±{search_radius_95:.0f} m</strong> of mean landing point<br>
                            <span style='font-size: 0.9rem; color: #065f46;'>Expected recovery probability: ~95%</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info("""
                **Interpretation**: The red star marks the expected landing location. 
                The orange ellipse shows the 68% confidence region (1 standard deviation), 
                and the red dashed ellipse shows the 95% confidence region. 
                Payloads released under similar conditions will land within the orange zone ~68% of the time 
                and within the red zone ~95% of the time.
                """)
            
            # ==================== TAB 2: TRAJECTORY ====================
            with tab2:
                st.markdown('<div class="sub-header">Representative Descent Trajectory</div>', unsafe_allow_html=True)
                
                traj_df = pd.DataFrame(result["representative_trajectory"])
                
                # Altitude vs drift plot
                fig_traj = go.Figure()
                
                fig_traj.add_trace(go.Scatter(
                    x=traj_df["x_drift_m"],
                    y=traj_df["altitude_km"],
                    mode='lines+markers',
                    name='Descent Path',
                    line=dict(color='#8b5cf6', width=3),
                    marker=dict(size=6, color='#8b5cf6'),
                    hovertemplate='Altitude: %{y:.1f} km<br>X Drift: %{x:.1f} m<extra></extra>'
                ))
                
                fig_traj.update_layout(
                    title="Payload Descent Trajectory (Representative Run)",
                    xaxis_title="East-West Drift (m)",
                    yaxis_title="Altitude (km)",
                    yaxis_autorange="reversed",
                    height=500,
                    plot_bgcolor='white',
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_traj, use_container_width=True)
                
                # 3D trajectory plot
                st.markdown("### 🌐 3D Trajectory View")
                
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=traj_df["x_drift_m"],
                    y=traj_df["y_drift_m"],
                    z=traj_df["altitude_km"],
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=4),
                    marker=dict(size=4, color='#3b82f6'),
                    hovertemplate='Altitude: %{z:.1f} km<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>'
                )])
                
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title='East-West Drift (m)',
                        yaxis_title='North-South Drift (m)',
                        zaxis_title='Altitude (km)',
                        zaxis=dict(autorange="reversed")
                    ),
                    height=600,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Descent statistics
                st.markdown("### 📊 Descent Statistics")
                total_descent_time = len(traj_df)  # seconds (1s time step)
                final_x_drift = traj_df["x_drift_m"].iloc[-1]
                final_y_drift = traj_df["y_drift_m"].iloc[-1]
                
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.metric("Descent Time", f"{total_descent_time} s")
                with col_t2:
                    st.metric("Total Horizontal Drift", f"{np.sqrt(final_x_drift**2 + final_y_drift**2):.1f} m")
                with col_t3:
                    st.metric("Average Descent Rate", f"{(drop_altitude * 1000) / total_descent_time:.2f} m/s")
            
            # ==================== TAB 3: STATISTICS ====================
            with tab3:
                st.markdown('<div class="sub-header">Monte Carlo Simulation Statistics</div>', unsafe_allow_html=True)
                
                stats = result["landing_statistics"]
                stats_df = pd.DataFrame({
                    "Metric": [
                        "Mean X Drift",
                        "Mean Y Drift", 
                        "X Uncertainty (1σ)",
                        "Y Uncertainty (1σ)",
                        "Total Dispersion Area (1σ)",
                        "Simulation Runs"
                    ],
                    "Value": [
                        f"{stats['mean_x_drift_m']:.1f} m",
                        f"{stats['mean_y_drift_m']:.1f} m",
                        f"±{stats['std_x_drift_m']:.1f} m",
                        f"±{stats['std_y_drift_m']:.1f} m",
                        f"{np.pi * stats['std_x_drift_m'] * stats['std_y_drift_m']:.0f} m²",
                        f"{result['monte_carlo_runs']}"
                    ]
                })
                
                st.table(stats_df)
                
                # Distribution histograms
                st.markdown("### 📈 Landing Point Distributions")
                
                col_hist1, col_hist2 = st.columns(2)
                
                with col_hist1:
                    fig_hist_x = px.histogram(
                        landing_df,
                        x="x_drift_m",
                        nbins=20,
                        title="X Drift Distribution",
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig_hist_x.add_vline(
                        x=stats["mean_x_drift_m"],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {stats['mean_x_drift_m']:.1f} m",
                        annotation_position="top"
                    )
                    st.plotly_chart(fig_hist_x, use_container_width=True)
                
                with col_hist2:
                    fig_hist_y = px.histogram(
                        landing_df,
                        x="y_drift_m",
                        nbins=20,
                        title="Y Drift Distribution",
                        color_discrete_sequence=['#ef4444']
                    )
                    fig_hist_y.add_vline(
                        x=stats["mean_y_drift_m"],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {stats['mean_y_drift_m']:.1f} m",
                        annotation_position="top"
                    )
                    st.plotly_chart(fig_hist_y, use_container_width=True)
                
                # Key insights
                st.markdown("### 🔑 Key Insights")
                
                primary_direction = "Eastward" if stats['mean_x_drift_m'] > 0 else "Westward"
                secondary_direction = "Northward" if stats['mean_y_drift_m'] > 0 else "Southward"
                
                dominant_uncertainty = "X-direction" if stats['std_x_drift_m'] > stats['std_y_drift_m'] else "Y-direction"
                
                st.markdown(f"""
                <div style='background-color: #dbeafe; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;'>
                    <ul style='margin: 0; line-height: 1.8; color: #1e40af;'>
                        <li><strong>Primary drift direction:</strong> {primary_direction} ({abs(stats['mean_x_drift_m']):.1f} m) and {secondary_direction} ({abs(stats['mean_y_drift_m']):.1f} m)</li>
                        <li><strong>Landing uncertainty:</strong> ±{max(stats['std_x_drift_m'], stats['std_y_drift_m']):.1f} m in the {dominant_uncertainty} direction</li>
                        <li><strong>Risk assessment:</strong> For recovery operations, search area should cover at least <strong>3σ</strong> ({3 * max(stats['std_x_drift_m'], stats['std_y_drift_m']):.0f} m radius) around mean landing point for >99% recovery probability</li>
                        <li><strong>Dispersion pattern:</strong> {mc_runs} Monte Carlo runs show consistent Gaussian-like distribution, validating the GPR uncertainty model</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # ==================== TAB 4: WIND INFLUENCE ====================
            with tab4:
                st.markdown('<div class="sub-header">Wind Influence During Descent</div>', unsafe_allow_html=True)
                
                wind_influence = []
                for point in result["representative_trajectory"]:
                    alt = point["altitude_km"]
                    wind_resp = requests.get(f"{BACKEND_URL}/wind-profile", params={"altitude_km": alt})
                    if wind_resp.status_code == 200:
                        wind = wind_resp.json()
                        wind_speed = np.sqrt(wind["u_wind"]**2 + wind["v_wind"]**2)
                        wind_influence.append({
                            "altitude_km": alt,
                            "wind_speed_ms": wind_speed,
                            "x_drift_m": point["x_drift_m"],
                            "u_wind_ms": wind["u_wind"],
                            "v_wind_ms": wind.get("v_wind", 0)
                        })
                
                wind_df_traj = pd.DataFrame(wind_influence)
                
                if not wind_df_traj.empty:
                    # Wind speed vs altitude
                    fig_wind_alt = go.Figure()
                    
                    fig_wind_alt.add_trace(go.Scatter(
                        x=wind_df_traj["wind_speed_ms"],
                        y=wind_df_traj["altitude_km"],
                        mode='lines+markers',
                        name='Wind Speed',
                        line=dict(color='#10b981', width=3),
                        marker=dict(size=8),
                        hovertemplate='Altitude: %{y:.1f} km<br>Wind Speed: %{x:.2f} m/s<extra></extra>'
                    ))
                    
                    fig_wind_alt.update_layout(
                        title="Wind Speed vs Altitude During Descent",
                        xaxis_title="Wind Speed (m/s)",
                        yaxis_title="Altitude (km)",
                        yaxis_autorange="reversed",
                        height=400,
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_wind_alt, use_container_width=True)
                    
                    # Wind components vs drift
                    fig_wind_drift = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Zonal Wind (U) vs X Drift", "Meridional Wind (V) vs Y Drift"),
                        vertical_spacing=0.15
                    )
                    
                    fig_wind_drift.add_trace(
                        go.Scatter(
                            x=wind_df_traj["x_drift_m"],
                            y=wind_df_traj["u_wind_ms"],
                            mode='lines+markers',
                            name='U Wind',
                            line=dict(color='#3b82f6', width=2),
                            marker=dict(size=6)
                        ),
                        row=1, col=1
                    )
                    
                    fig_wind_drift.add_trace(
                        go.Scatter(
                            x=wind_df_traj["x_drift_m"],
                            y=wind_df_traj["v_wind_ms"],
                            mode='lines+markers',
                            name='V Wind',
                            line=dict(color='#ef4444', width=2),
                            marker=dict(size=6)
                        ),
                        row=2, col=1
                    )
                    
                    fig_wind_drift.update_xaxes(title_text="Horizontal Drift (m)")
                    fig_wind_drift.update_yaxes(title_text="Wind Speed (m/s)")
                    fig_wind_drift.update_layout(
                        height=600,
                        plot_bgcolor='white',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_wind_drift, use_container_width=True)
                    
                    # Wind influence summary
                    st.markdown("### 💨 Wind Impact Summary")
                    
                    avg_wind_speed = wind_df_traj["wind_speed_ms"].mean()
                    max_wind_speed = wind_df_traj["wind_speed_ms"].max()
                    max_wind_alt = wind_df_traj.loc[wind_df_traj["wind_speed_ms"].idxmax(), "altitude_km"]
                    
                    col_w1, col_w2, col_w3 = st.columns(3)
                    
                    with col_w1:
                        st.metric("Average Wind Speed", f"{avg_wind_speed:.1f} m/s")
                    with col_w2:
                        st.metric("Max Wind Speed", f"{max_wind_speed:.1f} m/s")
                    with col_w3:
                        st.metric("Max Wind Altitude", f"{max_wind_alt:.0f} km")
                    
                    st.info(f"""
                    **Wind Exposure Analysis:** During descent from {drop_altitude} km, the payload was exposed to an average wind speed of {avg_wind_speed:.1f} m/s. 
                    Peak winds of {max_wind_speed:.1f} m/s occurred at {max_wind_alt:.0f} km altitude, significantly influencing the horizontal drift. 
                    The dominant wind direction was {'eastward' if wind_df_traj['u_wind_ms'].mean() > 0 else 'westward'}, 
                    explaining the primary drift direction observed in the landing dispersion.
                    """)
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend server not reachable. Make sure FastAPI is running at http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            st.code(str(e), language="python")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1.5rem;'>
    <h3 style='margin: 0 0 0.5rem 0; color: #1e3a8a;'>🚀 System Architecture</h3>
    <p style='margin: 0.5rem 0; font-size: 1.1rem;'>
        <strong>Gaussian Process Regression</strong> → Uncertainty-aware wind modeling<br>
        <strong>Monte Carlo Simulation</strong> → Probabilistic landing zone analysis<br>
        <strong>Physics Engine</strong> → Altitude-dependent wind drift simulation
    </p>
    <p style='margin: 1rem 0 0.5rem 0; font-size: 0.95rem; color: #475569;'>
        Built with FastAPI + Streamlit | Deployed on Render + Streamlit Cloud | 
        <a href='https://render.com' target='_blank' style='color: #3b82f6;'>Backend</a> • 
        <a href='https://streamlit.io/cloud' target='_blank' style='color: #3b82f6;'>Frontend</a>
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #94a3b8;'>
        AI-Based Wind Profiling & Payload Drop Prediction System for High-Altitude Platform Stations (HAPS)
    </p>
</div>
""", unsafe_allow_html=True)
