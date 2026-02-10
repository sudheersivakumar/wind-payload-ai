# HAPS Wind & Payload Drop AI 🌬️📦

An AI-powered system for High-Altitude Platform Stations (HAPS) to model stratospheric wind fields and predict payload landing zones with uncertainty quantification.

## 🚀 Features

- **AI Wind Modeling**: Uses **Gaussian Process Regression (GPR)** to model stratospheric wind vectors (u, v) from sparse data, providing both mean predictions and uncertainty estimates (confidence intervals).
- **Physics-Based Simulation**: Simulates payload descent trajectories accounting for altitude-dependent wind drift and drag.
- **Monte Carlo Analysis**: Runs probabilistic simulations to generate landing distribution heatmaps and 68%/95% confidence zones for recovery operations.
- **Interactive Dashboard**: Streamlit-based frontend for real-time visualization of wind profiles, trajectories, and landing statistics.
- **REST API**: FastAPI backend for model inference and simulation requests.

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **AI/ML**: Scikit-Learn (Gaussian Process Regression), NumPy, Pandas
- **Deployment**: Render (configuration included)

## 📂 Project Structure

```
wind-payload-ai/
├── backend/
│   ├── main.py          # FastAPI application entry point
│   ├── wind_model.py    # GPR model definition and training
│   ├── simulator.py     # Monte Carlo payload drop simulation logic
│   ├── physics.py       # Physics equations for trajectory calculation
│   └── utils.py         # Helper functions
├── frontend/
│   └── app.py           # Streamlit dashboard
├── data/
│   └── wind_sample.csv  # Sample wind data for training/testing
├── models/              # Directory for saving trained models (optional)
├── render.yaml          # Render deployment configuration
└── requirement.txt      # Python dependencies
```

## ⚡ Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/wind-payload-ai.git
    cd wind-payload-ai
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

3.  **Run the Backend (API)**:
    ```bash
    uvicorn backend.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`. API docs at `http://127.0.0.1:8000/docs`.

4.  **Run the Frontend (Dashboard)**:
    In a new terminal:
    ```bash
    streamlit run frontend/app.py
    ```
     The dashboard will open in your browser at `http://localhost:8501`.

## 🌐 Deployment

This project includes a `render.yaml` for easy deployment on [Render](https://render.com).

1.  Push your code to a GitHub repository.
2.  Connect your repo to Render.
3.  The `render.yaml` will automatically configure a Web Service for the backend.
4.  (Optional) Deploy the Streamlit frontend separately or serve it alongside the backend.

## 📊 Methodology

1.  **Wind Profiling**: We treat wind components ($u$, $v$) as continuous functions of altitude using Gaussian Processes ($f(z) \sim \mathcal{GP}$). This allows us to interpolate winds at any altitude and quantify uncertainty ($\sigma$) in regions with sparse data.
2.  **Trajectory Integration**:
    $\frac{dx}{dt}$ = $u_{wind}(z)$ + $\epsilon_u, \quad \frac{dy}{dt}$ = $v_{wind}(z)$ + $\epsilon_v, \quad \frac{dz}{dt}$ = $-v_{descent}$
    
    Where $\epsilon$ represents stochastic perturbations drawn from the GPR uncertainty estimate.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.
