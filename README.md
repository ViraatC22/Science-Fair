# 3D Physarum Simulation Platform

- Streamlit application for simulating slime mold growth across engineered scaffolds and analyzing metrics against literature benchmarks.
- Includes neural network surrogate modeling and a redesigned Export tab for sharing results.

## Quickstart

- Install Python 3.10+ and create a virtual environment.
- Install dependencies:
  - `pip install -r Streamlit_App/requirements.txt`
- Run the app:
  - `python -m streamlit run Streamlit_App/app/main.py`
- Open the app in your browser when prompted (local URL shown in the terminal).

## Core Features

- Simulation models:
  - 2.5D Surface (Pillar Tops)
  - 3D Porous (Channel Diffusion)
  - 3D Structured (Channel Flow)
- Results dashboard:
  - Summary metrics, 3D Plotly visualizations, theory references, and statistical analysis vs literature.
- Neural Network tab:
  - Trains a surrogate model on synthetic data via backend/nn.py.
  - Global search and parameter sensitivity with gradient-based importance.
  - Interactive NN pathway visualization:
    - Layout: Hierarchical or Radial
    - Labels: Auto, Important only, All, None
    - Edge threshold and directional arrows
    - Analysis vs Presentation mode
    - Top-path highlighting and optional activation sign coloring
- Export tab:
  - Formats: PDF, CSV, Excel, JSON
  - Range: Current view, Selected items, Custom index range
  - Destination: Local download (cloud/email configurable in future)
  - Batch export and ZIP compression
  - Metadata inclusion toggle
  - Scheduling UI for future exports
  - Parameter validation and error handling in the UI
  - **3D Model Export (STL):**
    - Generates manifold (watertight) STL files suitable for 3D printing and CAD.
    - Utilizes a "One Matrix" strategy, converting 2.5D simulation data into a solid model with a user-defined floor.
    - Features a **Global Z-Exaggeration** slider (up to 500x) to vertically scale subtle surface details, making them physically apparent in prints.
    - Constructs a solid base under the model to ensure printability.

## Directory Overview

- `Streamlit_App/app/main.py` — Streamlit app entry; UI and tab logic
- `Streamlit_App/simulation/export_utils.py` — Geometry engine for generating 3D printable STL files from 2D heightmaps.
- `Streamlit_App/simulation/physics.py` — Simulation and metric computation
- `backend/nn.py` — Surrogate model (PyTorch), training, and sensitivity
- `backend/opt.py` — Parameter sampling/validation and global search
- `backend/stats.py` — Statistical utilities (normality, t-test/MWU, Cohen’s d, power)
- `backend/vis.py` — Neural network pathway visualization

## Development Notes

- Run the app with headless mode or alternate ports as needed:
  - `python -m streamlit run Streamlit_App/app/main.py --server.port 8601 --server.headless true`
- The Export tab supports local downloads out-of-the-box; cloud/email integrations should be added via environment-based secrets and provider SDKs.
- The NN visualization is optimized for visual clarity and accessibility; tuning controls are available in the tab for dense architectures.

## License

- Educational project use. Please attribute if reused or adapted.
