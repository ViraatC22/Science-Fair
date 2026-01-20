# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview
This is a 3D Physarum polycephalum (slime mold) simulation platform for a science fair project. It combines physics-based simulation, neural network surrogate modeling, and genetic algorithm optimization to model slime mold growth on engineered scaffold geometries.

## Commands

### Running the Application
```bash
# From the project root
cd Streamlit_App
streamlit run app/main.py
```

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
cd Streamlit_App
pip install -r requirements.txt
```

### Dependencies
The project uses:
- `streamlit` - Web application framework
- `torch` - PyTorch for neural network surrogate model
- `numpy`, `scipy`, `pandas` - Scientific computing
- `plotly` - Interactive visualizations
- `scikit-learn` - ML utilities

## Code Architecture

### Two-Layer System
1. **Physics-Inspired Simulator**: Generates 3D voxel scaffolds with micropillar arrays, simulates growth fields, and computes biological/transport metrics
2. **Neural Network + Genetic Algorithm**: Learns to predict metrics from parameters, then searches parameter space to optimize scaffold designs

### Directory Structure
- `Streamlit_App/app/main.py` - Main Streamlit application with UI
- `Streamlit_App/simulation/physics.py` - Physics simulation logic, scaffold geometry generation, and metric computation
- `Streamlit_App/neural_network/predictor.py` - PyTorch neural network surrogate model (`SlimePredictor`, `ModelManager`)
- `Streamlit_App/optimization/genetic.py` - Genetic algorithm optimizer that uses the neural network for fast fitness evaluation

### Three Simulation Models
The app supports three distinct simulation paradigms:
1. **2.5D Surface (Pillar Tops)**: Growth constrained to pillar top surfaces
2. **3D Porous (Channel Diffusion)**: Isotropic diffusion through porous channels
3. **3D Structured (Channel Flow)**: Anisotropic flow in structured channel networks

Each model computes different metrics (permeability tensors, tortuosity, fractal dimensions, etc.)

### Key Data Flow
1. User inputs scaffold parameters (pillar count, size, channel width, stiffness) and biological parameters (DMEM components, ions, initial mass)
2. **Manual Mode**: Calls `run_simulation_logic()` → generates scaffold geometry → computes growth field → calculates metrics
3. **Neural Optimization Mode**: 
   - Trains neural network on synthetic data from `compute_metrics()`
   - Runs genetic algorithm using neural network for fast predictions
   - Verifies best candidate with full simulation
4. Results displayed via Plotly 3D visualizations and statistical analysis

### Scaffold Geometry Generation
Scaffold voxel volumes are built in `simulation/physics.py`:
- Base plate + grid of pillars (defined by `pillar_count`, `pillar_size_mm`, `channel_width_mm`)
- Optional channel nodes for 3D models
- Outputs: `scaffold_matrix` (solid/void), `pillar_tops` (2D mask), `channel_matrix` (growth space)

### Metric Computation
Metrics are the core biological/transport outputs:
- Driven by media chemistry (DMEM glucose/glutamine, ions) and environmental factors (light, depth, replenishment)
- Modified by `scaffold_effect` (geometry + mechanics) to ensure scaffold design impacts results
- Model-specific adjustments (porosity, tortuosity, permeability tensors)
- Key metrics: `avg_growth_rate`, `total_network_length`, `permeability_kappa_X/Y/iso`, `mean_tortuosity`, `fractal_dimension`, `mst_ratio`

### Neural Network Architecture
`SlimePredictor` in `neural_network/predictor.py`:
- Feedforward MLP with BatchNorm and optional attention mechanism
- Predicts 5 output metrics from 17 input parameters
- `ModelManager` handles training loop, optimizer/scheduler, and inference
- Trained on synthetic data sampled from the physics simulator's metric functions

### Genetic Algorithm
`GeneticOptimizer` in `optimization/genetic.py`:
- Fitness = z-score of `avg_growth_rate` vs literature data (`LITERATURE_DATA`)
- Uses neural network predictions for fast evaluation when trained
- Returns best parameter set after evolution

### Literature Validation
The app compares simulation outputs to published data:
- `LITERATURE_DATA` in `simulation/physics.py` contains reference values from papers (Kay 2022, Tero 2010, etc.)
- Statistical tests (t-tests, ANOVA) validate model accuracy
- Results displayed in the "Statistical Analysis" tab

## Important Patterns

### Session State Management
Streamlit session state stores:
- `model_manager` - Neural network wrapper
- `optimizer` - Genetic algorithm instance  
- `run_history` - List of all simulation results for statistical analysis
- `latest_result_full` - Most recent simulation result (metrics + params + visualization data)

### Reproducibility
- Seeds controlled via `ModelManager.set_seed()` and stored with each run
- Deterministic mode available for PyTorch operations
- All run parameters logged to `run_history`

### Visualization Data
Results include:
- `growth_data` - 3D numpy array for volumetric/surface visualization
- `scaffold_data` - 3D numpy array of scaffold geometry
- `metrics` - Dictionary of computed metrics
- `params` - Dictionary of input parameters

## Development Notes

### Adding New Metrics
1. Add computation logic to `compute_base_metrics()` or `compute_metrics()` in `simulation/physics.py`
2. Update neural network output dimension if needed (`output_dim` in `ModelManager`)
3. Update genetic algorithm fitness function if the metric should drive optimization
4. Add display logic to appropriate summary function in `app/main.py`

### Adding New Parameters
1. Add UI widgets in sidebar of `app/main.py`
2. Include in `parameters` dictionary passed to `run_simulation_logic()`
3. Update `scaffold_effect` or metric computation to use the new parameter
4. Update neural network input dimension and genetic algorithm parameter bounds

### Modifying Themes
Theme system defined in `_theme_map` dictionary in `app/main.py`. Each theme specifies colors for primary, secondary, accent, backgrounds, and supports dark mode.
