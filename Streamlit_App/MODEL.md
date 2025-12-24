# Model Overview (Concise)

This app has two layers:

1. A **physics-inspired simulator** that generates a scaffold volume, a growth field, and derived metrics.
2. A **neural-network surrogate + genetic algorithm (GA)** that searches scaffold/media parameters efficiently, then verifies the best candidate with the simulator.

## 1) Physics-Inspired Simulation (how outputs are generated)

### 1.1 Scaffold geometry (micropillar array)
The scaffold is built as a 3D voxel volume in `simulation_logic.py:15` (`generate_micropillar_geometry`):

- A solid **base plate** is created first.
- A grid of **pillars** is added using `pillar_count`, `pillar_size_mm`, and `channel_width_mm`.
- Optional **channel nodes** are added using `channel_node_size_mm`.
- The final outputs are:
  - `scaffold_matrix` (solid = 1, void = 0)
  - `pillar_tops` (2D “surface” mask for the 2.5D mode)
  - `channel_matrix` (void space where growth can occur)

### 1.2 Growth field generation
The simulator generates a synthetic “growth density” field in `simulation_logic.py:213` (`run_simulation_logic`):

- **2.5D Surface (Pillar Tops)**: combines `pillar_tops` with an anisotropic 2D fiber field (`_make_fibers2d`) to create surface growth.
- **3D Porous / 3D Structured**: combines a 3D fiber field (`_make_fibers3d`) with a smooth random field and masks it by `channel_matrix` so growth appears mainly in void regions.

The growth field is used for visualization; the reported metrics come from the metric functions below.

### 1.3 Metrics (the “biology/transport” outputs)
Core metrics are computed in:

- `simulation_logic.py:101` (`compute_base_metrics`)
- `simulation_logic.py:149` (`compute_metrics`)

Key ideas:

- **Media chemistry + environment** drives a baseline: DMEM (glucose/glutamine), calcium, light, etc.
- **Scaffold significance** is enforced via a multiplicative `scaffold_effect` that depends on geometry + mechanics + environment (pillar/channel sizes, stiffness, density, depth, replenishment frequency). This is designed to be strong enough to consistently move `avg_growth_rate` and `total_network_length` across runs, with reduced random noise.
- Model-specific adjustments (e.g., porosity `phi`) further shape 3D metrics like tortuosity and permeability.

## 2) Neural Network Surrogate (how the “AI model” is made)

### 2.1 Architecture
The surrogate is a feedforward network in `ml_modules.py`:

- `SlimePredictor` (`ml_modules.py:21`) is an MLP with:
  - BatchNorm on inputs
  - Optional simple self-attention (`AttentionBlock`, `ml_modules.py:9`)
  - Hidden layers controlled from the UI
  - Dropout for regularization
- It predicts multiple metrics at once (default 5 outputs).

`ModelManager` (`ml_modules.py:77`) wraps the model, optimizer, scheduler, training loop, and inference.

### 2.2 Training data (how it learns)
Training uses synthetic data generated from the simulator’s metric logic (not from the 3D voxel growth field):

- `train_initial_model` in `optimization_logic.py:171` samples random parameters within bounds
- For each sample, it calls `compute_metrics` to get target outputs
- It trains the neural network with MSE loss (`ModelManager.train`, `ml_modules.py:183`)

This makes the network a fast approximation of the metric function.

## 3) Genetic Algorithm Optimization (how “best scaffold” is found)

The GA lives in `optimization_logic.py`:

- Parameter bounds and encoding/decoding: `GeneticOptimizer` (`optimization_logic.py:6`)
- Fitness uses a **significance score vs literature**:
  - Computes a z-score for `avg_growth_rate` vs `LITERATURE_DATA` (`optimization_logic.py:45`)
  - Fitness is the z-score (higher is better), with basic validity penalties
- Evaluation:
  - If the neural net is trained, GA uses `ModelManager.predict` to score individuals quickly (`optimization_logic.py:62`)
  - If not trained, it falls back to `compute_metrics`
- After GA finishes, the app runs a full simulation once to verify the best candidate (`app_main.py:1205`).

## 4) Reproducibility

- Seeds are handled through `ModelManager.set_seed` (`ml_modules.py:152`) and are stored with each run in the run history (`app_main.py:1140` and `app_main.py:1210`).

## 5) Where to look in the UI

- The main app is `app_main.py`.
- “Manual Simulation” runs the simulator directly.
- “Neural Optimization” trains the surrogate, then runs GA, then verifies the best scaffold with the simulator.

