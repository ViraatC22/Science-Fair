# 3D Physarum Simulation Platform

An exploratory Streamlit application for studying how scaffold geometry and
environmental parameters could influence synthetic *Physarum polycephalum*
growth fields. It combines three simulation modes, statistical comparisons, a
PyTorch surrogate model, parameter search, interactive Plotly views, and
watertight STL export.

## Status

The recovered research prototype has a reproducible local workflow, guarded
parameter and mesh boundaries, 13 automated tests, a Streamlit application
smoke test, and GitHub Actions verification.

The outputs are synthetic and hypothesis-generating. They are not experimental
measurements, a validated biological digital twin, or a substitute for wet-lab
verification.

## Capabilities

- Simulate 2.5D pillar-top, 3D porous-diffusion, and 3D structured-flow models.
- Reproduce a run by reusing its simulation seed.
- Compare synthetic metrics across run history and literature reference values.
- Train an 18-input PyTorch surrogate and search feasible scaffold parameters.
- Inspect network pathways, sensitivities, and optimization distributions.
- Export tabular results and watertight STL/ZIP bundles for CAD workflows.

## Architecture

- `Streamlit_App/app/main.py` owns the Streamlit UI and session workflow.
- `Streamlit_App/simulation/physics.py` builds scaffold grids, synthetic growth
  fields, and model-specific metrics.
- `Streamlit_App/simulation/export_utils.py` validates and converts heightmaps
  or voxel fields into watertight meshes.
- `backend/opt.py` defines the 18-feature parameter contract and search helpers.
- `backend/nn.py` trains the surrogate model.
- `backend/stats.py` and `backend/vis.py` provide analysis and visualization.

The physics layer uses a 60-voxel grid representing a 200 mm domain. Parameter
validation rejects combinations whose full pillar/channel footprint cannot fit
that domain instead of silently clipping the requested structure.

## Prerequisites

- Python 3.12 or newer (local recovery was verified with Python 3.14)
- A virtual environment with enough space for PyTorch and the scientific stack

No environment variables, accounts, API keys, database, or external service are
required.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run Streamlit_App/app/main.py
```

The compatibility file at `Streamlit_App/requirements.txt` delegates to the
canonical root dependency manifest, so either documented install location
includes the mesh-export dependencies.

## Verify

With the virtual environment active:

```bash
PYTHON=python ./scripts/verify.sh
```

The verifier checks installed dependency compatibility, compiles all Python
sources into a temporary cache, runs unit and mesh-export tests, and executes
the default Streamlit simulation through Streamlit AppTest.

## Typical workflow

1. Choose a model and parameters in the sidebar.
2. Keep the default seed or enter a seed you want to record.
3. Run the simulation and inspect the dashboard.
4. Build run history before interpreting the statistical views.
5. Use neural optimization only within its validated parameter domain.
6. Export tabular data or a mesh bundle as needed.

Reuse both the parameters and seed to reproduce a synthetic result. Neural
training is deterministic for a fixed seed on the same supported runtime, but
floating-point results can vary slightly across hardware and PyTorch versions.

## Deployment

For Streamlit Community Cloud, select `Streamlit_App/app/main.py` as the entry
point. The service should discover the root `requirements.txt`. Do not add
secrets; this application has no credential requirement.

Before sharing a public deployment, review the included research-plan PDF for
personal information and confirm that every cited or bundled asset may be
redistributed.

## Repository structure

```text
.
├── backend/                 # Optimization, neural model, statistics, charts
├── Streamlit_App/
│   ├── app/main.py          # Active application
│   └── simulation/          # Physics and mesh export
├── tests/                   # Unit, export, reproducibility, and app smoke tests
├── scripts/verify.sh        # Canonical local verification
└── docs/                    # Recovery audit, plan, and final status
```

`Streamlit_App/app/old_app_main.py` is retained as a historical implementation
snapshot. It is not the application entry point.

## Known limitations

- Metric relationships are heuristic and have not been calibrated against the
  original experimental dataset.
- Statistical comparisons describe synthetic runs; they do not establish
  biological validity or causality.
- The neural model learns from the same synthetic equations and therefore
  cannot independently validate those equations.
- Mesh decimation depends on an optional backend; export safely skips
  decimation if that backend is unavailable.
- Real CAD import and physical-print quality require manual validation.

## Security and privacy

The application runs locally and does not transmit simulation data. Generated
exports are ignored by Git. The tracked research-plan PDF contains identifying
information in its filename and may contain additional personal information;
review it before making a fork or deployment public.

## Repository and license

Repository: [ViraatC22/Science-Fair](https://github.com/ViraatC22/Science-Fair)

No open-source license file is currently provided. Unless the repository owner
states otherwise, no permission to copy, modify, or redistribute the project is
granted.
