# Project Audit

Audit date: 2026-07-29

## 1. Project purpose

This repository is a science-fair research prototype for exploring synthetic
slime-mold growth on engineered scaffold geometries. The active product is a
Streamlit application with simulation, visualization, statistical comparison,
neural-surrogate optimization, and mesh export.

## 2. Existing architecture

The active UI is the 1,600+ line `Streamlit_App/app/main.py`. Physics and mesh
generation are separated under `Streamlit_App/simulation`; neural, optimization,
statistics, and visualization helpers are under `backend`. State is in the
Streamlit session and there is no server-side persistence or external API.

`Streamlit_App/app/old_app_main.py` is an inactive historical implementation.
The research-plan PDF is source material rather than executable code.

## 3. Current functionality

- Three synthetic scaffold/growth models run locally.
- Metrics and 2D/3D Plotly visualizations are rendered.
- Run history supports statistical views.
- A PyTorch surrogate and global search propose parameters.
- CSV, Excel, JSON, PDF, STL, and ZIP export paths exist.

## 4. Broken functionality found

- The optimization contract contained 18 parameters while `MLP` and application
  metadata defaulted to 17 inputs.
- The UI promised seed reproducibility, but physics and metric noise used
  process-global random calls and ignored that seed.
- Parameter validation allowed physically impossible footprints; geometry then
  silently omitted pillars outside the 200 mm voxel domain.
- Mesh functions raised opaque library errors for empty, undersized, uniform,
  non-finite, or invalid-isosurface inputs.
- The documented nested dependency install omitted `scikit-image` and
  `trimesh`, so documented mesh export setup was incomplete.
- Streamlit width arguments were past their deprecation deadline.

## 5. Missing functionality

The repository had no automated tests, canonical verification command, CI
workflow, audit, completion plan, or operational final-status report.

## 6. Build and runtime problems

The default app loaded and ran in the existing environment, but there was no
repeatable quality gate. Compilation initially attempted to write into ignored
project caches, which is avoided by the new verifier's temporary cache.

## 7. Dependency problems

Two divergent requirement files described different environments. The root
file is now canonical with bounded compatible versions; the nested file
delegates to it.

## 8. Security and scientific-integrity concerns

No credential inputs, hard-coded secrets, network data submission, database, or
upload surface were found. Generated exports are local.

The prior “digital twin” and “AI interpretation” language overstated what the
heuristic simulator can establish. The UI and README now identify results as
synthetic and hypothesis-generating. The tracked research-plan PDF contains
identifying information and must be reviewed before public redistribution.

## 9. Testing gaps

There was no coverage of parameter contracts, reproducibility, model selection,
mesh topology, invalid export inputs, or the primary Streamlit flow.

## 10. Documentation gaps

The README omitted verification, deployment detail, privacy, precise
limitations, repository structure, and accurate input dimensionality. `WARP.md`
also referred to directories and classes that do not exist in the current tree.

## 11. Deployment gaps

Streamlit is the evidenced deployment target, but no hosted deployment could be
verified locally. A GitHub Actions quality gate is appropriate; an automatic
deployment workflow is not supported by repository evidence.

## 12. Accessibility and usability gaps

The default Streamlit controls are keyboard-addressable, but the application
has not received a manual screen-reader or mobile-viewport audit. Invalid
geometry formerly failed indirectly; the active path now gives a clear error.

## 13. Completion definition

The recoverable prototype is complete when one dependency manifest installs the
whole application, fixed-seed runs reproduce, invalid geometry and mesh inputs
fail clearly, core functions and the principal UI flow are tested, documentation
states the scientific boundary, the canonical verifier passes, and tested work
is merged and pushed.

## 14. Prioritized implementation plan

1. Align and validate the 18-feature optimization/model contract.
2. Thread explicit random generators through physics and the UI.
3. Reject impossible scaffold footprints and invalid export surfaces.
4. Add regression, topology, reproducibility, and AppTest coverage.
5. Consolidate dependencies and add local/CI verification.
6. Correct product claims and complete operational documentation.

## 15. Known blockers and assumptions

No blocker prevents local completion. Empirical biological validation, physical
3D printing, and a hosted Streamlit check require external datasets, lab work,
hardware, or infrastructure and are explicitly outside the recovered local
prototype.
