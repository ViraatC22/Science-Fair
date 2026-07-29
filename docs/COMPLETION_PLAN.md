# Completion Plan

Updated: 2026-07-29

| ID | Task | Why | Files/modules | Dependencies | Acceptance criteria | Verification | Status | Commit |
|---|---|---|---|---|---|---|---|---|
| M0-01 | Reconcile repository and establish recovery branch | Protect clean upstream state | Git metadata | None | Clean `automation/project-recovery` starts at `45edcb9` | `git status --short --branch` | COMPLETED | pending recovery commit |
| M1-01 | Consolidate runtime dependencies | Documented install omitted mesh packages | `requirements.txt`, `Streamlit_App/requirements.txt` | Existing environment | One canonical manifest covers every import | `python -m pip check` | COMPLETED | pending recovery commit |
| M2-01 | Align the neural feature contract | Default network accepted 17 of 18 defined inputs | `backend/nn.py`, `backend/opt.py`, app | M1-01 | Model and vectors use 18 features; malformed data is rejected | `tests.test_optimization` | COMPLETED | pending recovery commit |
| M2-02 | Make simulation seeds effective | Reproducibility controls previously did not govern physics noise | physics and app | M2-01 | Same parameters/seed produce identical field and metrics | `tests.test_simulation` | COMPLETED | pending recovery commit |
| M2-03 | Enforce the physical domain | Large requested grids were silently clipped | optimization and physics | M2-01 | Footprints over 200 mm fail with a clear message | `tests.test_optimization`, `tests.test_simulation` | COMPLETED | pending recovery commit |
| M3-01 | Harden mesh export boundaries | Invalid arrays raised opaque downstream errors | export utilities | M1-01 | Invalid inputs fail clearly; representative meshes are watertight | `tests.test_export` | COMPLETED | pending recovery commit |
| M4-01 | Verify the primary UI flow | Compilation alone does not prove the app runs | app and tests | M2-02 | Default simulation completes with no Streamlit exceptions | `tests.test_app` | COMPLETED | pending recovery commit |
| M5-01 | Add canonical verification and CI | No repeatable project health gate existed | script, workflow | M1-01 through M4-01 | Dependency, compile, unit, mesh, and app checks run together | `PYTHON=python ./scripts/verify.sh` | COMPLETED | pending recovery commit |
| M6-01 | Synchronize product and engineering documentation | Setup and scientific claims were incomplete or overstated | README, audit, plan, `WARP.md` | Prior milestones | Commands and limitations match the tested code | Manual doc review | IN_PROGRESS | pending |
| M7-01 | Final verification, merge, push, and handoff | Produce a recoverable completed project state | all | M0-M6 | Clean main, passing checks, pushed commit, final report | Canonical verify plus Git checks | NOT_STARTED | pending |

## Deferred with reason

- Empirical calibration and wet-lab validation: `DEFERRED_WITH_REASON` because
  the necessary experimental dataset and lab work are not present.
- Hosted Streamlit and physical print validation: `DEFERRED_WITH_REASON`
  because external infrastructure and hardware are required.
