import numpy as np

ORDER = [
    "pillar_count",
    "pillar_size_mm",
    "channel_width_mm",
    "channel_node_size_mm",
    "scaffold_stiffness_kPa",
    "elasticity",
    "scaffold_density_g_cm3",
    "initial_mass_g",
    "media_depth_mm",
    "replenish_freq_hr",
    "dmem_glucose",
    "dmem_glutamine",
    "dmem_pyruvate",
    "ion_na",
    "ion_k",
    "ion_cl",
    "ion_ca",
    "light_lumens",
]

INTEGER_PARAMS = {"pillar_count", "replenish_freq_hr"}
DOMAIN_SIZE_MM = 200.0

PARAM_RANGES = {
    "pillar_count": (4, 20),
    "pillar_size_mm": (10.0, 50.0),
    "channel_width_mm": (5.0, 30.0),
    "channel_node_size_mm": (0.0, 15.0),
    "scaffold_stiffness_kPa": (1.0, 50.0),
    "elasticity": (0.0, 1.0),
    "scaffold_density_g_cm3": (1.0, 1.5),
    "initial_mass_g": (0.1, 2.0),
    "media_depth_mm": (0.5, 5.0),
    "replenish_freq_hr": (1, 48),
    "dmem_glucose": (0.0, 50.0),
    "dmem_glutamine": (0.0, 50.0),
    "dmem_pyruvate": (0.0, 5.0),
    "ion_na": (100.0, 200.0),
    "ion_k": (1.0, 10.0),
    "ion_cl": (100.0, 200.0),
    "ion_ca": (0.1, 5.0),
    "light_lumens": (0.0, 1000.0),
}

def sample_params(rng):
    for _ in range(1_000):
        p = {}
        for k in ORDER:
            lo, hi = PARAM_RANGES[k]
            if k in INTEGER_PARAMS:
                p[k] = int(rng.integers(int(lo), int(hi)+1))
            else:
                p[k] = float(rng.uniform(lo, hi))
        if validate_params(p)[0]:
            return p
    raise RuntimeError("could not sample a feasible parameter set")

def validate_params(p):
    missing = [key for key in ORDER if key not in p]
    if missing:
        return False, f"missing required parameters: {', '.join(missing)}"

    normalized = {}
    for key in ORDER:
        try:
            value = float(p[key])
        except (TypeError, ValueError):
            return False, f"{key} must be numeric"
        if not np.isfinite(value):
            return False, f"{key} must be finite"
        lo, hi = PARAM_RANGES[key]
        if value < lo or value > hi:
            return False, f"{key} must be between {lo} and {hi}"
        if key in INTEGER_PARAMS and not value.is_integer():
            return False, f"{key} must be an integer"
        normalized[key] = value

    if normalized["channel_node_size_mm"] > normalized["channel_width_mm"]:
        return False, "channel_node_size_mm must be ≤ channel_width_mm"
    footprint = (
        normalized["pillar_count"] * normalized["pillar_size_mm"]
        + (normalized["pillar_count"] + 1) * normalized["channel_width_mm"]
    )
    if footprint > DOMAIN_SIZE_MM:
        return False, f"scaffold footprint ({footprint:.1f} mm) exceeds the {DOMAIN_SIZE_MM:.0f} mm domain"
    return True, "ok"

def to_vector(p):
    ok, message = validate_params(p)
    if not ok:
        raise ValueError(message)
    return np.array([float(p[k]) for k in ORDER], dtype=np.float32)

def predict(model, scaler, X):
    import torch
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        out, _, _ = model(torch.tensor(Xs))
    return out.squeeze().detach().cpu().numpy()

def global_search(model, scaler, rng, n=10000):
    best = None
    for i in range(n):
        q = sample_params(rng)
        ok, _ = validate_params(q)
        if not ok:
            continue
        x = to_vector(q).reshape(1, -1)
        y = float(predict(model, scaler, x))
        if best is None or y > best[0]:
            best = (y, q)
    return best
