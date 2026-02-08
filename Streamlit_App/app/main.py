import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import time
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json
import sys
import os
import torch
import io
import zipfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add parent directory to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from simulation.physics import run_simulation_logic, LITERATURE_DATA, compute_metrics
from simulation.export_utils import generate_improved_mesh, generate_heightmap_mesh, export_to_bundle, calculate_voxel_size
try:
    from backend.nn import train_surrogate, gradient_sensitivity
except ModuleNotFoundError:
    import importlib.util
    nn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'nn.py'))
    nn_spec = importlib.util.spec_from_file_location("backend_nn", nn_path)
    backend_nn = importlib.util.module_from_spec(nn_spec) 
    nn_spec.loader.exec_module(backend_nn)
    train_surrogate = backend_nn.train_surrogate
    gradient_sensitivity = backend_nn.gradient_sensitivity
try:
    from backend.opt import sample_params as opt_sample_params, validate_params as opt_validate_params, to_vector as opt_to_vector, global_search as opt_global_search
except ModuleNotFoundError:
    import importlib.util
    opt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'opt.py'))
    opt_spec = importlib.util.spec_from_file_location("backend_opt", opt_path)
    backend_opt = importlib.util.module_from_spec(opt_spec)
    opt_spec.loader.exec_module(backend_opt)
    opt_sample_params = backend_opt.sample_params
    opt_validate_params = backend_opt.validate_params
    opt_to_vector = backend_opt.to_vector
    opt_global_search = backend_opt.global_search
    OPT_ORDER = backend_opt.ORDER
    OPT_PARAM_RANGES = backend_opt.PARAM_RANGES
if 'OPT_ORDER' not in globals():
    try:
        from backend.opt import ORDER as OPT_ORDER, PARAM_RANGES as OPT_PARAM_RANGES
    except Exception:
        pass
try:
    from backend.stats import normality as stats_normality, fit_gaussian as stats_fit_gaussian, ci_mean as stats_ci_mean, compare as stats_compare, power as stats_power
except ModuleNotFoundError:
    import importlib.util
    stats_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'stats.py'))
    stats_spec = importlib.util.spec_from_file_location("backend_stats", stats_path)
    backend_stats = importlib.util.module_from_spec(stats_spec)
    stats_spec.loader.exec_module(backend_stats)
    stats_normality = backend_stats.normality
    stats_fit_gaussian = backend_stats.fit_gaussian
    stats_ci_mean = backend_stats.ci_mean
    stats_compare = backend_stats.compare
    stats_power = backend_stats.power
try:
    import backend.vis as vis
    import importlib
    importlib.reload(vis)
except ModuleNotFoundError:
    import importlib.util
    vis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'vis.py'))
    vis_spec = importlib.util.spec_from_file_location("backend_vis", vis_path)
    backend_vis = importlib.util.module_from_spec(vis_spec)
    vis_spec.loader.exec_module(backend_vis)
    vis = backend_vis

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide", 
    page_title="3D Slime Mold Simulation",
    page_icon="🦠"
)

# Set the default plotly template
PLOTLY_TEMPLATE = "plotly_white"

# --- LOCAL ML/OPT HELPERS ---
PARAM_DISPLAY_NAMES = {
    "pillar_count": "Grid Density (Pillars)",
    "pillar_size_mm": "Pillar Diameter (mm)",
    "channel_width_mm": "Channel Width (mm)",
    "channel_node_size_mm": "Node Size (mm)",
    "scaffold_stiffness_kPa": "Stiffness (kPa)",
    "elasticity": "Elasticity (0-1)",
    "scaffold_density_g_cm3": "Scaffold Density (g/cm³)",
    "initial_mass_g": "Initial Biomass (g)",
    "media_depth_mm": "Media Depth (mm)",
    "replenish_freq_hr": "Feed Frequency (hrs)",
    "dmem_glucose": "Glucose (mM)",
    "dmem_glutamine": "Glutamine (mM)",
    "dmem_pyruvate": "Pyruvate (mM)",
    "ion_na": "Sodium (Na+)",
    "ion_k": "Potassium (K+)",
    "ion_cl": "Chloride (Cl-)",
    "ion_ca": "Calcium (Ca2+)",
    "light_lumens": "Light Intensity (lm)"
}

class ModelManager:
    def __init__(self, input_dim=17, output_dim=1):
        self.input_dim = input_dim
        self.output_dim = output_dim

class GeneticOptimizer:
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
    def run_optimization(self, model_type, pop_size=30, generations=10):
        rng = np.random.default_rng(42)
        best = None
        tries = 0
        n_search = 4000
        while tries < n_search:
            p = opt_sample_params(rng)
            p["model_type"] = model_type
            ok, _ = opt_validate_params(p)
            if ok:
                r = run_metrics_only(p, model_type)
                y = r["metrics"]["avg_growth_rate"]
                if best is None or y > best[0]:
                    best = (y, p)
            tries += 1
        return best[1], best[0], []

def train_initial_model(model_manager, n_samples=500):
    rng = np.random.default_rng(42)
    X_list = []
    y_list = []
    tries = 0
    while len(y_list) < n_samples and tries < n_samples * 10:
        p = opt_sample_params(rng)
        p["model_type"] = "3D Structured (Channel Flow)"
        ok, _ = opt_validate_params(p)
        if ok:
            r = run_metrics_only(p, p["model_type"])
            y = r["metrics"]["avg_growth_rate"]
            X_list.append(opt_to_vector(p))
            y_list.append(y)
        tries += 1
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    m, sc, _ = train_surrogate(X, y, seed=42, epochs=50, snapshot_stride=10)
    Xs = sc.transform(X).astype(np.float32)
    with torch.no_grad():
        pred, _, _ = m(torch.tensor(Xs))
        loss = torch.nn.functional.mse_loss(pred, torch.tensor(y))
    return float(loss.detach().cpu().numpy())

def run_metrics_only(params, model_type):
    m = compute_metrics(params, model_type)
    return {"params": params, "model_type": model_type, "metrics": m}

# --- INITIALIZE MODULES IN SESSION STATE ---
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager(input_dim=17, output_dim=5)

if 'optimizer' not in st.session_state:
    st.session_state.optimizer = GeneticOptimizer(st.session_state.model_manager)

# --- PLOT GENERATORS ---

# --- SUMMARY DISPLAYS ---
def display_structured_flow_summary(metrics, params):
    st.markdown("#### 📋 Input Parameters")
    st.markdown(f"""
    <div class="info-box" style="background: var(--surface); border-color: var(--accent);">
    <b>Pillars:</b> {params['pillar_count']}x{params['pillar_count']} | 
    <b>Pillar Size:</b> {params['pillar_size_mm']} mm | 
    <b>Channel Width:</b> {params['channel_width_mm']} mm | 
    <b>Node Size:</b> {params['channel_node_size_mm']} mm
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🔬 Key Metrics (Anisotropic Flow)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Porosity (ϕ)", f"{metrics['porosity_phi']:.2f}")
    c2.metric("Permeability (κ_x)", f"{metrics['permeability_kappa_X']:.1e} m²")
    c3.metric("Permeability (κ_y)", f"{metrics['permeability_kappa_Y']:.1e} m²")
    c4, c5, c6 = st.columns(3)
    c4.metric("Total Length", f"{metrics['total_network_length']:.1f} mm")
    c5.metric("Avg. Growth Rate", f"{metrics['avg_growth_rate']:.1f} mm/hr")
    c6.metric("Junctions", f"{metrics['num_junctions']}")
    
    st.divider()
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown(f"**🧠 AI Interpretation:** This structured grid creates anisotropic flow, with Kx ({metrics['permeability_kappa_X']:.1e}) and Ky ({metrics['permeability_kappa_Y']:.1e}) dominating transport.")
    st.markdown('</div>', unsafe_allow_html=True)

def display_porous_diffusion_summary(metrics, params):
    st.markdown("#### 📋 Input Parameters")
    st.markdown(f"""
    <div class="info-box" style="background: var(--surface); border-color: var(--accent);">
    <b>Pillars:</b> {params['pillar_count']}x{params['pillar_count']} | 
    <b>Pillar Size:</b> {params['pillar_size_mm']} mm | 
    <b>Channel Width:</b> {params['channel_width_mm']} mm | 
    <b>Node Size:</b> {params['channel_node_size_mm']} mm
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔬 Key Metrics (Isotropic Diffusion)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Porosity (ϕ)", f"{metrics['porosity_phi']:.2f}")
    c2.metric("Isotropic Permeability (κ)", f"{metrics['permeability_kappa_iso']:.1e} m²")
    c3.metric("Effective Modulus (E_eff)", f"{metrics['Eeff']:.1f} kPa")
    c4, c5, c6 = st.columns(3)
    c4.metric("Total Length", f"{metrics['total_network_length']:.1f} mm")
    c5.metric("Avg. Growth Rate", f"{metrics['avg_growth_rate']:.1f} mm/hr")
    c6.metric("Junctions", f"{metrics['num_junctions']}")
    
    st.divider()
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown(f"**🧠 AI Interpretation:** This geometry, modeled with diffusion, shows strong isotropic transport (κ ≈ {metrics['permeability_kappa_iso']:.1e}). Tortuosity is moderate ({metrics['mean_tortuosity']:.2f}).")
    st.markdown('</div>', unsafe_allow_html=True)

def display_pillar_top_summary(metrics, params):
    st.markdown("#### 📋 Input Parameters")
    st.markdown(f"""
    <div class="info-box" style="background: var(--surface); border-color: var(--accent);">
    <b>Pillars:</b> {params['pillar_count']}x{params['pillar_count']} | 
    <b>Pillar Size:</b> {params['pillar_size_mm']} mm | 
    <b>Channel Width:</b> {params['channel_width_mm']} mm
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🔬 Key Metrics (2D Surface)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Length", f"{metrics['total_network_length']:.1f} mm")
    c2.metric("Avg. Growth Rate", f"{metrics['avg_growth_rate']:.1f} mm/hr")
    c3.metric("Junctions", f"{metrics['num_junctions']}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Coverage", f"{metrics['coverage_fraction']:.1%}") 
    c5.metric("Fractal Dimension (Df)", f"{metrics['fractal_dimension']:.2f}")
    c6.metric("Pillar Adhesion", f"{metrics['pillar_adhesion_index']:.2f}")

    st.divider()
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown(f"**🧠 AI Interpretation:** Growth is confined to the 2D pillar tops, showing strong adhesion ({metrics['pillar_adhesion_index']:.2f}) and a high fractal dimension ({metrics['fractal_dimension']:.2f}).")
    st.markdown('</div>', unsafe_allow_html=True)
    
# --- HEX TO RGBA for CSS ---
def hex_to_rgba(hex, opacity):
    hex = hex.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

# --- THEME SELECTION ---
_theme_map = {
    "Ocean (Blue/Teal)": {"primary": "#1f77b4", "secondary": "#2c3e50", "accent": "#00b894", "bg": "#f7f9fc", "text": "#2c3e50", "surface": "#ffffff", "surface_border": "#e6edf5", "sidebar": "#fbfcfe", "dark": False},
    "Forest (Green)": {"primary": "#2ecc71", "secondary": "#1b5e20", "accent": "#27ae60", "bg": "#f6fbf6", "text": "#1b4332", "surface": "#ffffff", "surface_border": "#e4efe7", "sidebar": "#f7fbf8", "dark": False},
    "Sunset (Orange/Purple)": {"primary": "#ff7f0e", "secondary": "#5b2c6f", "accent": "#c0392b", "bg": "#fff8f2", "text": "#2d2a32", "surface": "#ffffff", "surface_border": "#f3e6dc", "sidebar": "#fffaf5", "dark": False},
    "Monochrome (Gray)": {"primary": "#6c757d", "secondary": "#343a40", "accent": "#868e96", "bg": "#f8f9fa", "text": "#343a40", "surface": "#ffffff", "surface_border": "#e9ecef", "sidebar": "#f8f9fa", "dark": False},
    "Dark Slate": {"primary": "#3b82f6", "secondary": "#e5e7eb", "accent": "#22c55e", "bg": "#0b1220", "text": "#e5e7eb", "surface": "#111827", "surface_border": "#1f2937", "sidebar": "#0f172a", "dark": True}
}

# --- Initialize Session State ---
if 'run_history' not in st.session_state:
    st.session_state.run_history = []

# --- SIDEBAR CONTENT (NOW WITH ALL PARAMS) ---
with st.sidebar:
    st.header("🔬 Simulation Configuration")
    
    color_scheme = st.selectbox("🎨 Color Scheme", _theme_map.keys())
    
    with st.expander("📐 Model Selection", expanded=True):
        model_type = st.radio("Choose Simulation Model", 
                              ["2.5D Surface (Pillar Tops)", 
                               "3D Porous (Channel Diffusion)", 
                               "3D Structured (Channel Flow)"])
    
    st.divider()
    st.subheader("⚙️ Scaffold Parameters (from CAD)")
    
    # --- NEW CAD-BASED PARAMS ---
    pillar_count = st.slider("Pillar Count (N x N)", 4, 20, 4, 1)
    pillar_size_mm = st.slider("Pillar Size (mm)", 10.0, 50.0, 30.0, 1.0)
    channel_width_mm = st.slider("Channel Width (mm)", 5.0, 30.0, 15.0, 0.5)
    
    # Conditional parameter for channel nodes
    if "3D" in model_type:
        channel_node_size_mm = st.slider("Channel Node Size (mm)", 0.0, 15.0, 5.0, 0.5)
    else:
        channel_node_size_mm = 0.0
    
    # --- SHARED PARAMS ---
    scaffold_stiffness_kPa = st.slider("Stiffness (Young's Modulus, kPa)", 1.0, 50.0, 25.0, 0.5)
    elasticity = st.slider("Elasticity (Unitless)", 0.0, 1.0, 0.5, 0.05) 
    scaffold_density_g_cm3 = st.slider("Scaffold Density (g/cm³)", 1.0, 1.5, 1.2, 0.01)
    
    st.divider()
    st.subheader("🧬 Biological Parameters")
    
    initial_mass_g = st.slider("Initial Slime Mold Mass (g)", 0.1, 2.0, 0.5, 0.1)
    dev_stage = st.selectbox("Developmental Stage", ["Active Plasmodium", "Sclerotium (Reactivated)", "Spore (Germinated)"])
    
    st.divider()
    st.subheader("🧪 Nutrient & Medium")
    
    nutrient_config = st.selectbox("Nutrient Configuration", ["Even Distribution", "Concentrated Pockets"])
    media_depth_mm = st.slider("Medium Depth (mm)", 0.5, 5.0, 2.0, 0.1)
    replenish_freq_hr = st.slider("Replenishment Frequency (hours)", 1, 48, 24, 1) 

    with st.expander("🧪 DMEM Components (mM)"): 
        dmem_glucose = st.number_input("Glucose", 0.0, 50.0, 25.0, 1.0)
        dmem_glutamine = st.number_input("Glutamine", 0.0, 50.0, 45.0, 1.0) 
        dmem_pyruvate = st.number_input("Sodium Pyruvate", 0.0, 5.0, 1.0, 0.1) 

    with st.expander("🧪 Ion Concentrations (mM)"): 
        ion_na = st.number_input("Na+", 100.0, 200.0, 154.0, 1.0)
        ion_k = st.number_input("K+", 1.0, 10.0, 5.4, 0.1)
        ion_cl = st.number_input("Cl-", 100.0, 200.0, 140.0, 1.0)
        ion_ca = st.number_input("Ca2+", 0.1, 5.0, 1.8, 0.1)

    st.divider()
    st.subheader("💡 Optional Parameters")
    
    with st.expander("💡 Light Exposure"): 
        light_type = st.selectbox("Light Type", ["None (Dark)", "Visible (White)", "Red (660nm)", "Blue (470nm)", "UV", "IR"])
        light_lumens = st.slider("Light Intensity (Lumens)", 0, 1000, 0, 50)

    with st.expander("🔬 Hydrogel Ratio (Optional)"): 
        gel_g = st.number_input("Gelatin (g)", 0.0, 10.0, 3.0, 0.1)
        dmem_ml = st.number_input("DMEM (ml)", 50.0, 200.0, 100.0, 1.0)
        alg_g = st.number_input("Sodium Alginate (g)", 0.0, 5.0, 1.0, 0.1)
        cacl_g = st.number_input("Calcium Chloride (g)", 0.0, 2.0, 0.3, 0.05)
    
    st.divider()
    run_button = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

# --- THEME INJECTION ---
_t = _theme_map.get(color_scheme, _theme_map["Ocean (Blue/Teal)"])
PLOTLY_TEMPLATE = "plotly_dark" if _t["dark"] else "plotly_white"
st.markdown(f"""
<style>
:root {{
    --primary: {_t['primary']}; --secondary: {_t['secondary']}; --accent: {_t['accent']};
    --bg: {_t['bg']}; --text: {_t['text']}; --surface: {_t['surface']};
    --surface-border: {_t['surface_border']}; --sidebar-bg: {_t['sidebar']};
}}
body {{ background-color: var(--bg); color: var(--text); }}
.stApp {{ background-color: var(--bg); }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }}
h1, h2, h3 {{ color: var(--text); }}
h1 {{ font-weight: 700; letter-spacing: 0.2px; }}
h2 {{ font-weight: 600; border-bottom: 2px solid var(--primary); padding-bottom: .4rem; margin-top: 1.25rem; }}
h3 {{ font-weight: 600; margin-top: .75rem; }}
section[data-testid="stSidebar"] {{ background-color: var(--sidebar-bg); border-right: 1px solid {_t['surface_border']}; }}
.streamlit-expanderHeader {{ background-color: {_t['surface']}; border: 1px solid {_t['surface_border']}; border-radius: 6px; font-weight: 600; color: var(--text); }}
div[data-testid="metric-container"] {{ background: var(--surface); border: 1px solid var(--surface-border); padding: 14px 16px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04); }}
.info-box {{ background: linear-gradient(135deg, {_t['surface']} 0%, {_t['bg']} 100%); border-left: 4px solid var(--primary); padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 10px; border: 1px solid var(--surface-border); }}
.success-box {{ background: linear-gradient(135deg, {hex_to_rgba(_t['accent'], 0.05)} 0%, {_t['bg']} 100%); border-left: 4px solid var(--accent); padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 10px; border: 1px solid var(--surface-border); }}
.hero {{ background: linear-gradient(100deg, {hex_to_rgba(_t['primary'], 0.12)}, {hex_to_rgba(_t['accent'], 0.12)}); border: 1px solid var(--surface-border); border-radius: 16px; padding: 22px 24px; margin-bottom: 16px; }}
.hero h1 {{ margin: 0 0 6px 0; color: var(--secondary); }}
.hero p {{ margin: 0; color: var(--text); opacity: 0.8; }}
.stButton>button {{ background: linear-gradient(180deg, {hex_to_rgba(_t['primary'], 0.9)}, {_t['primary']}); color: #fff; border: 1px solid {hex_to_rgba(_t['primary'], 0.9)}; border-radius: 10px; padding: 10px 16px; box-shadow: 0 6px 12px {hex_to_rgba(_t['primary'], 0.18)}; transition: transform .08s ease-in-out, box-shadow .08s ease-in-out; }}
.stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px {hex_to_rgba(_t['primary'], 0.22)}; }}
.stButton>button:focus {{ outline: none; }}
[data-testid="stTabs"] button[role="tab"] {{ background: {_t['surface']}; border: 1px solid var(--surface-border) !important; border-bottom-color: transparent !important; border-radius: 10px 10px 0 0; margin-right: 6px; color: var(--text); opacity: 0.7; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ background: var(--surface); color: var(--primary); opacity: 1; border-bottom: 2px solid var(--primary) !important; margin-bottom: -2px; }}
[data-testid="stTabs"] div[role="tablist"] {{ padding-bottom: 0; border-bottom: 2px solid var(--surface-border); }}
.plotly .legend {{ border-radius: 8px; border: 1px solid var(--surface-border); background-color: var(--surface); }}
/* Radio button styling for navigation */
div[role="radiogroup"] {{ display: flex; flex-direction: row; overflow-x: auto; gap: 8px; padding-bottom: 5px; border-bottom: 1px solid var(--surface-border); }}
div[role="radiogroup"] label {{ flex: 0 0 auto; background-color: var(--surface); border: 1px solid var(--surface-border); padding: 8px 16px; border-radius: 8px; transition: all 0.2s ease; margin-right: 0 !important; }}
div[role="radiogroup"] label:hover {{ border-color: var(--primary); color: var(--primary); }}
div[role="radiogroup"] label[data-checked="true"] {{ background-color: var(--primary) !important; border-color: var(--primary) !important; color: white !important; }}
div[role="radiogroup"] label[data-checked="true"] p {{ color: white !important; }}
</style>
""", unsafe_allow_html=True)

# --- MAIN PAGE LAYOUT ---
st.markdown(f"""
<div class="hero">
  <h1>🦠 3D Physarum Simulation Platform</h1>
  <p>An advanced digital twin for modeling slime mold growth on engineered scaffolds.</p>
  <p style="margin-top:8px;color:{_t['secondary']}; opacity: 0.7;">Configure your full experimental query in the sidebar and run the simulation.</p>
</div>
""", unsafe_allow_html=True)
st.divider()

if not run_button and not st.session_state.run_history:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 👋 Welcome to the Slime Mold Simulation Platform
    This tool simulates *Physarum polycephalum* growth on various scaffold geometries based on your research parameters.
    
    **To get started:**
    1.  Select a **Model Type** from the sidebar (e.g., "3D Structured").
    2.  Configure the full set of **Scaffold, Biological, and Nutrient** parameters.
    3.  Click **"▶️ Run Simulation"** to generate the results dashboard.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if run_button:
    parameters = {
        "model_type": model_type,
        "pillar_count": pillar_count,
        "pillar_size_mm": pillar_size_mm,
        "channel_width_mm": channel_width_mm,
        "channel_node_size_mm": channel_node_size_mm,
        "scaffold_stiffness_kPa": scaffold_stiffness_kPa,
        "elasticity": elasticity,
        "scaffold_density_g_cm3": scaffold_density_g_cm3,
        "initial_mass_g": initial_mass_g,
        "dev_stage": dev_stage,
        "nutrient_config": nutrient_config,
        "media_depth_mm": media_depth_mm,
        "replenish_freq_hr": replenish_freq_hr,
        "dmem_glucose": dmem_glucose,
        "dmem_glutamine": dmem_glutamine,
        "dmem_pyruvate": dmem_pyruvate,
        "ion_na": ion_na, "ion_k": ion_k, "ion_cl": ion_cl, "ion_ca": ion_ca,
        "light_type": light_type,
        "light_lumens": light_lumens,
        "gel_ratio": f"{gel_g}g Gel : {dmem_ml}ml DMEM : {alg_g}g Alg : {cacl_g}g CaCl2"
    }
    
    with st.spinner("Running simulation... (Computing specialized metrics...)"):
        new_result = run_simulation_logic(parameters, model_type)
    
    st.session_state.run_history.append(new_result['metrics']) # Append metrics to history
    st.session_state.latest_result_full = new_result # Store full result for viz
    
    st.markdown('<div class="success-box">✅ Simulation complete! The results dashboard is now available below.</div>', unsafe_allow_html=True)

# --- RESULTS DASHBOARD ---
if 'latest_result_full' in st.session_state:
    
    results = st.session_state.latest_result_full
    metrics = results['metrics']
    params = results['params']
    model_type = results['model_type']
    
    st.header(f"📊 Results Dashboard: {model_type}")
    
    # Navigation
    tabs = ["📈 Summary & Visualization", "🔬 Advanced Diagnostics", "📈 Statistical Analysis", "🧠 Neural Network", "Export"]
    
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = tabs[0]
        
    selected_tab = st.radio(
        "Navigation", 
        tabs, 
        horizontal=True, 
        label_visibility="collapsed",
        key="nav_selection"
    )

    if selected_tab == tabs[0]:
        st.markdown("## Summary & 3D Visualization")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### 📊 Key Metrics")
            if model_type == "3D Structured (Channel Flow)":
                display_structured_flow_summary(metrics, params)
            elif model_type == "3D Porous (Channel Diffusion)":
                display_porous_diffusion_summary(metrics, params)
            elif model_type == "2.5D Surface (Pillar Tops)":
                display_pillar_top_summary(metrics, params)
        with col2:
            st.markdown("### 🎨 Interactive 3D Visualization")
            growth_data = results['growth_data']
            scaffold_data = results.get('scaffold_data')
            fig = go.Figure()
            
            if model_type == "2.5D Surface (Pillar Tops)":
                fig.add_trace(go.Surface(
                    z=growth_data, colorscale='Greens',
                    colorbar=dict(title='Density'), name="Organism/Cells"
                ))
                fig.update_layout(title="2.5D Surface Growth (On Pillar Tops)", height=600, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="var(--text)"), margin=dict(l=10,r=10,t=60,b=10), scene_camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)))
            else:
                x, y, z = np.mgrid[:growth_data.shape[0], :growth_data.shape[1], :growth_data.shape[2]]
                if scaffold_data is not None:
                    # --- VISUALIZATION FIX ---
                    # Increased opacity to 0.2 to make sure it's visible
                    fig.add_trace(go.Isosurface(
                        x=x.flatten(), y=y.flatten(), z=z.flatten(),
                        value=scaffold_data.flatten(),
                        isomin=0.5, isomax=1.0, 
                        opacity=0.2, # <-- FIX: Was 0.15, now 0.2
                        colorscale='Blues', 
                        showscale=False, name="Collagen/Scaffold"
                    ))
                fig.add_trace(go.Isosurface(
                    x=x.flatten(), y=y.flatten(), z=z.flatten(),
                    value=growth_data.flatten(),
                    isomin=0.3, isomax=0.8, opacity=0.8, colorscale='Greens',
                    colorbar=dict(title='Density'), name="Organism/Cells"
                ))
                fig.update_layout(
                    title="3D Volumetric Growth (in Channels)", height=600, template=PLOTLY_TEMPLATE, 
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="var(--text)"), margin=dict(l=10,r=10,t=60,b=10),
                    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z (Depth)"),
                    legend=dict(bgcolor="var(--surface)", bordercolor="var(--surface-border)", borderwidth=1)
                )
            st.plotly_chart(fig, use_container_width=True, key="viz_main_fig")

    if selected_tab == tabs[1]:
        st.markdown(f"## Advanced Diagnostics: {model_type}")
        st.markdown("*Detailed analytical visualizations tracking evolutionary trends across runs.*")
        st.divider()
        
        # Prepare data
        run_df = pd.DataFrame(st.session_state.run_history) if 'run_history' in st.session_state else pd.DataFrame()
        
        if run_df.empty:
            st.info("Run at least one simulation to view diagnostics.")
        else:
            # Controls
            col_ctrl1, col_ctrl2 = st.columns([3, 1])
            with col_ctrl2:
                normalize = st.toggle("Normalize Metrics (0-1)", key="adv_norm_toggle")

            # Layout: 2x2 Grid of Key Topological Metrics
            col1, col2 = st.columns(2)
            
            # 1. Network Length (Structural - Blue)
            with col1:
                # Calculate baseline from first run if available
                base_len = run_df["total_network_length"].iloc[0] if not run_df.empty else 0
                ref_band = (base_len * 0.98, base_len * 1.02) if base_len > 0 else None
                
                fig1 = vis.draw_evolution_plot(
                    run_df, "total_network_length", "Network Length", 
                    subtitle="Total biological mass and reach of the organism.",
                    unit="mm",
                    color="#1f77b4", ref_band=ref_band, ref_name="Baseline ±2%",
                    normalize=normalize, template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig1, use_container_width=True)

            # 2. Junctions (Structural - Blue)
            with col2:
                fig2 = vis.draw_evolution_plot(
                    run_df, "num_junctions", "Connectivity (Junctions)", 
                    subtitle="Number of branching points (nodes) in the network.",
                    unit="count",
                    color="#1f77b4", ref_band=None, 
                    normalize=normalize, template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)

            # 3. Fractal Dimension (Complexity - Green)
            with col3:
                if "fractal_dimension" in run_df.columns:
                    fig3 = vis.draw_evolution_plot(
                        run_df, "fractal_dimension", "Fractal Dimension", 
                        subtitle="Complexity of the pattern (1.0 = Line, 2.0 = Plane).",
                        unit="Df",
                        color="#2ca02c", ref_band=(1.3, 1.7), ref_name="Biological Complexity",
                        normalize=normalize, template=PLOTLY_TEMPLATE
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Fractal dimension not available for this model.")

            # 4. Tortuosity/Efficiency (Transport Cost - Red)
            with col4:
                if "mean_tortuosity" in run_df.columns:
                    fig4 = vis.draw_evolution_plot(
                        run_df, "mean_tortuosity", "Transport Efficiency (Tortuosity)", 
                        subtitle="Path efficiency. 1.0 is a straight line (Optimal).",
                        unit="τ",
                        color="#d62728", ref_band=(1.0, 1.3), ref_name="Optimal Transport",
                        normalize=normalize, template=PLOTLY_TEMPLATE
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("Tortuosity not available for this model.")
        
        st.divider()

        st.markdown(f"## Governing Equations & Theory: {model_type}")
        st.markdown("*Mathematical foundations and derived metrics*")
        st.divider()
        
        if model_type == "3D Structured (Channel Flow)":
            # --- Calculations for Extra Metrics ---
            # Re: Characteristic length = channel width. Velocity ~ 10 um/s (creeping flow)
            # Pe: L*v/D. D ~ 2e-9 m2/s.
            # Dh: Hydraulic diameter
            L_char = params['channel_width_mm'] * 1e-3
            h_char = params['media_depth_mm'] * 1e-3
            v_char = 1e-5 # m/s
            rho = 1000 # kg/m3
            mu = 1e-3 # Pa.s
            D_free = 2e-9 # m2/s
            
            # Hydraulic Diameter Dh = 2wh/(w+h)
            Dh = (2 * L_char * h_char) / (L_char + h_char + 1e-9)
            
            Re = (rho * v_char * Dh) / mu
            Pe = (v_char * Dh) / D_free
            Shear = (6 * mu * v_char) / Dh # Parallel plate approximation
            
            # Pressure Drop (Hagen-Poiseuille) over 10mm
            L_chip = 10e-3 
            DeltaP = (12 * mu * L_chip * v_char) / (Dh**2 + 1e-12)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌊 Fluid Transport (Anisotropic)")
                st.latex(r"\text{Navier-Stokes: } \rho (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \mu \nabla^2 \mathbf{u}")
                st.latex(r"\text{Hydraulic Dia: } D_h = \frac{4A}{P} \approx \frac{2wh}{w+h}")
                st.latex(r"\text{Reynolds Number: } Re = \frac{\rho v D_h}{\mu}")
                st.latex(r"\text{Shear Stress: } \tau_w = \mu \frac{\partial u}{\partial y} \approx \frac{6\mu v}{D_h}")

            with col2:
                st.markdown("### 🧪 Mass Transport")
                st.latex(r"\frac{\partial C}{\partial t} + \mathbf{u}\cdot\nabla C = D_{\rm eff}\nabla^2 C - r(C)")
                st.latex(r"\text{Peclet Number: } Pe = \frac{v D_h}{D} = \frac{\text{Advection}}{\text{Diffusion}}")
                st.latex(r"\text{Pressure Drop: } \Delta P = \frac{12 \mu L v}{D_h^2}")
                st.latex(r"\text{Darcy Velocity: } \mathbf{u} = -\frac{\mathbf{K}}{\mu}\nabla p")
            st.divider()
            st.markdown("### 📊 Calculated Parameters")
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("Porosity (φ)", f"{metrics['porosity_phi']:.3f}")
            pcol1.metric("Reynolds No. (Re)", f"{Re:.2e}")
            
            pcol2.metric("Permeability (K_x)", f"{metrics['permeability_kappa_X']:.2e} m²")
            pcol2.metric("Peclet No. (Pe)", f"{Pe:.1f}")
            
            pcol3.metric("Hydraulic Dia (Dh)", f"{Dh*1000:.2f} mm")
            pcol3.metric("Shear Stress", f"{Shear:.2e} Pa")
            
            pcol4.metric("Pressure Drop", f"{DeltaP:.2f} Pa")
            pcol4.metric("Keff", f"{metrics['Keff']:.2f} kPa")

        elif model_type == "3D Porous (Channel Diffusion)":
            # --- Calculations ---
            # Thiele Modulus: phi = L * sqrt(k/Deff). k ~ 0.01 1/s
            # Effectiveness: eta = tanh(phi)/phi
            L_char = params['pillar_size_mm'] * 1e-3 # Characteristic pore size approx
            k_react = 0.01 
            Deff = metrics['Deff']
            phi_thiele = L_char * np.sqrt(k_react / (Deff + 1e-12))
            eta = np.tanh(phi_thiele) / (phi_thiele + 1e-6)
            
            # Specific Surface Area Sv = 6(1-phi)/d_p
            phi_void = metrics['porosity_phi']
            Sv_real = (6 * (1 - phi_void)) / (L_char + 1e-9)
            
            # Archie's Law m calculation: Deff/D0 = phi^m -> m = log(Deff/D0)/log(phi)
            D0 = 2e-9
            if phi_void > 0 and phi_void < 1:
                m_archie = np.log((Deff + 1e-15)/D0) / np.log(phi_void)
            else:
                m_archie = 1.0

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌊 Porous Media Transport")
                st.latex(r"\text{Kozeny-Carman: } \kappa \approx \frac{\phi^3}{C(1-\phi)^2 S_v^2}")
                st.latex(r"\text{Specific Surface: } S_v = \frac{surface}{volume} \approx \frac{6(1-\phi)}{d_p}")
                st.latex(r"\text{Thiele Modulus: } \phi_{Th} = L \sqrt{\frac{k}{D_{eff}}}")
                st.latex(r"\text{Effectiveness Factor: } \eta = \frac{\tanh(\phi_{Th})}{\phi_{Th}}")
            with col2:
                st.markdown("### 🧪 Advection-Diffusion")
                st.latex(r"\frac{\partial C}{\partial t} = \nabla\cdot(D_{\rm eff}\nabla C) - k C^n")
                st.latex(r"\text{Archie's Law: } D_{eff} = D_0 \phi^m")
                st.latex(r"\text{Damkohler Number: } Da = \frac{k L^2}{D_{eff}} = \phi_{Th}^2")
                st.latex(r"\text{Percolation: } P_\infty \sim (\phi-\phi_c)^\beta")
            st.divider()
            st.markdown("### 📊 Calculated Parameters")
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("Porosity (φ)", f"{metrics['porosity_phi']:.3f}")
            pcol1.metric("Thiele Modulus", f"{phi_thiele:.2f}")
            
            pcol2.metric("Permeability (κ)", f"{metrics['permeability_kappa_iso']:.2e} m²")
            pcol2.metric("Effectiveness (η)", f"{eta:.2f}")
            
            pcol3.metric("Archie's Exp (m)", f"{m_archie:.2f}")
            pcol3.metric("Spec. Surface (Sv)", f"{Sv_real:.1e} 1/m")
            
            pcol4.metric("Deff", f"{metrics['Deff']:.2e} m²/s")
            pcol4.metric("Eeff", f"{metrics['Eeff']:.2f} kPa")

        elif model_type == "2.5D Surface (Pillar Tops)":
            # --- Calculations ---
            # Diffusion length Ld = sqrt(4Dt). t=24h
            # Schmidt Sc = nu / D. nu = 1e-6 m2/s
            t_sec = 24 * 3600
            D_surf = 1e-10 # Slower on surface
            L_diff = np.sqrt(4 * D_surf * t_sec) * 1000 # to mm
            MSD = 4 * D_surf * t_sec * 1e6 # mm^2
            
            nu = 1e-6
            Sc = nu / D_surf
            
            # Deborah Number
            tau_growth = metrics['time_to_connection'] * 60 # seconds
            De = tau_growth / t_sec

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🧪 Surface Diffusion")
                st.latex(r"\frac{\partial C}{\partial t} = D_s \nabla^2 C - r(C)")
                st.latex(r"\text{Mean Sq. Disp: } \text{MSD} = \langle r^2 \rangle = 4 D t")
                st.latex(r"\text{Diffusion Length: } L_D = \sqrt{4 D t}")
                st.latex(r"\text{Schmidt Number: } Sc = \frac{\nu}{D} = \frac{\text{Viscous}}{\text{Diffusive}}")
            with col2:
                st.markdown("### 🗺️ Network Topology")
                st.latex(r"\text{Tortuosity: } \tau = \frac{L_{\text{path}}}{\| \mathbf{x}_{\text{end}} - \mathbf{x}_{\text{start}}\|}")
                st.latex(r"\text{Fractal Dimension: } D_f = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}")
                st.latex(r"\text{Deborah Number: } De = \frac{\tau_{growth}}{t_{obs}}")
                st.latex(r"\text{Coverage Rate: } R_c = \frac{dA}{dt} \approx k_{spread} (1-A)")
            st.divider()
            st.markdown("### 📊 Calculated Parameters")
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("Fractal Dim. (Df)", f"{metrics['fractal_dimension']:.3f}")
            pcol1.metric("Diffusion Len", f"{L_diff:.1f} mm")
            
            pcol2.metric("Tortuosity (τ)", f"{metrics['mean_tortuosity']:.3f}")
            pcol2.metric("Schmidt No. (Sc)", f"{Sc:.1e}")
            
            pcol3.metric("MSD (24h)", f"{MSD:.2f} mm²")
            pcol3.metric("Deborah No. (De)", f"{De:.2e}")
            
            pcol4.metric("Growth Rate", f"{metrics['avg_growth_rate']:.2f} mm/h")
            pcol4.metric("Coverage Rate", f"{metrics['coverage_fraction']:.2f}")

    # --- STATISTICAL ANALYSIS TAB ---
    if selected_tab == tabs[2]:
        st.markdown("## 📈 Statistical Analysis vs. Literature")
        st.markdown("*Compare your simulation runs against published data.*")
        st.divider()
        
        run_df = pd.DataFrame(st.session_state.run_history)
        
        if len(run_df) < 2:
            st.markdown('<div class="info-box">Run at least 2 simulations to perform statistical analysis.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"### Analysis based on {len(run_df)} simulation runs")
            
            st.markdown("#### 1. One-Sample t-tests (Your Model vs. Literature)")
            st.markdown("This test checks if your model's average output is *statistically different* from a known value.")
            
            lit_col1, lit_col2 = st.columns(2)
            
            # T-test vs. AutoAnalysis_TotalLength
            with lit_col1:
                lit_mean = LITERATURE_DATA["AutoAnalysis_TotalLength"]["mean"]
                sim_data = run_df['total_network_length'].dropna()
                t_stat, p_val = stats.ttest_1samp(sim_data, lit_mean)
                
                st.metric(f"Total Network Length (vs. {lit_mean} mm)", f"{sim_data.mean():.1f} mm", f"p-val: {p_val:.3f}")
                if p_val < 0.05:
                    st.error(f"**Significant Difference:** Your model's mean length ({sim_data.mean():.1f}) is statistically different from the literature value ({lit_mean}).")
                else:
                    st.success(f"**Good Match:** Your model's mean length ({sim_data.mean():.1f}) is *not* statistically different from the literature value ({lit_mean}).")
            
            # T-test vs. Tero_2010_MST_Ratio
            with lit_col2:
                lit_mean_tero = LITERATURE_DATA["Tero_2010_MST_Ratio"]["mean"]
                sim_data_tero = run_df['mst_ratio'].dropna()
                t_stat_tero, p_val_tero = stats.ttest_1samp(sim_data_tero, lit_mean_tero)
                
                st.metric(f"MST Ratio (vs. {lit_mean_tero})", f"{sim_data_tero.mean():.2f}", f"p-val: {p_val_tero:.3f}")
                if p_val_tero < 0.05:
                    st.error(f"**Significant Difference:** Your model's mean MST ratio ({sim_data_tero.mean():.2f}) is statistically different from Tero et al. ({lit_mean_tero}).")
                else:
                    st.success(f"**Good Match:** Your model's mean MST ratio ({sim_data_tero.mean():.2f}) is *not* statistically different from Tero et al. ({lit_mean_tero}).")

            st.divider()
            
            st.markdown("#### 2. ANOVA (Analysis of Variance)")
            st.markdown("This test checks if changing an *input parameter* (like Model Type) had a significant effect on an *output metric* (like Growth Rate).")
            
            # ANOVA on Model Type vs. Growth Rate
            model_groups = run_df.groupby('param_model_type')['avg_growth_rate'].apply(list)
            
            if len(model_groups) > 1:
                f_val, p_val_anova = stats.f_oneway(*model_groups)
                st.metric("ANOVA: Model Type vs. Growth Rate", f"p-val: {p_val_anova:.3f}")
                if p_val_anova < 0.05:
                    st.success("**Significant Effect:** The 'Model Type' you chose *does* have a statistically significant effect on the 'Average Growth Rate'.")
                else:
                    st.error("**No Significant Effect:** The 'Model Type' you chose does *not* have a statistically significant effect on the 'Average Growth Rate'.")
            else:
                st.info("Run simulations with different 'Model Types' to enable this ANOVA test.")

            st.divider()

        st.markdown("#### Note on Statistical Tests")
        st.markdown(f"""
        <div class="info-box" style="background: var(--surface); border-color: var(--accent);">
        You asked about **Chi-Square, t-tests, and ANOVA**. Here's how they're used:
        <ul>
            <li><b>t-tests</b> and <b>ANOVA</b> (which we use here) are perfect for comparing continuous data (like <i>length, rate, or time</i>) to see if there is a significant difference between group averages.</li>
            <li>A <b>Chi-Square</b> test is used for categorical data (like <i>counts, frequencies, or proportions</i>). For example, "Did Scaffold A have a 50% success rate and Scaffold B have a 65% success rate? Is that difference significant?"</li>
        </ul>
        Since your metrics are all continuous, we are using t-tests and ANOVA.
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 📜 Full Run History")
        st.dataframe(run_df)
        if st.button("Clear Run History"):
            st.session_state.run_history = []
            st.rerun()

    if selected_tab == tabs[3]:
        st.markdown("## 🧠 Neural Network")
        nn_col1 = st.container()
        with nn_col1:
            with st.expander("🎯 Surrogate Modeling", expanded=True):
                st.info("Train a Neural Network to predict metrics instantly, bypassing the simulation steps.")
                if st.button("Train Surrogate Model (PyTorch)", key="nn_ml_train"):
                    with st.spinner("Training Neural Network on synthetic data..."):
                        loss = train_initial_model(st.session_state.model_manager, n_samples=500)
                    st.success(f"✓ Model trained (Final Loss: {loss:.4f})")
                    st.session_state.model_trained = True
            
            with st.expander("🧠 AI-Driven Parameter Optimization", expanded=True):
                st.markdown("""
                Use the Neural Network to find the optimal scaffold parameters for maximizing slime mold growth.
                This process explores thousands of combinations instantly to find the best configuration.
                """)
                
                # --- Advanced Settings (Hidden by default) ---
                with st.expander("⚙️ Advanced Training Settings", expanded=False):
                    st.caption("Fine-tune how the AI learns and explores the parameter space.")
                    c1, c2 = st.columns(2)
                    with c1:
                        seed = st.number_input("Simulation Seed (Reproducibility)", 0, 1000000, 42, 1, key="nn_opt_seed", help="Controls the random number generator. Using the same seed ensures you get the exact same results every time.")
                    with c2:
                        n_search = st.number_input("Exploration Depth (Scenarios)", 1000, 50000, 4000, 1000, key="nn_opt_n_search", help="How many different parameter combinations the AI will test to find the best one. More scenarios = better results but slower.")
                    
                    colA, colB, colC = st.columns(3)
                    with colA:
                        epochs = st.number_input("AI Learning Cycles (Epochs)", 50, 1000, 150, 50, key="nn_opt_epochs", help="How many times the AI reviews the data to learn the patterns. More cycles = smarter AI, but takes longer.")
                    with colB:
                        snapshot_stride = st.number_input("Visual Update Rate", 1, 50, 10, 1, key="nn_opt_stride", help="How often (in cycles) to update the 3D brain visualization below.")
                    with colC:
                        connections_shown = st.number_input("Network Wiring Complexity", 50, 1000, 200, 50, key="nn_opt_conn", help="Limit the number of connections shown in the 3D visualization to keep it readable.")
                
                run_opt = st.button("🚀 Find Optimal Parameters", type="primary", key="nn_opt_run")
                
                if run_opt:
                    rng = np.random.default_rng(int(seed))
                    recs = []
                    X_list = []
                    y_list = []
                    tries = 0
                    while len(recs) < 150 and tries < 10000:
                        p = opt_sample_params(rng)
                        p["model_type"] = model_type
                        ok, msg = opt_validate_params(p)
                        if ok:
                            r = run_metrics_only(p, model_type)
                            y = r["metrics"]["avg_growth_rate"]
                            recs.append({"params": p, "metrics": r["metrics"], "y": y})
                            X_list.append(opt_to_vector(p))
                            y_list.append(y)
                        tries += 1
                    X = np.vstack(X_list).astype(np.float32)
                    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
                    m, sc, snaps = train_surrogate(X, y, seed=int(seed), epochs=int(epochs), snapshot_stride=int(snapshot_stride))
                    st.session_state.opt_baseline = y.squeeze()
                    st.session_state.opt_model = m
                    st.session_state.opt_scaler = sc
                    st.session_state.opt_snaps = snaps
                    best = opt_global_search(m, sc, rng, n=int(n_search))
                    best[1]["model_type"] = model_type
                    st.session_state.opt_candidate = best[1]
                    cand = []
                    for i in range(20):
                        rr = run_metrics_only(best[1], model_type)
                        cand.append(rr["metrics"]["avg_growth_rate"])
                    st.session_state.opt_candidate_samples = np.array(cand, dtype=np.float32)
                
                if "opt_candidate" in st.session_state:
                    st.divider()
                    st.markdown("### 📊 Optimization Results")
                    
                    base = st.session_state.opt_baseline
                    cand = st.session_state.opt_candidate_samples
                    
                    # Statistical Comparison
                    p_val_cand, d_cand = stats_compare(cand, base)
                    
                    # Interpretation Helper Strings
                    p_interp = "(Significant)" if p_val_cand < 0.05 else "(Not Sig.)"
                    d_interp = "Small"
                    if d_cand > 0.5: d_interp = "Medium"
                    if d_cand > 0.8: d_interp = "Large"
                    if d_cand > 1.2: d_interp = "Very Large"
                    
                    # Metrics Row
                    m1, m2, m3 = st.columns(3)
                    
                    # 1. Growth Gain
                    gain_pct = (cand.mean() - base.mean()) / base.mean() * 100
                    m1.metric("Projected Growth Gain", f"{gain_pct:+.1f}%", 
                              help="Relative increase in growth rate compared to the baseline random parameters.")
                    
                    # 2. P-Value
                    p_display = f"{p_val_cand:.2e}" if p_val_cand < 0.001 else f"{p_val_cand:.4f}"
                    m2.metric("Statistical Significance (p-value)", f"{p_display}", 
                              delta=p_interp, delta_color="normal" if p_val_cand < 0.05 else "off",
                              help="The probability that the observed improvement occurred by chance. p < 0.05 is standard for significance.")
                    
                    # 3. Cohen's d
                    m3.metric("Effect Size (Cohen's d)", f"{d_cand:.2f}", 
                              delta=d_interp, delta_color="normal",
                              help="A standardized measure of difference. d=0.2 is small, d=0.5 is medium, d=0.8 is large.")
                    
                    # Plot
                    mu, sigma = stats_fit_gaussian(base)
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=base, name="Baseline (Random)", opacity=0.6, marker_color="#6c757d"))
                    fig.add_trace(go.Histogram(x=cand, name="AI Optimized Model", opacity=0.8, marker_color="#2ecc71"))
                    fig.update_layout(
                        barmode="overlay", 
                        title="Growth Rate Distribution: Baseline vs. Optimized",
                        xaxis_title="Growth Rate (mm/hr)",
                        yaxis_title="Frequency",
                        template=PLOTLY_TEMPLATE,
                        legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig, use_container_width=True, key="nn_candidate_hist")
                    
                    # --- NEW: Feature Importance ---
                    st.divider()
                    st.markdown("### 🧠 AI Insights: What drives growth?")
                    st.markdown("The neural network analyzed which parameters had the biggest impact on the simulation outcome.")
                    
                    show_normalized = st.toggle("Show Normalized Impact (Relative %)", value=False, key="nn_imp_norm_toggle")
                    
                    if "opt_model" in st.session_state and "opt_candidate" in st.session_state:
                        # Calculate sensitivity for the optimal parameters
                        vec = opt_to_vector(st.session_state.opt_candidate)
                        # We need to reshape vec to (1, -1) inside gradient_sensitivity or here
                        # The function in nn.py handles reshape: xs = scaler.transform(x.reshape(1, -1))
                        # But wait, opt_to_vector returns a 1D array.
                        
                        sens = gradient_sensitivity(st.session_state.opt_model, st.session_state.opt_scaler, vec)
                        
                        if show_normalized:
                            sens = sens / (sens.max() + 1e-9) * 100
                        
                        # Create a DataFrame for plotting (Restored)
                        sens_df = pd.DataFrame({
                            "Parameter": [PARAM_DISPLAY_NAMES.get(k, k) for k in OPT_ORDER],
                            "Importance": sens
                        })

                        # --- 2. Create Group Mapping ---
                        # Define groups based on parameter keys
                        GROUPS = {
                            "pillar_count": "Geometry / Structure",
                            "pillar_size_mm": "Geometry / Structure",
                            "channel_width_mm": "Geometry / Structure",
                            "channel_node_size_mm": "Geometry / Structure",
                            "media_depth_mm": "Geometry / Structure",
                            
                            "scaffold_stiffness_kPa": "Material / Mechanics",
                            "elasticity": "Material / Mechanics",
                            "scaffold_density_g_cm3": "Material / Mechanics",
                            "initial_mass_g": "Material / Mechanics",
                            
                            "replenish_freq_hr": "Media / Transport",
                            "dmem_glucose": "Media / Transport",
                            "dmem_glutamine": "Media / Transport",
                            "dmem_pyruvate": "Media / Transport",
                            
                            "ion_na": "Ions / Environment",
                            "ion_k": "Ions / Environment",
                            "ion_cl": "Ions / Environment",
                            "ion_ca": "Ions / Environment",
                            "light_lumens": "Ions / Environment"
                        }
                        
                        # Add group column
                        sens_df["Group"] = [GROUPS.get(k, "Other") for k in OPT_ORDER]
                        
                        # --- 3. Sorting & Ranking ---
                        # Sort descending by importance for the chart (biggest at top)
                        sens_df = sens_df.sort_values(by="Importance", ascending=True) # Ascending for horizontal bar (bottom to top)
                        
                        # Calculate relative importance (0-1) for styling
                        max_imp = sens_df["Importance"].max()
                        sens_df["RelImp"] = sens_df["Importance"] / max_imp
                        
                        # --- 4. Styling Logic (Opacity & Color) ---
                        # Color mapping for groups
                        GROUP_COLORS = {
                            "Geometry / Structure": "#1f77b4", # Blue
                            "Material / Mechanics": "#2ca02c", # Green
                            "Media / Transport": "#ff7f0e",    # Orange
                            "Ions / Environment": "#9467bd"    # Purple
                        }
                        
                        # Generate colors with opacity based on importance
                        # Top 3 -> 1.0 opacity, Mid -> 0.7, Low -> 0.4
                        # We need to rank them descending first to determine top 3
                        rank_df = sens_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
                        top_3_idx = rank_df.index[:3]
                        
                        colors = []
                        opacities = []
                        text_labels = []
                        
                        for idx, row in sens_df.iterrows():
                            # Determine rank (0 is highest)
                            rank = rank_df[rank_df["Parameter"] == row["Parameter"]].index[0]
                            
                            # Opacity logic
                            if rank < 3:
                                op = 1.0
                                txt = f"{row['Importance']:.1f}%" if show_normalized else f"{row['Importance']:.2f}"
                            elif rank < 8:
                                op = 0.85
                                txt = ""
                            else:
                                op = 0.65
                                txt = ""
                                
                            c_base = GROUP_COLORS.get(row["Group"], "#7f7f7f")
                            colors.append(c_base)
                            opacities.append(op)
                            text_labels.append(txt)
                            
                        # --- 5. Build Chart (Grouped Traces) ---
                        fig_imp = go.Figure()
                        
                        # We must preserve the sorted order of parameters on the Y-axis
                        y_order = rank_df["Parameter"].iloc[::-1].tolist()
                        
                        # Iterate through groups to create separate traces for the legend
                        for g_name, g_color in GROUP_COLORS.items():
                            # Filter data for this group
                            g_df = sens_df[sens_df["Group"] == g_name]
                            
                            if g_df.empty:
                                continue
                                
                            # Calculate opacities and text for this subset
                            g_opacities = []
                            g_texts = []
                            
                            for _, row in g_df.iterrows():
                                # Determine rank based on global dataframe
                                rank = rank_df[rank_df["Parameter"] == row["Parameter"]].index[0]
                                if rank < 3:
                                    g_opacities.append(1.0)
                                    g_texts.append(f"{row['Importance']:.1f}%" if show_normalized else f"{row['Importance']:.2f}")
                                elif rank < 8:
                                    g_opacities.append(0.85)
                                    g_texts.append("")
                                else:
                                    g_opacities.append(0.65)
                                    g_texts.append("")

                            fig_imp.add_trace(go.Bar(
                                x=g_df["Importance"],
                                y=g_df["Parameter"],
                                name=g_name,
                                orientation='h',
                                marker=dict(color=g_color, opacity=g_opacities),
                                text=g_texts,
                                textposition='outside',
                                hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<br>Group: " + g_name + "<extra></extra>"
                            ))
                        
                        # --- 6. Layout Improvements ---
                        fig_imp.update_layout(
                            title=dict(
                                text="Parameter Sensitivity Analysis",
                                y=0.95,
                                x=0.0,
                                xanchor='left',
                                yanchor='top'
                            ),
                            # Subtitle via annotation
                            annotations=[
                                dict(
                                    x=0.0,
                                    y=1.08,
                                    xref='paper',
                                    yref='paper',
                                    text="Higher values indicate stronger influence on predicted growth rate",
                                    showarrow=False,
                                    font=dict(size=12, color="gray"),
                                    xanchor='left'
                                ),
                                # "Primary Driver" Callout - Adjusted to avoid overlap
                                dict(
                                    x=rank_df.iloc[0]["Importance"],
                                    y=rank_df.iloc[0]["Parameter"],
                                    xref='x',
                                    yref='y',
                                    text="Primary Driver",
                                    showarrow=True,
                                    arrowhead=2,
                                    ax=80, # Increased distance to clear the value label
                                    ay=0,
                                    standoff=30, # Stop arrow short so it doesn't cross the text
                                    xanchor="left",
                                    font=dict(color=GROUP_COLORS.get(rank_df.iloc[0]["Group"], "white"))
                                )
                            ],
                            xaxis=dict(
                                title=None, 
                                showgrid=True,
                                gridcolor='rgba(128,128,128,0.2)',
                                zeroline=False
                            ),
                            yaxis=dict(
                                title=None,
                                tickfont=dict(size=11),
                                categoryorder='array', # Enforce sorted order
                                categoryarray=y_order
                            ),
                            height=600, 
                            margin=dict(l=200, t=100, r=120, b=50), # Adjusted right margin for annotation
                            template=PLOTLY_TEMPLATE,
                            bargap=0.4,
                            legend=dict(
                                orientation="v",
                                yanchor="bottom",
                                y=0.02,
                                xanchor="right",
                                x=0.98,
                                bgcolor="#0f172a", # Solid dark slate
                                bordercolor="rgba(255, 255, 255, 0.1)",
                                borderwidth=1,
                                title_text="Parameter Category",
                                title_font=dict(size=12, color="rgba(255, 255, 255, 0.7)"),
                                font=dict(size=11, color="rgba(255, 255, 255, 0.9)"),
                                itemsizing="constant"
                            )
                        )
                        
                        st.plotly_chart(fig_imp, use_container_width=True, key="nn_feature_importance")

                    st.divider()
                    st.markdown("### 🎛️ Recommended Parameters")
                    st.markdown("The AI suggests these parameters to achieve the growth shown above. You can fine-tune them before confirming.")
                    
                    labels = OPT_ORDER
                    values = [float(st.session_state.opt_candidate[k]) for k in labels]
                    sliders = {}
                    
                    # Group sliders nicely
                    cols = st.columns(3)
                    for i, k in enumerate(labels):
                        lo, hi = OPT_PARAM_RANGES[k]
                        default = float(values[i])
                        display_name = PARAM_DISPLAY_NAMES.get(k, k.replace("_", " ").title())
                        with cols[i % 3]:
                            sliders[k] = st.slider(display_name, float(lo), float(hi), default, step=(0.01 if isinstance(lo, float) else 1.0), key=f"nn_opt_slider_{k}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_conf, col_dummy = st.columns([1, 2])
                    with col_conf:
                        confirm = st.button("✅ Apply These Parameters", type="primary", key="nn_opt_confirm", use_container_width=True)
                    
                    if confirm:
                        fp = {k: float(sliders[k]) for k in sliders}
                        fp["pillar_count"] = int(fp["pillar_count"])
                        fp["model_type"] = model_type
                        ok, msg = opt_validate_params(fp)
                        if not ok:
                            st.error(msg)
                        else:
                            st.session_state.opt_final = fp
                            fs = []
                            rng = np.random.default_rng(int(seed))
                            for i in range(30):
                                rr = run_simulation_logic(fp, model_type)
                                fs.append(rr["metrics"]["avg_growth_rate"])
                            st.session_state.opt_final_samples = np.array(fs, dtype=np.float32)
                
                if "opt_final_samples" in st.session_state:
                    st.divider()
                    st.markdown("### 🏆 Final Validation")
                    
                    base = st.session_state.opt_baseline
                    cand = st.session_state.opt_candidate_samples
                    fin = st.session_state.opt_final_samples
                    
                    p_val_fin, d_fin = stats_compare(fin, base)
                    
                    mb = float(np.mean(base)); mc = float(np.mean(cand)); mf = float(np.mean(fin))
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=base, name="Baseline", opacity=0.5, marker_color="gray"))
                    fig2.add_trace(go.Histogram(x=cand, name="AI Projected", opacity=0.5, marker_color="orange"))
                    fig2.add_trace(go.Histogram(x=fin, name="Actual Result", opacity=0.8, marker_color="green"))
                    
                    fig2.update_layout(
                        barmode="overlay", 
                        title="Validation: Model Performance Comparison",
                        xaxis_title="Growth Rate (mm/hr)",
                        template=PLOTLY_TEMPLATE
                    )
                    
                    # Add vertical lines for means
                    fig2.add_vline(x=mb, line_dash="dash", line_color="gray", annotation_text="Base Mean")
                    fig2.add_vline(x=mf, line_dash="dash", line_color="green", annotation_text="Final Mean")
                    
                    st.plotly_chart(fig2, use_container_width=True, key="nn_final_hist")
                    
                    st.success(f"**Success!** The optimized parameters achieved a mean growth rate of **{mf:.2f} mm/hr**, compared to the baseline of **{mb:.2f} mm/hr**.")
                    
                    # Network Viz
                    with st.expander("🕸️ Neural Network Internals (Visualization)", expanded=False):
                        idx = st.slider("Training Snapshot", 0, len(st.session_state.opt_snaps)-1, 0, 1, key="nn_snap_idx")
                        
                        col_v1, col_v2 = st.columns(2)
                        with col_v1:
                            view_mode_ui = st.selectbox("View Mode", ["Analysis","Presentation"], index=0, key="nn_view_mode")
                            layout_type = st.selectbox("Layout Style", ["Hierarchical","Radial"], index=0, key="nn_layout")
                        with col_v2:
                            act_src = st.selectbox("Show Activations For", ["None","Candidate","Final"], key="nn_act_src")
                            show_arrows = st.checkbox("Show Flow Arrows", value=True, key="nn_show_arrows")

                        # Hidden advanced viz settings
                        # layer_gap = st.slider("Layer spacing", 1.0, 4.0, 2.0, 0.1, key="nn_layer_gap")
                        # node_gap = st.slider("Node spacing", 0.8, 3.0, 1.5, 0.1, key="nn_node_gap")
                        # edge_thr = st.slider("Edge threshold", 0.0, 1.0, 0.15, 0.05, key="nn_edge_thr")
                        
                        # Use defaults for hidden ones
                        layer_gap = 2.0
                        node_gap = 1.5
                        edge_thr = 0.15
                        label_mode_ui = "Auto"
                        top_paths_k = 8
                        act_sign = False

                        acts = None
                        if act_src == "Candidate" and "opt_candidate" in st.session_state:
                            xv = opt_to_vector(st.session_state.opt_candidate)
                            xs = st.session_state.opt_scaler.transform(xv.reshape(1, -1)).astype(np.float32)
                            t = torch.tensor(xs)
                            o, h1, h2 = st.session_state.opt_model(t)
                            acts = {"in": xs.squeeze(), "h1": h1.squeeze().detach().cpu().numpy(), "h2": h2.squeeze().detach().cpu().numpy(), "out": float(o.squeeze().detach().cpu().numpy())}
                        elif act_src == "Final" and "opt_final" in st.session_state:
                            xv = opt_to_vector(st.session_state.opt_final)
                            xs = st.session_state.opt_scaler.transform(xv.reshape(1, -1)).astype(np.float32)
                            t = torch.tensor(xs)
                            o, h1, h2 = st.session_state.opt_model(t)
                            acts = {"in": xs.squeeze(), "h1": h1.squeeze().detach().cpu().numpy(), "h2": h2.squeeze().detach().cpu().numpy(), "out": float(o.squeeze().detach().cpu().numpy())}
                        
                        import importlib
                        importlib.reload(vis)
                        fig4 = vis.draw_nn(
                            st.session_state.opt_snaps[idx],
                            activations=acts,
                            layer_gap=layer_gap,
                            node_gap=node_gap,
                            show_key=True,
                            max_edges=int(connections_shown),
                            layout=("radial" if layout_type == "Radial" else "hierarchical"),
                            label_mode={"Auto":"auto","Important only":"important","All":"all","None":"none"}[label_mode_ui],
                            edge_threshold=float(edge_thr),
                            show_arrows=bool(show_arrows),
                            top_paths_k=int(top_paths_k),
                            show_activation_sign=bool(act_sign),
                            view_mode=("presentation" if view_mode_ui == "Presentation" else "analysis"),
                            palette="colorblind",
                        )
                        st.plotly_chart(fig4, use_container_width=True, key="nn_nn_viz")

    if selected_tab == tabs[4]:
        st.markdown("## Export")
        
        # --- 3D GEOMETRY EXPORT (NEW) ---
        st.subheader("📦 3D Geometry Export (CAD/STL)")
        st.caption("Generate a high-quality, manifold STL mesh for SolidWorks/manufacturing.")
        
        if 'latest_result_full' not in st.session_state:
            st.warning("⚠️ No simulation data available. Please run a simulation first.")
        else:
            res = st.session_state.latest_result_full
            # Ensure scaffold data exists
            if "scaffold_data" not in res:
                st.error("Scaffold geometry data missing from the latest run.")
            else:
                # Select what to export
                export_target = st.radio(
                    "Export Target (What to Visualize)", 
                    ["Scaffold Structure (Gray)", "Biological Growth (Green)", "Combined Model"],
                    index=1,
                    horizontal=True,
                    help="Select 'Biological Growth' to export the high-fidelity organic structure. 'Combined Model' includes the pillars."
                )

                with st.expander("🛠️ Advanced Mesh Settings", expanded=True):
                    z_scale = st.slider(
                        "Z Scale Factor (Vertical Exaggeration)", 
                        min_value=1.0, 
                        max_value=500.0, 
                        value=50.0, 
                        step=1.0,
                        help="Multiplies the height of the features to make them visible in 3D prints. Use 100+ for very subtle textures."
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        smooth_sigma = st.slider("Smoothing Sigma (Voxels)", 0.0, 2.0, 0.4, 0.1, help="Lower values (0.4) preserve ridges. Higher values (1.0+) make blobs.")
                    with c2:
                        upsample = st.checkbox("High Quality (Upsample 2x)", value=True, help="Increases resolution to capture fine ridges.")
                        decimate = st.checkbox("Decimate Mesh", value=True)
                        target_faces = st.number_input("Target Face Count", 1000, 500000, 50000, 1000, disabled=not decimate)
                
                if st.button("Generate Onshape-Ready Bundle", type="primary", key="btn_gen_stl"):
                    with st.spinner("Processing geometry (SDF → Marching Cubes → Conditioning)..."):
                        try:
                            scaffold = res.get("scaffold_data")
                            growth = res.get("growth_data")
                            params = res.get("params", {})
                            # Estimate voxel size
                            v_size = calculate_voxel_size(params)
                            
                            # Parameters
                            floor_thresh = 0.8
                            
                            if export_target == "Biological Growth (Green)":
                                if growth is None:
                                    st.error("No growth data available in this result.")
                                    st.stop()
                                
                                # One Matrix Strategy: Use 2D heightmap
                                if growth.ndim == 3:
                                    # If 3D, project max to 2D? Or just use as is?
                                    # User specifically requested "One Matrix" 2D strategy.
                                    # Assuming growth is 2D for this mode usually.
                                    data_2d = np.max(growth, axis=2)
                                else:
                                    data_2d = growth
                                
                                # Generate Heightmap Mesh
                                # Use voxel_size_mm=1.0 to match the user's "60-unit base" mental model
                                # and ensure the Z-scale of 20.0 produces the expected 20-30% height ratio.
                                mesh = generate_heightmap_mesh(
                                    data_2d, 
                                    floor_threshold=floor_thresh, 
                                    voxel_size_mm=1.0, 
                                    z_scale=z_scale,
                                    base_z=0.0
                                )
                                filename_suffix = "growth"

                            elif export_target == "Combined Model":
                                if growth is None or scaffold is None:
                                    st.error("Data missing for combined export.")
                                    st.stop()
                                
                                # Prepare 2D Scaffold
                                if scaffold.ndim == 3:
                                    scaffold_2d = np.max(scaffold, axis=2)
                                else:
                                    scaffold_2d = scaffold
                                
                                # Prepare 2D Growth
                                if growth.ndim == 3:
                                    growth_2d = np.max(growth, axis=2)
                                else:
                                    growth_2d = growth
                                    
                                # Combine: Max of both
                                combined_2d = np.maximum(scaffold_2d, growth_2d)
                                
                                # Generate Heightmap Mesh
                                # Use voxel_size_mm=1.0 to match the user's "60-unit base" mental model
                                mesh = generate_heightmap_mesh(
                                    combined_2d, 
                                    floor_threshold=floor_thresh, 
                                    voxel_size_mm=1.0, 
                                    z_scale=z_scale,
                                    base_z=0.0
                                )
                                filename_suffix = "combined"
                            else:
                                # Scaffold Only (keep old logic or update?)
                                # Let's update to heightmap for consistency if 2D
                                if scaffold.ndim == 3:
                                    # Use old 3D method for full 3D scaffold if needed
                                    data_to_mesh = scaffold
                                    mesh = generate_improved_mesh(
                                        data_to_mesh, 
                                        voxel_size_mm=v_size, 
                                        smoothing_sigma=smooth_sigma, 
                                        target_faces=target_faces if decimate else None,
                                        is_binary=True,
                                        iso_level=0.0
                                    )
                                else:
                                    # 2D Scaffold
                                    mesh = generate_heightmap_mesh(
                                        scaffold, 
                                        floor_threshold=0.1, # Pillars exist > 0
                                        voxel_size_mm=1.0, 
                                        z_scale=z_scale,
                                        base_z=0.0
                                    )
                                filename_suffix = "scaffold"
                                st.info("💡 Tip: To export the detailed organic texture, select 'Biological Growth' or 'Combined Model'.")

                            # Export to Bundle (ZIP with Manifest)
                            zip_bytes = export_to_bundle(mesh, filename_base=f"{filename_suffix}_{params.get('model_type', 'custom')}")
                            
                            # Info
                            st.success(f"Mesh Generated! Target: {export_target}, Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
                            
                            # Download
                            st.download_button(
                                label="Download Onshape Bundle (ZIP)",
                                data=zip_bytes,
                                file_name=f"scaffold_onshape_{int(time.time())}.zip",
                                mime="application/zip",
                                type="primary"
                            )
                            
                        except Exception as e:
                            st.error(f"Mesh generation failed: {str(e)}")
                            # st.exception(e) # Uncomment for debugging

        st.divider()
        st.subheader("📊 Data Export (CSV/JSON/PDF)")
        run_df = pd.DataFrame(st.session_state.run_history) if 'run_history' in st.session_state else pd.DataFrame()
        formats = st.multiselect("File format options", ["PDF", "CSV", "Excel", "JSON"], default=["CSV"], key="export_formats")
        range_choice = st.radio("Export range", ["Current view", "Selected items", "Custom range"], index=0, key="export_range")
        df_sel = pd.DataFrame()
        if range_choice == "Current view":
            if 'latest_result_full' not in st.session_state:
                st.error("No current view available")
            else:
                df_sel = pd.DataFrame([st.session_state.latest_result_full['metrics']])
        elif range_choice == "Selected items":
            if run_df.empty:
                st.error("No run history available")
            else:
                idxs = st.multiselect("Select runs", options=list(range(len(run_df))), key="export_selected_idxs")
                if idxs:
                    df_sel = run_df.iloc[idxs]
        else:
            if run_df.empty:
                st.error("No run history available")
            else:
                start_idx = st.number_input("Start index", 0, max(0, len(run_df)-1), 0, 1, key="export_range_start")
                end_idx = st.number_input("End index", 0, max(0, len(run_df)-1), max(0, len(run_df)-1), 1, key="export_range_end")
                if end_idx < start_idx:
                    st.error("End index must be ≥ start index")
                else:
                    df_sel = run_df.iloc[int(start_idx):int(end_idx)+1]
        include_meta = st.checkbox("Include metadata", value=True, key="export_include_meta")
        dest = st.selectbox("Destination", ["Local download", "Cloud storage", "Email attachment"], index=0, key="export_destination")
        with st.expander("PDF settings", expanded=False):
            pdf_dpi = st.number_input("DPI", 100, 600, 300, 50, key="export_pdf_dpi")
        with st.expander("CSV settings", expanded=False):
            csv_delim = st.text_input("Delimiter", ",", key="export_csv_delim")
            csv_header = st.checkbox("Include header", value=True, key="export_csv_header")
        with st.expander("Excel settings", expanded=False):
            xls_sheet = st.text_input("Sheet name", "Runs", key="export_xls_sheet")
        with st.expander("JSON settings", expanded=False):
            json_indent = st.number_input("Indent", 0, 8, 2, 1, key="export_json_indent")
            json_ascii = st.checkbox("Ensure ASCII", value=False, key="export_json_ascii")
        compress = st.checkbox("Compress to ZIP", value=len(formats) > 1, key="export_compress")
        comp_level = st.slider("Compression level", 0, 9, 5, 1, key="export_comp_level")
        sched = st.checkbox("Schedule export", value=False, key="export_schedule")
        if sched:
            sch_date = st.date_input("Date", key="export_schedule_date")
            sch_time = st.time_input("Time", key="export_schedule_time")
            if st.button("Add scheduled export", key="export_schedule_add"):
                if 'scheduled_exports' not in st.session_state:
                    st.session_state.scheduled_exports = []
                st.session_state.scheduled_exports.append({"date": str(sch_date), "time": str(sch_time), "formats": formats, "range": range_choice})
                st.success("Scheduled")
        ok_to_export = True
        if range_choice == "Current view" and 'latest_result_full' not in st.session_state:
            ok_to_export = False
        if range_choice != "Current view" and run_df.empty:
            ok_to_export = False
        if not formats:
            ok_to_export = False
        if dest != "Local download":
            st.info("Cloud and email destinations are not configured")
        if st.button("Prepare export", type="primary", key="export_prepare"):
            if not ok_to_export:
                st.error("Invalid export parameters")
            else:
                files = []
                latest = st.session_state.latest_result_full if 'latest_result_full' in st.session_state else None
                meta = {}
                if include_meta:
                    if latest:
                        meta = {"model_type": latest["model_type"], "params": latest.get("params", {})}
                if "CSV" in formats:
                    if df_sel.empty and latest:
                        df_csv = pd.DataFrame([latest["metrics"]])
                    else:
                        df_csv = df_sel
                    try:
                        csv_bytes = df_csv.to_csv(index=False, sep=csv_delim, header=csv_header).encode("utf-8")
                        files.append(("runs.csv", "text/csv", csv_bytes))
                    except Exception as e:
                        st.error(f"CSV export failed: {e}")
                if "Excel" in formats:
                    if df_sel.empty and latest:
                        df_xls = pd.DataFrame([latest["metrics"]])
                    else:
                        df_xls = df_sel
                    try:
                        bio_xls = io.BytesIO()
                        with pd.ExcelWriter(bio_xls) as writer:
                            df_xls.to_excel(writer, sheet_name=xls_sheet, index=False)
                        files.append(("runs.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", bio_xls.getvalue()))
                    except Exception as e:
                        st.error(f"Excel export failed: {e}")
                if "JSON" in formats:
                    payload = {}
                    if not df_sel.empty:
                        payload["runs"] = df_sel.to_dict(orient="records")
                    elif latest:
                        payload["runs"] = [latest["metrics"]]
                    if include_meta:
                        payload["meta"] = meta
                    try:
                        json_bytes = json.dumps(payload, indent=int(json_indent), ensure_ascii=bool(json_ascii)).encode("utf-8")
                        files.append(("runs.json", "application/json", json_bytes))
                    except Exception as e:
                        st.error(f"JSON export failed: {e}")
                if "PDF" in formats:
                    try:
                        bio_pdf = io.BytesIO()
                        with PdfPages(bio_pdf) as pdf:
                            fig = plt.figure(figsize=(8.5, 11))
                            txt = ""
                            if latest:
                                txt += f"Model: {latest['model_type']}\n"
                                for k, v in latest["metrics"].items():
                                    txt += f"{k}: {v}\n"
                            elif not df_sel.empty:
                                txt += f"Runs: {len(df_sel)}\n"
                                for k in df_sel.columns[:10]:
                                    txt += f"{k}: {df_sel[k].iloc[0]}\n"
                            fig.text(0.1, 0.95, "Simulation Report", fontsize=16)
                            fig.text(0.1, 0.9, txt[:2000], fontsize=10)
                            pdf.savefig(fig, dpi=int(pdf_dpi))
                            plt.close(fig)
                        files.append(("report.pdf", "application/pdf", bio_pdf.getvalue()))
                    except Exception as e:
                        st.error(f"PDF export failed: {e}")
                if not files:
                    st.error("No files generated")
                else:
                    if compress or len(files) > 1:
                        bio_zip = io.BytesIO()
                        with zipfile.ZipFile(bio_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=int(comp_level)) as zf:
                            for fn, mt, by in files:
                                zf.writestr(fn, by)
                        st.download_button("Download bundle", bio_zip.getvalue(), "export_bundle.zip", "application/zip", use_container_width=True, key="export_download_zip")
                    else:
                        fn, mt, by = files[0]
                        st.download_button("Download file", by, fn, mt, use_container_width=True, key="export_download_single")

# --- FOOTER ---
st.divider()
st.markdown(f"""
<div style='text-align: center; color: var(--text); opacity: 0.7; padding: 20px;'>
    <p><strong>3D Physarum Simulation Platform</strong> | Version 5.0 (CAD Fix)</p>
    <p>Built with Streamlit • Plotly • SciPy • NumPy</p>
</div>
""", unsafe_allow_html=True)
