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
def get_placeholder_radar_chart(metrics):
    categories = ['Permeability', 'Coverage', 'Stiffness', 'Redundancy', 'Printability']
    p_k = metrics.get('permeability_kappa_iso', metrics.get('permeability_kappa_X', 0))
    values = [
        np.clip(p_k * 1e12, 0.1, 1.0),
        metrics.get('coverage_fraction', 0),
        np.clip(metrics.get('Eeff', metrics.get('Keff', 0)) / 25.0, 0.1, 1.0),
        metrics.get('lcc_fraction', np.random.uniform(0.8, 1.0)),
        np.clip(metrics.get('printability_min_pore', 0.5) / 1.0, 0.1, 1.0)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill='toself', name='Scaffold Quality', line=dict(color='var(--primary)', width=2)
    ))
    fig.update_layout(title="Scaffold Quality Indices", height=400, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="var(--text)"), margin=dict(l=40, r=40, t=80, b=40))
    return fig

def get_placeholder_histogram(title="Channel Width Distribution"):
    np.random.seed(42)
    mean_val = st.session_state.latest_result_full['params'].get('channel_width_mm', 0.9)
    data = np.random.normal(mean_val, 0.05, 500)
    fig = ff.create_distplot([data], ['Channel Width (mm)'], bin_size=.01, show_rug=False)
    fig.update_layout(title=title, height=400, showlegend=False, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="var(--text)"), margin=dict(l=10, r=10, t=50, b=10))
    return fig

def get_placeholder_slice_plot(title="Cross-Sectional Slice"):
    data = np.random.rand(20, 20)
    fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis', colorbar={"title": "Concentration"}))
    fig.update_layout(title=title, height=400, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="var(--text)"), margin=dict(l=10, r=10, t=50, b=10))
    return fig

def get_placeholder_streamline_plot():
    x_vec = np.arange(0, 10)
    y_vec = np.arange(0, 10)
    x, y = np.meshgrid(x_vec, y_vec)
    u = np.ones_like(x) * 1.0
    v = np.sin(x) * 0.5
    fig = ff.create_streamline(x_vec, y_vec, u, v, arrow_scale=.1)
    fig.update_layout(title="Darcy Velocity Streamlines", height=400, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="var(--text)"), margin=dict(l=10, r=10, t=50, b=10))
    return fig

def get_placeholder_anisotropy_plot():
    x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    fig = go.Figure()
    fig.add_trace(go.Cone(x=x.flatten(), y=y.flatten(), z=np.zeros(100), u=np.ones(100), v=np.zeros(100), w=np.zeros(100), sizemode="absolute", sizeref=0.5, anchor="tip", colorscale='Reds', showscale=False))
    fig.add_trace(go.Cone(x=x.flatten(), y=y.flatten(), z=np.zeros(100), u=np.zeros(100), v=np.ones(100), w=np.zeros(100), sizemode="absolute", sizeref=0.2, anchor="tip", colorscale='Blues', showscale=False))
    fig.update_layout(title="Anisotropy Map (D∥ vs D⊥)", height=400, template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="var(--text)"), margin=dict(l=10, r=10, t=50, b=10))
    return fig

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
.plotly .legend {{ border-radius: 8px; border: 1px solid var(--surface-border); background-color: var(--surface) !important; }}
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
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Summary & Visualization", 
        "🔬 Advanced Diagnostics", 
        "⚙️ Theory & Equations", 
        "📈 Statistical Analysis", 
        "🧠 Neural Network",
        "Export"
    ])

    with tab1:
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

    with tab2:
        st.markdown(f"## Advanced Diagnostics: {model_type}")
        st.markdown("*Detailed analytical visualizations for in-depth analysis*")
        st.divider()
        
        if model_type == "3D Structured (Channel Flow)":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Scaffold Quality Assessment")
                st.plotly_chart(get_placeholder_radar_chart(metrics), use_container_width=True, key="diag_struct_radar")
            with col2:
                st.markdown("### Flow Field Analysis")
                st.plotly_chart(get_placeholder_streamline_plot(), use_container_width=True, key="diag_struct_stream")
            st.divider()
            st.markdown("### Depth-Dependent Transport")
            depth_data = pd.DataFrame({'Depth (mm)': np.linspace(0, 10, 20), 'Tortuosity': 1.0 + 0.5 * np.sin(np.linspace(0, 3.14, 20))})
            st.line_chart(depth_data.set_index('Depth (mm)'))

        elif model_type == "3D Porous (Channel Diffusion)":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Channel Width Distribution")
                st.plotly_chart(get_placeholder_histogram(title="Channel Width"), use_container_width=True, key="diag_porous_hist_width")
            with col2:
                st.markdown("### Cross-Sectional Analysis")
                st.plotly_chart(get_placeholder_slice_plot(), use_container_width=True, key="diag_porous_slice")
            st.divider()
            st.markdown("### Percolation Analysis")
            perc_data = pd.DataFrame({'Porosity': np.linspace(0.1, 0.8, 30), 'LCC_Fraction': 1 / (1 + np.exp(-15*(np.linspace(0.1, 0.8, 30) - 0.31)))})
            st.line_chart(perc_data.set_index('Porosity'))

        elif model_type == "2.5D Surface (Pillar Tops)":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Path Tortuosity Distribution")
                st.plotly_chart(get_placeholder_histogram(title="Path Tortuosity"), use_container_width=True, key="diag_surface_hist_tort")
            with col2:
                st.markdown("### Pillar Adhesion Heatmap")
                st.plotly_chart(get_placeholder_slice_plot(title="Growth Density on Pillar Tops"), use_container_width=True, key="diag_surface_heat_pillars")
            st.divider()
            st.markdown("### Network Topology")
            st.image("https://i.imgur.com/g8fS1qK.png", caption="Placeholder for a 2D Network Graph visualization.")

    with tab3:
        st.markdown(f"## Governing Equations & Theory: {model_type}")
        st.markdown("*Mathematical foundations and derived metrics*")
        st.divider()
        
        if model_type == "3D Structured (Channel Flow)":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌊 Fluid Transport (Anisotropic)")
                st.latex(r"\text{Darcy's Law: } \mathbf{u} = -\frac{\mathbf{K}}{\mu}\nabla p")
                st.latex(r"\text{Permeability Tensor: } \mathbf{K} = \begin{pmatrix} K_x & 0 \\ 0 & K_y \end{pmatrix}")
            with col2:
                st.markdown("### 🧪 Mass Transport")
                st.latex(r"\frac{\partial C}{\partial t} + \mathbf{u}\cdot\nabla C = D_{\rm eff}\nabla^2 C - r(C)")
                st.latex(r"\text{Stiffness: } K_{\rm eff} = \sum_{i}\frac{EA_i}{L_i}")
            st.divider()
            st.markdown("### 📊 Calculated Parameters")
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Porosity (φ)", f"{metrics['porosity_phi']:.3f}")
            pcol1.metric("Permeability (K_x)", f"{metrics['permeability_kappa_X']:.2e} m²")
            pcol2.metric("Permeability (K_y)", f"{metrics['permeability_kappa_Y']:.2e} m²")
            pcol2.metric("Tortuosity (τ)", f"{metrics['mean_tortuosity']:.3f}")
            pcol3.metric("Deff", f"{metrics['Deff']:.2e} m²/s")
            pcol3.metric("Keff", f"{metrics['Keff']:.2f} kPa")

        elif model_type == "3D Porous (Channel Diffusion)":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌊 Porous Media Transport (Isotropic)")
                st.latex(r"\text{Permeability: } \kappa \approx \frac{\phi^3}{C(1-\phi)^2 S_v^2}")
                st.latex(r"\text{Stiffness (Gibson–Ashby): } E_{\text{eff}} \sim (1-\phi)^n")
            with col2:
                st.markdown("### 🧪 Advection-Diffusion")
                st.latex(r"\frac{\partial C}{\partial t}+\mathbf{u}\cdot\nabla C = \nabla\cdot(D_{\rm eff}\nabla C)-r(C)")
                st.latex(r"\text{Percolation: } P_\infty \sim (\phi-\phi_c)^\beta")
            st.divider()
            st.markdown("### 📊 Calculated Parameters")
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Porosity (φ)", f"{metrics['porosity_phi']:.3f}")
            pcol1.metric("Permeability (κ)", f"{metrics['permeability_kappa_iso']:.2e} m²")
            pcol2.metric("Tortuosity (τ)", f"{metrics['mean_tortuosity']:.3f}")
            pcol2.metric("Fractal Dim. (Df)", f"{metrics['fractal_dimension']:.3f}")
            pcol3.metric("Deff", f"{metrics['Deff']:.2e} m²/s")
            pcol3.metric("Eeff", f"{metrics['Eeff']:.2f} kPa")

        elif model_type == "2.5D Surface (Pillar Tops)":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🧪 Surface Diffusion")
                st.latex(r"\frac{\partial C}{\partial t} = D\nabla^2 C - r(C)")
                st.latex(r"\text{Reaction-Diffusion on a 2D surface}")
            with col2:
                st.markdown("### 🗺️ Network Topology")
                st.latex(r"\text{Tortuosity: } \tau = L_{\text{path}} / \| \mathbf{x}_{\text{end}} - \mathbf{x}_{\text{start}}\|")
                st.latex(r"\text{Fractal Dimension: } D_f = -d\ \log N(\epsilon) / d\ \log \epsilon")
            st.divider()
            st.markdown("### 📊 Calculated Parameters")
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Fractal Dim. (Df)", f"{metrics['fractal_dimension']:.3f}")
            pcol2.metric("Tortuosity (τ)", f"{metrics['mean_tortuosity']:.3f}")
            pcol3.metric("Pillar Adhesion", f"{metrics['pillar_adhesion_index']:.3f}")

    # --- STATISTICAL ANALYSIS TAB ---
    with tab4:
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

    with tab5:
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
            with st.expander("🧠 Neural Optimization Mode", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    seed = st.number_input("Seed", 0, 1000000, 42, 1, key="nn_opt_seed")
                with c2:
                    n_search = st.number_input("Global Search Samples", 1000, 50000, 4000, 1000, key="nn_opt_n_search")
                colA, colB, colC = st.columns(3)
                with colA:
                    epochs = st.number_input("Surrogate epochs", 50, 1000, 150, 50, key="nn_opt_epochs")
                with colB:
                    snapshot_stride = st.number_input("Snapshot stride", 1, 50, 10, 1, key="nn_opt_stride")
                with colC:
                    connections_shown = st.number_input("Connections shown", 50, 1000, 200, 50, key="nn_opt_conn")
                run_opt = st.button("Run Neural Optimization", type="primary", key="nn_opt_run")
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
                    base = st.session_state.opt_baseline
                    cand = st.session_state.opt_candidate_samples
                    mu, sigma = stats_fit_gaussian(base)
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=base, name="Baseline", opacity=0.55))
                    fig.add_trace(go.Histogram(x=cand, name="Candidate", opacity=0.55))
                    fig.update_layout(barmode="overlay", title="Baseline vs Candidate")
                    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
                    xs = np.linspace(base.min(), base.max(), 200)
                    pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mu)/sigma)**2)
                    pdf_scaled = pdf * (base.size * (xs[1]-xs[0]))
                    fig.add_trace(go.Scatter(x=xs, y=pdf_scaled, mode="lines", name="Baseline Normal Fit"))
                    st.plotly_chart(fig, use_container_width=True, key="nn_candidate_hist")
                    p_val_cand, d_cand = stats_compare(cand, base)
                    pow_cand = stats_power(d_cand, len(cand), len(base))
                    st.metric("Candidate p-value", f"{p_val_cand:.4f}")
                    st.metric("Candidate Cohen's d", f"{d_cand:.3f}")
                    st.metric("Candidate Power", f"{pow_cand:.3f}")
                    labels = OPT_ORDER
                    values = [float(st.session_state.opt_candidate[k]) for k in labels]
                    sliders = {}
                    cols = st.columns(3)
                    for i, k in enumerate(labels):
                        lo, hi = OPT_PARAM_RANGES[k]
                        default = float(values[i])
                        with cols[i % 3]:
                            sliders[k] = st.slider(k, float(lo), float(hi), default, step=(0.01 if isinstance(lo, float) else 1.0), key=f"nn_opt_slider_{k}")
                    confirm = st.button("Confirm Final Parameters", type="primary", key="nn_opt_confirm")
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
                    base = st.session_state.opt_baseline
                    cand = st.session_state.opt_candidate_samples
                    fin = st.session_state.opt_final_samples
                    mb = float(np.mean(base)); mc = float(np.mean(cand)); mf = float(np.mean(fin))
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=base, name="Baseline", opacity=0.55))
                    fig2.add_trace(go.Histogram(x=cand, name="Candidate", opacity=0.55))
                    fig2.add_trace(go.Histogram(x=fin, name="Final", opacity=0.55))
                    fig2.update_layout(barmode="overlay", title="Baseline vs Candidate vs Final")
                    fig2.update_traces(marker_line_width=0.5, marker_line_color="white")
                    fig2.add_shape(type="line", x0=mb, x1=mb, y0=0, y1=1, yref="paper", line=dict(color="blue", dash="dash"))
                    fig2.add_shape(type="line", x0=mc, x1=mc, y0=0, y1=1, yref="paper", line=dict(color="orange", dash="dash"))
                    fig2.add_shape(type="line", x0=mf, x1=mf, y0=0, y1=1, yref="paper", line=dict(color="green", dash="dash"))
                    st.plotly_chart(fig2, use_container_width=True, key="nn_final_hist")
                    p_val_fin, d_fin = stats_compare(fin, base)
                    pow_fin = stats_power(d_fin, len(fin), len(base))
                    st.metric("Final p-value", f"{p_val_fin:.4f}")
                    st.metric("Final Cohen's d", f"{d_fin:.3f}")
                    st.metric("Final Power", f"{pow_fin:.3f}")
                    xvec = opt_to_vector(st.session_state.opt_final)
                    sens = gradient_sensitivity(st.session_state.opt_model, st.session_state.opt_scaler, xvec)
                    fig3 = go.Figure(go.Bar(x=labels, y=sens))
                    fig3.update_layout(title="Parameter Sensitivity (|gradients|)", xaxis_title="Parameter", yaxis_title="Sensitivity")
                    st.plotly_chart(fig3, use_container_width=True, key="nn_sensitivity_bar")
                    idx = st.slider("NN Snapshot Index", 0, len(st.session_state.opt_snaps)-1, 0, 1, key="nn_snap_idx")
                    layer_gap = st.slider("Layer spacing", 1.0, 4.0, 2.0, 0.1, key="nn_layer_gap")
                    node_gap = st.slider("Node spacing", 0.8, 3.0, 1.5, 0.1, key="nn_node_gap")
                    act_src = st.selectbox("Activation source", ["None","Candidate","Final"], key="nn_act_src")
                    layout_type = st.selectbox("Layout", ["Hierarchical","Radial"], index=0, key="nn_layout")
                    label_mode_ui = st.selectbox("Labels", ["Auto","Important only","All","None"], index=0, key="nn_label_mode")
                    edge_thr = st.slider("Edge threshold", 0.0, 1.0, 0.15, 0.05, key="nn_edge_thr")
                    show_arrows = st.checkbox("Show arrows on edges", value=True, key="nn_show_arrows")
                    view_mode_ui = st.selectbox("View mode", ["Analysis","Presentation"], index=0, key="nn_view_mode")
                    top_paths_k = st.number_input("Top paths highlighted", 1, 20, 8, 1, key="nn_top_paths_k")
                    act_sign = st.checkbox("Color by activation sign", value=False, key="nn_activation_sign")
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

    with tab6:
        st.markdown("## Export")
        st.divider()
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
