import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy import stats
import time

# --- LITERATURE DATA ---
LITERATURE_DATA = {
    "Tero_2010_MST_Ratio": {"mean": 1.75, "std": 0.30},
    "Tero_2010_Transport_Ratio": {"mean": 0.85, "std": 0.04},
    "AutoAnalysis_TotalLength": {"mean": 550, "std": 50}, 
    "AutoAnalysis_NodeDensity": {"mean": 16, "std": 1},
    "Kay_2022_GrowthRate_High": {"mean": 10.0, "std": 1.0}
} 

def generate_micropillar_geometry(params):
    """
    Generates the 3D numpy matrix for the micropillar array.
    """
    grid_size = 60 # Main grid resolution
    scaffold_matrix = np.zeros((grid_size, grid_size, grid_size))    
    
    # Get normalized dimensions from params
    n_pillars = int(params['pillar_count'])
    pillar_mm = float(params.get("pillar_size_mm", 0.0))
    channel_mm = float(params.get("channel_width_mm", 0.0))
    node_mm = float(params.get("channel_node_size_mm", 0.0))

    if n_pillars <= 0 or pillar_mm <= 0 or channel_mm <= 0:
        return np.ones((grid_size, grid_size, grid_size)), np.ones((grid_size, grid_size)), np.zeros((grid_size, grid_size, grid_size))

    pillar_width_px = max(1, int(grid_size * (pillar_mm / 200.0)))
    channel_width_px = max(1, int(grid_size * (channel_mm / 200.0)))
    node_width_px = max(0, int(grid_size * (node_mm / 200.0)))
    
    if pillar_width_px == 0 or channel_width_px == 0: 
        return np.ones((grid_size, grid_size, grid_size)), np.ones((grid_size, grid_size)), np.zeros((grid_size, grid_size, grid_size))

    # --- 1. Create Base ---
    base_height = int(grid_size * 0.2)
    scaffold_matrix[:, :, :base_height] = 1.0
    
    # --- 2. Create Large Pillars ---
    pillar_height = int(grid_size * 0.8)
    step_size = pillar_width_px + channel_width_px
    
    pillar_starts = [channel_width_px + i * step_size for i in range(n_pillars)]
    
    for x_start in pillar_starts:
        x_end = x_start + pillar_width_px
        for y_start in pillar_starts:
            y_end = y_start + pillar_width_px
            if x_end <= grid_size and y_end <= grid_size:
                scaffold_matrix[x_start:x_end, y_start:y_end, base_height:pillar_height] = 1.0

    # --- 3. Create Small Channel Nodes ---
    if node_width_px > 0:
        node_height = base_height + int((pillar_height - base_height) * 0.4)
        channel_centers = [int(i * step_size + channel_width_px / 2) for i in range(n_pillars + 1)]
        
        for cx in channel_centers:
            for cy in channel_centers:
                x_start = max(0, cx - node_width_px // 2)
                x_end = min(grid_size, cx + node_width_px // 2)
                y_start = max(0, cy - node_width_px // 2)
                y_end = min(grid_size, cy + node_width_px // 2)
                
                scaffold_matrix[x_start:x_end, y_start:y_end, base_height:node_height] = 1.0

    # --- 4. Define the final matrices ---
    pillar_tops = (scaffold_matrix[:, :, pillar_height-1] > 0).astype(float)
    channel_matrix = 1.0 - scaffold_matrix
    
    return scaffold_matrix, pillar_tops, channel_matrix

def _make_fibers2d(n, density=0.65):
    acc = np.zeros((n, n))
    for s in (2, 4, 8):
        noise = np.random.rand(n, n) - 0.5
        anis = gaussian_filter(noise, sigma=(s, 1))
        ang = np.random.uniform(0, 180)
        rot = rotate(anis, angle=ang, reshape=False, order=1, mode='reflect')
        acc += rot
    acc = np.abs(acc)
    acc /= (acc.max() + 1e-6)
    thr = np.quantile(acc, 1.0 - density)
    acc = (acc > thr).astype(float)
    acc = gaussian_filter(acc, sigma=1.2)
    return acc

def _make_fibers3d(n, n_orients=4):
    vol = np.zeros((n, n, n))
    base_sig = (1, 4, 1)
    for i in range(n_orients):
        noise = np.random.rand(n, n, n) - 0.5
        sig = (base_sig[i % 3], base_sig[(i+1) % 3], base_sig[(i+2) % 3])
        vol += gaussian_filter(noise, sigma=sig)
    vol = np.abs(vol)
    vol /= (vol.max() + 1e-6)
    return vol

def compute_base_metrics(params):
    """Computes all shared dependent variables."""
    metrics = {}
    base_growth = (params['dmem_glucose'] / 25.0) * (params['dmem_glutamine'] / 45.0)
    ion_effect = params['ion_ca'] / 1.8 
    light_effect = 1.0 if params['light_lumens'] == 0 else 0.5
    
    # Introduce more deterministic relationships for the NN to learn
    # But keep some noise to simulate biological variability
    n = float(params.get("pillar_count", 0))
    p = float(params.get("pillar_size_mm", 0.0))
    c = float(params.get("channel_width_mm", 0.0))
    node_w = float(params.get("channel_node_size_mm", 0.0))
    stiff = float(params.get("scaffold_stiffness_kPa", 0.0))
    dens = float(params.get("scaffold_density_g_cm3", 0.0))
    depth = float(params.get("media_depth_mm", 0.0))
    repl = float(params.get("replenish_freq_hr", 24.0))

    geom_effect = 1.0 + 0.035 * (c - 2.0) + 0.012 * (node_w - 2.0) - 0.010 * (n - 6.0) + 0.018 * (p - 2.0)
    mech_effect = 1.0 - 0.006 * (stiff - 10.0) - 0.25 * (dens - 0.6)
    env_effect = 1.0 + 0.020 * (depth - 2.0) - 0.003 * (repl - 24.0)
    scaffold_effect = float(np.clip(geom_effect * mech_effect * env_effect, 0.35, 1.80))

    metrics['avg_growth_rate'] = base_growth * ion_effect * light_effect * scaffold_effect * (7.0 + 0.15 * p)
    metrics['total_network_length'] = 520 * (1.0 + float(params.get("initial_mass_g", 0.5))) * (0.85 + 0.35 * scaffold_effect)
    
    # Add noise
    metrics['avg_growth_rate'] *= np.random.uniform(0.97, 1.03)
    metrics['total_network_length'] *= np.random.uniform(0.97, 1.03)
    
    metrics['num_junctions'] = int(metrics['total_network_length'] / 30.0)
    metrics['num_edges'] = int(metrics['num_junctions'] * 1.5)
    metrics['avg_branches_per_node'] = metrics['num_edges'] / (metrics['num_junctions'] + 1e-6)
    metrics['graph_density'] = metrics['num_edges'] / (metrics['num_junctions'] * (metrics['num_junctions'] - 1) + 1e-6)
    metrics['largest_component_size'] = metrics['total_network_length'] * 0.9
    
    cov_base = 0.6 * light_effect * (1.0 - float(params.get('elasticity', 0.5)))
    metrics['coverage_fraction'] = float(np.clip(cov_base, 0.1, 0.95))
    
    metrics['dye_diffusion_rate'] = 1.0 * ion_effect
    metrics['time_to_connection'] = 60.0 / (metrics['avg_growth_rate'] + 1e-6) # minutes
    metrics['time_to_reconnection'] = metrics['time_to_connection'] * 1.3
    metrics['mst_ratio'] = 1.75 * (1.0 / (params['elasticity'] + 0.5)) 
    metrics['path_efficiency'] = 0.85
    metrics['param_model_type'] = params['model_type']
    metrics['param_stiffness'] = params['scaffold_stiffness_kPa']
    return metrics

def compute_metrics(params, model_type):
    metrics = compute_base_metrics(params)
    
    if model_type == "2.5D Surface (Pillar Tops)":
        metrics['fractal_dimension'] = 1.9
        metrics['mean_tortuosity'] = 1.4
        metrics['pillar_adhesion_index'] = 0.8
        metrics['penetration_depth'] = 0.0
        metrics['avg_growth_rate'] *= 1.0 + 0.010 * (float(params.get("pillar_count", 0)) - 6.0)
        
    elif model_type == "3D Porous (Channel Diffusion)":
        n, p, c = params['pillar_count'], params['pillar_size_mm'], params['channel_width_mm']
        node_w = params.get('channel_node_size_mm', 0)
        
        total_area = (n*p + (n+1)*c)**2
        pillar_area = (n*p)**2
        node_area = ((n+1)**2) * (node_w**2)
        void_area = total_area - pillar_area - node_area
        phi = float(void_area / (total_area + 1e-12))
        phi = float(np.clip(phi, 1e-6, 0.95))
        
        denom = (1.0 - phi)
        kappa = (phi**3) / (5.0 * (denom**2 + 1e-12) * (2000**2))
        tortuosity = 1.0 / np.sqrt(phi + 1e-12)
        
        metrics['fractal_dimension'] = 2.3
        metrics['mean_tortuosity'] = tortuosity
        metrics['porosity_phi'] = phi
        metrics['permeability_kappa_iso'] = kappa
        metrics['Deff'] = (2.0e-9) / tortuosity
        metrics['Eeff'] = params['scaffold_stiffness_kPa'] * (1.0 - phi)**2
        metrics['penetration_depth'] = params['media_depth_mm'] * 0.8
        metrics['avg_growth_rate'] *= float(np.clip(0.55 + 0.85 * phi, 0.35, 1.35))
        metrics['total_network_length'] *= float(np.clip(0.75 + 0.55 * phi, 0.50, 1.35))

    elif model_type == "3D Structured (Channel Flow)":
        n, p, c = params['pillar_count'], params['pillar_size_mm'], params['channel_width_mm']
        node_w = params.get('channel_node_size_mm', 0)
        
        total_area = (n*p + (n+1)*c)**2
        pillar_area = (n*p)**2
        node_area = ((n+1)**2) * (node_w**2)
        void_area = total_area - pillar_area - node_area
        phi = float(void_area / (total_area + 1e-12))
        phi = float(np.clip(phi, 1e-6, 0.95))
        
        denom = (1.0 - phi)
        kappa_x = (phi**3) / (5.0 * (denom**2 + 1e-12) * (1900**2))
        kappa_y = (phi**3) / (5.0 * (denom**2 + 1e-12) * (2000**2))
        tortuosity = 1.0 + 0.5 * (1-phi)
        
        metrics['fractal_dimension'] = 2.1
        metrics['mean_tortuosity'] = tortuosity
        metrics['porosity_phi'] = phi
        metrics['permeability_kappa_X'] = kappa_x
        metrics['permeability_kappa_Y'] = kappa_y
        metrics['Deff'] = (2.0e-9) / tortuosity
        metrics['Keff'] = params['scaffold_stiffness_kPa'] * (1.0 - phi) * (1.0 + params['elasticity'])
        metrics['printability_min_pore'] = params['channel_width_mm'] - node_w
        metrics['avg_growth_rate'] *= float(np.clip(0.60 + 0.80 * phi, 0.35, 1.35))
        metrics['total_network_length'] *= float(np.clip(0.70 + 0.60 * phi, 0.50, 1.35))

    return metrics

def run_simulation_logic(params, model_type):
    scaffold_matrix, pillar_tops, channel_matrix = generate_micropillar_geometry(params)
    
    if model_type == "2.5D Surface (Pillar Tops)":
        fibers2d = _make_fibers2d(pillar_tops.shape[0])
        growth_data = np.clip(0.75 * pillar_tops + 1.15 * fibers2d, 0, 1.5)
        scaffold_data = scaffold_matrix 
        
    elif model_type == "3D Porous (Channel Diffusion)":
        fibers3d = _make_fibers3d(channel_matrix.shape[0], 5)
        rnd = gaussian_filter(np.random.rand(60, 60, 60), sigma=2)
        field = 0.65 + 0.35 * rnd
        growth_data = channel_matrix * np.clip(fibers3d * field, 0, 1.0)
        scaffold_data = scaffold_matrix
        
    elif model_type == "3D Structured (Channel Flow)":
        fibers3d = _make_fibers3d(channel_matrix.shape[0], 3)
        rnd = gaussian_filter(np.random.rand(60, 60, 60), sigma=3)
        field = 0.60 + 0.40 * rnd
        growth_data = channel_matrix * np.clip(fibers3d * field, 0, 1.0)
        scaffold_data = scaffold_matrix
    
    metrics = compute_metrics(params, model_type)
    
    return {
        "params": params, "model_type": model_type, "growth_data": growth_data, 
        "scaffold_data": scaffold_data, "metrics": metrics
    }
