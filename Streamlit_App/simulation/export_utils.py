
import numpy as np
import scipy.ndimage as nd
from skimage import measure
import trimesh
import io

def generate_heightmap_mesh(data_matrix, floor_threshold=0.8, voxel_size_mm=1.0, z_scale=10.0, base_z=0.0):
    """
    Generates a manifold mesh from a 2D heightmap using the "One Matrix" strategy.
    
    Args:
        data_matrix (np.ndarray): 2D numpy array of height values.
        floor_threshold (float): Values below this are clamped to this level.
        voxel_size_mm (float): XY scaling factor.
        z_scale (float): Z scaling factor (to make vertical features visible).
        base_z (float): The Z-coordinate of the flat bottom base.
        
    Returns:
        trimesh.Trimesh: The watertight mesh.
    """
    # 1. Data Normalization ("One Matrix" Rule)
    # Clamp data to the floor threshold
    height_map = np.maximum(data_matrix, floor_threshold)
    
    # Dimensions
    rows, cols = height_map.shape
    
    # 2. Generate Vertices
    # We need a grid of vertices for the top surface
    # and a corresponding grid (or just a flat face) for the bottom.
    # To ensure it's a solid block, we'll create a full bottom grid to match topology.
    
    # Create XY grid
    x = np.arange(cols) * voxel_size_mm
    y = np.arange(rows) * voxel_size_mm
    X, Y = np.meshgrid(x, y) # shape (rows, cols)
    
    # Top vertices (Z comes from height_map)
    # Apply Global Z-Exaggeration as requested by user
    # Formula: Final_Height = (Raw_Z - floor_threshold) * Z_SCALE_FACTOR
    # This scales the relief relative to the floor.
    
    # Calculate relief (height above floor)
    relief = (height_map - floor_threshold) * z_scale
    
    # Z_top is the relief + base_z
    # We add a small offset (e.g. 0.0) if we want the floor to be exactly at base_z.
    Z_top = relief + base_z
    
    # Bottom vertices (Z is fixed)
    # To ensure the mesh is a solid block with volume (not zero thickness at the floor),
    # we place the bottom face slightly below the lowest point of the top face.
    # If the lowest Z_top is base_z (0.0), we put Z_bottom at base_z - 2.0 (2mm base).
    base_thickness = 2.0
    Z_bottom = np.full_like(Z_top, base_z - base_thickness)
    
    # Flatten arrays to list of points (x, y, z)
    # Order: Row by row
    # Top vertices: 0 to N-1
    # Bottom vertices: N to 2N-1
    
    # Ravel assumes C-order (row-major), which matches meshgrid default if we are careful.
    # meshgrid(x, y) gives X where rows vary y, cols vary x.
    # So X[r, c] is x[c]. Y[r, c] is y[r].
    
    verts_top = np.column_stack((X.ravel(), Y.ravel(), Z_top.ravel()))
    verts_bottom = np.column_stack((X.ravel(), Y.ravel(), Z_bottom.ravel()))
    
    vertices = np.vstack((verts_top, verts_bottom))
    
    # 3. Generate Faces (Triangulation)
    faces = []
    
    # Helper to get index in flat array
    def get_idx(r, c):
        return r * cols + c
    
    n_pixels = rows * cols
    
    # A. Top Surface & Bottom Surface
    # For each square in the grid (r, c) to (r+1, c+1)
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Top indices
            t_tl = get_idx(r, c)
            t_tr = get_idx(r, c+1)
            t_bl = get_idx(r+1, c)
            t_br = get_idx(r+1, c+1)
            
            # Bottom indices (offset by n_pixels)
            b_tl = t_tl + n_pixels
            b_tr = t_tr + n_pixels
            b_bl = t_bl + n_pixels
            b_br = t_br + n_pixels
            
            # Top Face (Counter-Clockwise)
            # Triangle 1: TL, BL, TR
            faces.append([t_tl, t_bl, t_tr])
            # Triangle 2: TR, BL, BR
            faces.append([t_tr, t_bl, t_br])
            
            # Bottom Face (Clockwise / Inverted to point down)
            # Triangle 1: TL, TR, BL (swapped 2&3) -> No, standard is CCW from outside.
            # Looking from bottom, we want CW relative to top view.
            # So: TL, TR, BL
            faces.append([b_tl, b_tr, b_bl])
            faces.append([b_tr, b_br, b_bl])
            
    # B. Wall Extrusion (Sides)
    # We need to stitch the edges.
    
    # Top Edge (Row 0)
    for c in range(cols - 1):
        t_l = get_idx(0, c)
        t_r = get_idx(0, c+1)
        b_l = t_l + n_pixels
        b_r = t_r + n_pixels
        # Quad: t_l, t_r, b_r, b_l
        # Tri 1: t_l, t_r, b_l
        faces.append([t_l, t_r, b_l])
        # Tri 2: b_l, t_r, b_r
        faces.append([b_l, t_r, b_r])
        
    # Bottom Edge (Row rows-1)
    for c in range(cols - 1):
        t_l = get_idx(rows-1, c)
        t_r = get_idx(rows-1, c+1)
        b_l = t_l + n_pixels
        b_r = t_r + n_pixels
        # This is the "back" wall. Normals must point out (Y positive).
        # t_l is (X, MaxY).
        # Quad: t_r, t_l, b_l, b_r (Order reversed for CCW facing Y+)
        faces.append([t_r, t_l, b_r])
        faces.append([b_r, t_l, b_l])
        
    # Left Edge (Col 0)
    for r in range(rows - 1):
        t_t = get_idx(r, 0)
        t_b = get_idx(r+1, 0)
        b_t = t_t + n_pixels
        b_b = t_b + n_pixels
        # This is "left" (X=0). Normals point X negative.
        # Quad: t_b, t_t, b_t, b_b
        faces.append([t_b, t_t, b_b])
        faces.append([b_b, t_t, b_t])
        
    # Right Edge (Col cols-1)
    for r in range(rows - 1):
        t_t = get_idx(r, cols-1)
        t_b = get_idx(r+1, cols-1)
        b_t = t_t + n_pixels
        b_b = t_b + n_pixels
        # This is "right" (X=Max). Normals point X positive.
        # Quad: t_t, t_b, b_b, b_t
        faces.append([t_t, t_b, b_t])
        faces.append([b_t, t_b, b_b])

    # 4. Create Mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 5. Validate & Repair
    mesh.fix_normals()
    
    return mesh

def generate_improved_mesh(data_matrix, voxel_size_mm=1.0, smoothing_sigma=0.4, target_faces=None, is_binary=True, iso_level=0.0, upsample_factor=1):
    """
    Generates a high-quality mesh from a voxel grid using Marching Cubes.
    
    Args:
        data_matrix (np.ndarray): 3D numpy array (binary or continuous).
        voxel_size_mm (float): Physical size of one voxel in mm.
        smoothing_sigma (float): Gaussian smoothing sigma (in voxels).
        target_faces (int, optional): Target number of faces for decimation.
        is_binary (bool): If True, converts binary mask to SDF before meshing.
        iso_level (float): The isosurface level to extract (default 0.0 for SDF).
        upsample_factor (int): Factor to upscale the grid for higher fidelity (default 1).
        
    Returns:
        trimesh.Trimesh: The processed mesh object.
    """
    if is_binary:
        # 1. Convert to Signed Distance Field (SDF)
        # distance_transform_edt computes distance to the nearest zero (background)
        # We want solid to be positive, void to be negative.
        # dist_inside: distance to nearest void (inside solid)
        dist_inside = nd.distance_transform_edt(data_matrix)
        # dist_outside: distance to nearest solid (inside void)
        dist_outside = nd.distance_transform_edt(1 - data_matrix)
        
        # SDF: Positive inside, negative outside
        data_field = dist_inside - dist_outside
        level = 0.0
    else:
        # Use the continuous field directly
        data_field = data_matrix
        level = iso_level
    
    # NEW: Upsampling for Detail Preservation
    if upsample_factor > 1:
        # cubic interpolation (order=3) creates smooth transitions
        data_field = nd.zoom(data_field, upsample_factor, order=3)
        # Adjust voxel size because voxels are now smaller
        voxel_size_mm /= upsample_factor

    # 2. Apply Sub-Voxel Smoothing
    # Smooth the field to preserve topology while rounding corners
    if smoothing_sigma > 0:
        field_smooth = nd.gaussian_filter(data_field, sigma=smoothing_sigma)
    else:
        field_smooth = data_field
        
    # 3. Marching Cubes
    # spacing defines the physical dimensions of the voxel
    verts, faces, normals, values = measure.marching_cubes(
        field_smooth, 
        level=level,
        spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm)
    )
    
    # 4. Create Trimesh Object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # 5. Mesh Conditioning (Repair)
    # Merge vertices that are close (within 1e-5 by default)
    mesh.merge_vertices()
    
    # Remove degenerate faces (using boolean mask)
    # Trimesh 4.x: update_faces takes a mask
    mesh.update_faces(mesh.nondegenerate_faces())
    
    # Remove duplicate faces
    mesh.update_faces(mesh.unique_faces())
    
    # Fix normals (ensure they point outwards)
    trimesh.repair.fix_normals(mesh)
    # Fix inversion (if any faces are flipped)
    trimesh.repair.fix_inversion(mesh)
    
    # 6. Decimation (Optional)
    if target_faces and len(mesh.faces) > target_faces:
        try:
            # Try to use open3d or other backends if available
            # If not, trimesh might fallback or fail. 
            # We wrap in try/except to ensure robustness.
            # 'simplify_quadratic_decimation' is the preferred method for shape preservation
            mesh_simplified = mesh.simplify_quadratic_decimation(target_faces)
            mesh = mesh_simplified
        except Exception:
            # Fallback or just skip decimation if backend missing
            pass
            
    return mesh

def export_mesh_to_bytes(mesh, file_type='stl'):
    """
    Exports the mesh to a byte stream.
    """
    # Create a BytesIO object
    file_obj = io.BytesIO()
    # Export
    mesh.export(file_obj, file_type=file_type)
    file_obj.seek(0)
    return file_obj.getvalue()

import zipfile
import time

def generate_onshape_manifest():
    """
    Generates a text file with specific instructions for Onshape import.
    """
    text = """ONSHAPE IMPORT INSTRUCTIONS
---------------------------
To ensure this file imports correctly into Onshape, please use the following settings in the "Import" dialog:

1. [ ] Orient imported models with Y Axis Up  --> UNCHECKED (Leave blank)
   (This model is already Z-up aligned)

2. [ ] Create a composite part              --> UNCHECKED (Leave blank)
   (This ensures a single mesh body is created)

3. [ ] Join adjacent surfaces               --> UNCHECKED (Leave blank)
   (Not required for STL mesh files)

4. Units                                    --> Millimeter
   (The model is scaled 1:1 in millimeters)

---------------------------
Generated by 3D Physarum Simulation Platform
"""
    return text

def export_to_bundle(mesh, filename_base="scaffold"):
    """
    Exports the mesh and a manifest to a ZIP bundle.
    """
    # 1. Export STL
    stl_data = export_mesh_to_bytes(mesh, file_type='stl')
    
    # 2. Generate Manifest
    manifest_text = generate_onshape_manifest()
    
    # 3. Create ZIP
    bio_zip = io.BytesIO()
    with zipfile.ZipFile(bio_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add STL
        zf.writestr(f"{filename_base}.stl", stl_data)
        # Add Manifest
        zf.writestr("ONSHAPE_IMPORT_SETTINGS.txt", manifest_text)
        
    bio_zip.seek(0)
    return bio_zip.getvalue()


def calculate_voxel_size(params, grid_size=60):
    """
    Estimates the physical voxel size based on the hardcoded 200.0 normalization factor
    in physics.py.
    """
    # In physics.py: pillar_width_px = grid_size * (pillar_mm / 200.0)
    # This implies 200.0 represents the full scale of the grid dimension in mm?
    # Or rather, the grid represents a 200mm box?
    # Let's assume the domain size is 200mm for now based on the code logic.
    return 200.0 / float(grid_size)
