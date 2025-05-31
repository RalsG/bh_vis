import numpy as np
import sxs
from sxs.waveforms import WaveformModes # For type hints
import spherical
import quaternionic
from mayavi import mlab
from scipy.special import sph_harm
import imageio.v2 as imageio # clarification 
import os
import time
import inspect
import sys

# --- Configuration Parameters ---
SXS_ID = "SXS:BBH:0001"
OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_hplus_surface_movie.mp4"
NUM_FRAMES = 50
FPS = 24

# GW Surface Visualization Parameters
N_R_GW = 80
N_PHI_GW = 160
MIN_R_GW = 10.0
MAX_R_GW = 500.0
AMPLITUDE_SCALE = 75.0 # Factor to scale h+ for z-displacement (TUNE THIS!)
GW_SURFACE_COLOR = (0.3, 0.6, 1.0) # Uniform color for the GW surface

CAMERA_INITIAL_DISTANCE = MAX_R_GW * 0.9
CAMERA_FINAL_DISTANCE = MAX_R_GW * 1.8
CAMERA_ZOOM_START_TIME_FRAC = 0.3
CAMERA_ZOOM_END_TIME_FRAC = 0.7 # Extended zoom duration

# --- Helper Functions ---

# load_simulation_data remains the same
def load_simulation_data(sxs_id_str):
    print(f"Loading simulation: {sxs_id_str}")
    try:
        simulation = sxs.load(sxs_id_str, download=True, progress=True, ignore_deprecation=True)
    except Exception as e: print(f"Error loading simulation {sxs_id_str}: {e}"); raise
    strain_modes = None

    if hasattr(simulation, 'h') and simulation.h is not None: strain_modes = simulation.h
    
    if strain_modes is None or (hasattr(strain_modes, 'size') and strain_modes.size == 0) or \
       (hasattr(strain_modes, 'data') and strain_modes.data.size == 0):
        raise ValueError(f"Strain not found or empty for simulation {sxs_id_str}.")
    
    print(f"strain modes loaded. Time range: {strain_modes.t[0]:.2f}M to {strain_modes.t[-1]:.2f}M.")
    
    horizons_data = getattr(simulation, 'horizons', None)
    if horizons_data: print("Horizons data loaded.")
    else: print("Warning: No 'horizons' attribute.")
    return strain_modes, horizons_data


# get_bh_mesh_data remains the same
def get_bh_mesh_data(horizons_obj, bh_label, time_val, n_theta_bh=20, n_phi_bh=40):
    if horizons_obj is None or not hasattr(horizons_obj, bh_label):
        return np.empty((0,0)), np.empty((0,0)), np.empty((0,0))
    try:
        bh = getattr(horizons_obj, bh_label)
        com = bh.coord_center_inertial_spline(time_val)
        R_horizon_approx = 1.0
        if time_val > 0 and bh_label == 'A': R_horizon_approx = 1.5
        u, v = np.linspace(0, 2*np.pi, n_phi_bh), np.linspace(0, np.pi, n_theta_bh)
        x = R_horizon_approx * np.outer(np.cos(u), np.sin(v)) + com[0]
        y = R_horizon_approx * np.outer(np.sin(u), np.sin(v)) + com[1]
        z = R_horizon_approx * np.outer(np.ones_like(u), np.cos(v)) + com[2]
        return x, y, z
    except Exception:
        return np.empty((0,0)), np.empty((0,0)), np.empty((0,0))

def reconstruct_hplus_on_xy_plane_at_time_t(
    sxs_strain_modes: sxs.waveforms.WaveformModes,
    lab_time_t: float,
    r_coords: np.ndarray,
    phi_coords: np.ndarray,
    spin_weight: int = -2
):
    """
    Reconstructs h+ on a specified x-y polar grid at a given lab time.
    Returns a 2D array for z-displacement (scaled h+).
    Shape will be (len(r_coords), len(phi_coords)).
    """
    z_displacement_grid = np.zeros((len(r_coords), len(phi_coords)))
    theta_val_equator = np.pi / 2.0

    # Prepare the 1D theta array for evaluation.
    theta_for_evaluation = np.full_like(phi_coords, theta_val_equator)

    # Prepare a 1D array of zeros for the third Euler angle, gamma.
    # This is needed when converting to quaternions if gamma isn't explicitly used
    # in the evaluation of the scalar field (h+).
    gamma_for_evaluation = np.zeros_like(phi_coords)

    # Convert Euler angles (alpha, beta, gamma) to quaternions (rotors).
    # alpha corresponds to phi, beta to theta.
    # This creates an array of quaternion objects, one for each (phi, theta) point.
    rotors_for_evaluation = quaternionic.array.from_euler_angles(phi_coords, theta_for_evaluation, gamma_for_evaluation)

    for i, r_val in enumerate(r_coords):
        retarded_time_scalar = lab_time_t - r_val
        retarded_time_for_interp = np.array([retarded_time_scalar])

        try:
            strain_modes_at_retarded_t_obj = sxs_strain_modes.interpolate(retarded_time_for_interp)
            h_coeffs_at_ret_t = strain_modes_at_retarded_t_obj.data.ravel()
            
            if h_coeffs_at_ret_t.size == 0:
                z_displacement_grid[i, :] = 0.0
                continue

        except ValueError:
            z_displacement_grid[i, :] = 0.0
            continue

        
        sph_modes_obj = spherical.Modes(h_coeffs_at_ret_t,
                                spin_weight=spin_weight,
                                ell_min=sxs_strain_modes.ell_min,
                                ell_max=sxs_strain_modes.ell_max)

        complex_strain_values_at_r = sph_modes_obj.evaluate(rotors_for_evaluation)
        # Could simply use the scri.WaveformModes.to_grid() method and avoid quaternions,
        # but that creates the whole spherical grid and we only need a ring

        z_displacement_grid[i, :] = complex_strain_values_at_r.real * AMPLITUDE_SCALE
        
    return z_displacement_grid


# --- Main Animation Logic ---
def create_merger_movie():
    script_init_time = time.time()
    strain_modes_sxs, horizons_data = load_simulation_data(SXS_ID)

    sxs_times_all = strain_modes_sxs.t
    peak_strain_time = strain_modes_sxs.max_norm_time()
    anim_start_time = peak_strain_time - 500.0 
    anim_end_time = peak_strain_time + 100.0
    anim_start_time = max(anim_start_time, sxs_times_all[0])
    anim_end_time = min(anim_end_time, sxs_times_all[-1])
    anim_time_indices = np.linspace(
        np.argmin(np.abs(sxs_times_all - anim_start_time)),
        np.argmin(np.abs(sxs_times_all - anim_end_time)),
        NUM_FRAMES, dtype=int, endpoint=True
    )
    anim_lab_times = sxs_times_all[anim_time_indices]
    print(f"Animation time: {anim_lab_times[0]:.2f}M to {anim_lab_times[-1]:.2f}M over {len(anim_lab_times)} frames.")



    r_gw_axis = np.linspace(MIN_R_GW, MAX_R_GW, N_R_GW)
    phi_gw_axis = np.linspace(0, 2 * np.pi, N_PHI_GW, endpoint=True)
    PHI_GW_MESH, R_GW_MESH = np.meshgrid(phi_gw_axis, r_gw_axis)
    X_GW_SURF = R_GW_MESH * np.cos(PHI_GW_MESH)
    Y_GW_SURF = R_GW_MESH * np.sin(PHI_GW_MESH)


    mlab.figure(size=(1280, 1024), bgcolor=(0.05, 0.05, 0.05))
    # mlab.options.offscreen = True # Ensure offscreen rendering for saving frames without GUI pop-up


    """bh1_pos_init = get_bh_mesh_data(horizons_data, 'A', anim_lab_times[0])
    bh_A_plot = mlab.mesh(bh1_pos_init[0], bh1_pos_init[1], bh1_pos_init[2], color=(0,0,0))
    bh2_pos_init = get_bh_mesh_data(horizons_data, 'B', anim_lab_times[0])
    bh_B_plot = mlab.mesh(bh2_pos_init[0], bh2_pos_init[1], bh2_pos_init[2], color=(0,0,0))"""


    frame_files = []
    frames_dir_path = "frames" # Changed directory name as requested
    os.makedirs(frames_dir_path, exist_ok=True)




    total_physical_anim_time = anim_lab_times[-1] - anim_lab_times[0]
    if total_physical_anim_time == 0: total_physical_anim_time = 1.0 # Avoid division by zero
    


    print("Starting frame rendering loop...")
    for i_frame, current_lab_time in enumerate(anim_lab_times):
        frame_render_start_time = time.time()
        print(f"Processing frame {i_frame+1}/{NUM_FRAMES} for lab_time = {current_lab_time:.2f} M")

        z_gw_frame = reconstruct_hplus_on_xy_plane_at_time_t(
            strain_modes_sxs, current_lab_time, r_gw_axis, phi_gw_axis
        )

        mlab.clf()
        mlab.mesh(X_GW_SURF, Y_GW_SURF, z_gw_frame,
                        color=GW_SURFACE_COLOR,
                        representation='wireframe',
                        name="GW h+ Surface",
                        opacity=0.8,
                        # warp_scale='auto'
                        )



        """x_a, y_a, z_a = get_bh_mesh_data(horizons_data, 'A', current_lab_time)
        bh_A_plot.mlab_source.reset(x=x_a, y=y_a, z=z_a)
        if current_lab_time < peak_psi4_time + 20:
            x_b, y_b, z_b = get_bh_mesh_data(horizons_data, 'B', current_lab_time)
            bh_B_plot.mlab_source.reset(x=x_b, y=y_b, z=z_b)
            bh_B_plot.actor.actor.visibility = True if x_b.size > 0 else False
        else:
            bh_B_plot.mlab_source.reset(x=np.empty((0,0)), y=np.empty((0,0)), z=np.empty((0,0)))
            bh_B_plot.actor.actor.visibility = False"""
        
        current_anim_frac = (current_lab_time - anim_lab_times[0]) / total_physical_anim_time if total_physical_anim_time > 0 else 0
        if current_anim_frac < CAMERA_ZOOM_START_TIME_FRAC: cam_dist = CAMERA_INITIAL_DISTANCE
        elif current_anim_frac > CAMERA_ZOOM_END_TIME_FRAC: cam_dist = CAMERA_FINAL_DISTANCE
        else:
            progress = (current_anim_frac - CAMERA_ZOOM_START_TIME_FRAC) / \
                       (CAMERA_ZOOM_END_TIME_FRAC - CAMERA_ZOOM_START_TIME_FRAC)
            cam_dist = CAMERA_INITIAL_DISTANCE + (CAMERA_FINAL_DISTANCE - CAMERA_INITIAL_DISTANCE) * progress
        
        mlab.view(azimuth=30 + i_frame*0.15, elevation=60, distance=cam_dist, focalpoint=(0,0,0))
        mlab.text(0.02, 0.92, f"Time: {current_lab_time:.1f} M", width=0.25)
        
        mlab.orientation_axes()
        frame_filename = f"{frames_dir_path}/frame_{i_frame:04d}.png"

        mlab.savefig(frame_filename)
        frame_files.append(frame_filename)
        print(f"  Frame saved to {frame_filename}. Rendered in {time.time() - frame_render_start_time:.2f}s")

    print("All animation frames rendered.")
    print("Compiling movie...")

    images = []
    for f_name in frame_files:
        if not os.path.exists(f_name):
            print(f"{f_name} not found")
            continue
        images.append(imageio.imread(f_name))

    imageio.mimsave(OUTPUT_MOVIE_FILENAME, images, fps=FPS)
    print(f"Movie saved to {OUTPUT_MOVIE_FILENAME}")
    
    mlab.close(all=True)
    print(f"Total script execution time: {time.time() - script_init_time:.2f} seconds.")

if __name__ == "__main__":
    create_merger_movie()
