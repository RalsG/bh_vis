import numpy as np
import sxs
import spherical
import quaternionic
from mayavi import mlab
import matplotlib.pyplot as plt
import vtk
import imageio.v2 as imageio # clarification
import cv2
import os
import time
import inspect
import sys

# --- Configuration Parameters ---
SXS_ID = "SXS:BBH:0001"
OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_PiP_movie.mp4"
auto_loop_bool = False # option to make many movies
sxs_idx_start = 160
loop_size = 4
representation_str = 'surface'
NUM_FRAMES = 50
FPS = 24

# GW Surface Visualization Parameters
MAX_R_STEP = 40.0 # In units of M for the adaptive grid sampling.
MIN_R_STEP = 0.7
POINTS_PER_WAVE = 20.0 # Higher means more compute but smoother surface waves
MIN_R_GW = 12.0
MAX_R_GW = 800.0
NUM_PHI_GW = 500
AMPLITUDE_SCALE = 0.45 * MAX_R_GW # Factor to scale h+ for z-displacement (TUNE THIS!)
GW_SURFACE_COLOR = (0.3, 0.6, 1.0) # Uniform color for the GW surface
BG_COLOR = (0.4, 0.4, 0.4)
SPIN_ARROW_COLOR = (0.85, 0.85, 0.1)

PIP_CAMERA_DISTANCE = MIN_R_GW * 2.5
MAIN_CAMERA_DISTANCE = MAX_R_GW * 1.8
PIP_SCALE = 2.7 # This is the ratio of window to PiP
PROGRESS_WAVE_SCALE = 8 # ratio of window to progress waveform
antial_bool = False


# --- Helper Functions ---
def load_simulation_data(sxs_id_str):
    print(f"Loading simulation: {sxs_id_str}")
    try:
        simulation = sxs.load(sxs_id_str, download=True, progress=True, ignore_deprecation=True)
    except Exception as e: print(f"Error loading simulation {sxs_id_str}: {e}"); raise
    
    strain_modes = getattr(simulation, 'h', None)
    if strain_modes is not None: print(f"strain modes loaded. Time range: {strain_modes.t[0]:.2f}M to {strain_modes.t[-1]:.2f}M.")
    else: raise ValueError(f"Strain not found or empty for simulation {sxs_id_str}.")
    
    horizons_data = getattr(simulation, 'horizons', None)
    if horizons_data is not None: print("Horizons data loaded.")
    else: print(f"Warning: horizons not found or empty for simulation {sxs_id_str}.")

    return strain_modes, horizons_data

def pseudo_uniform_times(
        sample_times: np.ndarray,
        peak_strain_time: float,
        start_back_prop: float = 0.6,
        end_for_prop: float = 0.15
):
    sim_start_time = sample_times[0]
    sim_end_time = sample_times[-1]
    sim_total_time = sim_end_time - sim_start_time
    anim_start_time = peak_strain_time - (start_back_prop * sim_total_time)
    anim_end_time = (end_for_prop * sim_total_time) + peak_strain_time
    anim_start_time = max(anim_start_time, sim_start_time)
    anim_end_time = min(anim_end_time, sim_end_time)

    uniform_time_array = np.linspace(
        anim_start_time,
        anim_end_time,
        NUM_FRAMES, endpoint=True
    )
    anim_time_indices = np.searchsorted(sample_times, uniform_time_array)
    anim_time_array = sample_times[anim_time_indices]

    return anim_time_array, anim_time_indices


def make_progress_signal_plot(dom_mode_signal: sxs.waveforms.WaveformModes,
                            anim_lab_times: np.ndarray,
                            anim_time_indices: np.ndarray,
                            current_frame_index: int,
                            width: int,
                            height: int,
                            bg_color: tuple[float, float, float]
):
    trimmed_strain_signal = dom_mode_signal[anim_time_indices[0]:(anim_time_indices[-1] + 1)]
    time_array = trimmed_strain_signal.t


    h_complex = np.asarray(trimmed_strain_signal.data)
    real_h_array = h_complex.real

    dpi = 100
    figsize_inches = (width / dpi, height / dpi)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Plot the full waveform in the background for context
    ax.plot(time_array, real_h_array, color=(0.95, 0.85, 0.95), alpha=0.5, linewidth=1.0)

    # Plot the "live" part of the waveform up to the current frame
    start_time_idx = anim_time_indices[0]
    current_time_idx = anim_time_indices[current_frame_index]
    live_time = time_array[:(current_time_idx - start_time_idx + 1)]
    live_strain = real_h_array[:(current_time_idx - start_time_idx + 1)]
    ax.plot(live_time, live_strain, color='cyan', linewidth=1.5)

    # Add a vertical "now" line
    ax.axvline(x=live_time[-1], color='red', linestyle='--', linewidth=1.0, alpha=0.8)

    y_min = real_h_array.min() * 1.1
    y_max = real_h_array.max() * 1.1
    ax.set_xlim(time_array[0], time_array[-1])
    ax.set_ylim(y_min, y_max)

    # Remove all chart junk (ticks, labels, spines) for a clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Ensure there is no padding around the plot
    fig.tight_layout(pad=0)

    # --- Rendering to NumPy array ---
    # Draw the canvas and get the RGB buffer
    fig.canvas.draw()
    buffer_rgb = fig.canvas.buffer_rgba()
    # Convert the buffer to a NumPy array of the correct shape
    plot_array = np.asarray(buffer_rgb)[:, :, :3]
    # Close the figure to free up memory, crucial for running in a loop
    plt.close(fig)

    return plot_array


def get_bh_mesh_data(horizons_obj, time_vals, n_theta_bh=15, n_phi_bh=20):
    # --- Initial Data Extraction and Interpolation ---
    # If .interpolate() fails (e.g., time_vals outside original range AND
    # the TimeSeries.interpolate doesn't handle fill_value/bounds_error itself
    # in a way that produces NaNs), this could still be an issue.
    # Let's assume TimeSeries.interpolate will return data aligned with time_vals,
    # possibly with NaNs if interpolation is outside original range and it's configured to do so.

    # For coordinates (expected shape of .data: (len(time_vals), 3))
    A_coords_ts_at_times = horizons_obj.A.coord_center_inertial.interpolate(time_vals)
    B_coords_ts_at_times = horizons_obj.B.coord_center_inertial.interpolate(time_vals)
    C_coords_ts_at_times = horizons_obj.C.coord_center_inertial.interpolate(time_vals)

    # For areal mass (expected shape of .data: (len(time_vals),))
    A_mass_ts_at_times = horizons_obj.A.areal_mass.interpolate(time_vals)
    B_mass_ts_at_times = horizons_obj.B.areal_mass.interpolate(time_vals)
    C_mass_ts_at_times = horizons_obj.C.areal_mass.interpolate(time_vals)

    # Pull spins (as Cartesian vectors) and spin magnitudes for norming vectors to radius
    A_chi_at_times = horizons_obj.A.chi_inertial.interpolate(time_vals)
    B_chi_at_times = horizons_obj.B.chi_inertial.interpolate(time_vals)
    C_chi_at_times = horizons_obj.C.chi_inertial.interpolate(time_vals)
    
    # chi_inertial_mag is *supposed* to be a TimeSeries, but isn't for all the simulations I've pulled
    # so we just use NumPy's interpolator. Also the time (t) attribute should be the same across A and B, not sure about C
    A_chi_mag_data = np.interp(time_vals, horizons_obj.A.time, horizons_obj.A.chi_inertial_mag)
    B_chi_mag_data = np.interp(time_vals, horizons_obj.B.time, horizons_obj.B.chi_inertial_mag)
    C_chi_mag_data = np.interp(time_vals, horizons_obj.C.time, horizons_obj.C.chi_inertial_mag)

    # Extract NumPy data arrays from the TimeSeries objects
    A_coords_data = np.asarray(A_coords_ts_at_times.data)
    B_coords_data = np.asarray(B_coords_ts_at_times.data)
    C_coords_data = np.asarray(C_coords_ts_at_times.data)

    A_rad_data = 2 * np.asarray(A_mass_ts_at_times.data)
    B_rad_data = 2 * np.asarray(B_mass_ts_at_times.data)
    C_rad_data = 2 * np.asarray(C_mass_ts_at_times.data)

    A_chi_data = np.asarray(A_chi_at_times.data)
    B_chi_data = np.asarray(B_chi_at_times.data)
    C_chi_data = np.asarray(C_chi_at_times.data)

    # Use spin components, radii, and center coords of the BHs
    # to make the spin vectors start at the horizon of each BH in the direction of the spin
    # There had to be a better way but figuring out a loop was gonna take as long as hardcoding :|
    A_vec_pos_t = [A_coords_data[:, 0] + A_rad_data * (A_chi_data[:, 0]/A_chi_mag_data),
                   A_coords_data[:, 1] + A_rad_data * (A_chi_data[:, 1]/A_chi_mag_data),
                   A_coords_data[:, 2] + A_rad_data * (A_chi_data[:, 2]/A_chi_mag_data)]
    B_vec_pos_t = [B_coords_data[:, 0] + B_rad_data * (B_chi_data[:, 0]/B_chi_mag_data),
                   B_coords_data[:, 1] + B_rad_data * (B_chi_data[:, 1]/B_chi_mag_data),
                   B_coords_data[:, 2] + B_rad_data * (B_chi_data[:, 2]/B_chi_mag_data)]
    C_vec_pos_t = [C_coords_data[:, 0] + C_rad_data * (C_chi_data[:, 0]/C_chi_mag_data),
                   C_coords_data[:, 1] + C_rad_data * (C_chi_data[:, 1]/C_chi_mag_data),
                   C_coords_data[:, 2] + C_rad_data * (C_chi_data[:, 2]/C_chi_mag_data)]

    chi_arrays_to_plot = [np.asarray([*A_vec_pos_t, A_chi_data[:, 0], A_chi_data[:, 1], A_chi_data[:, 2]]),
                          np.asarray([*B_vec_pos_t, B_chi_data[:, 0], B_chi_data[:, 1], B_chi_data[:, 2]]),
                          np.asarray([*C_vec_pos_t, C_chi_data[:, 0], C_chi_data[:, 1], C_chi_data[:, 2]])
                          ]
    chi_max = max(np.max(A_chi_mag_data), np.max(B_chi_mag_data), np.max(C_chi_mag_data))

    # Now for actual BH surfaces
    all_coords_over_time = [A_coords_data, B_coords_data, C_coords_data]
    all_radii_over_time = [A_rad_data, B_rad_data, C_rad_data]
    surfaces_along_time = [[], [], []] # To store [(x,y,z)_t0, (x,y,z)_t1, ...] for each BH

    # --- Pre-calculate unit sphere points ---
    # These define the SHAPE of the sphere. They are 2D arrays.
    # Theta: polar angle from z-axis (0 to pi)
    # Phi: azimuthal angle in x-y plane (0 to 2pi)
    theta_1d = np.linspace(0, np.pi, n_theta_bh)  # n_theta_bh points from 0 to pi
    phi_1d = np.linspace(0, 2 * np.pi, n_phi_bh)    # n_phi_bh points from 0 to 2pi

    # Create 2D grids. For mlab.mesh, shapes should be (n_theta_bh, n_phi_bh)
    # Using indexing='ij' makes theta_grid vary along rows, phi_grid along columns.
    # If you used `meshgrid(phi_1d, theta_1d)`, then shapes would be (n_phi_bh, n_theta_bh)
    # Let's stick to (n_rows=n_theta_bh, n_cols=n_phi_bh) for the mesh.
    # Theta (polar) varies along the first dimension, Phi (azimuthal) along the second.
    phi_grid, theta_grid = np.meshgrid(phi_1d, theta_1d) # phi_grid & theta_grid are (n_theta_bh, n_phi_bh)

    # Unit sphere coordinates (2D arrays)
    x_unit_sphere = np.sin(theta_grid) * np.cos(phi_grid)
    y_unit_sphere = np.sin(theta_grid) * np.sin(phi_grid)
    z_unit_sphere = np.cos(theta_grid)
    # These are now all shape (n_theta_bh, n_phi_bh)

    # --- Loop through time and generate sphere surfaces ---
    for t_idx in range(len(time_vals)):
        for i in range(3): # For BH A, B, C
            current_center = all_coords_over_time[i][t_idx]  # Should be [cx, cy, cz] for this time
            current_radius = all_radii_over_time[i][t_idx]  # Should be scalar radius for this time

            # Handle cases where data might be NaN (e.g., interpolation failure, horizon not existing)
            if np.isnan(current_radius) or current_radius <= 1e-9 or np.any(np.isnan(current_center)):
                # If radius is NaN, zero, negative or center is NaN, store None or empty arrays
                # to indicate no valid surface at this time for this BH.
                # Mayavi's mlab.mesh can handle x,y,z being single points if you want to "hide" it,
                # or you can explicitly pass None and handle it in the animation loop.
                surfaces_along_time[i].append(None) # Or (np.array([]), np.array([]), np.array([]))
                continue

            # Scale and translate the unit sphere to the actual horizon position and size
            # These operations are vectorized because x_unit_sphere etc. are 2D arrays.
            x_surface = current_center[0] + current_radius * x_unit_sphere
            y_surface = current_center[1] + current_radius * y_unit_sphere
            z_surface = current_center[2] + current_radius * z_unit_sphere
            # x_surface, y_surface, z_surface are now 2D arrays of shape (n_theta_bh, n_phi_bh)

            surfaces_along_time[i].append((x_surface, y_surface, z_surface))
            
    omega_orbit = horizons_obj.omega

    return surfaces_along_time, chi_arrays_to_plot, chi_max, omega_orbit


def generate_adaptive_r_coords(
    r_axis: np.ndarray,
    angular_velocity_ts: sxs.TimeSeries,
    lab_time_t: float,
    lab_time_step: float,
    r_min: float,
    r_max: float,
    max_r_step: float = 40.0,
    min_r_step: float = 0.5,
    samples_per_wavelength = 20.0
) -> np.ndarray:
    """
    Generates a non-uniform radial grid by adapting to the local spatial
    wavelength of the gravitational wave.

    Args:
        angular_velocity_ts: A sxs.TimeSeries object representing the
                             binary's orbital angular velocity vector over time.
        lab_time_t: The current lab time of the visualization.
        r_min, r_max: The radial domain.
        samples_per_wavelength: Samples per wavelength. Must be > 2.
                                Higher values (e.g., 5-8) give smoother results.

    Returns:
        A 1D numpy array of non-uniformly spaced radial coordinates.
    """
    if r_axis is not None:
        temp = np.searchsorted(r_axis, r_max - lab_time_step, side='left')
        r_points = r_axis[:temp] + lab_time_step
        num_chopped = len(r_axis) - temp
        print(num_chopped)
        if r_points[-1] < r_max:
           r_points = np.append(r_points, [r_max])
        r_max = r_points[0]
    else:
        r_points = np.array([r_max])
        num_chopped = 0
    current_r = r_max

    # Create an interpolator for the angular velocity for efficiency.
    # This is much faster than recreating it inside the loop.
    ang_vel_interp = angular_velocity_ts.interpolate

    # Define a minimum frequency to prevent infinitely large steps
    # during the very early, slow inspiral.
    min_omega_gw = 2*np.pi / (max_r_step * samples_per_wavelength)

    while current_r > r_min:
        retarded_time = np.array([lab_time_t - current_r])

        try:
            omega_orb = ang_vel_interp(retarded_time).data[0]
            # The dominant GW frequency is twice the orbital frequency.
            omega_gw = 2.0 * omega_orb

        except ValueError:
            # If time is out of bounds, use the minimum frequency to step through.
            omega_gw = min_omega_gw

        # Ensure frequency is not zero to avoid division by zero.
        omega_gw = max(omega_gw, min_omega_gw)

        # Calculate the desired step size. The spatial wavelength of the
        # wave is (2*pi / omega_gw). We sample this wavelength N times.
        spatial_wavelength = 2.0 * np.pi / omega_gw
        delta_r = spatial_wavelength / samples_per_wavelength
        delta_r = max(delta_r, min_r_step)
        # Advance the current radius
        current_r -= delta_r
        
        # Append the new point, but avoid overshooting the boundary.
        if current_r > r_min:
            r_points = np.insert(r_points, 0, current_r)
        else:
            break

    # Ensure the last point is exactly r_min for a clean boundary.
    if r_points[0] > r_min:
        r_points = np.insert(r_points, 0, r_min)
    
    return r_points, num_chopped


def reconstruct_hplus_on_xy_plane_at_time_t(
    old_z_grid: np.ndarray,
    old_color_scalars: np.ndarray,
    sxs_strain_modes: sxs.waveforms.WaveformModes,
    lab_time_t: float,
    num_points_chopped: float,
    r_coords: np.ndarray,
    phi_coords: np.ndarray,
    spin_weight: int = -2
):
    """
    Reconstructs h_+ and h_x on a specified x-y polar grid at a given lab time.
    Returns a 2D array for z-displacement (scaled h_+) and surface color (scaled h_x).
    Shape will be (len(r_coords), len(phi_coords)).
    Now takes in old strain arrays and steps them forward in time
    """
    if old_z_grid is not None:
        buffer_size = len(r_coords) - len(old_z_grid) + num_points_chopped
        grid_buffer = np.zeros((buffer_size, len(old_z_grid[0])))
        z_displacement_grid = np.concatenate((grid_buffer, old_z_grid[:-num_points_chopped]), axis=0)
        print(grid_buffer.shape)
        color_scalars = np.concatenate((grid_buffer, old_color_scalars[:-num_points_chopped]), axis=0)
        r_coords_for_eval = r_coords[:buffer_size]
        print(len(r_coords_for_eval), len(phi_coords))

    else:
        z_displacement_grid = np.zeros((len(r_coords), len(phi_coords)))
        color_scalars = np.zeros((len(r_coords), len(phi_coords)))
        r_coords_for_eval = r_coords

    # --- Vectorized Interpolation ---
    # Create a 1D array of all unique retarded times needed.
    # r_coords parameter should be INCREASING
    r_coords_flipped = np.flip(r_coords_for_eval)
    retarded_times_vec = np.asarray(lab_time_t - r_coords_flipped)

    # Times must be strictly INCREASING

    # try:
    # Perform one single, fast interpolation for all radii at once.
    temp_time = time.time()
    strain_modes_at_times_obj = sxs_strain_modes.interpolate(retarded_times_vec)
    print(f"interpolation took {(time.time() - temp_time):.2f}s")
    # h_coeffs_matrix will have shape (len(r_coords), n_modes)
    h_coeffs_matrix_flipped = strain_modes_at_times_obj.data
    h_coeffs_matrix = np.flip(h_coeffs_matrix_flipped, axis=0)
    """except ValueError:
        print(f"Out of bounds at lab time {lab_time_t}")
        sys.exit()
        # This will be raised if any time in retarded_times_vec is out of bounds.
        # We return a flat plane as a fallback.
        # A more robust solution could identify which times are valid and set others to zero.
        # For simplicity here, we assume if one is bad, all are bad for this frame.
        return z_displacement_grid, color_scalars"""

    theta_val_equator = np.pi / 2.0
    theta_for_evaluation = np.full_like(phi_coords, theta_val_equator)
    # Convert spherical coordinates to quaternions (rotors).
    # This creates an array of quaternion objects, one for each (phi, theta) point.
    temp_time = time.time()
    rotors_for_evaluation = quaternionic.array.from_spherical_coordinates(theta_for_evaluation, phi_coords)
    print(f"making quaternions took {(time.time() - temp_time):.2f}s")


    ell_min, ell_max = sxs_strain_modes.ell_min, sxs_strain_modes.ell_max

    sph_modes_time_sum = 0
    sph_eval_time_sum = 0

    # Now, loop through the results of the interpolation. This loop is much faster
    # because the expensive part (interpolation) is already done.
    for i in range(len(r_coords_for_eval)):
        # Get the row of coefficients for the i-th radius
        h_coeffs_for_this_r = h_coeffs_matrix[i, :]
        
        if np.all(h_coeffs_for_this_r == 0):
            # This can happen if the interpolation returned zeros for out-of-bounds times
            # that were handled internally by sxs.
            z_displacement_grid[i, :] = 0.0
            color_scalars[i, :] = 0.0
            continue
            
        temp_time = time.time()    
        sph_modes_obj = spherical.Modes(
            h_coeffs_for_this_r,
            spin_weight=spin_weight,
            ell_min=ell_min,
            ell_max=ell_max
        )
        sph_modes_time_sum += time.time() - temp_time

        temp_time = time.time() 
        complex_strain_values_at_r = sph_modes_obj.evaluate(rotors_for_evaluation)
        # Could simply use the scri.WaveformModes.to_grid() method and avoid quaternions,
        # but that creates the whole spherical grid and we only need a ring
        sph_eval_time_sum += time.time() - temp_time

        z_displacement_grid[i, :] = complex_strain_values_at_r.real * AMPLITUDE_SCALE
        color_scalars[i, :] = complex_strain_values_at_r.imag

    print(f"making spherical.Modes object took {sph_modes_time_sum:.2f}s")
    print(f"evaluating SWSHs took {sph_eval_time_sum:.2f}s")

    return z_displacement_grid, color_scalars


# --- Main Animation Logic ---
def create_merger_movie():
    script_init_time = time.time()
    strain_modes_sxs, horizons_data = load_simulation_data(SXS_ID)
    data_loaded_time = time.time()
    print(f"Data loading took {data_loaded_time - script_init_time:.2f}s")
    start_back_prop = 0.1 # fraction of total sim time to go back from peak strain for the start
    end_for_prop = 0.05 # fraction of total sim time to go forwards from peak strain for the end

    dom_l, dom_m = 2, 2
    h_lm_signal = strain_modes_sxs[:, strain_modes_sxs.index(dom_l, dom_m)]
    
    # Ideally we wouldn't be interpolating and would just use the closest sim sample time
    # HOWEVER: the horizon and waveform data are sampled at completely different timesteps (smart)
    # so this is where we're at. Would still be good to at least avoid interpolating the waveform data, add later
    peak_strain_time = strain_modes_sxs.max_norm_time()
    sample_times = strain_modes_sxs.t
    
    anim_lab_times, anim_time_indices = pseudo_uniform_times(sample_times, peak_strain_time, start_back_prop, end_for_prop)

    print(f"Animation time: {anim_lab_times[0]:.2f}M to {anim_lab_times[-1]:.2f}M over {len(anim_lab_times)} frames.")

    common_horizon_start = (horizons_data.A.time[-1] + horizons_data.C.time[0])/2
    phi_gw_axis = np.linspace(0, 2 * np.pi, NUM_PHI_GW, endpoint=True)

    mlab.figure(size=(1280, 1024), bgcolor=BG_COLOR)
    # mlab.options.offscreen = True # Ensure offscreen rendering for saving frames without GUI pop-up

    bh_surfs, spin_vectors, spin_arrow_size, orbital_vel_t = get_bh_mesh_data(horizons_data, anim_lab_times)
    spin_arrow_size *= 8

    frame_files = []
    frames_dir_path = "frames" # Changed directory name as requested
    os.makedirs(frames_dir_path, exist_ok=True)

    total_physical_anim_time = anim_lab_times[-1] - anim_lab_times[0]
    if total_physical_anim_time == 0: total_physical_anim_time = 1.0 # Avoid division by zero

    print(f"Processing and surface building took {time.time() - data_loaded_time:.2f}s")
    print("Starting frame rendering loop...")
    r_points_sum = 0
    r_gw_axis = None
    z_gw_frame = None
    GW_color_scalars = None

    for i_frame, current_lab_time in enumerate(anim_lab_times):
        frame_render_start_time = time.time()
        if not auto_loop_bool:
            print(f"Processing frame {i_frame+1}/{NUM_FRAMES} for lab_time = {current_lab_time:.2f} M")
        
        lab_time_step = current_lab_time - anim_lab_times[i_frame - 1] # It will be the whole animation time on frame 0, but is not used then

        temp_time = time.time()
        r_gw_axis, num_points_chopped = generate_adaptive_r_coords(r_gw_axis, orbital_vel_t, current_lab_time,
                                               lab_time_step, MIN_R_GW,
                                               MAX_R_GW, MAX_R_STEP, MIN_R_STEP, POINTS_PER_WAVE)
        num_r_points = len(r_gw_axis)
        r_points_sum += num_r_points
        PHI_GW_MESH, R_GW_MESH = np.meshgrid(phi_gw_axis, r_gw_axis)
        X_GW_SURF = R_GW_MESH * np.cos(PHI_GW_MESH)
        Y_GW_SURF = R_GW_MESH * np.sin(PHI_GW_MESH)
        print(f"Making and meshing a grid took {(time.time() - temp_time):.2f}s")
        
        temp_time = time.time()
        z_gw_frame, GW_color_scalars = reconstruct_hplus_on_xy_plane_at_time_t(
            z_gw_frame, GW_color_scalars, strain_modes_sxs, current_lab_time,
            num_points_chopped, r_gw_axis, phi_gw_axis)
        print(f"whole function for making surface data took {(time.time() - temp_time):.2f}s")

        mlab.clf()
        
        temp_time = time.time()
        if current_lab_time < common_horizon_start:
            # plot BH1
            mlab.mesh(*bh_surfs[0][i_frame], opacity=1, color=(0, 0, 0), name='Event Horizon 1')
            spin1_obj = mlab.quiver3d(*spin_vectors[0][:, i_frame], color=SPIN_ARROW_COLOR,
                        line_width = 0.7*spin_arrow_size, scale_factor = spin_arrow_size, name='Spin 1')
            # plot BH2
            mlab.mesh(*bh_surfs[1][i_frame], opacity=1, color=(0, 0, 0), name='Event Horizon 2')
            spin2_obj = mlab.quiver3d(*spin_vectors[1][:, i_frame], color=SPIN_ARROW_COLOR,
                        line_width = 0.7*spin_arrow_size, scale_factor = spin_arrow_size, name='Spin 2')
        else:
            # plot merged BH
            mlab.mesh(*bh_surfs[2][i_frame], opacity=1, color=(0, 0, 0), name='Event Horizon 3')
            spin3_obj = mlab.quiver3d(*spin_vectors[2][:, i_frame], color=SPIN_ARROW_COLOR,
                        line_width = 0.7*spin_arrow_size, scale_factor = spin_arrow_size, name='Spin 3')
        print(f"rendering BHs took {(time.time() - temp_time):.2f}s")

        mlab.view(azimuth=30, elevation=60, distance=PIP_CAMERA_DISTANCE, focalpoint=(0,0,0))

        pip_arr_large = mlab.screenshot(antialiased=True)
        if current_lab_time < common_horizon_start:
            spin1_obj.remove()
            spin2_obj.remove()
        else:
            spin3_obj.remove()

        # plot GW surface
        temp_time = time.time()
        mlab.mesh(X_GW_SURF, Y_GW_SURF, z_gw_frame,
                        scalars=GW_color_scalars, colormap='winter',
                        representation = representation_str,
                        name="GW h+ Surface", opacity=0.9)
        print(f"rendering GW surface took {(time.time() - temp_time):.2f}s")

        mlab.view(azimuth=30, elevation=60, distance=MAIN_CAMERA_DISTANCE, focalpoint=(0,0,0))
        main_arr = mlab.screenshot(antialiased=True)
        temp_time = time.time()
        # Resize PiP image
        orig_pip_h, orig_pip_w, _ = pip_arr_large.shape
        new_pip_h, new_pip_w = int(orig_pip_h / PIP_SCALE), int(orig_pip_w / PIP_SCALE)
        pip_arr_resized = cv2.resize(pip_arr_large, (new_pip_w, new_pip_h), interpolation=cv2.INTER_AREA)
        # Paste the resized PiP array onto the main array
        main_arr[:new_pip_h, (orig_pip_w - new_pip_w):] = pip_arr_resized
        progress_plot_w, progress_plot_h = int(orig_pip_w / PROGRESS_WAVE_SCALE), int(orig_pip_h // PROGRESS_WAVE_SCALE)
        progress_plot_array = make_progress_signal_plot(h_lm_signal, anim_lab_times, anim_time_indices, i_frame,
                                                       progress_plot_w, progress_plot_h, BG_COLOR)
        h_buff = w_buff = 10
        main_arr[h_buff:(progress_plot_h + h_buff), w_buff:(progress_plot_w + w_buff)] = progress_plot_array
        print(f"slicing arrays and making the progress plot took {(time.time() - temp_time):.2f}s")
        temp_time = time.time()
        frame_filename = f"{frames_dir_path}/frame_{i_frame:04d}.png"
        combined_frame = cv2.cvtColor(main_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_filename, combined_frame)
        print(f"saving frame took {(time.time() - temp_time):.2f}s")

        frame_files.append(frame_filename)
        if not auto_loop_bool:
            print(f"Frame saved to {frame_filename}. Rendered in {time.time() - frame_render_start_time:.2f}s using {num_r_points} radial points.")

    print("All animation frames rendered.")
    print(f"{r_points_sum/NUM_FRAMES} average radial points used")
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
    if auto_loop_bool:
        surface_change_bool = False
        if representation_str != 'surface' or None:
            surface_change_bool = True
        else:
            representation_str = 'surface'

        for sxs_idx in range(sxs_idx_start, sxs_idx_start + loop_size):
            SXS_ID = f"SXS:BBH:{sxs_idx:04d}"
            OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_PiP_movie.mp4"
            create_merger_movie()
            if surface_change_bool:
                if representation_str == 'surface':
                    representation_str = 'wireframe'
                else:
                    representation_str = 'surface'
    else:
        create_merger_movie()
