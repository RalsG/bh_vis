import numpy as np
import sxs
from mayavi import mlab
import matplotlib.pyplot as plt
import imageio.v2 as imageio # clarification
import cv2
import os
import time
import inspect
import sys

# --- Configuration Parameters ---
SXS_ID = "SXS:BBH:0308"
OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_3D_strain_displacement_test.mp4"
auto_loop_bool = False # option to make many movies
sxs_idx_start = 308
loop_size = 4
representation_str = 'surface'
NUM_FRAMES = 300
FPS = 20
timing_bool = False

# GW Surface Visualization Parameters
MAX_R_STEP = 10.0 # In units of M for the adaptive grid sampling.
MIN_R_STEP = 0.05
POINTS_PER_WAVE = 20.0 # Higher means more compute but smoother surface waves
MIN_R = 10.
MAX_R = 100.
NUM_R = 40
NUM_PHI = 40
NUM_THETA = NUM_PHI//2
STRAIN_SCALE = 0.45 * MAX_R # Factor to scale h+ for z-displacement (TUNE THIS!)
GW_SURFACE_COLOR = (0.3, 0.6, 1.0) # Uniform color for the GW surface
BG_COLOR = (0.4, 0.4, 0.4)
SPIN_ARROW_COLOR = (0.85, 0.85, 0.1)


PIP_CAMERA_DISTANCE = 30
MAIN_CAMERA_DISTANCE = MAX_R * 3.
BH_ELEVATION = 0 # Set 0 or a positive float to have the BHs rotate above the strain surface
PIP_SCALE = 4.0 # This is the ratio of window to PiP
PROGRESS_WAVE_SCALE = 8 # ratio of window to progress waveform
antial_bool = False


# --- Helper Functions ---
def load_simulation_data(sxs_id_str):
    print(f"Loading simulation: {sxs_id_str}")
    try:
        simulation = sxs.load(sxs_id_str, download=True, progress=True,
                              ignore_deprecation=True,
                              # auto_supersede=True # I learned I've been misspelling this all my life
                              )
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
    ax.axvline(x=live_time[-1], color='red', linestyle='-', linewidth=1.0, alpha=0.4)
    # Ensures the waveform will be the same size across simulations, avoids auto-scaling
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


def get_bh_mesh_data(horizons_obj, time_vals, n_theta_bh=15, n_phi_bh=20, bh_elevation=0.):
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
    A_coords_data[:, 2] = A_coords_data[:, 2] + bh_elevation
    B_coords_data[:, 2] = B_coords_data[:, 2] + bh_elevation
    C_coords_data[:, 2] = C_coords_data[:, 2] + bh_elevation

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


def generate_3d_grid(num_r, r_min, r_max, num_theta, num_phi) -> np.ndarray:
    r_axis = np.linspace(r_min, r_max, num_r)
    theta_axis = np.linspace(0, np.pi, num_theta)
    phi_axis = np.linspace(0, 2*np.pi, num_phi)
    r_grid, theta_grid, phi_grid = np.meshgrid(r_axis, theta_axis, phi_axis, indexing='ij')
    print(r_grid, theta_grid, phi_grid)

    """# Thin out the grid closer to the center
    r_grid_out = np.array()
    theta_grid_out = np.array()
    phi_grid_out = np.array()

    for i, r in enumerate(r_axis):
        if r == 0:
            continue
        scale_diff_slice = int(r_max / r)
        
        np.append(r_grid_out, r_grid[i, ::scale_diff_slice, ::scale_diff_slice], axis=0)
        np.append(theta_grid_out, theta_grid[i, ::scale_diff_slice, ::scale_diff_slice], axis=0)
        np.append(phi_grid_out, phi_grid[i, ::scale_diff_slice, ::scale_diff_slice], axis=0)"""
    return r_axis, r_grid, theta_grid, phi_grid

def strain_in_3D(
        sxs_strain_modes: sxs.waveforms.WaveformModes,
        phi_axis: np.ndarray,
        theta_axis: np.ndarray
):
    equiangular = np.array([
        [
            [theta, phi]
            for phi in phi_axis
        ]
        for theta in theta_axis
    ])

    return sxs_strain_modes.evaluate(equiangular)

def disturb_the_points(strain_grid, lab_time, r_grid, phi_grid, theta_grid, amplitude_scale: float = 10.) -> np.ndarray:
    # Convert the base spherical grids to Cartesian
    x_grid = r_grid * np.cos(phi_grid) * np.sin(theta_grid)
    y_grid = r_grid * np.sin(phi_grid) * np.sin(theta_grid)
    z_grid = r_grid * np.cos(theta_grid)
    r_axis = r_grid[:, 0, 0]

    ret_times_vec = np.asarray(lab_time - np.flip(r_grid[:, 0, 0]))
    strain_interpolated_obj = strain_grid.interpolate(ret_times_vec)
    h_at_times = np.flip(np.asarray(strain_interpolated_obj), axis=0)

    for r_idx, radius in enumerate(r_axis):
        if radius == 0: # Avoid division by zero
            continue
        h_at_times[r_idx] = h_at_times[r_idx] / radius

    # 1. Extract plus and cross polarizations from the complex strain object
    h_plus = h_at_times.real
    h_cross = h_at_times.imag

    # 2. Calculate the Cartesian components of the local spherical basis vectors e_theta and e_phi
    # These vectors form the basis of the transverse plane at each point.
    ct, st = np.cos(theta_grid), np.sin(theta_grid)
    cp, sp = np.cos(phi_grid), np.sin(phi_grid)

    e_theta_x, e_theta_y, e_theta_z = ct * cp,  ct * sp, -st
    e_phi_x,   e_phi_y,   e_phi_z   = -sp,      cp,       0

    # 3. Construct the polarization basis tensors e_plus_ij and e_cross_ij
    # e_plus = e_theta ⊗ e_theta - e_phi ⊗ e_phi
    # e_cross = e_theta ⊗ e_phi + e_phi ⊗ e_theta
    e_plus_xx = e_theta_x**2 - e_phi_x**2
    e_plus_yy = e_theta_y**2 - e_phi_y**2
    e_plus_zz = e_theta_z**2 - e_phi_z**2
    e_plus_xy = e_theta_x * e_theta_y - e_phi_x * e_phi_y
    e_plus_xz = e_theta_x * e_theta_z - e_phi_x * e_phi_z
    e_plus_yz = e_theta_y * e_theta_z - e_phi_y * e_phi_z

    e_cross_xx = 2 * e_theta_x * e_phi_x
    e_cross_yy = 2 * e_theta_y * e_phi_y
    e_cross_zz = 2 * e_theta_z * e_phi_z
    e_cross_xy = e_theta_x * e_phi_y + e_phi_x * e_theta_y
    e_cross_xz = e_theta_x * e_phi_z + e_phi_x * e_theta_z
    e_cross_yz = e_theta_y * e_phi_z + e_phi_y * e_theta_z

    # 4. Construct the full strain tensor h_ij at each point
    h_xx = h_plus * e_plus_xx + h_cross * e_cross_xx
    h_yy = h_plus * e_plus_yy + h_cross * e_cross_yy
    h_zz = h_plus * e_plus_zz + h_cross * e_cross_zz
    h_xy = h_plus * e_plus_xy + h_cross * e_cross_xy
    h_xz = h_plus * e_plus_xz + h_cross * e_cross_xz
    h_yz = h_plus * e_plus_yz + h_cross * e_cross_yz

    # 5. Calculate the displacement vector using δx_i = 0.5 * h_ij * x_j
    delta_x = amplitude_scale * (h_xx * x_grid + h_xy * y_grid + h_xz * z_grid)
    delta_y = amplitude_scale * (h_xy * x_grid + h_yy * y_grid + h_yz * z_grid)
    delta_z = amplitude_scale * (h_xz * x_grid + h_yz * z_grid + h_zz * z_grid)

    # 6. Apply the displacement to the Cartesian grid objects
    x_grid = x_grid + delta_x
    y_grid = y_grid + delta_y
    z_grid = z_grid + delta_z

    displacement_scalars = np.sqrt((delta_x ** 2) + (delta_y ** 2) + (delta_z ** 2))

    return  x_grid, y_grid, z_grid, displacement_scalars

# --- Main Animation Logic ---
def create_merger_movie():
    script_init_time = time.time()
    strain_modes_sxs, horizons_data = load_simulation_data(SXS_ID)
    data_loaded_time = time.time()
    print(f"Data loading took {data_loaded_time - script_init_time:.2f}s")
    
    start_back_prop = 0.2 # fraction of total sim time to go back from peak strain for the start
    end_for_prop = 0.05 # fraction of total sim time to go forwards from peak strain for the end
    dom_l, dom_m = 2, 2

    common_horizon_start = (horizons_data.A.time[-1] + horizons_data.C.time[0])/2
    h_lm_signal = strain_modes_sxs[:, strain_modes_sxs.index(dom_l, dom_m)]
    peak_strain_time = strain_modes_sxs.max_norm_time()
    sample_times = strain_modes_sxs.t

    anim_lab_times, anim_time_indices = pseudo_uniform_times(sample_times, peak_strain_time, start_back_prop, end_for_prop)
    print(f"Animation time: {anim_lab_times[0]:.2f}M to {anim_lab_times[-1]:.2f}M over {len(anim_lab_times)} frames.")
    mlab.figure(size=(1280, 1024), bgcolor=BG_COLOR)
    # mlab.options.offscreen = True # Ensure offscreen rendering for saving frames without GUI pop-up

    r_axis = np.linspace(MIN_R, MAX_R, NUM_R)
    theta_buffer = 1.2 * np.pi/(NUM_THETA)
    theta_axis = np.linspace(theta_buffer, np.pi - theta_buffer, NUM_THETA)
    phi_axis = np.linspace(0, 2*np.pi, NUM_PHI)
    r_grid, theta_grid, phi_grid = np.meshgrid(r_axis, theta_axis, phi_axis, indexing='ij')
    temp_time = time.time()
    strain_grid = strain_in_3D(strain_modes_sxs, phi_axis, theta_axis)
    print(f"evaluating across all directions and times took {(time.time() - temp_time):.2f}s")

    bh_surfs, spin_vectors, spin_arrow_size, orbital_vel_t = get_bh_mesh_data(horizons_data, anim_lab_times, bh_elevation = BH_ELEVATION)
    spin_arrow_size *= 8

    frame_files = []
    frames_dir_path = "frames" # Changed directory name as requested
    os.makedirs(frames_dir_path, exist_ok=True)

    total_physical_anim_time = anim_lab_times[-1] - anim_lab_times[0]
    if total_physical_anim_time == 0: total_physical_anim_time = 1.0 # Avoid division by zero

    print(f"Processing and surface building took {time.time() - data_loaded_time:.2f}s")
    print("Starting frame rendering loop...")

    for i_frame, current_lab_time in enumerate(anim_lab_times):
        frame_render_start_time = time.time()
        if not auto_loop_bool:
            print(f"Processing frame {i_frame+1}/{NUM_FRAMES} for lab_time = {current_lab_time:.2f} M")
        
        lab_time_step = current_lab_time - anim_lab_times[i_frame - 1] # It will be the whole animation time on frame 0, but is not used then

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
        if timing_bool:
            print(f"rendering BHs took {(time.time() - temp_time):.2f}s")

        mlab.view(azimuth=30, elevation=60, distance=PIP_CAMERA_DISTANCE, focalpoint=(0,0,BH_ELEVATION))

        pip_arr_large = mlab.screenshot(antialiased=True)
        if current_lab_time < common_horizon_start:
            spin1_obj.remove()
            spin2_obj.remove()
        else:
            spin3_obj.remove()

        # plot points moving with strain
        x_grid_for_plotting, y_grid_for_plotting, z_grid_for_plotting, displacement_scalars = disturb_the_points(
            strain_grid, current_lab_time, r_grid, phi_grid, theta_grid, STRAIN_SCALE)
        mlab.points3d(x_grid_for_plotting, y_grid_for_plotting, z_grid_for_plotting, -displacement_scalars,
                      colormap='cool', opacity=0.4, scale_mode='none', scale_factor=0.5)
        
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

        progress_plot_array = make_progress_signal_plot(h_lm_signal, anim_time_indices, i_frame,
                                                       progress_plot_w, progress_plot_h, BG_COLOR)
        h_buff = w_buff = 10
        main_arr[h_buff:(progress_plot_h + h_buff), w_buff:(progress_plot_w + w_buff)] = progress_plot_array
        if timing_bool:
            print(f"slicing arrays and making the progress plot took {(time.time() - temp_time):.2f}s")
        temp_time = time.time()
        frame_filename = f"{frames_dir_path}/frame_{i_frame:04d}.png"
        combined_frame = cv2.cvtColor(main_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_filename, combined_frame)
        if timing_bool:
            print(f"saving frame took {(time.time() - temp_time):.2f}s")

        frame_files.append(frame_filename)
        if not auto_loop_bool:
            print(f"Frame saved to {frame_filename}. Rendered in {time.time() - frame_render_start_time:.2f}s.")

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
