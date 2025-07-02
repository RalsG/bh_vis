import numpy as np
import sxs
from mayavi import mlab
import matplotlib.pyplot as plt
from tvtk.api import tvtk
import h5py
import scipy # Eventually switch over to full moviepy, not urgent
import cv2
import moviepy
import os
import time
import inspect
import sys

# --- Configuration Parameters ---
SXS_ID = "SXS:BBH:0001"
OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_h_volume_with_noise_and_arms_2.mp4"
auto_loop_bool = False # option to make many movies
sxs_idx_start = 164
loop_size = 3
NUM_FRAMES = 300
FPS = 24
timing_bool = False
cone_test_bool = False
RIT_bool = True
RIT_filename = '/home/guest/Downloads/ExtrapStrain_RIT-eBBH-1460-n100.h5'
if RIT_bool:
    OUTPUT_MOVIE_FILENAME = f"{RIT_filename[:-3].replace('/home/guest/Downloads/ExtrapStrain_', '')}_h_volume.mp4"

# Strain Visualization Parameters
MAX_XYZ = 100.
POINTS_PER_DIM = 50 # resolution on each axis
if POINTS_PER_DIM % 2 == 1:
    POINTS_PER_DIM += 1 # MUST BE EVEN to avoid divide by zero errors/theta pole singularities
GW_SURFACE_COLOR = (0.3, 0.6, 1.0) # Uniform color for the GW surface
BG_COLOR = (0.2, 0.2, 0.2)
SPIN_ARROW_COLOR = (0.85, 0.85, 0.1)
OPACITY_SPIKE_WIDTH = 0.8 # Width of each opaque-ish region as a percentage of total data range 
MAX_OPACITY = 0.4
BASE_OPACITY = 0.0
STRAIN_COLORMAP = 'gist_ncar'
SPIKE_SHAPE = 'triangle' # options are 'triangle', 'box', or 'gaussian'
VALUES_TO_BE_OPAQUE = np.array([-0.6, -0.35, 0.35, 0.6]) # fractions of peak strain to make an opaque region around
# VALUES_TO_BE_OPAQUE = np.delete(np.linspace(0.1, 0.6, 6), 5)
CLIP_FRAC = 0.9

NUM_RINGS = 1
ARM_LENGTH = 0.6 * MAX_XYZ
r_axis = np.array([MAX_XYZ])
AZIMUTHAL_ANGLE = np.pi/10
STRAIN_SCALE = 2.4 * MAX_XYZ
RIT_STRAIN_SCALE = MAX_XYZ / 5
CYLINDER_RADIUS = 0.04 # Relative to scale of the glyph

PIP_CAMERA_DISTANCE = 30
MAIN_CAMERA_DISTANCE = MAX_XYZ * 4.5
BH_ELEVATION = 0 # Set 0 or a positive float to have the BHs rotate above the strain surface
PIP_SCALE = 4.0 # This is the ratio of window to PiP
PROGRESS_WAVE_SCALE = 8 # ratio of window to progress waveform heigt at bottom
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
    if strain_modes is not None: print(f"Strain modes loaded. Time range: {strain_modes.t[0]:.2f}M to {strain_modes.t[-1]:.2f}M.")
    else: raise ValueError(f"Strain not found or empty for simulation {sxs_id_str}.")
    
    horizons_data = getattr(simulation, 'horizons', None)
    if horizons_data is not None: print("Horizons data loaded.")
    else: print(f"Warning: horizons not found or empty for simulation {sxs_id_str}.")

    return strain_modes, horizons_data

def load_RIT_data(filename: str, strain_scale: float = 35, ell_min: int = 2,
                  ell_max: int = 4, spin_weight: int = -2,
                  dom_ell: int = 2, dom_em: int = 2):
    """
    Usual format of files is 'NRTimes', then a group for the amplitude of each mode,
    then a usually empty auxiliary-info group, then a group for the phase of each mode.
    They seem to have the same deg scalar for constructing the splines, but I haven't seen
    that confirmed explicitly anywhere, so it's pulled each time. They do have VERY different
    coarse array sizes, so those CANNOT be used interchangeably.
    
    Also note that the modes are ordered -1, ..., -ell, 0, 1, ..., ell which is not
    the convention SXS uses.
    """
    input_h5_file  = h5py.File(filename,'r')
    intended_time_axis = input_h5_file['NRTimes'][...]
    num_modes = ((ell_max + 1) ** 2) - (ell_min ** 2)

    dom_phase_group = input_h5_file[f'phase_l{dom_ell}_m{dom_em}']
    dom_phase_spline = scipy.interpolate.make_interp_spline(
        dom_phase_group['X'][...],
        dom_phase_group['Y'][...], # note: RIT data has DECREASING phase
        k=dom_phase_group['deg'][...]
    )
    angular_phase = dom_phase_spline(intended_time_axis)
    omega_spline = dom_phase_spline.derivative()
    angular_velocity_BHs = 0.5 * omega_spline(intended_time_axis)
    omega_calculated_TS = sxs.TimeSeries(angular_velocity_BHs, intended_time_axis)

    """plt.plot(intended_time_axis, angular_phase, label='phase')
    # plt.plot(intended_time_axis, angular_velocity_BHs, label='omega')
    plt.legend()
    plt.show()"""

    modes_data = np.zeros((len(intended_time_axis), num_modes), dtype=complex)
    for ell in range(ell_min, ell_max + 1):
        for em in range(-ell, ell + 1):
            amp_group = input_h5_file[f'amp_l{ell}_m{em}']
            phase_group = input_h5_file[f'phase_l{ell}_m{em}']
            amp_spline = scipy.interpolate.make_interp_spline(
                    amp_group['X'][...],
                    amp_group['Y'][...],
                    k=amp_group['deg'][...]
                )
            amp_t_array = amp_spline(intended_time_axis)
            phase_spline = scipy.interpolate.make_interp_spline(
                    phase_group['X'][...],
                    phase_group['Y'][...],
                    k=phase_group['deg'][...]
                )
            phase_t_array = phase_spline(intended_time_axis)
            waveform = amp_t_array * np.exp(1j * phase_t_array)
            out_idx = ell + em + (ell ** 2) - (ell_min ** 2)
            modes_data[:, out_idx] = waveform * strain_scale

            print(ell, em, out_idx)
            plt.plot(intended_time_axis, waveform, label=f'{ell}, {em}')
        print('\n')
        plt.legend()
        plt.show()

    strain_waveforms_obj = sxs.waveforms.WaveformModes(
        modes_data, intended_time_axis, time_axis=0, modes_axis=1,
        ell_min=ell_min, ell_max=ell_max, spin_weight=spin_weight
    )
    print(f"Strain modes loaded. Time range: {strain_waveforms_obj.t[0]:.2f}M to {strain_waveforms_obj.t[-1]:.2f}M.")

    return strain_waveforms_obj, omega_calculated_TS

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


def figure_it_out(strain_modes: sxs.waveforms.WaveformModes, lab_time_t: float,
                  x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray, r_grid: np.ndarray,
                  theta_grid: np.ndarray, phi_grid: np.ndarray):
    """# Step 1: Same as before
    og_shape = r_grid.shape
    ret_times_flat = (lab_time_t - r_grid).ravel()
    unique_times, inverse_indices = np.unique(ret_times_flat, return_inverse=True)

    # Step 2: Interpolate for all unique times at once
    h_lm_at_unique_times = strain_modes.interpolate(unique_times) # Shape: (U, num_modes)

    # Step 3: Fan out coefficients to align with the flattened grid
    # This is the key step that avoids the n^6 array
    h_lm_for_each_grid_point = h_lm_at_unique_times.data[inverse_indices] # Shape: (n^3, num_modes)

    # Step 4: Prepare flattened directions
    directions_flat = np.transpose(np.array([theta_grid.ravel(), phi_grid.ravel()])) # Shape: (n^3, 2)

    # Step 5: Evaluate in a single, aligned shot
    # We create a temporary WaveformModes object to use the evaluate method.
    # The time array here is just a placeholder, as the time-dependence is now baked into the coefficients.
    print(h_lm_for_each_grid_point.shape)
    temp_waveform = sxs.WaveformModes(
        h_lm_for_each_grid_point, 
        time=np.arange(len(h_lm_for_each_grid_point)),
        modes_axis=1,
        ell_min=strain_modes.ell_min,
        ell_max=strain_modes.ell_max,
        spin_weight=-2
    )
    waveform_evaluator = temp_waveform.evaluate
    strain_values_flat = temp_waveform.evaluate(directions_flat) # Shape: (n^3,)

    # Step 6: Reshape to the final grid shape
    final_strain_grid = np.reshape(strain_values_flat, og_shape)"""

    og_shape = r_grid.shape
    ret_times_flat = (lab_time_t - r_grid).ravel()
    unique_times, inverse_indices = np.unique(ret_times_flat, return_inverse=True)

    # Step 2: Initialize output
    final_strain_grid = np.zeros(og_shape)
    final_strain_flat = final_strain_grid.ravel() # A flattened view for easy assignment

    # Also flatten directions for easier indexing
    flat_theta = theta_grid.ravel()
    flat_phi = phi_grid.ravel()
    strain_modes_interpolator = strain_modes.interpolate

    # Step 3: Loop over unique times
    for i, time_val in enumerate(unique_times):
        # Interpolate modes ONCE for this time
        strain_coeffs_at_time = strain_modes_interpolator([time_val])

        # Find all grid points that need this evaluation
        target_indices = np.where(inverse_indices == i)[0]

        # Gather the directions for only those points
        directions_for_this_time = np.array([flat_theta[target_indices], flat_phi[target_indices]]).T
        
        # Evaluate strain just for this group of points
        strain_values = strain_coeffs_at_time.evaluate(directions_for_this_time)

        # Place the results into the flattened output array
        final_strain_flat[target_indices] = strain_values.real

    return final_strain_grid, np.min(final_strain_flat), np.max(final_strain_flat)


def generate_ring_cartesian_coords(
    num_observatories: int,
    arm_length: float,
    radial_axis: np.ndarray,
    azimuthal_angle: float = np.pi/8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates Cartesian coordinates (x, y, z) for multiple model "LIGOs" (2 orthogonal vectors).

    The geometry consists of 'num_rings' vertical rings, whose centers are
    evenly spaced on a circle. This entire configuration is then generated
    at each radius specified in the 'radial_axis'.

    Args:
        num_rings (int): The number of vertical rings in a single set.
        num_points (int): The number of points to generate for each ring.
        ring_radius (float): The 3D radius of the individual rings.
        radial_axis np.ndarray: An array of radii. The entire set of
            'num_rings' will be generated for each radius in this array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
            x, y, and z coordinate matrices of shape (len(radial_axis),
            num_rings, num_points).
    """
    # Handle empty or invalid inputs
    if num_observatories <= 0 or len(radial_axis) == 0:
        return (np.array([]), np.array([]), np.array([]))

    # 1. Define the base angles for ring centers and points within a ring
    ring_center_phis = np.array([azimuthal_angle])

    # 2. Create 2D grids for observatory center radii and arms using meshgrid.
    # This prepares the data for vectorized operations
    r_centers, phi_centers = np.meshgrid(radial_axis, ring_center_phis)

    # 3. Use broadcasting to calculate coordinates for all points at once.
    # Reshape array to (num_observatories, num_radii, 1) and (1, 1, 2)
    # so that NumPy can broadcast them to a final shape of
    # (num_observatories, num_radii, 3).
    r_broadcast = r_centers[..., np.newaxis]
    phi_broadcast = phi_centers[..., np.newaxis]

    x_centers = r_broadcast * np.cos(phi_broadcast)
    y_centers = r_broadcast * np.sin(phi_broadcast)
    z_centers = np.zeros_like(x_centers)

    x_arms = np.array([[[-np.sin(azimuthal_angle)*arm_length, 0]]])
    y_arms = np.array([[[np.cos(azimuthal_angle)*arm_length, 0]]])
    z_arms = np.array([[[0, arm_length]]])

    x_cart = x_centers + x_arms
    y_cart = y_centers + y_arms
    z_cart = z_centers + z_arms

    x_centers = (x_cart - x_arms).transpose(1, 0, 2)
    y_centers = (y_cart - y_arms).transpose(1, 0, 2)
    z_centers = (z_cart - z_arms).transpose(1, 0, 2)
    
    x_final = x_cart.transpose(1, 0, 2)
    y_final = y_cart.transpose(1, 0, 2)
    z_final = z_cart.transpose(1, 0, 2)

    return x_final, y_final, z_final, x_centers, y_centers, z_centers

def generate_ring_data(
    strain_modes: sxs.waveforms.WaveformModes,
    num_rings: int,
    ring_radius: float,
    radial_axis: np.ndarray,
    azimuthal_angle: float = np.pi/8
) -> tuple:
    """
    Generates data for points on rings and for the centers of those rings.

    This function prepares all geometric information needed for the strain
    calculation, including point coordinates, point directions, ring center
    coordinates, and ring center directions.

    Args:
        num_rings: The number of vertical rings in a single concentric set.
        num_points: The number of points to generate for each ring.
        ring_radius: The 3D radius of the individual rings.
        radial_axis: An array of radii for the ring centers.

    Returns:
        A tuple containing:
        - x_grid, y_grid, z_grid: Cartesian coordinates of each point.
          Shape: (len(radial_axis), num_rings, num_points)
        - directions_grid: Spherical [theta, phi] of each point.
          Shape: (len(radial_axis), num_rings, num_points, 2)
        - center_directions: Spherical [theta, phi] of each ring center.
          Shape: (len(radial_axis), num_rings, 2)
        - center_radii: Radial distance of each ring center.
          Shape: (len(radial_axis), num_rings)
    """
    # Generate the Cartesian coordinates of every point on the rings
    x_grid, y_grid, z_grid, x_centers, y_centers, z_centers = generate_ring_cartesian_coords(
        num_rings, ring_radius, radial_axis
    )

    # --- Calculate data for the center of each ring ---
    # The radii of the centers are simply the values from the input radial_axis
    num_radii = len(radial_axis)

    # The directions of the centers are on the equator (theta=pi/2)
    # with phi angles distributed evenly.
    ring_center_phis = np.array([azimuthal_angle])
    center_theta = np.full((num_radii, num_rings), np.pi / 2)
    center_phi = np.tile(ring_center_phis, (num_radii, 1))
    center_directions = np.stack((center_theta, center_phi), axis=-1)

    # 2. Evaluate the strain at each ring center's direction. This returns
    # an array of TimeSeries objects, one for each ring center.
    strain_at_centers = strain_modes.evaluate(center_directions[0])

    return x_grid, y_grid, z_grid, center_directions, strain_at_centers, x_centers, y_centers, z_centers


def disturb_the_points(
    strain_at_centers: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    center_directions: np.ndarray,
    radial_axis: np.ndarray,
    lab_time: float,
    amplitude_scale: float = 50.0
) -> tuple:
    """
    Displaces points near 'centers' using strain evaluated at each center.

    Args:
        sxs_strain_modes: The SXS waveform object containing strain modes.
        x_grid, y_grid, z_grid: Cartesian coordinates of the points on the rings.
        directions_grid: Spherical coordinates [theta, phi] for each point.
        center_radii: Radial distance for each ring CENTER.
        lab_time: The observer's laboratory time.
        amplitude_scale: A scaling factor for the displacement visualization.

    Returns:
        A tuple with displaced coordinates, displacement vectors, and magnitude.
    """
    # Calculate the retarded time for each RING CENTER
    retarded_times_centers = np.flip(lab_time - radial_axis)

    # Interpolate each TimeSeries at the corresponding retarded time.
    strain_interpolated_obj = strain_at_centers.interpolate(retarded_times_centers)
    h_at_centers = np.flip(np.asarray(strain_interpolated_obj), axis=0)
    
    # Apply 1/r falloff using the radius of the ring centers
    radial_axis = radial_axis[:, np.newaxis]
    h_at_centers = np.divide(h_at_centers, radial_axis, where=(radial_axis != 0))
    

    h_plus = amplitude_scale * h_at_centers.real # shape (num_r, num_rings, num_points)
    h_cross = amplitude_scale * h_at_centers.imag

    # Calculate polarization basis using the RING CENTER'S direction.
    # This ensures the "stretch" and "squeeze" axes are the same for all points on one ring.
    center_theta = center_directions[..., 0]
    center_phi = center_directions[..., 1]

    cos_theta, sin_theta = np.cos(center_theta), np.sin(center_theta)
    cos_phi, sin_phi = np.cos(center_phi), np.sin(center_phi)
    """center_x = 
    center_y = 
    center_z = """

    e_theta_x, e_theta_y, e_theta_z = cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta
    e_phi_x, e_phi_y, e_phi_z = -sin_phi, cos_phi, 0

    # Construct the polarization basis tensors e_plus_ij and e_cross_ij
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
    delta_x = h_xx[:, :, np.newaxis] * x_grid + h_xy[:, :, np.newaxis] * y_grid + h_xz[:, :, np.newaxis] * z_grid
    delta_y = h_xy[:, :, np.newaxis] * x_grid + h_yy[:, :, np.newaxis] * y_grid + h_yz[:, :, np.newaxis] * z_grid
    delta_z = h_xz[:, :, np.newaxis] * x_grid + h_yz[:, :, np.newaxis] * y_grid + h_zz[:, :, np.newaxis] * z_grid

    # 6. Apply the displacement to the Cartesian grid objects
    x_grid_out = x_grid + delta_x - x_centers
    y_grid_out = y_grid + delta_y - y_centers
    z_grid_out = z_grid + delta_z - z_centers

    length_scalars = np.sqrt((x_grid_out ** 2) + (y_grid_out ** 2) + (z_grid_out ** 2))

    return x_grid_out, y_grid_out, z_grid_out, delta_x, delta_y, delta_z, length_scalars


def sonify_strain(dom_strain_mode: sxs.TimeSeries,
                angular_velocity_TS: sxs.TimeSeries,
                anim_time_indices: np.ndarray,
                num_seconds: float,
                sample_rate = 44100
):
    trimmed_strain_signal = dom_strain_mode[anim_time_indices[0]:(anim_time_indices[-1] + 1)]
    sample_times = trimmed_strain_signal.t
    uniform_time_array = np.linspace(sample_times[0], sample_times[-1], int(sample_rate * num_seconds))
    uniform_strain_array = trimmed_strain_signal.interpolate(uniform_time_array)
    amplitude = uniform_strain_array.abs.ndarray
    angular_velocity_interpolated = angular_velocity_TS.interpolate(uniform_time_array)

    raw_frequency = angular_velocity_interpolated.ndarray
    idx = np.argmax(amplitude)
    num_points = num_seconds * sample_rate

    # artistic choice to make the whole range hearable
    general_multiplier = 8000/raw_frequency[int(idx - 0.01*num_points)]
    frequency_to_hear = general_multiplier * raw_frequency
    phase = np.cumsum(frequency_to_hear / sample_rate)

    amplitude_normalized = amplitude / np.max(amplitude)
    # waveform = A(t) * sin(phase(t))
    waveform = amplitude_normalized * np.sin(phase)

    # Convert to 16-bit integer format (for WAV file)
    # The range for 16-bit is -32768 to 32767
    waveform_16bit = np.int16(waveform * 32767)

    # Save the Audio File
    filename = "gravitational_wave_sonification.wav"
    scipy.io.wavfile.write(filename, sample_rate, waveform_16bit)
    print(f"Audio saved to {filename}")

    return filename


def create_color_opacity_transfer_functions(
    isosurface_values: list[float],
    data_range: tuple[float, float],
    spike_width_percent: float = 2.0,
    max_opacity: float = 0.3,
    base_opacity: float = 0.0,
    colormap: str = 'hsv',
    spike_shape: str = 'triangle' # or 'box' or 'gaussian'
) -> tuple[tvtk.ColorTransferFunction, tvtk.PiecewiseFunction]:
    """
    Creates customized color and opacity transfer functions for volume rendering.

    This function generates opacity spikes at specified scalar values, allowing for
    the simulation of multiple isosurfaces or highlighted regions within a volume.

    Args:
        values (List[Scalar]):
            A list of floats where opacity spikes should be centered, ranging from -1. to 1.
        data_range (Tuple[Scalar, Scalar]):
            The (min, max) range of the scalar data.
        width_percent (float, optional):
            The width of each opacity spike as a percentage of the data_range.
            Defaults to 2.0.
        max_opacity (float, optional):
            The peak opacity for the spikes (0.0 to 1.0). Defaults to 0.8.
        base_opacity (float, optional):
            The baseline opacity for all other data values. Defaults to 0.0.
        colormap (str, optional):
            The name of the Matplotlib colormap to use. Defaults to 'viridis'.
        spike_shape (SpikeShape, optional):
            The shape of the opacity spike: 'triangle', 'box', or 'gaussian'.
            Defaults to 'triangle'.

    Returns:
        Tuple[tvtk.ColorTransferFunction, tvtk.PiecewiseFunction]:
            A tuple containing the configured tvtk color and opacity functions.
    """
    data_min, data_max = data_range
    if data_min >= data_max:
        # Return empty, valid objects for an invalid data range
        return tvtk.ColorTransferFunction(), tvtk.PiecewiseFunction()

    # 1. Create the transfer function objects
    otf = tvtk.PiecewiseFunction()
    ctf = tvtk.ColorTransferFunction()

    # 2. Get the colormap object from Matplotlib
    # Use the modern, recommended API
    cmap = plt.get_cmap(colormap)
    cmap_for_lut = plt.get_cmap(colormap, 256)(np.arange(256))
    lut_out = (cmap_for_lut * 255).astype(np.uint8)

    # 3. Define the base opacity across the entire data range
    # This ensures regions without spikes have the specified base opacity.
    otf.add_point(data_min, base_opacity)
    otf.add_point(data_max, base_opacity)
    data_width = data_max - data_min

    # 4. Create opacity spikes and corresponding color points
    # Calculate the absolute width of the spike from the percentage
    scalar_width = (spike_width_percent / 100.0) * data_width
    half_width = scalar_width / 2.0

    for val in sorted(isosurface_values):
        opacity_multiplier = abs(val) ** 1/5
        val_opacity = max_opacity * opacity_multiplier
        # Normalize the value to sample the colormap
        norm_val = (val/2) + 0.5
        data_val = (norm_val * data_width) + data_min
        color = cmap(np.clip(norm_val, 0.0, 1.0)) # Clip to handle values outside range

        # Add the color point at the center of the spike
        ctf.add_rgb_point(data_val, color[0], color[1], color[2])

        # Define the spike region, ensuring it stays within the data range
        start = np.clip(data_val - half_width, data_min, data_max)
        end = np.clip(data_val + half_width, data_min, data_max)
        center = np.clip(data_val, data_min, data_max)

        # Add points to the opacity function based on the chosen shape
        if spike_shape == 'triangle':
            # Smooth 5-point spike
            otf.add_point(start, base_opacity)
            otf.add_point(np.clip(data_val - half_width / 2.0, data_min, data_max), val_opacity * 0.3)
            otf.add_point(center, val_opacity)
            otf.add_point(np.clip(data_val + half_width / 2.0, data_min, data_max), val_opacity * 0.3)
            otf.add_point(end, base_opacity)
        elif spike_shape == 'box':
            # Sharp 4-point "top-hat" spike
            epsilon = data_width / 10000.0 # Tiny offset for vertical lines
            otf.add_point(np.clip(start - epsilon, data_min, data_max), base_opacity)
            otf.add_point(start, val_opacity)
            otf.add_point(end, val_opacity)
            otf.add_point(np.clip(end + epsilon, data_min, data_max), base_opacity)
        elif spike_shape == 'gaussian':
            # Create a smoother, more realistic falloff
            # We add several points to approximate the curve
            epsilon = data_width / 10000.0
            otf.add_point(np.clip(start - epsilon, data_min, data_max), base_opacity)
            otf.add_point(np.clip(end + epsilon, data_min, data_max), base_opacity)
            for i in np.linspace(-1, 1, 15): # Use 15 points to define the curve
                # Map i from [-1, 1] to the scalar range [start, end]
                point_val = data_val + i * half_width
                
                # Gaussian function: exp(-x^2 / (2*sigma^2))
                # Here, we use a simpler form where i is our normalized x
                opacity = val_opacity * np.exp(-(i**2) * 2.5) # The 2.5 is a shape factor
                
                # Clip the point and ensure it doesn't dip below base_opacity
                clipped_val = np.clip(point_val, data_min, data_max)
                final_opacity = max(opacity, base_opacity)
                otf.add_point(clipped_val, final_opacity)

    return ctf, otf, lut_out

# --- Main Animation Logic ---
def create_merger_movie():
    script_init_time = time.time()
    if RIT_bool: strain_modes_obj, orbital_vel_TS = load_RIT_data(RIT_filename, RIT_STRAIN_SCALE)
    else: strain_modes_obj, horizons_data = load_simulation_data(SXS_ID)
    data_loaded_time = time.time()
    print(f"Data loading took {data_loaded_time - script_init_time:.2f}s")
    
    start_back_prop = 0.6 # fraction of total sim time to go back from peak strain for the start
    end_for_prop = 0.1 # fraction of total sim time to go forwards from peak strain for the end
    dom_l, dom_m = 2, 2

    if RIT_bool: common_horizon_start = 0.0
    else: common_horizon_start = (horizons_data.A.time[-1] + horizons_data.C.time[0])/2
    h_lm_signal = strain_modes_obj[:, strain_modes_obj.index(dom_l, dom_m)]
    peak_strain_time = strain_modes_obj.max_norm_time()
    sample_times = strain_modes_obj.t

    anim_lab_times, anim_time_indices = pseudo_uniform_times(sample_times, peak_strain_time, start_back_prop, end_for_prop)
    print(f"Animation time: {anim_lab_times[0]:.2f}M to {anim_lab_times[-1]:.2f}M over {len(anim_lab_times)} frames.")
    print(f"Peak strain is around {peak_strain_time:.2f}M.")
    mlab.figure(size=(1280, 1024), bgcolor=BG_COLOR)
    # mlab.options.offscreen = True # Ensure offscreen rendering for saving frames without GUI pop-up

    x_axis = np.linspace(-MAX_XYZ, MAX_XYZ, POINTS_PER_DIM)
    y_axis = np.linspace(-MAX_XYZ, MAX_XYZ, POINTS_PER_DIM)
    z_axis = np.linspace(-MAX_XYZ, MAX_XYZ, POINTS_PER_DIM)
    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
    r_grid = np.sqrt((x_grid ** 2) + (y_grid ** 2) + (z_grid ** 2))
    theta_grid = np.arctan(np.sqrt((x_grid ** 2) + (y_grid ** 2))/z_grid)
    phi_grid = np.arctan(y_grid/x_grid)
    _, min_strain, max_strain = figure_it_out(strain_modes_obj, peak_strain_time, x_grid, y_grid,
                                    z_grid, r_grid, theta_grid, phi_grid)
    print(min_strain, max_strain)
    max_strain, min_strain = CLIP_FRAC*max_strain, CLIP_FRAC*min_strain
    avg_peak_strain_amp = (max_strain - min_strain)/2
    if not RIT_bool:
        bh_surfs, spin_vectors, spin_arrow_size, orbital_vel_TS = get_bh_mesh_data(
        horizons_data, anim_lab_times, bh_elevation = BH_ELEVATION)
        spin_arrow_size *= 7

    wav_filename = sonify_strain(h_lm_signal, orbital_vel_TS, anim_time_indices, NUM_FRAMES/FPS)
    frame_files = []
    frames_dir_path = "frames"
    os.makedirs(frames_dir_path, exist_ok=True)

    color_transfer_function, opacity_transfer_function, colorbar_lut = create_color_opacity_transfer_functions(
        VALUES_TO_BE_OPAQUE, (min_strain, max_strain), OPACITY_SPIKE_WIDTH, MAX_OPACITY,
        BASE_OPACITY, STRAIN_COLORMAP, SPIKE_SHAPE)
    bar_top = 0.923
    bar_bottom = 0.149
    bar_height = bar_top - bar_bottom
    x_LIGO, y_LIGO, z_LIGO, center_directions, strain_at_centers, x_centers, y_centers, z_centers = generate_ring_data(
        strain_modes_obj, NUM_RINGS, ARM_LENGTH, r_axis, AZIMUTHAL_ANGLE)
    
    print(f"Processing and surface building took {time.time() - data_loaded_time:.2f}s")
    print("Starting frame rendering loop...")

    for i_frame, current_lab_time in enumerate(anim_lab_times):
        if cone_test_bool:
            if (i_frame != NUM_FRAMES - 1) and (i_frame != 0):
                continue
        frame_render_start_time = time.time()
        if not auto_loop_bool:
            print(f"Processing frame {i_frame+1}/{NUM_FRAMES} for lab_time = {current_lab_time:.2f} M")
        
        lab_time_step = current_lab_time - anim_lab_times[i_frame - 1] # It will be the whole animation time on frame 0, but is not used then
        temp_time = time.time()
        strain_grid, _, _ = figure_it_out(strain_modes_obj, current_lab_time, x_grid, y_grid,
                                    z_grid, r_grid, theta_grid, phi_grid)
        x_grid_for_plotting, y_grid_for_plotting, z_grid_for_plotting, x_displace, y_displace, z_displace, displacement_scalars = disturb_the_points(
            strain_at_centers, x_LIGO, y_LIGO, z_LIGO, x_centers, y_centers, z_centers, center_directions, r_axis, current_lab_time, STRAIN_SCALE)
        if timing_bool:
            print(f"Evaluating strain grid took {(time.time() - temp_time):.2f}s")
        mlab.clf()

        if not RIT_bool:
            temp_time = time.time()
            if current_lab_time < common_horizon_start:
                # plot BH1
                mlab.mesh(*bh_surfs[0][i_frame], opacity=1, color=(0, 0, 0), name='Event Horizon 1')
                spin1_obj = mlab.quiver3d(*spin_vectors[0][:, i_frame], color=SPIN_ARROW_COLOR, mode='arrow',
                            line_width = 0.4*spin_arrow_size, scale_factor = spin_arrow_size, name='Spin 1', opacity=0.7)
                # plot BH2
                mlab.mesh(*bh_surfs[1][i_frame], opacity=1, color=(0, 0, 0), name='Event Horizon 2')
                spin2_obj = mlab.quiver3d(*spin_vectors[1][:, i_frame], color=SPIN_ARROW_COLOR, mode='arrow',
                            line_width = 0.4*spin_arrow_size, scale_factor = spin_arrow_size, name='Spin 2', opacity=0.7)
            else:
                # plot merged BH
                mlab.mesh(*bh_surfs[2][i_frame], opacity=1, color=(0, 0, 0), name='Event Horizon 3')
                spin3_obj = mlab.quiver3d(*spin_vectors[2][:, i_frame], color=SPIN_ARROW_COLOR, mode='arrow',
                            line_width = 0.4*spin_arrow_size, scale_factor = spin_arrow_size, name='Spin 3', opacity=0.7)
            if timing_bool:
                print(f"rendering BHs took {(time.time() - temp_time):.2f}s")
        mlab.view(azimuth=45, elevation=60, distance=PIP_CAMERA_DISTANCE, focalpoint=(0,0,BH_ELEVATION))
        pip_arr_large = mlab.screenshot(antialiased=True)

        if not RIT_bool:
            if current_lab_time < common_horizon_start:
                spin1_obj.remove()
                spin2_obj.remove()
            else:
                spin3_obj.remove()

        # plot strain volume
        temp_time = time.time()
        strain_field_source = mlab.pipeline.scalar_field(x_grid, y_grid, z_grid, strain_grid)
        strain_cloud = mlab.pipeline.volume(strain_field_source, vmin=min_strain, vmax=max_strain)
        volume_property = strain_cloud._volume_property
        volume_property.set_scalar_opacity(opacity_transfer_function)
        volume_property.set_color(color_transfer_function)
        strain_cloud.module_manager.scalar_lut_manager.lut.table = colorbar_lut
        mlab.colorbar(object=strain_cloud, orientation='vertical', nb_labels=0)
        mlab.text(0.01, 0.96, "Real Polarized Strain (unscaled)", width=0.33, line_width=6)
        for frac in VALUES_TO_BE_OPAQUE:
            mlab.text(0.053, bar_bottom + ((frac/2) + 0.5)*bar_height, f"{(frac*avg_peak_strain_amp):.3f}", width=0.05)
        if timing_bool:
            print(f"Rendering strain volume took {(time.time() - temp_time):.2f}s")

        temp_time = time.time()
        mlab.points3d(x_centers, y_centers, z_centers, mode='cube', color=(0.45, 0.45, 0.45), scale_factor=MAX_XYZ/12.5, opacity=0.9)
        cylinder1_obj = mlab.quiver3d(x_centers[..., 0], y_centers[..., 0], z_centers[..., 0],
                                      x_grid_for_plotting[..., 0], y_grid_for_plotting[..., 0],
                                      z_grid_for_plotting[..., 0], scalars=displacement_scalars[..., 0],
                                      mode='cylinder', color=(0.5, 0.5, 0.5), opacity=0.7,
                                      scale_mode='vector', scale_factor=1, resolution=24)
        adjustment_factor = ARM_LENGTH / displacement_scalars[0, 0, 0]
        cylinder1_obj.glyph.glyph_source.glyph_source.radius = CYLINDER_RADIUS * adjustment_factor

        cylinder2_obj = mlab.quiver3d(x_centers[..., 1], y_centers[..., 1], z_centers[..., 1],
                                      x_grid_for_plotting[..., 1], y_grid_for_plotting[..., 1],
                                      z_grid_for_plotting[..., 1], scalars=displacement_scalars[..., 1],
                                      mode='cylinder', color=(0.5, 0.5, 0.5), opacity=0.7,
                                      scale_mode='vector', scale_factor=1, resolution=24)
        adjustment_factor = ARM_LENGTH / displacement_scalars[0, 0, 1]
        cylinder2_obj.glyph.glyph_source.glyph_source.radius = CYLINDER_RADIUS * adjustment_factor
        mlab.quiver3d(x_centers, y_centers, z_centers, x_grid_for_plotting, y_grid_for_plotting,
                      z_grid_for_plotting, scalars=displacement_scalars, mode='2ddash',
                      color=(1.0, 0.15, 0.15), opacity=1, scale_mode='vector', scale_factor=1, line_width=4.)
        if timing_bool:
            print(f"Modeling LIGO took {(time.time() - temp_time):.2f}s")

        temp_time = time.time()
        mlab.view(azimuth=45, elevation=70, distance=MAIN_CAMERA_DISTANCE, focalpoint=(0,0,0))
        main_arr = mlab.screenshot(antialiased=True)
        # Resize PiP image
        orig_pip_h, orig_pip_w, _ = pip_arr_large.shape
        new_pip_h, new_pip_w = int(orig_pip_h / PIP_SCALE), int(orig_pip_w / PIP_SCALE)
        pip_arr_resized = cv2.resize(pip_arr_large, (new_pip_w, new_pip_h), interpolation=cv2.INTER_AREA)
        # Paste the resized PiP array onto the main array
        # main_arr[:new_pip_h, (orig_pip_w - new_pip_w):] = pip_arr_resized
        w_buff = 10
        progress_plot_w, progress_plot_h = int(orig_pip_w - 2*w_buff), int(orig_pip_h // PROGRESS_WAVE_SCALE)

        progress_plot_array = make_progress_signal_plot(h_lm_signal, anim_time_indices, i_frame,
                                                       progress_plot_w, progress_plot_h, BG_COLOR)
        main_arr[-progress_plot_h:, w_buff:-w_buff] = progress_plot_array
        if timing_bool:
            print(f"Screenshotting, slicing arrays, and progress wave took {(time.time() - temp_time):.2f}s")
        temp_time = time.time()
        frame_filename = f"{frames_dir_path}/frame_{i_frame:04d}.png"
        combined_frame = cv2.cvtColor(main_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_filename, combined_frame)
        if timing_bool:
            print(f"saving frame took {(time.time() - temp_time):.2f}s")

        frame_files.append(frame_filename)
        if not auto_loop_bool:
            print(f"Frame saved to {frame_filename}. Rendered in {time.time() - frame_render_start_time:.2f}s.")

    if cone_test_bool:
        mlab.close(all=True)
        return
    
    print("All animation frames rendered.")
    print("Compiling movie...")
    video_clip = moviepy.ImageSequenceClip(frame_files, fps=FPS)
    audio_clip = moviepy.AudioFileClip(wav_filename)
    combined_clip = video_clip.with_audio(audio_clip)
    if combined_clip.duration > audio_clip.duration:
        print(f"Video was {combined_clip.duration:.2f}s long,")
        print(f"while audio was only {audio_clip.duration:.2f}s.")
        print("Video cut to audio length.")
        combined_clip = combined_clip.with_duration(audio_clip.duration)
    # Write the result to a file using high-quality, standard codecs.
    combined_clip.write_videofile(
        OUTPUT_MOVIE_FILENAME, 
        codec='libx264', 
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a', # Recommended for stability
        remove_temp=True
    )

    print(f"Movie saved to {OUTPUT_MOVIE_FILENAME}")
    
    mlab.close(all=True)
    print(f"Total script execution time: {time.time() - script_init_time:.2f} seconds.")

if __name__ == "__main__":
    if auto_loop_bool:
        #for sxs_idx in range(sxs_idx_start, sxs_idx_start + loop_size):
        for sxs_idx in [1, 165, 166, 150]:
            SXS_ID = f"SXS:BBH:{sxs_idx:04d}"
            OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_g_volume_with_noise_and_arms_auto.mp4"
            create_merger_movie()
            
    else:
        create_merger_movie()
