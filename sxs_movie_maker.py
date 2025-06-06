import numpy as np
import sxs
from sxs.waveforms import WaveformModes # For type hints
import spherical
import quaternionic
from mayavi import mlab
import vtk
import imageio.v2 as imageio # clarification
import cv2
import os
import time
import inspect
import sys

# --- Configuration Parameters ---
SXS_ID = "SXS:BBH:0149"
OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_PiP_movie.mp4"
auto_loop_bool = False # option to make many movies
sxs_idx_start = 150
loop_size = 10

NUM_FRAMES = 400
FPS = 24

# GW Surface Visualization Parameters
MAX_R_STEP = 60 # In units of M for the adaptive grid sampling
N_PHI_GW = 240
MIN_R_GW = 10.0
MAX_R_GW = 800.0
AMPLITUDE_SCALE = 0.45 * MAX_R_GW # Factor to scale h+ for z-displacement (TUNE THIS!)
GW_SURFACE_COLOR = (0.3, 0.6, 1.0) # Uniform color for the GW surface

PIP_CAMERA_DISTANCE = MIN_R_GW * 2.2
MAIN_CAMERA_DISTANCE = MAX_R_GW * 1.8
PIP_SCALE = 3.5 # This is the ratio of main scene to PiP
antial_bool = False

# --- Helper Functions ---
def update_or_create_progress_circle(scene, current_frame, total_frames,
                                     circle_actors=None,
                                     diameter_pixels=20,
                                     center_norm_coords=(0.05, 0.95), # (x_center, y_center) from bottom-left
                                     color=(0.9, 0.9, 0.9),
                                     outline_thickness=2):
    """
    Creates or updates a clockwise progress circle overlay in a Mayavi scene.
    This version assumes mlab.clf() removes actors from the renderer,
    so it re-adds them each frame if they exist.
    It also incorporates robust window size detection.
    """
    if scene is None:
        figure = mlab.gcf()
        if figure is None:
            print("Error: No Mayavi scene/figure found for progress circle.")
            return None
        scene = figure.scene
    
    renderer = scene.renderer
    render_window = scene.render_window
    if renderer is None: # Added check for renderer
        print("Error: Scene has no renderer for progress circle.")
        return None
    
    # Robust window size determination (combining ideas from Expert 1 and 2)
    window_width, window_height = 0, 0
    valid_size_obtained = False

    if render_window:
        window_width, window_height = render_window.size
        if window_width > 0 and window_height > 0:
            valid_size_obtained = True

    if not valid_size_obtained:
        # Try scene.parent.size (often the figure size, from Expert 1)
        if hasattr(scene, 'parent') and hasattr(scene.parent, 'size') and \
           scene.parent.size[0] > 0 and scene.parent.size[1] > 0:
            window_width, window_height = scene.parent.size
            valid_size_obtained = True
        else: # Try renderer's viewport size (from original and Expert 1 & 2)
            viewport_size = renderer.GetSize()
            if viewport_size[0] > 0 and viewport_size[1] > 0:
                window_width, window_height = viewport_size
                valid_size_obtained = True

    # Calculate normalized dimensions for the circle
    if valid_size_obtained:
        norm_diameter_w = float(diameter_pixels) / window_width
        norm_diameter_h = float(diameter_pixels) / window_height
    else: # Fallback if size is still unknown (e.g., offscreen, pre-first-render)
        print(f"Warning: Window/Renderer size is {window_width}x{window_height} for progress circle. "
              f"Using fallback based on diameter_pixels and assumed 800px dimension.")
        # Fallback inspired by Expert 1: pixel-based using a default assumed dimension
        default_dimension_for_pixel_calc = 800.0
        norm_diameter_w = float(diameter_pixels) / default_dimension_for_pixel_calc
        norm_diameter_h = float(diameter_pixels) / default_dimension_for_pixel_calc
        # Store 0,0 to ensure resize logic triggers on next frame if actual size becomes known
        window_width, window_height = 0,0

    actor_pos_x = center_norm_coords[0] - norm_diameter_w / 2.0
    actor_pos_y = center_norm_coords[1] - norm_diameter_h / 2.0

    if circle_actors is None:
        circle_actors = {}

        # Outline Actor
        outline_source = vtk.vtkRegularPolygonSource()
        outline_source.SetNumberOfSides(60)
        outline_source.SetRadius(0.07)
        outline_source.SetCenter(0, 0, 0)
        outline_source.GeneratePolygonOff()
        outline_source.GeneratePolylineOn()
        # outline_source.Update() # Mapper will call it

        outline_mapper = vtk.vtkPolyDataMapper2D()
        outline_mapper.SetInputConnection(outline_source.GetOutputPort())
        transform_coords_outline = vtk.vtkCoordinate()
        transform_coords_outline.SetCoordinateSystemToNormalizedViewport()
        outline_mapper.SetTransformCoordinate(transform_coords_outline)

        outline_actor = vtk.vtkActor2D()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(color)
        outline_actor.GetProperty().SetLineWidth(outline_thickness)
        outline_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        outline_actor.GetPositionCoordinate().SetValue(actor_pos_x, actor_pos_y)
        outline_actor.SetWidth(norm_diameter_w)
        outline_actor.SetHeight(norm_diameter_h)
        
        renderer.add_actor(outline_actor)
        circle_actors['outline_source'] = outline_source
        circle_actors['outline_actor'] = outline_actor

        # Fill Actor
        fill_source = vtk.vtkSectorSource()
        fill_source.SetInnerRadius(0.0)
        fill_source.SetOuterRadius(0.07)
        fill_source.SetCircumferentialResolution(60)

        fill_mapper = vtk.vtkPolyDataMapper2D()
        fill_mapper.SetInputConnection(fill_source.GetOutputPort())
        transform_coords_fill = vtk.vtkCoordinate()
        transform_coords_fill.SetCoordinateSystemToNormalizedViewport()
        fill_mapper.SetTransformCoordinate(transform_coords_fill)

        fill_actor = vtk.vtkActor2D()
        fill_actor.SetMapper(fill_mapper)
        fill_actor.GetProperty().SetColor(color)
        fill_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        fill_actor.GetPositionCoordinate().SetValue(actor_pos_x, actor_pos_y)
        fill_actor.SetWidth(norm_diameter_w)
        fill_actor.SetHeight(norm_diameter_h)

        renderer.add_actor(fill_actor)
        circle_actors['fill_source'] = fill_source
        circle_actors['fill_actor'] = fill_actor
        
        circle_actors['last_window_size'] = (window_width, window_height) # Could be (0,0)
        circle_actors['last_center_norm_coords'] = center_norm_coords
        circle_actors['last_diameter_pixels'] = diameter_pixels

    else: # circle_actors dictionary exists, VTK objects are already created.
        outline_actor = circle_actors['outline_actor']
        fill_actor = circle_actors['fill_actor']
        fill_source = circle_actors['fill_source']

        # Update properties that might change frame-to-frame (inspired by Expert 3)
        outline_actor.GetProperty().SetColor(color)
        outline_actor.GetProperty().SetLineWidth(outline_thickness)
        fill_actor.GetProperty().SetColor(color)

        # IMPORTANT: Re-add actors to the renderer, as mlab.clf() is assumed to remove them. (Expert 1's core logic)
        # To be robust, remove first in case of any weird state, then add.
        renderer.remove_actor(outline_actor) 
        renderer.add_actor(outline_actor)
        renderer.remove_actor(fill_actor)
        renderer.add_actor(fill_actor)
        
        # Check if window size or other geometry-defining parameters changed
        current_params_window_size = (window_width, window_height) # The size used for calculations in *this* call.
        
        if current_params_window_size != circle_actors.get('last_window_size') or \
           center_norm_coords != circle_actors.get('last_center_norm_coords') or \
           diameter_pixels != circle_actors.get('last_diameter_pixels'):

            # Update actor position and size properties only if current_params_window_size reflects a valid size
            # The variables actor_pos_x, actor_pos_y, norm_diameter_w, norm_diameter_h reflect current params.
            if window_width > 0 and window_height > 0: # Ensure we are updating with valid dimensions
                outline_actor.GetPositionCoordinate().SetValue(actor_pos_x, actor_pos_y)
                outline_actor.SetWidth(norm_diameter_w)
                outline_actor.SetHeight(norm_diameter_h)

                fill_actor.GetPositionCoordinate().SetValue(actor_pos_x, actor_pos_y)
                fill_actor.SetWidth(norm_diameter_w)
                fill_actor.SetHeight(norm_diameter_h)
                
                circle_actors['last_window_size'] = current_params_window_size
                circle_actors['last_center_norm_coords'] = center_norm_coords
                circle_actors['last_diameter_pixels'] = diameter_pixels

    # Update progress display
    if total_frames <= 0: 
        progress = 0.0
    else: 
        progress = min(1.0, max(0.0, float(current_frame + 1) / total_frames)) 

    if progress == 0.0:
        circle_actors['fill_actor'].SetVisibility(0)
    elif progress >= 0.9999: # Use a threshold for "full" to handle float precision
        circle_actors['fill_actor'].SetVisibility(1)
        fill_source.SetStartAngle(90.0)
        fill_source.SetEndAngle(90.0 + 359.99)
    else:
        circle_actors['fill_actor'].SetVisibility(1)
        sweep_angle_deg = progress * 360.0
        fill_source.SetStartAngle(450.0 - sweep_angle_deg)
        fill_source.SetEndAngle(450.0)
    
    fill_source.Update() # Crucial to update the vtkSectorSource geometry
    return circle_actors

# load_simulation_data remains the same
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
    chi_max = max(max(A_chi_mag_data), max(B_chi_mag_data), max(C_chi_mag_data))

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
    angular_velocity_ts: sxs.TimeSeries,
    lab_time_t: float,
    r_min: float,
    r_max: float,
    max_r_step: float = 40,
    samples_per_wavelength: float = 5.0
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
    r_points = [r_min]
    current_r = r_min

    # Create an interpolator for the angular velocity for efficiency.
    # This is much faster than recreating it inside the loop.
    ang_vel_interp = angular_velocity_ts.interpolate

    # Define a minimum frequency to prevent infinitely large steps
    # during the very early, slow inspiral. This corresponds to a max
    # step size of about 25 M.
    min_omega_gw = 2*np.pi / max_r_step

    while current_r < r_max:
        retarded_time = np.array([lab_time_t - current_r])

        try:
            omega_orb = np.asarray(ang_vel_interp(retarded_time).data).ravel()
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
        
        # Advance the current radius
        current_r += delta_r
        
        # Append the new point, but avoid overshooting the boundary.
        if current_r < r_max:
            r_points.append(current_r)
        else:
            break


    # Ensure the last point is exactly r_max for a clean boundary.
    if r_points[-1] < r_max:
        r_points.append(r_max)
        
    r_points_out = np.flip(np.array(r_points))
    return r_points_out

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

    # --- Vectorized Interpolation ---
    # Create a 1D array of all unique retarded times needed.
    # This is the core optimization.
    retarded_times_vec = np.asarray(lab_time_t - r_coords)
    # Times must be strictly INCREASING

    try:
        # Perform one single, fast interpolation for all radii at once.
        strain_modes_at_times_obj = sxs_strain_modes.interpolate(retarded_times_vec)
        # h_coeffs_matrix will have shape (len(r_coords), n_modes)
        h_coeffs_matrix = strain_modes_at_times_obj.data
    except ValueError:
        print(f"Out of bounds at lab time {lab_time_t}")
        sys.exit()
        # This will be raised if any time in retarded_times_vec is out of bounds.
        # We return a flat plane as a fallback.
        # A more robust solution could identify which times are valid and set others to zero.
        # For simplicity here, we assume if one is bad, all are bad for this frame.
        return z_displacement_grid

    theta_val_equator = np.pi / 2.0
    theta_for_evaluation = np.full_like(phi_coords, theta_val_equator)
    gamma_for_evaluation = np.zeros_like(phi_coords)
    # Convert Euler angles (alpha, beta, gamma) to quaternions (rotors).
    # alpha corresponds to phi, beta to theta.
    # This creates an array of quaternion objects, one for each (phi, theta) point.
    rotors_for_evaluation = quaternionic.array.from_euler_angles(phi_coords, theta_for_evaluation, gamma_for_evaluation)
    
    ell_min, ell_max = sxs_strain_modes.ell_min, sxs_strain_modes.ell_max

    # Now, loop through the results of the interpolation. This loop is much faster
    # because the expensive part (interpolation) is already done.
    for i in range(len(r_coords)):
        # Get the row of coefficients for the i-th radius
        h_coeffs_for_this_r = h_coeffs_matrix[i, :]
        
        if np.all(h_coeffs_for_this_r == 0):
            # This can happen if the interpolation returned zeros for out-of-bounds times
            # that were handled internally by sxs.
            z_displacement_grid[i, :] = 0.0
            continue
            
        sph_modes_obj = spherical.Modes(
            h_coeffs_for_this_r,
            spin_weight=spin_weight,
            ell_min=ell_min,
            ell_max=ell_max
        )

        complex_strain_values_at_r = sph_modes_obj.evaluate(rotors_for_evaluation)
        # Could simply use the scri.WaveformModes.to_grid() method and avoid quaternions,
        # but that creates the whole spherical grid and we only need a ring

        z_displacement_grid[i, :] = complex_strain_values_at_r.real * AMPLITUDE_SCALE
        
    return z_displacement_grid


# --- Main Animation Logic ---
def create_merger_movie():
    script_init_time = time.time()
    strain_modes_sxs, horizons_data = load_simulation_data(SXS_ID)
    data_loaded_time = time.time()
    print(f"Data loading took {data_loaded_time - script_init_time:.2f}s")

    peak_strain_time = strain_modes_sxs.max_norm_time()
    sim_start_time = strain_modes_sxs.t[0]
    sim_end_time = strain_modes_sxs.t[-1]
    sim_total_time = sim_end_time - sim_start_time
    anim_start_time = peak_strain_time - (0.4 * sim_total_time)
    anim_end_time = (0.08 * sim_total_time) + peak_strain_time
    anim_start_time = max(anim_start_time, sim_start_time)
    anim_end_time = min(anim_end_time, sim_end_time)
    anim_lab_times = np.linspace(
        anim_start_time,
        anim_end_time,
        NUM_FRAMES, endpoint=True
    )

    print(f"Animation time: {anim_lab_times[0]:.2f}M to {anim_lab_times[-1]:.2f}M over {len(anim_lab_times)} frames.")

    common_horizon_start = (horizons_data.A.time[-1] + horizons_data.C.time[0])/2
    print(common_horizon_start)
    print(peak_strain_time)

    phi_gw_axis = np.linspace(0, 2 * np.pi, N_PHI_GW, endpoint=True)

    mlab.figure(size=(1280, 1024), bgcolor=(0.3, 0.3, 0.3))
    # mlab.options.offscreen = True # Ensure offscreen rendering for saving frames without GUI pop-up

    bh_surfs, spin_vectors, spin_arrow_size, orbital_vel_t = get_bh_mesh_data(horizons_data, anim_lab_times)
    spin_arrow_size *= 9

    frame_files = []
    frames_dir_path = "frames" # Changed directory name as requested
    os.makedirs(frames_dir_path, exist_ok=True)

    total_physical_anim_time = anim_lab_times[-1] - anim_lab_times[0]
    if total_physical_anim_time == 0: total_physical_anim_time = 1.0 # Avoid division by zero

    progress_circle_actors = None 
    # Parameters for the progress circle (can be defined once if static)
    circle_display_params = {
        "diameter_pixels": 60,
        "center_norm_coords": (0.1, 0.95), 
        "color": (0.95, 0.85, 0.95), # Light purple
        "outline_thickness": 2
    }
    no_circle_display_params = {
        "diameter_pixels": 60,
        "center_norm_coords": (-1, 2), 
        "color": (0.3, 0.3, 0.3), # same as background
        "outline_thickness": 2
    }


    print(f"Processing and surface building took {time.time() - data_loaded_time:.2f}s")
    print("Starting frame rendering loop...")

    for i_frame, current_lab_time in enumerate(anim_lab_times):
        frame_render_start_time = time.time()
        if not auto_loop_bool:
            print(f"Processing frame {i_frame+1}/{NUM_FRAMES} for lab_time = {current_lab_time:.2f} M")

        r_gw_axis = generate_adaptive_r_coords(orbital_vel_t, current_lab_time, MIN_R_GW, MAX_R_GW, MIN_R_STEP)
        PHI_GW_MESH, R_GW_MESH = np.meshgrid(phi_gw_axis, r_gw_axis)
        X_GW_SURF = R_GW_MESH * np.cos(PHI_GW_MESH)
        Y_GW_SURF = R_GW_MESH * np.sin(PHI_GW_MESH)



        z_gw_frame = reconstruct_hplus_on_xy_plane_at_time_t(
            strain_modes_sxs, current_lab_time, r_gw_axis, phi_gw_axis
        )

        mlab.clf()
        current_scene = mlab.gcf().scene 
        if not current_scene or not current_scene.renderer:
            print(f"Error: Could not get valid Mayavi scene/renderer for frame {i_frame}. Skipping.")
            continue
        
        if current_lab_time < common_horizon_start:
            # plot BH1
            mlab.mesh(*bh_surfs[0][i_frame], opacity=1, color=(0, 0, 0))
            mlab.quiver3d(*spin_vectors[0][:, i_frame], color=(0.5, 0.5, 0.5), line_width = 0.4*spin_arrow_size, scale_factor = spin_arrow_size)
            # plot BH2
            mlab.mesh(*bh_surfs[1][i_frame], opacity=1, color=(0, 0, 0))
            mlab.quiver3d(*spin_vectors[1][:, i_frame], color=(0.5, 0.5, 0.5), line_width = 0.4*spin_arrow_size, scale_factor = spin_arrow_size)
        else:
            # plot merged BH
            mlab.mesh(*bh_surfs[2][i_frame], opacity=1, color=(0, 0, 0))
            mlab.quiver3d(*spin_vectors[2][:, i_frame], color=(0.5, 0.5, 0.5), line_width = 0.4*spin_arrow_size, scale_factor = spin_arrow_size)
        
        mlab.view(azimuth=30, elevation=60, distance=PIP_CAMERA_DISTANCE, focalpoint=(0,0,0))

        # Make progress circle same color as background and move way out of frame
        progress_circle_actors = update_or_create_progress_circle(
           scene=current_scene,
           current_frame=i_frame, 
           total_frames=NUM_FRAMES, 
           circle_actors=progress_circle_actors,
           **no_circle_display_params # Pass display parameters
        )

        pip_arr_large = mlab.screenshot(antialiased=True)

        # plot GW surface
        mlab.mesh(X_GW_SURF, Y_GW_SURF, z_gw_frame,
                        color=GW_SURFACE_COLOR,
                        # representation='wireframe',
                        name="GW h+ Surface",
                        opacity=0.75,
                        )

        mlab.view(azimuth=30, elevation=60, distance=MAIN_CAMERA_DISTANCE, focalpoint=(0,0,0))
        # <<<< PROGRESS CIRCLE UPDATE >>>>
        progress_circle_actors = update_or_create_progress_circle(
           scene=current_scene,
           current_frame=i_frame, 
           total_frames=NUM_FRAMES, 
           circle_actors=progress_circle_actors,
           **circle_display_params # Pass display parameters
        )
        
        main_arr = mlab.screenshot(antialiased=True)

        # Resize PiP image
        orig_pip_h, orig_pip_w, _ = pip_arr_large.shape
        new_pip_h, new_pip_w = int(orig_pip_h / PIP_SCALE), int(orig_pip_w / PIP_SCALE)
        pip_arr_resized = cv2.resize(pip_arr_large, (new_pip_w, new_pip_h), interpolation=cv2.INTER_AREA)
        # Paste the resized PiP array onto the main array
        main_arr[:new_pip_h, (orig_pip_w - new_pip_w):] = pip_arr_resized

        frame_filename = f"{frames_dir_path}/frame_{i_frame:04d}.png"
        combined_frame = cv2.cvtColor(main_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_filename, combined_frame)

        frame_files.append(frame_filename)
        if not auto_loop_bool:
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
    if auto_loop_bool:
        for sxs_idx in range(sxs_idx_start, sxs_idx_start + loop_size):
            SXS_ID = f"SXS:BBH:{sxs_idx:04d}"
            OUTPUT_MOVIE_FILENAME = f"{SXS_ID.replace(':', '_')}_PiP_movie.mp4"
            create_merger_movie()
    else:
        create_merger_movie()
