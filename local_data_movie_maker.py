import numpy as np
import spherical
import quaternionic
from mayavi import mlab
from scipy.interpolate import interp1d
import imageio.v2 as imageio # clarification 
import os
import glob
import time
import inspect
import sys

# --- Configuration Parameters ---
input_data_dir = "<strain input path>"
extraction_radius_str = "100"
# Define the folder where the .gp (horizon) files are stored
horizon_folder = "<horizon input path>" # change if separate locations

data_name = "Einstein_toolkit_combined_data"
OUTPUT_MOVIE_FILENAME = f"{data_name}_hplus_ah_surfaces_movie.mp4"
NUM_FRAMES = 240
FPS = 24

# Apparent Horizon Data Shape Parameters
N_THETA_AH = 32
N_PHI_AH = 65
ITERATION_TIME_STEP = 0.55 # Time separating *supplied* AH data files (not the time step in between actual AHA interations, see below)
ITERATION_NAME_STEP = 44 # Usually only every x iterations are actually supplied (e.g. data goes from h.t0000000.ah1.gp to h.t0000044.ah1.gp)

# GW Surface Visualization Parameters
N_R_GW = 80
N_PHI_GW = 160
MIN_R_GW = 8.5
MAX_R_GW = 130.0
AMPLITUDE_SCALE = 7e5 # Factor to scale h+ for z-displacement (TUNE THIS!)
GW_SURFACE_COLOR = (0.3, 0.6, 1.0) # Uniform color for the GW surface

CAMERA_INITIAL_DISTANCE = MAX_R_GW * 0.78
CAMERA_FINAL_DISTANCE = MAX_R_GW * 1.8
CAMERA_ZOOM_START_TIME_FRAC = 0.6
CAMERA_ZOOM_END_TIME_FRAC = 0.9 # Extended zoom duration

# --- Helper Class for Interpolated Data ---
class _InterpolatedDataContainer:
    """
    A simple container class that holds interpolated mode data.
    It provides a .data attribute, mimicking the relevant structure
    expected by the reconstruction function.
    """
    def __init__(self, data_1d_array_complex: np.ndarray):
        """
        Initializes the container with a 1D array of complex mode coefficients.
        
        :param data_1d_array_complex: A 1D NumPy array of complex mode coefficients.
        """
        if not isinstance(data_1d_array_complex, np.ndarray) or data_1d_array_complex.ndim != 1:
            raise TypeError("Input for _InterpolatedDataContainer must be a 1D NumPy array.")
        # The .data attribute stores this as a 2D array of shape (1, num_modes)
        self.data = data_1d_array_complex.reshape(1, -1)

# --- Main Data Loader Class ---
class StrainDataFromFile:
    """
    Loads gravitational wave strain data from multiple text files,
    aggregates it, and provides an interpolation interface compatible
    with the `reconstruct_hplus_on_xy_plane_at_time_t` function.
    """
    def __init__(self, data_directory: str, radius_str: str):
        """
        Initializes the data loader.

        :param data_directory: Path to the directory containing the mode data files.
        :param radius_str: String representation of the extraction radius (e.g., "0200.0"),
                           used to construct filenames and scale mode data.
        """
        if not os.path.isdir(data_directory):
            raise FileNotFoundError(f"Data directory not found: {data_directory}")
        self.data_directory = data_directory
        self.radius_str = radius_str

        try:
            self.extraction_radius = float(radius_str)
            if self.extraction_radius == 0:
                raise ValueError("Extraction radius must be non-zero for data scaling.")
        except ValueError as e:
            raise ValueError(f"Invalid radius_str: '{radius_str}'. Cannot convert to float: {e}") from e

        self.ell_min = 2
        self.ell_max = 8

        self.time_array: np.ndarray = None  # Populated by _load_data()
        self.modes_data_at_times: np.ndarray = None  # Shape (num_times, num_total_modes), complex
        self._interpolators: list = []  # List of complex interpolators, one for each mode

        self._load_data()
        self._setup_interpolators()

    def _load_data(self):
        """
        Loads and processes data from individual mode files.
        Populates self.time_array and self.modes_data_at_times.
        Assumes files are named 'h_l<ELL>_r<RADIUS>.txt'.
        """
        all_l_modes_data_list = []
        time_array_loaded = False

        for ell in range(self.ell_min, self.ell_max + 1):
            filename = f"h_l{ell}_r{self.radius_str}.txt"
            filepath = os.path.join(self.data_directory, filename)

            try:
                raw_file_data = np.loadtxt(filepath, comments='#')
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Data file not found: {filepath}. "
                    f"Ensure all files for l={self.ell_min} to l={self.ell_max} exist in '{self.data_directory}'."
                )
            except Exception as e:
                raise IOError(f"Error loading or parsing data from {filepath}: {e}")

            if raw_file_data.ndim == 0:
                raise ValueError(f"File {filepath} is empty or contains insufficient data (loaded as scalar).")
            if raw_file_data.ndim == 1: # Single line of data
                raw_file_data = raw_file_data.reshape(1, -1)
            if raw_file_data.shape[0] == 0: # No data rows after comments
                raise ValueError(f"File {filepath} contains no data rows after parsing comments.")
            
            expected_num_cols = 1 + 2 * (2 * ell + 1) # 1 (time) + 2*(Re/Im) * (2*ell+1 modes)
            if raw_file_data.shape[1] != expected_num_cols:
                raise ValueError(
                    f"File {filepath} has {raw_file_data.shape[1]} columns, "
                    f"but {expected_num_cols} were expected for l={ell}."
                )

            current_file_time_array = raw_file_data[:, 0]

            if not time_array_loaded:
                self.time_array = current_file_time_array.copy()
                if len(self.time_array) < 4: # Cubic spline (k=3) needs at least k+1=4 points
                    raise ValueError(
                        f"Time data in {filepath} (l={ell}) has {len(self.time_array)} points. "
                        f"At least 4 data points are required for cubic spline interpolation."
                    )
                if not np.all(np.diff(self.time_array) > 0):
                    raise ValueError(
                        f"Time data in {filepath} (l={ell}) must be strictly increasing."
                    )
                time_array_loaded = True
            else:
                if not np.array_equal(self.time_array, current_file_time_array):
                     raise ValueError(
                        f"Time column in {filepath} (l={ell}) does not match "
                        f"the time column from the first loaded file (e.g., l={self.ell_min})."
                    )

            num_times = len(self.time_array)
            num_m_modes_for_l = 2 * ell + 1
            current_l_modes_data_complex = np.zeros((num_times, num_m_modes_for_l), dtype=np.complex128)

            for m_idx in range(num_m_modes_for_l): # m_idx from 0 to 2*ell
                real_col_idx = 1 + (2 * m_idx)
                imag_col_idx = 1 + (2 * m_idx) + 1
                
                real_part = raw_file_data[:, real_col_idx]
                imag_part = raw_file_data[:, imag_col_idx]
                
                current_l_modes_data_complex[:, m_idx] = (real_part + 1j * imag_part) / self.extraction_radius
            
            all_l_modes_data_list.append(current_l_modes_data_complex)

        if not time_array_loaded:
             raise RuntimeError("No data was successfully loaded. Ensure data_directory and files are correct.")

        self.modes_data_at_times = np.concatenate(all_l_modes_data_list, axis=1)

    def _setup_interpolators(self):
        """
        Creates cubic spline interpolators for each complex mode time series.
        Requires SciPy >= 1.0.0 for interp1d to handle complex data directly.
        """
        if self.modes_data_at_times is None or self.time_array is None:
            raise RuntimeError("Data must be loaded via _load_data() before setting up interpolators.")
        
        if self.time_array.shape[0] < 4: # Should have been caught in _load_data, but defensive check
             raise ValueError(
                 f"Insufficient time points ({self.time_array.shape[0]}) "
                 "to perform cubic interpolation. At least 4 points are required."
             )

        num_total_modes = self.modes_data_at_times.shape[1]
        for i in range(num_total_modes):
            # interp1d can directly handle complex data since SciPy 1.0.0
            # bounds_error=True will raise ValueError for out-of-bounds requests.
            complex_mode_coeffs_over_time = self.modes_data_at_times[:, i]
            interpolator = interp1d(
                self.time_array, 
                complex_mode_coeffs_over_time, 
                kind='cubic', 
                bounds_error=True
            )
            self._interpolators.append(interpolator)

    def interpolate(self, target_times_np_array: np.ndarray):
        """
        Interpolates all mode coefficients to the time(s) specified.

        :param target_times_np_array: A 1D NumPy array of target time(s).
                                      Expected to contain a single retarded time value.
        :return: An `_InterpolatedDataContainer` object. Its `.data` attribute is a 
                 2D NumPy array of shape (1, num_total_modes) containing the 
                 complex interpolated mode coefficients for the (first) target time.
        :raises ValueError: If a target_time is outside bounds, input array is invalid.
        :raises TypeError: If target_times_np_array is not a NumPy array.
        """
        if not isinstance(target_times_np_array, np.ndarray):
            raise TypeError("Input times must be a NumPy array.")
        if target_times_np_array.ndim != 1:
            raise ValueError("Input times must be a 1D NumPy array.")
        if target_times_np_array.size == 0:
            raise ValueError("Input times array cannot be empty.")

        target_time = target_times_np_array[0]
        
        if not self._interpolators: # Should not happen if init was successful
            raise RuntimeError("Interpolation called before interpolators were set up.")

        num_modes = len(self._interpolators)
        interpolated_coeffs = np.zeros(num_modes, dtype=np.complex128)
        
        try:
            for i in range(num_modes):
                interpolated_coeffs[i] = self._interpolators[i](target_time) # Can raise ValueError
        except ValueError as e:
            raise ValueError(
                f"Time {target_time} is out of bounds for interpolation. "
                f"Data available in time range: [{self.time_array[0]}, {self.time_array[-1]}]."
            ) from e

        return _InterpolatedDataContainer(interpolated_coeffs)


# --- Helper Functions ---
# Function that can handle potential blank lines or irregularities in the horizon file
def parse_gp_file_safe(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith("#") and line.strip():  # Skip header and blank lines
                try:
                    # Attempt to parse the line into floats
                    data.append(list(map(float, line.split())))
                except ValueError:
                    # Ignore lines that cannot be parsed
                    continue
    return np.array(data)


def get_bh_mesh_data(horizons_dir : str, time_vals : np.array,time_step=0.55, name_step = 44):
    # Get all .gp files in sorted order
    horizon_data_list = [[], [], []]
    for i in range(1,4):
        for frame, t in enumerate(time_vals):
            actualdx = round(t/time_step) * name_step
            filename = f"h.t{actualdx:07d}.ah{i}.gp"
            filepath = os.path.join(horizons_dir, filename)
            try:
                horizon_data_list[i-1].append(parse_gp_file_safe(filepath))
                if i==2:
                    merger_frame_out = frame
            except FileNotFoundError:
                if i==3:
                    merger_start_frame = frame

    return merger_frame_out, merger_start_frame, horizon_data_list



    


    

def reconstruct_hplus_on_xy_plane_at_time_t(
    strain_data_source: StrainDataFromFile,
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
            strain_modes_at_retarded_t_obj = strain_data_source.interpolate(retarded_time_for_interp)
            h_coeffs_at_ret_t = strain_modes_at_retarded_t_obj.data.ravel()
            
            if h_coeffs_at_ret_t.size == 0:
                z_displacement_grid[i, :] = 0.0
                print("no h_coeffs")
                continue

        except ValueError:
            z_displacement_grid[i, :] = 0.0
            continue


        sph_modes_obj = spherical.Modes(h_coeffs_at_ret_t,
                                spin_weight=spin_weight,
                                ell_min=strain_data_source.ell_min,
                                ell_max=strain_data_source.ell_max)

        complex_strain_values_at_r = sph_modes_obj.evaluate(rotors_for_evaluation)
        # Could simply use the scri.WaveformModes.to_grid() method and avoid quaternions,
        # but that creates the whole spherical grid and we only need a ring

        z_displacement_grid[i, :] = complex_strain_values_at_r.real * AMPLITUDE_SCALE
        
    return z_displacement_grid


# --- Main Animation Logic ---
def create_merger_movie():
    script_init_time = time.time()


    print(f"\nLoading strain data from '{input_data_dir}' with radius_str='{extraction_radius_str}'...")
    strain_modes = StrainDataFromFile(data_directory=input_data_dir, radius_str=extraction_radius_str)
    print("Data loaded successfully.")
    print(f"  Time range: [{strain_modes.time_array[0]:.2f}, {strain_modes.time_array[-1]:.2f}]")
    print(f"  Total modes: {strain_modes.modes_data_at_times.shape[1]}")
    print(f"  l_min={strain_modes.ell_min}, l_max={strain_modes.ell_max}")
    

    times_all = strain_modes.time_array
    # peak_strain_time = strain_modes.max_norm_time()
    # anim_start_time = peak_strain_time - 500.0 
    # anim_end_time = peak_strain_time + 100.0
    anim_start_time = times_all[0] + 750 # max(anim_start_time, sxs_times_all[0])
    anim_end_time = times_all[-1] - 650 # min(anim_end_time, sxs_times_all[-1])
    anim_lab_times = np.linspace(
        anim_start_time,
        anim_end_time,
        NUM_FRAMES, endpoint=True
    )

    print(f"Animation time: {anim_lab_times[0]:.2f}M to {anim_lab_times[-1]:.2f}M over {len(anim_lab_times)} frames.")

    separate_horizons_end_frame, common_horizon_start_frame, ah_surfs = get_bh_mesh_data(horizon_folder, anim_lab_times, ITERATION_TIME_STEP, ITERATION_NAME_STEP)
    print(common_horizon_start_frame, separate_horizons_end_frame)

    r_gw_axis = np.linspace(MIN_R_GW, MAX_R_GW, N_R_GW)
    phi_gw_axis = np.linspace(0, 2 * np.pi, N_PHI_GW, endpoint=True)
    PHI_GW_MESH, R_GW_MESH = np.meshgrid(phi_gw_axis, r_gw_axis)
    X_GW_SURF = R_GW_MESH * np.cos(PHI_GW_MESH)
    Y_GW_SURF = R_GW_MESH * np.sin(PHI_GW_MESH)


    mlab.figure(size=(1280, 1026), bgcolor=(0.3, 0.3, 0.3))
    # mlab.options.offscreen = True # Ensure offscreen rendering for saving frames without GUI pop-up





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
            strain_modes, current_lab_time, r_gw_axis, phi_gw_axis
        )

        mlab.clf()
        # plot GW surface
        mlab.mesh(X_GW_SURF, Y_GW_SURF, z_gw_frame,
                        color=GW_SURFACE_COLOR,
                        representation='wireframe',
                        name="GW h+ Surface",
                        opacity=0.9,
                        )
        

        
        if i_frame <= separate_horizons_end_frame:
            # plot BH1
            x_grid, y_grid, z_grid = ah_surfs[0][i_frame][:, 0].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[0][i_frame][:, 1].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[0][i_frame][:, 2].reshape(N_THETA_AH, N_PHI_AH)
            mlab.mesh(x_grid, y_grid, z_grid, opacity=1.0, color=(0, 0, 0))
            # blot BH2
            x_grid, y_grid, z_grid = ah_surfs[1][i_frame][:, 0].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[1][i_frame][:, 1].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[1][i_frame][:, 2].reshape(N_THETA_AH, N_PHI_AH)
            mlab.mesh(x_grid, y_grid, z_grid, opacity=1.0, color=(0, 0, 0))
            if i_frame > common_horizon_start_frame:
                x_grid, y_grid, z_grid = ah_surfs[2][i_frame - common_horizon_start_frame - 1][:, 0].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[2][i_frame - common_horizon_start_frame - 1][:, 1].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[2][i_frame - common_horizon_start_frame - 1][:, 2].reshape(N_THETA_AH, N_PHI_AH)
                mlab.mesh(x_grid, y_grid, z_grid, opacity=0.5, color=(1, 0.2, 0.3))
        else:
            x_grid, y_grid, z_grid = ah_surfs[2][i_frame - common_horizon_start_frame - 1][:, 0].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[2][i_frame - common_horizon_start_frame - 1][:, 1].reshape(N_THETA_AH, N_PHI_AH), ah_surfs[2][i_frame - common_horizon_start_frame - 1][:, 2].reshape(N_THETA_AH, N_PHI_AH)
            mlab.mesh(x_grid, y_grid, z_grid, opacity=1.0, color=(0, 0, 0))




        
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
