"""A script to visualize gravitational wave strain data and black hole trajectories.

Process gravitational wave strain data and black hole positional data,
apply spin-weighted spherical harmonics to the data, and create a Mayavi animation
of the black holes and their gravitational waves. At each state, move the black holes
to their respective positions and save the render as a .png file.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

import os
import sys
import csv
import time
from math import erf
from typing import Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
import quaternionic
import spherical
import cv2 # pip install opencv-python
import vtk  # Unused, but Required by TVTK.
from tvtk.api import tvtk
from mayavi import mlab
from mayavi.api import Engine
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.sources.parametric_surface import ParametricSurface
from mayavi.modules.surface import Surface
from mayavi.modules.scalar_cut_plane import ScalarCutPlane
import psi4_FFI_to_strain as psi4strain


BH_DIR = "../data/GW150914_data/r100" # changeable with sys arguments
MOVIE_DIR = "../data/GW150914_data/movies" # changeable with sys arguments
ELL_MAX = 8
ELL_MIN = 2
S_MODE = -2
EXT_RAD = 100 # changeable with sys arguments
USE_SYS_ARGS = False
STATUS_MESSAGES = True

def swsh_summation_angles(colat: float, azi: NDArray[np.float64], mode_data: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Add up all strain modes with corresponding spin-weighted spherical harmonics.

    This function calculates the superimposed wave at specified angles by factoring in
    spin-weighted spherical harmonics for each mode. The result is an array
    corresponding to [angle, time].

    :param colat: The colatitude angle for the SWSH factor.
    :param azi: The azimuthal angle for the SWSH factor.
    :param mode_data: A NumPy array containing complex-valued strain data for all modes.
                      Expected shape is (n_modes, n_times).
    :return: A complex-valued NumPy array of the superimposed wave, with shape (n_pts, n_times).

    DocTests:
    >>> mode_data = np.zeros((77, 3), dtype=complex)
    >>> mode_idx = 0
    >>> for l in range(2, 9):
    ...     for m in range(-l, l+1):
    ...         mode_data[mode_idx] = np.array([1+1j, 2+3j, 4+5j])
    ...         mode_idx += 1
    >>> np.round(swsh_summation_angles(np.pi/2, np.array([0]), mode_data), 5)
    array([[ 4.69306 +4.69306j,  9.38612+14.07918j, 18.77224+23.4653j ]])
    """
    # Adds up all the strain modes after factoring in corresponding spin-weighted spherical harmonic
    # to specified angle in the mesh. Stored as an array corresponding to [angle, time] time_idxs.
    quat_arr = quaternionic.array.from_spherical_coordinates(colat, azi)
    winger = spherical.Wigner(ELL_MAX, ELL_MIN)
    # Create an swsh array shaped like (n_modes, n_quaternions)
    swsh_arr = winger.sYlm(S_MODE, quat_arr).T
    # mode_data has shape (n_modes, n_times), swsh_arr has shape (n_mode, n_pts).
    # Pairwise multiply and sum over modes: the result has shape (n_pts, n_times).
    pairwise_product = mode_data[:, np.newaxis, :] * swsh_arr[:, :, np.newaxis]
    return np.sum(pairwise_product, axis=0)

def generate_interpolation_points(
    time_array: NDArray[np.float64],
    radius_values: NDArray[np.float64],
    r_ext: float,
) -> NDArray[np.float64]:
    """Generate adjusted time values for linear interpolation of wave strain.

    This function creates a 2D array of time values. The first index represents
    the simulation time (which mesh), and the second index represents the radial
    distance to interpolate to. The time values are adjusted based on radius and
    extraction radius.

    :param time_array: NumPy array of strain time indices (simulation times).
    :param radius_values: NumPy array of radial points on the mesh.
    :param r_ext: The extraction radius of the original data.
    :return: A 2D NumPy array of time values with shape (n_radius, n_times).

    DocTests:
    >>> time_arr = np.array([10.0, 11.0, 12.0])
    >>> r_vals = np.array([100.0, 101.0, 102.0])
    >>> r_ext = 100.0
    >>> expected_times = np.array([[10., 11., 12.], [10., 11., 12.], [10., 11., 12.]]) # time - radius + r_ext. e.g. 10-100+100=10, 11-100+100=11, 12-100+100=12 (for first row)
    >>> np.array_equal(generate_interpolation_points(time_arr, r_vals, r_ext), expected_times)
    True
    >>> time_arr_short = np.array([5.0, 6.0])
    >>> r_vals_short = np.array([10.0, 11.0])
    >>> r_ext_short = 10.0
    >>> expected_times_short = np.array([[5., 6.], [5., 6.]])
    >>> np.array_equal(generate_interpolation_points(time_arr_short, r_vals_short, r_ext_short), expected_times_short)
    True
    """
    # Fills out a 2D array of adjusted time values for the wave strain to be
    # linearly interpolated to. First index of the result represents the simulation
    # time time_idx (aka which mesh), and the second index represents radial distance to
    # interpolate to.
    # Repeat time_array and radius_values to match the shape of the output array
    time_repeated = np.repeat(time_array[np.newaxis, :], len(radius_values), axis=0)
    radius_repeated = np.repeat(radius_values[:, np.newaxis], len(time_array), axis=1)

    target_times = time_repeated - radius_repeated + r_ext
    # Shape is (n_radius, n_times)
    filtered_target_times = np.clip(target_times, time_array.min(), time_array.max())
    return filtered_target_times

def interpolate_coords_by_time(
    old_times: NDArray[np.float64],
    e1: NDArray[np.float64],
    e2: NDArray[np.float64],
    e3: NDArray[np.float64],
    new_times: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate 3D coordinates to a new set of time states.

    :param old_times: Original time array corresponding to the input coordinates.
    :param e1: Array of first coordinates (e.g., X-coordinates).
    :param e2: Array of second coordinates (e.g., Y-coordinates).
    :param e3: Array of third coordinates (e.g., Z-coordinates).
    :param new_times: Array of new time states to which the coordinates should be interpolated.
    :return: A tuple containing three NumPy arrays: interpolated e1, e2, and e3 coordinates.

    DocTests:
    >>> old_t = np.array([0.0, 1.0, 2.0])
    >>> x_coords = np.array([0.0, 1.0, 2.0])
    >>> y_coords = np.array([0.0, 0.5, 1.0])
    >>> z_coords = np.array([0.0, 0.2, 0.4])
    >>> new_t = np.array([0.5, 1.5])
    >>> ix, iy, iz = interpolate_coords_by_time(old_t, x_coords, y_coords, z_coords, new_t)
    >>> np.round(ix, 1)
    array([0.5, 1.5])
    >>> np.round(iy, 1)
    array([0.2, 0.8])
    >>> np.round(iz, 1)
    array([0.1, 0.3])
    >>> old_t_single = np.array([0.0])
    >>> x_coords_single = np.array([5.0])
    >>> y_coords_single = np.array([6.0])
    >>> z_coords_single = np.array([7.0])
    >>> new_t_single = np.array([0.0])
    >>> ix_s, iy_s, iz_s = interpolate_coords_by_time(old_t_single, x_coords_single, y_coords_single, z_coords_single, new_t_single)
    >>> np.array_equal(ix_s, np.array([5.0]))
    True
    """
    # Interpolates the 3D coordinates to the given time state.
    new_e1 = interp1d(old_times, e1, fill_value="extrapolate")(new_times)
    new_e2 = interp1d(old_times, e2, fill_value="extrapolate")(new_times)
    new_e3 = interp1d(old_times, e3, fill_value="extrapolate")(new_times)
    return new_e1, new_e2, new_e3

def initialize_tvtk_grid(num_azi: int, num_radius: int) -> Tuple[tvtk.FloatArray, tvtk.UnstructuredGrid, tvtk.Points]:
    """Set initial parameters for mesh generation and create a circular, polar mesh.

    This function initializes a circular, polar mesh with the specified number of
    azimuthal and radial points, along with the necessary TVTK objects to write
    and save data.

    :param num_azi: The number of azimuthal points on the mesh.
    :param num_radius: The number of radial points on the mesh.
    :returns: A tuple containing:
              - strain_array (tvtk.FloatArray): An array to store strain scalar data.
              - grid (tvtk.UnstructuredGrid): The unstructured grid representing the mesh.
              - points (tvtk.Points): The points object that stores the mesh coordinates.

    DocTests:
    >>> strain_array, grid, points = initialize_tvtk_grid(3, 4)
    >>> isinstance(strain_array, tvtk.FloatArray)
    True
    >>> isinstance(grid, tvtk.UnstructuredGrid)
    True
    >>> isinstance(points, tvtk.Points)
    True
    >>> grid.number_of_cells
    6
    >>> grid.number_of_points
    12
    """
    # Sets initial parameters for the mesh generation module and returns
    # a circular, polar mesh with manipulation objects to write and save data.

    # Create tvtk objects
    points = tvtk.Points()
    grid = tvtk.UnstructuredGrid()
    strain_array = tvtk.FloatArray(
        name="Strain", number_of_components=1, number_of_tuples=num_azi * num_radius
    )

    # Create cells
    cell_array = tvtk.CellArray()
    for j in range(num_radius - 1):
        for i in range(num_azi):
            cell = tvtk.Quad()
            point_ids = [
                i + j * num_azi,
                (i + 1) % num_azi + j * num_azi,
                (i + 1) % num_azi + (j + 1) * num_azi,
                i + (j + 1) % num_azi,
            ]
            for idx, pid in enumerate(point_ids):
                cell.point_ids.set_id(idx, pid)
            cell_array.insert_next_cell(cell)
    # Set grid properties
    # grid.points = points
    grid.set_cells(tvtk.Quad().cell_type, cell_array)

    return strain_array, grid, points

def create_gw(
    engine: Engine,
    grid: tvtk.UnstructuredGrid,
    color: Tuple[float, float, float],
    display_radius: int,
    wireframe: bool = False,
) -> None:
    """Create and display a gravitational wave strain from a given grid.

    The gravitational wave is visualized as a surface, with optional wireframe contours.

    :param engine: The Mayavi engine instance.
    :param grid: The `tvtk.UnstructuredGrid` object containing the gravitational wave data.
    :param color: The color of the gravitational wave surface as an RGB tuple, with
                  components ranging from 0.0 to 1.0.
    :param display_radius: The maximum radius up to which the wave is displayed.
    :param wireframe: If True, additional wireframe contours are displayed; otherwise,
                      only the surface is shown.

    DocTests:
    >>> # This function modifies the Mayavi scene directly, so doctests are tricky.
    >>> # A "mock" setup or integration test would be more appropriate.
    >>> # For now, we'll demonstrate it with a simplified call (without scene setup).
    >>> # This won't run a full Mayavi render, but checks argument handling.
    >>> # from mayavi.api import Engine
    >>> # engine = Engine()
    >>> # engine.start()
    >>> # grid = tvtk.UnstructuredGrid()
    >>> # create_gw(engine, grid, (1.0, 0.0, 0.0), 100, True)
    >>> # engine.stop()
    """
    # Creates and displays a gravitational wave strain from a given grid.
    scene = engine.scenes[0]
    gw = VTKDataSource(data=grid)
    engine.add_source(gw, scene)
    s = Surface()
    engine.add_filter(s, gw)
    s.actor.mapper.scalar_visibility = False
    s.actor.property.color = color

    def gen_contour(coord: NDArray[np.float64], normal: NDArray[np.float64]) -> None:
        contour = ScalarCutPlane()
        engine.add_filter(contour, gw)
        contour.implicit_plane.widget.enabled = False
        contour.implicit_plane.plane.origin = coord
        contour.implicit_plane.plane.normal = normal
        contour.actor.property.line_width = 5
        contour.actor.property.opacity = 0.5

    if wireframe:
        wire_intervals = np.linspace(-display_radius, display_radius, 14)

        for c in wire_intervals:
            gen_contour(np.array([c, 0, 0]), np.array([1, 0, 0]))
            gen_contour(np.array([0, c, 0]), np.array([0, 1, 0]))
        '''
        s.actor.property.representation = "wireframe"
        s.actor.property.color = (0, 0, 0)
        s.actor.property.line_width = 0.005
        s.actor.property.opacity = 0.5
        '''


def create_sphere(
    engine: Engine, radius: float = 1, color: Tuple[float, float, float] = (1, 0, 0)
) -> Surface:
    """Create and display a spherical surface.

    :param engine: The Mayavi engine instance.
    :param radius: The radius of the sphere. Defaults to 1.
    :param color: The color of the sphere as an RGB tuple, with components
                  ranging from 0.0 to 1.0. Defaults to red (1, 0, 0).
    :return: The Mayavi Surface object representing the created sphere.

    DocTests:
    >>> # Similar to create_gw, this function interacts with the Mayavi scene.
    >>> # Direct doctests are hard without a full Mayavi setup.
    >>> # from mayavi.api import Engine
    >>> # engine = Engine()
    >>> # engine.start()
    >>> # sphere = create_sphere(engine, radius=5, color=(0, 1, 0))
    >>> # isinstance(sphere, Surface)
    >>> # True
    >>> # engine.stop()
    """
    # Creates and displays a spherical surface with the given parameters.
    scene = engine.scenes[0]
    ps = ParametricSurface()
    ps.function = "ellipsoid"
    ps.parametric_function.x_radius = radius
    ps.parametric_function.y_radius = radius
    ps.parametric_function.z_radius = radius

    engine.add_source(ps, scene)
    s = Surface()
    engine.add_filter(s, ps)
    s.actor.mapper.scalar_visibility = False
    s.actor.property.color = color
    return s

def change_object_position(obj: Surface, position: Tuple[float, float, float]) -> None:
    """Change the Cartesian position of a Mayavi surface object.

    :param obj: The Mayavi Surface object whose position is to be changed.
    :param position: A tuple (x, y, z) specifying the new Cartesian coordinates.

    DocTests:
    >>> # This function directly manipulates a Mayavi object.
    >>> # Mocking Mayavi objects for doctests can be complex.
    >>> # A conceptual test might look like:
    >>> class MockActor:
    ...     def __init__(self):
    ...         self.position = (0.0, 0.0, 0.0)
    >>> class MockSurface:
    ...     def __init__(self):
    ...         self.actor = MockActor()
    >>> mock_surface = MockSurface()
    >>> change_object_position(mock_surface, (1.0, 2.0, 3.0))
    >>> mock_surface.actor.actor.position
    array([1., 2., 3.])
    """
    # Changes the Cartesian position of a Mayavi surface to the given parameters.
    position = np.array(position)
    obj.actor.actor.position = position

def change_view(
    engine: Engine,
    position: Optional[Tuple[float, float, float]] = None,
    focal_point: Optional[Tuple[float, float, float]] = None,
    view_up: Optional[Tuple[float, float, float]] = None,
    view_angle: Optional[float] = None,
    clipping_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Change the view parameters of the Mayavi engine's camera.

    :param engine: The Mayavi engine instance.
    :param position: The new position of the camera (x, y, z). If None, the current
                     position is retained.
    :param focal_point: The new focal point of the camera (x, y, z). If None, the current
                        focal point is retained.
    :param view_up: The new view-up vector of the camera (x, y, z). If None, the current
                    view-up vector is retained.
    :param view_angle: The new view angle of the camera in degrees. If None, the current
                       view angle is retained. Note: This function hardcodes it to 30.0 if not None.
    :param clipping_range: The new clipping range (near, far) for the camera. If None,
                           the current clipping range is retained.

    DocTests:
    >>> # As with other Mayavi scene manipulation functions, direct doctests are
    >>> # difficult without a running Mayavi engine.
    >>> # Conceptual example (requires Mayavi setup to actually run):
    >>> # from mayavi.api import Engine
    >>> # engine = Engine()
    >>> # engine.start()
    >>> # mlab.figure(engine=engine)
    >>> # change_view(engine, position=(10,10,10), focal_point=(0,0,0))
    >>> # engine.scenes[0].scene.camera.position # Check if position changed
    >>> # (10.0, 10.0, 10.0)
    >>> # engine.stop()
    """
    # Changes the view of the Mayavi engine to the given parameters.
    scene = engine.scenes[0]
    if position is not None:
        scene.scene.camera.position = position
    if focal_point is not None:
        scene.scene.camera.focal_point = focal_point
    if view_up is not None:
        scene.scene.camera.view_up = view_up
    if view_angle is not None:
        scene.scene.camera.view_angle = 30.0
    if clipping_range is not None:
        scene.scene.camera.clipping_range = clipping_range
    scene.scene.camera.compute_view_plane_normal()

def dhms_time(seconds: float) -> str:
    """Convert a given number of seconds into a formatted string indicating time.

    The format includes days, hours, and minutes, omitting units with zero values.

    :param seconds: The total number of seconds to convert.
    :return: A string representing the time in "D days H hours M minutes" format.

    DocTests:
    >>> dhms_time(0)
    ''
    >>> dhms_time(60)
    '1 minutes'
    >>> dhms_time(3600)
    '1 hours'
    >>> dhms_time(3661)
    '1 hours 1 minutes'
    >>> dhms_time(86400)
    '1 days'
    >>> dhms_time(90061)
    '1 days 1 hours 1 minutes'
    >>> dhms_time(7200)
    '2 hours'
    >>> dhms_time(123456)
    '1 days 10 hours 17 minutes'
    """
    # Converts a given number of seconds into a string indicating the remaining time.
    days_in_seconds = 24 * 60 * 60
    hours_in_seconds = 60 * 60
    minutes_in_seconds = 60

    # Calculate remaining days, hours, and minutes
    days = int(seconds // days_in_seconds)
    remaining_seconds = seconds % days_in_seconds
    hours = int(remaining_seconds // hours_in_seconds)
    remaining_seconds = remaining_seconds % hours_in_seconds
    minutes = int(remaining_seconds // minutes_in_seconds)

    # Build the output string
    output = ""
    if days > 0:
        output += f"{days} days"
    if hours > 0:
        if days > 0:
            output += " "
        output += f"{hours} hours"
    if minutes > 0:
        if days > 0 or hours > 0:
            output += " "
        output += f"{minutes} minutes"
    return output

def convert_to_movie(input_path: str, movie_name: str, fps: int = 24) -> None:
    """Convert a series of PNG files into a movie using OpenCV.

    Reads all .png files from the specified input path, sorts them, and then
    compiles them into an MP4 video file.

    :param input_path: Path to the directory containing the .png frame files.
    :param movie_name: The desired name for the output movie file (without extension).
    :param fps: Frames per second for the output video. Defaults to 24.

    DocTests:
    >>> # This function directly interacts with the file system and OpenCV.
    >>> # Mocking these interactions for a doctest is complex and would be fragile.
    >>> # A conceptual test would involve creating dummy files and checking if a
    >>> # video file is created, which is outside the scope of simple doctests.
    """
    # Converts a series of .png files into a movie using OpenCV.
    frames = [f for f in os.listdir(input_path) if f.endswith(".png")]
    frames.sort()
    # Create a movie from the frames
    if not frames:
        print(f"No PNG frames found in {input_path}. Skipping movie creation.")
        return # Added return to handle empty frames list gracefully
    ref = cv2.imread(os.path.join(input_path, frames[0]))
    height, width, _ = ref.shape
    video = cv2.VideoWriter(
        os.path.join(input_path, f"{movie_name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        f = cv2.imread(os.path.join(input_path, frame))
        video.write(f)
    video.release()

def ask_user(message: str) -> bool:
    """Prompt the user for a Yes/No response in the command terminal.

    Returns a boolean based on the user's input. Any response other than 'y' (case-insensitive)
    is considered 'No'.

    :param message: The message to display to the user, prompting for Y/N input.
    :return: True if the user responds 'y' or 'Y', False otherwise.

    DocTests:
    >>> import builtins
    >>> original_input = builtins.input
    >>> builtins.input = lambda _: "y"
    >>> ask_user("Test Y input? (Y/N): ")
    True
    >>> builtins.input = lambda _: "N"
    >>> ask_user("Test N input? (Y/N): ")
    False
    >>> builtins.input = lambda _: "Yes"
    >>> ask_user("Test non-Y input? (Y/N): ")
    False
    >>> builtins.input = original_input # Restore original input
    """
    # Allows user input in the command terminal to a Yes/No response.
    # Returns boolean based on input.
    response = input(message)
    if response.lower() != "y":
        return False
    else:
        return True

def main() -> None:
    """Main function to simulate and animate black hole mergers and gravitational waves.

    Reads gravitational wave strain data and black hole positional data,
    calculates and factors in spin-weighted spherical harmonics, linearly interpolates
    the strain to fit the mesh points, and creates a Mayavi animation. The meshes
    represent the full superimposed waveform at the polar angle pi/2 (the plane of
    the binary black hole merger). At each state, the black holes are moved to their
    respective positions, and the render is saved as a .png file.
    """

    # Convert psi4 data to strain using imported script
    # psi4_to_strain.main()

    # Check initial parameters
    time0 = time.time()
    if USE_SYS_ARGS:
        if len(sys.argv) != 5:
            raise RuntimeError(
                """Please include path to merger data as well as the psi4 extraction radius of that data.
                Usage (spaces between arguments): python3
                                                  scripts/animation_main.py
                                                  <path to data folder>
                                                  <extraction radius (r/M) (4 digits, e.g. 0100)>
                                                  <mass of one black hole>
                                                  <mass of other black hole>"""
            )
        else:
            # change directories and extraction radius based on inputs
            bh_dir = str(sys.argv[1])
            psi4_output_dir = os.path.join(bh_dir, "converted_strain")
            ext_rad = float(sys.argv[2])
            bh1_mass = float(sys.argv[3])
            bh2_mass = float(sys.argv[4])
            movie_dir = os.path.join(str(sys.argv[1]), "movies")
        #if ask_user(
        #    f"Save converted strain to {bh_dir} ? (Y/N): "
        #):
        #    psi4strain.WRITE_FILES = True
    else:
        bh_dir = BH_DIR
        movie_dir = MOVIE_DIR
        ext_rad = EXT_RAD
        psi4_output_dir = os.path.join(bh_dir, "converted_strain")
        # mass ratio for default system GW150914
        bh1_mass = 1
        bh2_mass = 1.24

    # File names
    bh_file_name = "puncture_posns_vels_regridxyzU.txt"
    bh_file_path = os.path.join(bh_dir, bh_file_name)

    movie_dir_name = "real_movie2"
    movie_file_path = os.path.join(movie_dir, movie_dir_name)

    if os.path.exists(movie_file_path):
        if ask_user(
            f"'{movie_file_path}' already exists. Would you like to overwrite it? (Y/N): "
        ) == False:
            raise ValueError("Movie directory already exists and overwrite was declined. Please choose a different directory.")
        for file in os.listdir(movie_file_path):
            os.remove(os.path.join(movie_file_path, file))
    else:
        os.makedirs(movie_file_path)
    time1 = time.time()
    # Mathematical parameters
    n_rad_pts = 450
    n_azi_pts = 180
    display_radius = 300
    amplitude_scale_factor = 200
    omitted_radius_length = 10

    colat = np.pi / 2  # colatitude angle representative of the plane of merger

    # Cosmetic parameters
    wireframe = True
    frames_per_second = 24
    save_rate = 10  # Saves every Nth frame
    resolution = (1920, 1080)
    gw_color = (0.28, 0.46, 1.0)
    bh_color = (0.1, 0.1, 0.1)
    time2 = time.time()
    # ---Preliminary Calculations---
    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Initializing grid points..."""
        )
    strain_array, grid, points = initialize_tvtk_grid(n_azi_pts, n_rad_pts)
    width = 0.5 * omitted_radius_length
    dropoff_radius = width + omitted_radius_length
    time3=time.time()
    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Converting psi4 data to strain..."""
        )

    # Import strain data
    time_array, mode_data = psi4strain.psi4_ffi_to_strain(bh_dir, psi4_output_dir, ELL_MAX, ext_rad)
    n_times = len(time_array)
    n_frames = int(n_times / save_rate)

    # theta and radius values for the mesh
    radius_values = np.linspace(0, display_radius, n_rad_pts)
    azimuth_values = np.linspace(0, 2 * np.pi, n_azi_pts, endpoint=False)

    rv, az = np.meshgrid(radius_values, azimuth_values, indexing="ij")
    x_values = rv * np.cos(az)
    y_values = rv * np.sin(az)
    time4=time.time()
    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Constructing mesh points in 3D..."""
        )

    # Apply spin-weighted spherical harmonics, superimpose modes, and interpolate to mesh points
    strain_azi = swsh_summation_angles(colat, azimuth_values, mode_data).real
    lerp_times = generate_interpolation_points(time_array, radius_values, ext_rad)

    strain_to_mesh = np.zeros((n_rad_pts, n_azi_pts, n_times))
    for i in range(n_azi_pts):
        # strain_azi, a function of time_array, is evaluated at t = lerp_times.
        strain_to_mesh[:, i, :] = np.interp(lerp_times, time_array, strain_azi[i, :])

    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Calculating black hole trajectories..."""
        )
    time5=time.time()
    # Import black hole data
    if bh1_mass > bh2_mass:  # then swap
        bh1_mass, bh2_mass = bh2_mass, bh1_mass

    with open(bh_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=" ")
        # _ = next(reader)  # uncomment to skip the header row
        bh_data = np.array(list(reader)).astype(np.float64)
    bh_time = bh_data[:, 0]
    # x is flipped because the data is in a different coordinate system
    bh1_x = -bh_data[:, 5]
    # z axis in the data is interpreted as y axis in the visualization
    bh1_y = bh_data[:, 7]
    bh1_z = np.zeros(len(bh1_x))
    bh1_x, bh1_y, bh1_z = interpolate_coords_by_time(
        bh_time, bh1_x, bh1_y, bh1_z, time_array
    )

    bh_mass_ratio = bh1_mass / bh2_mass
    bh2_x = -bh1_x * bh_mass_ratio
    bh2_y = -bh1_y * bh_mass_ratio
    bh2_z = -bh1_z * bh_mass_ratio
    time6=time.time()
    if STATUS_MESSAGES:
        print(
            """**********************************************************************
    Initializing animation..."""
        )

    # Create Mayavi objects
    #mlab.options.offscreen = True
    engine = Engine()
    engine.start()
    # engine.scenes[0].scene.jpeg_quality = 100
    # mlab.options.offscreen = True
    mlab.figure(engine=engine, size=resolution)

    create_gw(engine, grid, gw_color, display_radius, wireframe)

    bh_scaling_factor = 1
    bh1 = create_sphere(engine, bh1_mass * bh_scaling_factor, bh_color)
    bh2 = create_sphere(engine, bh2_mass * bh_scaling_factor, bh_color)
    mlab.view(
        azimuth=60, elevation=50, distance=80, focalpoint=(0, 0, 0)
    )  # Initialize viewpoint

    start_time = time.time()
    percentage = list(np.round(np.linspace(0, n_times, 100)).astype(int))
    time6=time.time()
    print(f"0:{time1-time0}\n1:{time2-time1}\n2:{time3-time2}\n3:{time4-time3}\n4:{time5-time4}\n5:{time6-time5}\na:{time6-time0}\n")
    @mlab.animate(delay=10,ui=False)  # ui=False) This doesn't work for some reason?
    def anim() -> None:
        for time_idx in range(n_times):
            if time_idx % save_rate != 0:
                continue  # Skip all but every nth iteration

            # Print status messages
            if time_idx == 10 * save_rate:
                end_time = time.time()
                eta = (end_time - start_time) * n_frames / 10
                print(
                    f"""Creating {n_frames} frames and saving them to:
{movie_file_path}
Estimated time: {dhms_time(eta)}"""
                )
            if STATUS_MESSAGES and time_idx != 0 and time_idx > percentage[0]:
                eta = ((time.time() - start_time) / time_idx) * (n_times - time_idx)
                print(
                    f"{int(time_idx * 100 / n_times)}% done, "
                    f"{dhms_time(eta)} remaining",
                    end="\r",
                )
                percentage.pop(0)

            # Change the position of the black holes
            bh1_xyz = (bh1_x[time_idx], bh1_y[time_idx], bh1_z[time_idx])
            bh2_xyz = (bh2_x[time_idx], bh2_y[time_idx], bh2_z[time_idx])
            change_object_position(bh1, bh1_xyz)
            change_object_position(bh2, bh2_xyz)

            points.reset()
            index = 0
            for rad_idx, radius in enumerate(radius_values):
                dropoff_factor = 0.5 + 0.5 * erf((radius - dropoff_radius) / width)
                for azi_idx, _ in enumerate(azimuth_values):
                    x = x_values[rad_idx, azi_idx]
                    y = y_values[rad_idx, azi_idx]
                    if radius <= omitted_radius_length:
                        strain_value = np.nan
                    else:
                        strain_value = strain_to_mesh[rad_idx][azi_idx][time_idx]
                    z = strain_value * amplitude_scale_factor * dropoff_factor
                    points.insert_next_point(x, y, z)
                    strain_array.set_tuple1(index, strain_value)
                    index += 1

            grid._set_points(points)
            grid._get_point_data().add_array(strain_array)
            grid.modified()
            mlab.view(
                # azimuth=min(60 + time_idx * 0.018, 78),
                elevation=max(50 - time_idx * 0.016, 34),
                distance=(
                    80 if time_idx < 2000 else min(80 + (time_idx - 2000) * 0.175, 370)
                ),
                focalpoint=(0, 0, 0),
            )

            # Save the frame
            frame_num = int(time_idx / save_rate)
            frame_path = os.path.join(movie_file_path, f"z_frame_{frame_num:05d}.png")
            mlab.savefig(frame_path, magnification=1)

            if time_idx >= (n_frames * save_rate) - 1:  # Smoothly exit the program
                total_time = time.time() - start_time
                print("Done", end="\r")
                print(
                    f"\nSaved {n_frames} frames to {movie_file_path} "
                    f"in {dhms_time(total_time)}."
                )
                print("Creating movie...")
                convert_to_movie(movie_file_path, movie_dir_name, frames_per_second)
                print(f"Movie saved to {movie_file_path}/{movie_dir_name}.mp4")
                sys.exit(0)
            yield
    _ = anim()
    mlab.show()


# This should automatically create the movie file...
# if it doesn't work, run the following in the movie directory:
# $ffmpeg -framerate 24 -i frame_%05d.png <movie_name>.mp4

if __name__ == "__main__":
    # run doctests first
    import doctest

    results = doctest.testmod()
    p4s_results = doctest.testmod(psi4strain)

    if p4s_results.failed > 0:
        print(
            f"""Doctest in {psi4strain} failed:
{p4s_results.failed} of {p4s_results.attempted} test(s) passed"""
        )
        sys.exit(1)
    else:
        print(
            f"""Doctest in {psi4strain} passed:
All {p4s_results.attempted} test(s) passed"""
        )

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
    # run main() after tests
    main()
