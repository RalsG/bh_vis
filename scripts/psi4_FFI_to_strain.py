"""Process gravitational wave data from numerical relativity simulations.

Read a directory of ASCII files containing waveform data for various ell values,
compute phase and amplitude data, calculate a minimum data frequency using a
quadratic fit of the monotonic phase of ell2 m2 data, and use a Fast Fourier Transform
to compute the second time integral of the waveform (the strain).
Compute a second derivative of the result to check against the original data.

Save the phase and amplitude, the second integral, and the twice integrated-twice differentiated
data to text files. The primary function returns the second integral data as a NumPy array
with the various ell-values.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

import sys
import os
from typing import Union, List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import scipy.special # For erf in quad_fit_intercept

def psi4_ffi_to_strain(
    data_dir: str,
    output_dir: str,
    ell_max: int = 8,
    ext_rad: float = 100.0,
    interval: float = 200.0,
    cutoff_factor: float = 0.75,
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Calculate the strain modes from PSI4 data using the Fixed-Frequency Integration (FFI) method.

    Read each of the ell-modes stored at `data_dir` from 2 to `ell_max` inclusive.
    Use the ell=2, m=2 mode to extrapolate a minimum frequency, and scale it by a cutoff factor.
    Perform a Fast Fourier Transform (FFT) and then divide by frequencies to integrate the psi4 data twice.
    Store the resulting strain data and its double time derivative, and return the strain data
    as an array of the various modes, where each mode is an array of complex data at various timestates.

    :param data_dir: The directory path where raw psi4 ell-mode data is read from.
    :param output_dir: The directory path to write output files to. If an empty string, no files are written.
    :param ell_max: The maximum ell value to read data for. Defaults to 8.
    :param ext_rad: The extraction radius of psi4 data, used as a location to sample for minimum frequency. Defaults to 100.0.
    :param interval: The size of the sampling interval to extrapolate a minimum frequency. Defaults to 200.0.
    :param cutoff_factor: A scaling factor applied to the minimum frequency, providing a cutoff for integration.
                          Must be between 0 and 1. Lower values risk nonphysical noise, higher may filter physical data. Defaults to 0.75.
    :return: A tuple containing:
             - time_arr (NDArray[np.float64]): A NumPy array of time values.
             - strain_modes (NDArray[np.complex128]): A NumPy array of complex-valued strain modes.
    :raises OSError: If there is an error reading the PSI4 data or writing strain data.
    :raises ValueError: If the lengths of the time and data arrays are not equal during data processing.
    """
    ell_min = 1  # if not 2, also adjust calls to read_psi4_dir(), modes_index(), and index_modes()

    try:
        time_arr, psi4_modes_data = read_psi4_dir(data_dir, ell_max, ell_min)
    except OSError as e: # Catch OSError as read_psi4_dir can raise FileNotFoundError
        raise OSError(f"Error reading PSI4 data: {e}") from e

    ell2em2_wave = psi4_phase_and_amplitude(
        time_arr, psi4_modes_data[modes_index(2, 2, ell_min)]
    )

    # The estimated minimum wave frequency for ell=2, m=2, scaled by a cutoff factor.
    min_freq = quad_fit_intercept(time_arr, ell2em2_wave[3], ext_rad, interval)
    freq_cutoff = min_freq * cutoff_factor

    # Initialize arrays for strain modes and their second time derivatives with the same shape as psi4_modes_data.
    strain_modes = np.zeros_like(psi4_modes_data, dtype=np.complex128)
    strain_modes_ddot = np.zeros_like(psi4_modes_data, dtype=np.complex128)

    # Calculate frequency list for FFT based on time array.
    freq_list = np.fft.fftfreq(len(time_arr), time_arr[1] - time_arr[0]) * 2 * np.pi

    # Loop over modes and perform FFT for integration.
    mode_idx = 0
    for ell in range(ell_min, ell_max + 1):
        for em in range(-ell, ell + 1):
            # Apply FFT and filter, see Eq. 27 in https://arxiv.org/abs/1006.1632
            # The filter ensures that very low frequencies (below cutoff) are handled appropriately
            # to avoid division by zero or large unphysical values.
            fft_result = np.fft.fft(psi4_modes_data[mode_idx])
            for i, freq in enumerate(freq_list):
                if np.fabs(freq) <= np.fabs(freq_cutoff):
                    # For frequencies below or equal to cutoff, divide by (i * freq_cutoff)^2.
                    # This prevents division by zero for freq=0 and handles low frequencies.
                    if freq_cutoff == 0.0: # Handle case where cutoff is exactly zero
                        fft_result[i] = 0.0 # Or some other appropriate value if cutoff is 0
                    else:
                        fft_result[i] *= 1 / (1j * freq_cutoff) ** 2
                else:
                    # For frequencies above cutoff, divide by (i * freq)^2 as per standard FFI.
                    if freq == 0.0: # This case should ideally be handled by the cutoff logic
                        fft_result[i] = 0.0
                    else:
                        fft_result[i] *= 1 / (1j * freq) ** 2
            # Inverse FFT to get strain (h).
            strain_modes[mode_idx] = np.fft.ifft(fft_result)

            # Calculate second time derivative (h_ddot) for validation against psi4.
            # psi4_ddot should approximately equal psi4 (original data) if integration is perfect.
            strain_modes_ddot[mode_idx] = second_time_derivative(
                time_arr, strain_modes[mode_idx]
            )
            mode_idx += 1

    # Save ell=2, m=2 psi4 wave data, and all modes strain data if output_dir is specified.
    if output_dir != "":
        labels = [
            "# Col 0: Time",
            "# Col 1: Amplitude",
            "# Col 2: Cumulative_Phase",
            "# Col 3: Angular Frequency",
        ]
        filename = f"Rpsi4_r{ext_rad:06.1f}_ell2_m2_phase_amp_omega.txt"
        try:
            arrays_to_txt(labels, ell2em2_wave, filename, output_dir)
        except OSError as e:
            raise OSError(f"Error saving {filename}: {e}") from e

        for ell in range(ell_min, ell_max + 1):
            strain_filename = f"Rpsi4_r{ext_rad:06.1f}_l{ell}_conv_to_strain.txt"
            ddot_filename = f"Rpsi4_r{ext_rad:06.1f}_l{ell}_from_strain.txt"
            labels_strain_ddot = []
            strain_cols: List[NDArray[np.float64]] = [] # Initialize as list of float arrays
            ddot_cols: List[NDArray[np.float64]] = [] # Initialize as list of float arrays
            col = 0

            labels_strain_ddot.append(f"# column {col}: t-R_ext = [retarded time]")
            strain_cols.append(time_arr)
            ddot_cols.append(time_arr) # Time column is the same for both
            col += 1

            for em in range(-ell, ell + 1):
                mode_data = strain_modes[modes_index(ell, em, ell_min)]
                ddot_data = strain_modes_ddot[modes_index(ell, em, ell_min)]

                labels_strain_ddot.append(f"# column {col}: Re(h_{{l={ell},m={em}}}) * R_ext")
                strain_cols.append(mode_data.real)
                ddot_cols.append(ddot_data.real)
                col += 1

                labels_strain_ddot.append(f"# column {col}: Im(h_{{l={ell},m={em}}}) * R_ext")
                strain_cols.append(mode_data.imag)
                ddot_cols.append(ddot_data.imag)
                col += 1
            try:
                arrays_to_txt(labels_strain_ddot, strain_cols, strain_filename, output_dir)
                arrays_to_txt(labels_strain_ddot, ddot_cols, ddot_filename, output_dir)
            except OSError as e:
                raise OSError(f"Error saving strain/ddot files for l={ell}: {e}") from e

    return time_arr, strain_modes


def read_psi4_dir(
    data_dir: str, ell_max: int, ell_min: int = 2
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Read data from the PSI4 output directory and return time and mode data.

    This function iterates through expected file names based on `ell_min` and `ell_max`,
    loads time and complex-valued mode data for each `ell` value, and performs consistency checks
    on time array lengths.

    :param data_dir: The directory path where raw psi4 ell-mode data files are located.
    :param ell_max: The maximum ell value to read data for.
    :param ell_min: The minimum ell value to read data for. Defaults to 2.
    :return: A tuple containing:
             - time_data (NDArray[np.float64]): A 1D NumPy array of time values (shape: (n_times,)).
             - mode_data (NDArray[np.complex128]): A 2D NumPy array of complex-valued mode data.
                                                 Shape is (n_modes, n_times), where n_modes
                                                 is the total number of (ell, em) modes from ell_min to ell_max.
    :raises FileNotFoundError: If any expected mode file is not found.
    :raises ValueError: If time arrays across different ell modes have inconsistent lengths.
    :raises OSError: If there are issues reading files.
    """
    time_data: NDArray[np.float64]
    psi4_modes_data: List[NDArray[np.complex128]] = []

    n_times = -1
    for ell in range(ell_min, ell_max + 1):
        try:
            filepath = find_file_for_l(data_dir, ell)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing data file for l={ell}: {e}") from e
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                # Read all lines from the file and filter out lines that start with '#'.
                lines = [line for line in file.readlines() if not line.startswith("#")]
            # Convert filtered lines into a NumPy array of float64.
            data = np.array([np.array(line.split(), dtype=np.float64) for line in lines])
        except OSError as e:
            raise OSError(f"Error reading file {filepath}: {e}") from e

        # Use np.unique to sort data by time (first column) and remove duplicates.
        time_data_current, indicies = np.unique(data[:, 0], return_index=True)
        data = data[indicies]  # sort data accordingly based on unique times

        if n_times == -1:
            n_times = len(time_data_current)
        if n_times != len(time_data_current):
            raise ValueError(
                f"Inconsistent time array lengths for l={ell}. Expected {n_times}, but got {len(time_data_current)}."
            )

        real_idx = 1
        for _ in range(2 * ell + 1): # Iterate over m-modes for the current ell
            # Reconstruct complex mode data from real and imaginary parts.
            psi4_modes_data.append(data[:, real_idx] + 1j * data[:, real_idx + 1])
            real_idx += 2 # Move to the next mode's real part
    return np.array(time_data_current), np.array(psi4_modes_data)


def find_file_for_l(data_dir: str, ell: int) -> str:
    """Find the file path with the corresponding ell value in the given directory.

    Searches for files whose names contain the pattern "_l[L]-" where [L] is the
    specified ell value.

    :param data_dir: The directory to search within.
    :param ell: The l spherical harmonics mode number to search for.
    :return: The absolute path to the found file.
    :raises FileNotFoundError: If no file matching the pattern is found in the directory.

    DocTests:
    >>> import tempfile
    >>> import shutil
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create dummy files
    ...     open(os.path.join(tmpdir, "waveform_l2-m0.txt"), "w").close()
    ...     open(os.path.join(tmpdir, "data_l3-m-1.txt"), "w").close()
    ...     open(os.path.join(tmpdir, "another_file.txt"), "w").close()
    ...     # Test finding an existing file
    ...     expected_path_l2 = os.path.join(tmpdir, "waveform_l2-m0.txt")
    ...     find_file_for_l(tmpdir, 2) == expected_path_l2
    True
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Test not finding a file
    ...     try:
    ...         find_file_for_l(tmpdir, 99)
    ...     except FileNotFoundError as e:
    ...         str(e)
    "File with mode l=99 not found."
    """
    # Find the file path with the corresponding ell value in the given directory.
    for filename in os.listdir(data_dir):
        if f"_l{ell}-" in filename:
            return os.path.join(data_dir, filename)
    raise FileNotFoundError(f"File with mode l={ell} not found.")


def modes_index(ell: int, em: int, ell_min: int = 2) -> int:
    """Return the array index for mode data given (ell, em).

    The indexing scheme assumes modes are flattened into a 1D array, ordered
    first by `em` (inner loop) and then by `ell` (outer loop), starting from `ell_min`.

    :param ell: The l spherical harmonics mode number.
    :param em: The m spherical harmonics mode number.
    :param ell_min: The minimum ell value used in the array's indexing. Defaults to 2.
    :return: The 0-based index in the flattened mode data array for the given (ell, em) pair.

    DocTests:
    >>> modes_index(3, 1, 2)
    9
    >>> modes_index(2, -2, 2) # First mode for ell=2
    0
    >>> modes_index(2, 2, 2) # Last mode for ell=2 (2*2+1 - 1 = 4)
    4
    >>> modes_index(3, -3, 2) # First mode for ell=3 (2*2+1 = 5 modes before it)
    5
    >>> modes_index(3, 3, 2) # Last mode for ell=3 (5 + 2*3+1 - 1 = 11)
    11
    """
    # The index begins with 0 and progresses through m (inner loop) then l (outer loop).
    # The formula computes the sum of (2*l'+1) for l' from ell_min to ell-1, plus (em + ell) for current ell.
    # A simpler formula can be derived as (ell^2 + ell + em) - (ell_min^2 + ell_min - ell_min) = ell^2 + ell + em - ell_min^2
    # The given formula `ell**2 + ell + em - ell_min**2` implicitly covers this.
    return ell**2 + ell + em - ell_min**2


def index_modes(idx: int, ell_min: int = 2) -> Tuple[int, int]:
    """Given the array index, return the (ell, em) mode numbers.

    This is the inverse operation of `modes_index`.

    :param idx: The 0-based mode data array index.
    :param ell_min: The minimum ell value used in the array's indexing. Defaults to 2.
    :return: A tuple containing the (ell, em) mode numbers corresponding to the index.

    DocTests:
    >>> index_modes(9, 2)
    (3, 1)
    >>> index_modes(0, 2)
    (2, -2)
    >>> index_modes(4, 2)
    (2, 2)
    >>> index_modes(5, 2)
    (3, -3)
    >>> index_modes(11, 2)
    (3, 3)
    """
    idx_adjusted = idx + ell_min**2
    # Determine ell by finding the largest integer ell_val such that ell_val^2 + ell_val <= idx_adjusted.
    # This comes from (ell_val^2 + ell_val + em) where em varies from -ell_val to ell_val.
    # Max index for a given ell is (ell^2 + ell + ell) = ell^2 + 2*ell.
    # Min index for a given ell is (ell^2 + ell - ell) = ell^2.
    # So, ell is `floor(sqrt(idx_adjusted))`.
    ell = int(np.sqrt(idx_adjusted))
    # Correct for cases where idx_adjusted falls within the next ell block's range but corresponds to current ell's max m.
    # For instance, if ell_min=2, and idx_adjusted=5 (for ell=3, m=-3, where min index is 3^2=9 with default ell_min),
    # the formula ell^2+ell+em is for ell_min=0. We use ell_min in modes_index to offset.
    # The adjusted index `idx_adjusted = idx + ell_min**2` effectively maps back to an `ell_min=0` scheme for calculation.
    # For a flat list of modes, the number of modes up to `ell-1` (inclusive) is `ell^2 - ell_min^2`.
    # Thus, current mode's index within its `ell` block is `idx - (ell^2 - ell_min^2)`.
    # And `em` is `(current_index_in_block) - ell`.
    # The formula `em = idx_adjusted - ell**2 - ell` is correct for the `ell_min=0` equivalent flat list.
    em = idx_adjusted - ell**2 - ell
    return ell, em


def arrays_to_txt(
    labels: List[str],
    collection: Union[
        NDArray[np.float64], List[NDArray[np.float64]], Tuple[NDArray[np.float64], ...]
    ],
    filename: str,
    dir_path: str,
) -> None:
    """Write a collection of NumPy arrays to a text file, formatting each row with labels.

    This function creates the specified directory if it doesn't exist, then writes
    the provided labels as header comments, followed by the data rows. Each row is
    formed by zipping corresponding elements from the input `collection` arrays.

    :param labels: A list of strings, each representing a comment line to be written at the top of the file.
    :param collection: A collection (e.g., list or tuple) of 1D NumPy arrays. Each array
                       represents a column in the output file, and they must all have the same length.
    :param filename: The name of the file to write to.
    :param dir_path: The path to the directory where the file will be saved.
    :raises OSError: If there is an error creating the directory or writing to the file.

    DocTests:
    >>> import tempfile
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     test_labels = ["# Col 0: A", "# Col 1: B"]
    ...     test_data = [np.array([1.0, 2.0]), np.array([10.0, 20.0])]
    ...     test_filename = "test_output.txt"
    ...     arrays_to_txt(test_labels, test_data, test_filename, tmpdir)
    ...     file_content = open(os.path.join(tmpdir, test_filename), "r", encoding="utf-8").read()
    ...     print(file_content.strip())
    # Col 0: A
    # Col 1: B
    1.000000000000000 10.000000000000000
    2.000000000000000 20.000000000000000
    >>> # Test empty collection
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     test_labels = ["# Empty test"]
    ...     test_data_empty = []
    ...     test_filename_empty = "empty_output.txt"
    ...     arrays_to_txt(test_labels, test_data_empty, test_filename_empty, tmpdir)
    ...     file_content = open(os.path.join(tmpdir, test_filename_empty), "r", encoding="utf-8").read()
    ...     print(file_content.strip())
    # Empty test
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)

        with open(file_path, mode="w", encoding="utf-8") as file:
            # Write labels as comment lines
            file.write("".join([f"{label}\n" for label in labels]))
            # Write data rows, formatting each item to 15 decimal places.
            # `zip(*collection)` transposes the data from columns to rows.
            for row in zip(*collection):
                file.write(" ".join([f"{item:.15f}" for item in row]) + "\n")
        print(f"File '{filename}' saved to '{dir_path}'")
    except OSError as e: # Catch OS-related errors during file/directory operations
        raise OSError(f"Error saving data to file: {e}") from e


def first_time_derivative(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the first time derivative of the input data.

    This function uses a second-order central finite difference stencil for interior points
    and a first-order forward/backward difference for the endpoints.

    :param time: A 1D NumPy array containing time values, assumed to be uniformly spaced.
    :param data: A 1D NumPy array containing the data to be differentiated, corresponding to `time`.
    :return: A 1D NumPy array containing the time derivative of the input data, with the same shape as `data`.

    DocTests:
    >>> time_arr = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data_arr = np.array([0, 1, 4, 9, 16], dtype=np.float64) # y = x^2, dy/dx = 2x
    >>> first_time_derivative(time_arr, data_arr)
    array([1., 2., 4., 6., 7.])
    >>> time_arr_linear = np.array([0., 0.5, 1.0, 1.5], dtype=np.float64)
    >>> data_arr_linear = np.array([0., 2.0, 4.0, 6.0], dtype=np.float64) # y = 4x, dy/dx = 4
    >>> first_time_derivative(time_arr_linear, data_arr_linear)
    array([4., 4., 4., 4.])
    >>> time_arr_sin = np.array([0, np.pi/2, np.pi], dtype=np.float64)
    >>> data_arr_sin = np.array([0, 1, 0], dtype=np.float64) # y = sin(x), dy/dx = cos(x)
    >>> np.round(first_time_derivative(time_arr_sin, data_arr_sin), 5)
    array([ 0.63662,  0.     , -0.63662])
    """
    delta_t = time[1] - time[0]
    data_dt = np.zeros_like(data)

    # Second-order in the interior using central difference (f(x+h) - f(x-h)) / (2h)
    data_dt[1:-1] = (data[2:] - data[:-2]) / (2 * delta_t)

    # Drop to first-order at the endpoints using forward/backward difference
    # Forward difference for the first point: (f(x+h) - f(x)) / h
    data_dt[0] = (data[1] - data[0]) / delta_t
    # Backward difference for the last point: (f(x) - f(x-h)) / h
    data_dt[-1] = (data[-1] - data[-2]) / delta_t

    return data_dt


def second_time_derivative(
    time: NDArray[np.float64], data: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Compute the second time derivative of the input complex-valued data.

    This function uses a second-order central finite difference method for interior points.
    For the endpoints, it uses a 4-point, second-order accurate forward (at index 0)
    and backward (at index n-1) finite difference stencils.

    :param time: A 1D NumPy array containing time values, assumed to be uniformly spaced.
    :param data: A 1D NumPy array containing complex-valued function data to differentiate.
    :return: A 1D NumPy array containing the second time derivative of the function data,
             with the same shape and complex dtype as `data`.

    DocTests:
    >>> time_arr = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data_arr = np.array([0, 1, 4, 9, 16], dtype=np.complex128) # y = x^2, d^2y/dx^2 = 2
    >>> second_time_derivative(time_arr, data_arr)
    array([2.+0.j, 2.+0.j, 2.+0.j, 2.+0.j, 2.+0.j])
    >>> time_arr_cubic = np.array([0., 1., 2., 3., 4., 5.], dtype=np.float64)
    >>> data_arr_cubic = np.array([x**3 for x in time_arr_cubic], dtype=np.complex128) # y=x^3, y''=6x
    >>> np.round(second_time_derivative(time_arr_cubic, data_arr_cubic), 5)
    array([ 0.+0.j,  6.+0.j, 12.+0.j, 18.+0.j, 24.+0.j, 30.+0.j])
    """
    delta_t = time[1] - time[0]
    data_dtdt = np.zeros_like(data)

    # Interior points using central finite difference (f(x+h) - 2f(x) + f(x-h)) / h^2
    data_dtdt[1:-1] = (data[:-2] - 2 * data[1:-1] + data[2:]) / (delta_t**2)

    # Endpoint 0: 4-point, second-order accurate forward finite difference (downwind)
    # Stencil: (2*f[0] - 5*f[1] + 4*f[2] - f[3]) / h^2
    if len(data) >= 4:
        data_dtdt[0] = (2 * data[0] - 5 * data[1] + 4 * data[2] - data[3]) / (delta_t**2)
    elif len(data) == 3: # Fallback for very short arrays
        data_dtdt[0] = (data[0] - 2 * data[1] + data[2]) / (delta_t**2) # Central diff if possible
    elif len(data) == 2:
        data_dtdt[0] = 0.0 # Cannot compute second derivative, or use a simpler, less accurate method
    else: # len(data) < 2
        data_dtdt[0] = 0.0 # Cannot compute

    # Endpoint n-1: 4-point, second-order accurate backward finite difference (upwind)
    # Stencil: (2*f[n-1] - 5*f[n-2] + 4*f[n-3] - f[n-4]) / h^2
    if len(data) >= 4:
        data_dtdt[-1] = (2 * data[-1] - 5 * data[-2] + 4 * data[-3] - data[-4]) / (delta_t**2)
    elif len(data) == 3: # Fallback for very short arrays
        data_dtdt[-1] = (data[0] - 2 * data[1] + data[2]) / (delta_t**2) # Central diff
    elif len(data) == 2:
        data_dtdt[-1] = 0.0
    else: # len(data) < 2
        data_dtdt[-1] = 0.0

    return data_dtdt


def psi4_phase_and_amplitude(
    time: NDArray[np.float64], cmplx: NDArray[np.complex128]
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Calculate the amplitude and cumulative phase of a complex gravitational wave signal.

    This function computes the instantaneous amplitude (`np.abs`), the raw phase (`np.angle`),
    and then unwraps the phase to get a cumulative phase. Finally, it computes the
    time derivative of the cumulative phase to obtain the angular frequency.

    :param time: A 1D NumPy array containing time values.
    :param cmplx: A 1D NumPy array containing the complex-valued signal (e.g., psi4 mode data).
    :return: A tuple containing four 1D NumPy arrays:
             - time (NDArray[np.float64]): The input time array.
             - amplitudes (NDArray[np.float64]): The instantaneous amplitude of the complex signal.
             - cumulative_phase (NDArray[np.float64]): The unwrapped, cumulative phase of the signal.
             - cumulative_phase_derivative (NDArray[np.float64]): The time derivative of the cumulative phase (angular frequency).
    :raises ValueError: If the lengths of the `time` and `cmplx` arrays are not equal.

    DocTests:
    >>> time_arr = np.array([0, 1, 2, 3], dtype=np.float64)
    >>> # Test a simple increasing phase
    >>> cmplx_linear_phase = np.array([1, np.cos(1)+1j*np.sin(1), np.cos(2)+1j*np.sin(2), np.cos(3)+1j*np.sin(3)], dtype=np.complex128)
    >>> t, amp, cum_phase, cum_phase_dt = psi4_phase_and_amplitude(time_arr, cmplx_linear_phase)
    >>> np.round(amp, 5)
    array([1., 1., 1., 1.])
    >>> np.round(cum_phase, 5)
    array([0.     , 1.     , 2.     , 3.     ])
    >>> np.round(cum_phase_dt, 5)
    array([1., 1., 1., 1.])
    >>> # Test phase wrapping
    >>> cmplx_wrap_phase = np.array([1, np.cos(pi)+1j*np.sin(pi), np.cos(3*pi)+1j*np.sin(3*pi)], dtype=np.complex128)
    >>> t_wrap = np.array([0, 1, 2], dtype=np.float64)
    >>> t, amp, cum_phase, cum_phase_dt = psi4_phase_and_amplitude(t_wrap, cmplx_wrap_phase)
    >>> np.round(cum_phase, 5)
    array([0.     , 3.14159, 9.42478]) # 0, pi, 3pi (wrapped 1pi becomes 3pi, 5pi becomes 9pi)
    """
    if len(time) != len(cmplx):
        raise ValueError(
            f"Time array length ({len(time)}) and complex data array length ({len(cmplx)}) must be equal."
        )

    amplitudes = np.abs(cmplx)
    phases = np.angle(cmplx)
    cycles = 0
    cum_phase = np.zeros_like(time)
    last_phase = phases[0]

    for i, ph in enumerate(phases):
        # Identify phase wrapping (jump of approx. 2*pi)
        # If the difference between current and last phase is greater than pi,
        # it indicates a wrap. Adjust `cycles` accordingly.
        if np.abs(ph - last_phase) >= np.pi:
            cycles += -1 if ph > 0 else 1 # If current phase is positive after wrap, it means it went from positive to negative (e.g. pi to -pi), so add 1 cycle. If it went from negative to positive, subtract 1 cycle.

        cum_phase[i] = ph + 2 * np.pi * cycles
        last_phase = ph

    cum_phase_dt = first_time_derivative(time, cum_phase)
    return time, amplitudes, cum_phase, cum_phase_dt


def quadratic(x: float, a: float, b: float, c: float) -> float:
    """Evaluate a quadratic polynomial (ax^2 + bx + c).

    :param x: The independent variable.
    :param a: The coefficient of the x^2 term.
    :param b: The coefficient of the x term.
    :param c: The constant term (y-intercept).
    :return: The value of the quadratic function at x.

    DocTests:
    >>> quadratic(1, 1, 2, 3)
    6.0
    >>> quadratic(0, 5, -2, 7)
    7.0
    >>> quadratic(-2, 1, 0, 0)
    4.0
    """
    # Evaluate a quadratic polynomial (ax^2 + bx + c).
    return a * x**2 + b * x + c

def quad_fit_intercept(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    ext_rad: float,
    interval: float,
    verbose: bool = False,
) -> float:
    """Sample data from a time interval, apply a quadratic fit, and return the absolute value of the y-intercept.

    This function is primarily intended for analyzing l=2, m=2 angular frequency data
    to estimate a minimum frequency by fitting a quadratic polynomial to a specific
    time interval and taking the y-intercept as the minimum frequency.

    :param time: A 1D NumPy array containing time values.
    :param data: A 1D NumPy array containing data values (e.g., angular frequency)
                 corresponding to the time values.
    :param ext_rad: The starting time for the sampling interval.
    :param interval: The length of the sampling interval (time - ext_rad to time - ext_rad + interval).
    :param verbose: If True, print details about the quadratic fit and its extremum. Defaults to False.
    :return: The absolute value of the constant term (y-intercept) of the fitted quadratic curve,
             representing the estimated minimum frequency.
    :raises ValueError: If the lengths of `time` and `data` arrays are not equal, or if no data points are
                        found within the specified interval for fitting.
    """
    if len(time) != len(data):
        raise ValueError(
            f"Time array length ({len(time)}) and data array length ({len(data)}) must be equal."
        )

    # Filter data points within the specified interval [ext_rad, ext_rad + interval].
    time_filtered = time[(ext_rad <= time) & (time <= ext_rad + interval)]
    data_filtered = data[(ext_rad <= time) & (time <= ext_rad + interval)]

    if len(time_filtered) < 3: # Need at least 3 points for a quadratic fit
        raise ValueError(
            f"Not enough data points ({len(time_filtered)}) in the interval "
            f"[{ext_rad}, {ext_rad + interval}] for quadratic fitting. At least 3 points are required."
        )

    # Fit a quadratic curve to the data using nonlinear least squares.
    # The `curve_fit` function returns optimal parameters `popt` and covariance matrix `pcov`.
    params, _ = curve_fit(quadratic, time_filtered, data_filtered)

    # Extract coefficients: a (x^2), b (x), c (constant/intercept).
    a, b, c = params
    # Calculate the x-coordinate of the extremum (vertex) of the quadratic.
    # For ax^2 + bx + c, the vertex is at x = -b / (2a).
    extremum_x = -b / (2 * a)
    # Evaluate the quadratic fit at its extremum.
    quad_fit_extremum = quadratic(extremum_x, a, b, c)

    if verbose:
        print(
            f"Quadratic Vertex at (time = {extremum_x:.7e}, value = {quad_fit_extremum:.7e}).\n"
            f"Params: a = {a:.7e}, b = {b:.7e}, c = {c:.7e}, Intercept magnitude: {np.fabs(c):.7e}"
        )
    # The intercept `c` (value at x=0) is used as the estimated minimum frequency.
    return float(np.fabs(c))


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    # Special handling for doctests for psi4_ffi_to_strain itself, as it requires file access.
    # Skipping direct doctest for psi4_ffi_to_strain, as it's an integration-level function.
    # The overall `doctest.testmod()` call covers the unit functions.

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    # Command-line argument parsing for main execution
    if len(sys.argv) > 7:
        # Replaced print+sys.exit with raise ValueError
        raise ValueError(
            "Error: Too many Arguments.\n"
            "Usage: psi4_ffi_to_strain.py <input dir> <output dir> "
            "[maxmimum l] [extraction radius] [sample interval] [cutoff factor]"
        )

    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(workspace, "data", "GW150914_data", "r100")
    # Initialize args with default values, ensuring correct types for `psi4_ffi_to_strain`.
    # sys.argv[0] is the script name itself, so we start from index 1 for actual arguments.
    # The provided args list needs to correspond to the function signature:
    # (data_dir, output_dir, ell_max, ext_rad, interval, cutoff_factor)
    _args_defaults = ["dummy_script_name", default_input, "", "8", "100.0", "200.0", "0.75"]
    
    # Override defaults with provided command-line arguments.
    # Slicing sys.argv[1:] to exclude the script name itself.
    _args_defaults[1:1+len(sys.argv[1:])] = sys.argv[1:]
    
    # Cast arguments to their correct types for the function call.
    try:
        psi4_ffi_to_strain(
            str(_args_defaults[1]),      # data_dir
            str(_args_defaults[2]),      # output_dir
            int(_args_defaults[3]),      # ell_max
            float(_args_defaults[4]),    # ext_rad
            float(_args_defaults[5]),    # interval
            float(_args_defaults[6]),    # cutoff_factor
        )
    except (OSError, ValueError) as e:
        # Catch and re-raise specific errors that psi4_ffi_to_strain might raise.
        # This acts as a top-level catch for known errors during execution.
        raise RuntimeError(f"An error occurred during strain calculation: {e}") from e
