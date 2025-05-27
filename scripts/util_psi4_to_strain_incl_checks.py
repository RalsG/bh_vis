"""Process gravitational wave data from numerical relativity simulations.

This module provides functions to read waveform data files, compute derivatives,
process wave data, perform complex Fast Fourier Transforms (FFT), and fit data
to a quadratic function. The main functionality involves reading gravitational wave
data, extracting relevant information like the phase and amplitude, and performing
analysis like FFT and quadratic fitting to extract physical properties from the waveforms.
It's designed to work with ASCII files containing gravitational wave data from simulations.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""
import sys
import os
from typing import Tuple, Dict, Union, List

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import scipy.special # for erf


def construct_generic_filename(radius: float) -> str:
    """Construct a filename based on the input radius following a specific format.

    The format is 'Rpsi4_l[MODENUM]-rXXXX.X.txt', where [MODENUM] is a placeholder
    for the `ell` value and XXXX.X is the radius formatted to one decimal place.

    :param radius: The radius value to be included in the filename.
    :return: A string representing the constructed filename.

    DocTests:
    >>> construct_generic_filename(24.0)
    'Rpsi4_l[MODENUM]-r0024.0.txt'
    >>> construct_generic_filename(1124.2)
    'Rpsi4_l[MODENUM]-r1124.2.txt'
    >>> construct_generic_filename(5.0)
    'Rpsi4_l[MODENUM]-r0005.0.txt'
    """
    return f"Rpsi4_l[MODENUM]-r{radius:06.1f}.txt"


def read_BHaH_psi4_files(
    generic_file_name: str,
    psi4_folder_path: str
) -> Tuple[
    NDArray[np.float64],
    Dict[Tuple[int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]],
]:
    """Read an ASCII file with a header describing the real and imaginary parts of the data for each mode.

    Return the data in a format to access the real and imaginary parts given (ell, m) values.
    This function reads data for each mode (l from 2 to 8) from files within
    the specified `psi4_folder_path`. It processes the data by skipping
    comment lines, sorting by time, removing duplicate time entries,
    and then storing the real and imaginary parts of the mode data in a
    dictionary keyed by (l, m) tuples.

    :param generic_file_name: The generic filename pattern, e.g., 'Rpsi4_l[MODENUM]-r0024.0.txt'.
                              `[MODENUM]` will be replaced by the actual 'ell' value.
    :param psi4_folder_path: The absolute path to the directory containing the psi4 data files.
    :return: A tuple containing:
             - NDArray[np.float64]: A 1D NumPy array of time values.
             - Dict[Tuple[int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
               A dictionary where keys are (ell, m) tuples and values are tuples of
               (real_part_array, imaginary_part_array) for each mode.
    :raises ValueError: If the length of time data is inconsistent across different ell values.
    :raises FileNotFoundError: If any of the expected data files cannot be found.
    :raises OSError: For other file system related errors during reading.

    DocTests:
    >>> import tempfile
    >>> import os
    >>> # Create a temporary directory and dummy files for testing
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create dummy files for ell=2 and ell=3
    ...     file_l2 = os.path.join(tmpdir, 'Rpsi4_l2-r100.0.txt')
    ...     file_l3 = os.path.join(tmpdir, 'Rpsi4_l3-r100.0.txt')
    ...     with open(file_l2, 'w') as f:
    ...         f.write('# Header for l=2\\n')
    ...         f.write('0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0  10.0\\n')
    ...         f.write('1.0  1.1  2.1  3.1  4.1  5.1  6.1  7.1  8.1  9.1  10.1\\n')
    ...     with open(file_l3, 'w') as f:
    ...         f.write('# Header for l=3\\n')
    ...         f.write('0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2  1.3\\n')
    ...         f.write('1.0  1.1  1.2  1.3  1.4  1.5  1.6  1.7  1.8  1.9  2.0  2.1  2.2  2.3\\n')
    ...     generic_name = 'Rpsi4_l[MODENUM]-r100.0.txt'
    ...     time_data, mode_data = read_BHaH_psi4_files(generic_name, tmpdir)
    ...     # Check time data
    ...     np.array_equal(time_data, np.array([0.0, 1.0]))
    True
    >>>     # Check a specific mode (e.g., ell=2, m=2)
    >>>     # For l=2, m ranges from -2 to 2 (5 modes).
    >>>     # Columns are t, Re(-2), Im(-2), Re(-1), Im(-1), Re(0), Im(0), Re(1), Im(1), Re(2), Im(2)
    >>>     # So (2,2) real part is column 9 (index 8), imag is column 10 (index 9)
    >>>     # Correct index for (2,2) would be 1 + 2 * (2+2) = 9 for real
    >>>     np.array_equal(mode_data[(2, 2)][0], np.array([9.0, 9.1]))
    True
    >>>     np.array_equal(mode_data[(2, 2)][1], np.array([10.0, 10.1]))
    True
    >>>     # Test inconsistent time data size (l=3 file has more time points)
    >>>     with open(file_l3, 'w') as f: # Overwrite l3 file with different time size
    ...         f.write('# Header for l=3\\n')
    ...         f.write('0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  1.1  1.2  1.3\\n')
    ...         f.write('1.0  1.1  1.2  1.3  1.4  1.5  1.6  1.7  1.8  1.9  2.0  2.1  2.2  2.3\\n')
    ...         f.write('2.0  2.1  2.2  2.3  2.4  2.5  2.6  2.7  2.8  2.9  3.0  3.1  3.2  3.3\\n')
    ...     try:
    ...         read_BHaH_psi4_files(generic_name, tmpdir)
    ...     except ValueError as e:
    ...         print(e)
    Inconsistent time data size for ell=3. Expected 2, got 3.
    >>>     # Test FileNotFoundError
    >>>     try:
    ...         read_BHaH_psi4_files('non_existent_l[MODENUM].txt', tmpdir)
    ...     except FileNotFoundError as e:
    ...         print(e)
    [Errno 2] No such file or directory: '.../non_existent_l2.txt'
    """
    mode_data: Dict[
        Tuple[int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = {}

    time_data_size: int = -1
    time_data: NDArray[np.float64] = np.array([]) # Initialize outside loop

    for ell in range(2, 9): # Loops for ell from 2 to 8, inclusive
        file_name = generic_file_name.replace("[MODENUM]", str(ell))
        file_path = os.path.abspath(
            os.path.join(psi4_folder_path, file_name)
        )
        print(f"Reading file {file_name}...")
        try:
            with open(file_path, mode="r", encoding="utf-8") as file:
                # Read the lines and ignore lines starting with '#' (comments).
                lines = [line for line in file.readlines() if not line.startswith("#")]

            # Convert lines to NumPy arrays of float64.
            # Each line is split by whitespace, and elements are converted to float.
            data: NDArray[np.float64] = np.array(
                [list(map(np.float64, line.split())) for line in lines]
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: '{file_path}'") from e
        except OSError as e:
            raise OSError(f"Error reading file '{file_path}': {e}") from e
        except ValueError as e:
            raise ValueError(f"Error parsing data in '{file_path}': {e}. Ensure data is numeric and correctly formatted.") from e

        # Sort the entire array by the first column (time) to ensure chronological order.
        data = data[np.argsort(data[:, 0])]

        # Remove duplicate time entries, keeping the first occurrence.
        time_data_current_ell, index = np.unique(data[:, 0], return_index=True)
        data = data[index]

        # Store time data from the current ell file. This will be the reference time array.
        time_data = time_data_current_ell # Assign the current time data

        # Check for consistency in time data size across different ell values.
        if time_data_size < 0:
            time_data_size = len(time_data)
        else:
            if time_data_size != len(time_data):
                raise ValueError(
                    f"Inconsistent time data size for ell={ell}. Expected {time_data_size}, got {len(time_data)}."
                )

        # Loop through columns and store real and imaginary parts in mode_data dictionary.
        # The column index for the real part of (ell, m) mode is 1 + 2 * (m + ell) based on file format.
        for m in range(-ell, ell + 1):
            idx = 1 + 2 * (m + ell)  # Calculate the index of the real part column.
            # Store a tuple of (real_part_array, imaginary_part_array) for the current mode.
            mode_data[(ell, m)] = (data[:, idx], data[:, idx + 1])

    return time_data, mode_data


def compute_first_derivative_in_time(
    time: NDArray[np.float64], data: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Calculate the first time derivative of the input data using a second-order finite difference stencil.

    This function uses a second-order central finite difference stencil for interior points
    and a first-order forward/backward difference for the endpoints.

    :param time: A NumPy array containing time values, assumed to be uniformly spaced.
    :param data: A NumPy array containing the data to be differentiated.
    :return: A NumPy array containing the time derivative of the input data.

    DocTests:
    >>> time_test = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data_test = np.array([0, 1, 4, 9, 16], dtype=np.float64) # y = x^2, dy/dx = 2x
    >>> compute_first_derivative_in_time(time_test, data_test)
    array([1., 2., 4., 6., 7.])
    >>> time_linear = np.array([0., 0.5, 1.0, 1.5], dtype=np.float64)
    >>> data_linear = np.array([0., 2.0, 4.0, 6.0], dtype=np.float64) # y = 4x, dy/dx = 4
    >>> compute_first_derivative_in_time(time_linear, data_linear)
    array([4., 4., 4., 4.])
    """
    dt = time[1] - time[0] # Calculate the time step size.
    derivative = np.zeros_like(data) # Initialize an array for the derivative with zeros.

    # Apply a second-order central finite difference stencil for interior points.
    # (f(x+h) - f(x-h)) / (2h)
    derivative[1:-1] = (data[2:] - data[:-2]) / (2 * dt)

    # Use first-order forward difference at the beginning.
    # (f(x+h) - f(x)) / h
    derivative[0] = (data[1] - data[0]) / dt
    # Use first-order backward difference at the end.
    # (f(x) - f(x-h)) / h
    derivative[-1] = (data[-1] - data[-2]) / dt

    return derivative


def compute_second_derivative_in_time(
    time: NDArray[np.float64], data: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute the second time derivative of the input data.

    Uses the second-order central finite difference method for interior points,
    with 4-point, second-order accurate forward/backward stencils for the endpoints.

    :param time: A NumPy array containing time values, assumed to be uniformly spaced.
    :param data: A NumPy array containing data for which the second time derivative is to be calculated.
    :return: A NumPy array containing the second time derivative of the input data.

    DocTests:
    >>> time_test = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data_test = np.array([0, 1, 4, 9, 16], dtype=np.float64) # y = x^2, d^2y/dx^2 = 2
    >>> compute_second_derivative_in_time(time_test, data_test)
    array([2., 2., 2., 2., 2.])
    >>> time_cubic = np.array([0., 1., 2., 3., 4., 5.], dtype=np.float64)
    >>> data_cubic = np.array([x**3 for x in time_cubic], dtype=np.float64) # y=x^3, y''=6x
    >>> np.round(compute_second_derivative_in_time(time_cubic, data_cubic), 5)
    array([ 0.,  6., 12., 18., 24., 30.])
    """
    dt = time[1] - time[0] # Calculate the time step size.
    n = len(data)
    second_derivative = np.zeros(n) # Initialize an array for the second derivative.

    # Interior points using central finite difference: (f(x+h) - 2f(x) + f(x-h)) / h^2
    second_derivative[1:-1] = (data[:-2] - 2 * data[1:-1] + data[2:]) / (dt**2)

    # Endpoint 0: 4-point, second-order accurate forward finite difference (downwind).
    # Stencil: (2*f[0] - 5*f[1] + 4*f[2] - f[3]) / h^2
    # This stencil requires at least 4 data points.
    if n >= 4:
        second_derivative[0] = (2 * data[0] - 5 * data[1] + 4 * data[2] - data[3]) / (
            dt**2
        )
    elif n == 3: # Fallback to a simpler central difference if only 3 points are available
        second_derivative[0] = (data[0] - 2 * data[1] + data[2]) / (dt**2)
    else: # Less than 3 points, cannot reliably compute second derivative
        second_derivative[0] = 0.0

    # Endpoint n-1: 4-point, second-order accurate backward finite difference (upwind).
    # Stencil: (2*f[n-1] - 5*f[n-2] + 4*f[n-3] - f[n-4]) / h^2
    # This stencil requires at least 4 data points.
    if n >= 4:
        second_derivative[-1] = (2 * data[-1] - 5 * data[-2] + 4 * data[-3] - data[-4]) / (
            dt**2
        )
    elif n == 3: # Fallback to a simpler central difference
        second_derivative[-1] = (data[0] - 2 * data[1] + data[2]) / (dt**2)
    else: # Less than 3 points
        second_derivative[-1] = 0.0

    return second_derivative


def compute_psi4_wave_phase_and_amplitude(
    time: NDArray[np.float64], real: NDArray[np.float64], imag: NDArray[np.float64]
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Calculate the cumulative phase and amplitude of a gravitational wave signal.

    This function takes time, real, and imaginary components of a complex
    gravitational wave signal. It computes the instantaneous amplitude,
    unwraps the phase to get a cumulative phase, and calculates the
    time derivative of the cumulative phase to obtain the angular frequency.

    :param time: A NumPy array containing time values.
    :param real: A NumPy array containing the real part of the signal.
    :param imag: A NumPy array containing the imaginary part of the signal.
    :return: A tuple containing four NumPy arrays:
             (time, cumulative_phase, amplitude, cumulative_phase_derivative).
    :raises ValueError: If the lengths of time, real, and imag arrays are not equal.

    DocTests:
    >>> time_test = np.array([0, 1, 2, 3], dtype=np.float64)
    >>> # Test a simple increasing phase (complex exponential e^(i*t))
    >>> real_linear_phase = np.array([np.cos(t) for t in time_test], dtype=np.float64)
    >>> imag_linear_phase = np.array([np.sin(t) for t in time_test], dtype=np.float64)
    >>> t, cum_ph, amp, cum_ph_dt = compute_psi4_wave_phase_and_amplitude(time_test, real_linear_phase, imag_linear_phase)
    >>> np.round(amp, 5)
    array([1., 1., 1., 1.])
    >>> np.round(cum_ph, 5)
    array([0.     , 1.     , 2.     , 3.     ])
    >>> np.round(cum_ph_dt, 5)
    array([1., 1., 1., 1.])

    >>> # Test phase wrapping (e.g., jump from pi to -pi should be treated as continuous)
    >>> time_wrap = np.array([0, 1, 2], dtype=np.float64)
    >>> real_wrap = np.array([1, -1, -1], dtype=np.float64) # Represents 0, pi, 3pi
    >>> imag_wrap = np.array([0, 0, 0], dtype=np.float64)
    >>> t_w, cum_ph_w, amp_w, cum_ph_dt_w = compute_psi4_wave_phase_and_amplitude(time_wrap, real_wrap, imag_wrap)
    >>> np.round(cum_ph_w, 5)
    array([0.     , 3.14159, 6.28319]) # Should be 0, pi, 2pi (from 3pi) - previous docstring was confused. Corrected: -pi to pi maps to (-pi, pi], so -1 is pi. 2nd -1 (0+2pi) means pi+2pi. This should be 0, pi, 2pi.
    """

    if not len(time) == len(real) == len(imag):
        raise ValueError("The lengths of time, real, and imag arrays must be equal.")

    # Calculate the amplitude of the gravitational wave signal.
    amplitude = np.sqrt(real**2 + imag**2)

    # Calculate the instantaneous phase of the gravitational wave signal using arctan2 for full quadrant range.
    phase = np.arctan2(imag, real)

    # Initialize variables for cumulative phase calculation.
    cycles = 0 # Counts the number of 2*pi cycles completed.
    cum_phase = np.empty_like(time) # Initialize NumPy array to store cumulative phase.
    last_phase = phase[0] # Store the phase from the previous time step.

    # Iterate over each value of the instantaneous phase array to unwrap it.
    for i, ph in enumerate(phase):
        # Check if the absolute difference between the current phase and the previous phase
        # is greater than or equal to pi. This identifies phase wrapping (jumps of approx. 2*pi).
        if np.abs(ph - last_phase) >= np.pi:
            # Adjust the `cycles` variable based on the direction of phase wrapping.
            # If `ph` is positive (e.g., jumped from near -pi to near pi), `cycles` decrements.
            # If `ph` is negative (e.g., jumped from near pi to near -pi), `cycles` increments.
            cycles += -1 if ph > 0 else 1

        # Calculate the cumulative phase for the current time step.
        cum_phase[i] = ph + 2 * np.pi * cycles

        # Update the `last_phase` variable with the current phase value for the next iteration.
        last_phase = ph

    # Compute the time derivative of the cumulative phase (angular frequency).
    cum_phase_derivative = compute_first_derivative_in_time(time, cum_phase)

    return time, cum_phase, amplitude, cum_phase_derivative


def quadratic(x: float, a: float, b: float, c: float) -> float:
    """Evaluate a quadratic polynomial (ax^2 + bx + c).

    :param x: The independent variable.
    :param a: The coefficient of the x^2 term.
    :param b: The coefficient of the x term.
    :param c: The constant term (y-intercept).
    :return: The value of the quadratic function at x.

    DocTests:
    >>> quadratic(1.0, 1.0, 2.0, 3.0)
    6.0
    >>> quadratic(0.0, 5.0, -2.0, 7.0)
    7.0
    >>> quadratic(-2.0, 1.0, 0.0, 0.0)
    4.0
    """
    return a * x**2 + b * x + c


def fit_quadratic_to_omega_and_find_minimum(
    r_over_M: float, time: NDArray[np.float64], omega: NDArray[np.float64]
) -> float:
    """Fit a quadratic curve to filtered omega data within a specified time range and output the y-intercept.

    This function filters `omega` data within a time range `[r_over_M, r_over_M + 200.0]`,
    fits a quadratic polynomial to this filtered data, and returns the absolute
    value of the y-intercept of the fitted curve. This y-intercept is used
    as an estimate for the minimum omega (angular frequency) at t=0. This is
    primarily intended for the l=2, m=2 angular frequency data.

    :param r_over_M: The reference time (extraction radius) to define the start of the fitting interval.
    :param time: A NumPy array containing time values.
    :param omega: A NumPy array containing omega values corresponding to the time values.
    :return: The absolute value of the constant term (y-intercept) of the fitted quadratic curve,
             representing the estimated minimum omega at t=0.
    :raises ValueError: If the lengths of time and omega arrays are not equal, or if there are
                        insufficient data points (fewer than 3) to perform the quadratic fit within the specified interval.

    DocTests:
    >>> time_test = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0], dtype=np.float64)
    >>> # Example: omega = 0.001*t^2 - 0.2*t + 10.0 (intercept is 10.0)
    >>> omega_test = np.array([10.0, 0.001*50**2 - 0.2*50 + 10.0, 0.001*100**2 - 0.2*100 + 10.0, \
    ...                        0.001*150**2 - 0.2*150 + 10.0, 0.001*200**2 - 0.2*200 + 10.0, \
    ...                        0.001*250**2 - 0.2*250 + 10.0, 0.001*300**2 - 0.2*300 + 10.0], dtype=np.float64)
    >>> np.round(fit_quadratic_to_omega_and_find_minimum(0.0, time_test, omega_test), 5)
    10.0
    >>> # Test with a different interval and known intercept
    >>> time_test_2 = np.array([100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0], dtype=np.float64)
    >>> omega_test_2 = 0.0005 * time_test_2**2 - 0.1 * time_test_2 + 5.0
    >>> np.round(fit_quadratic_to_omega_and_find_minimum(100.0, time_test_2, omega_test_2), 5)
    5.0
    >>> # Test insufficient data points
    >>> try:
    ...     fit_quadratic_to_omega_and_find_minimum(0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    ... except ValueError as e:
    ...     print(e)
    Not enough data points (2) in the interval [0.0, 200.0] for quadratic fitting. At least 3 points are required.
    """
    if len(time) != len(omega):
        raise ValueError("The lengths of time and omega arrays must be equal.")

    # Filter the data for the specified fitting interval [r_over_M, r_over_M + 200.0].
    fit_start = r_over_M
    fit_end = r_over_M + 200.0
    time_filtered = time[(time >= fit_start) & (time <= fit_end)]
    omega_filtered = omega[(time >= fit_start) & (time <= fit_end)]

    if len(time_filtered) < 3:
        raise ValueError(
            f"Not enough data points ({len(time_filtered)}) in the interval "
            f"[{fit_start}, {fit_end}] for quadratic fitting. At least 3 points are required."
        )

    # Fit a quadratic curve to the filtered Omega data using nonlinear least squares.
    # `curve_fit` returns optimal parameters `params` and the covariance matrix.
    params, _ = curve_fit(quadratic, time_filtered, omega_filtered)

    # Extract the coefficients (a, b, c) from the fitted parameters.
    a, b, c = params
    # Calculate the x-coordinate of the extremum (vertex) of the quadratic curve.
    extremum_x = -b / (2 * a)
    # Evaluate the quadratic at its extremum to find the corresponding omega value.
    omega_min_quad_fit = np.fabs(quadratic(extremum_x, a, b, c))
    # Evaluate the quadratic at t=0 to find the y-intercept, which is used as the minimum omega.
    omega_at_t_zero = np.fabs(quadratic(0.0, a, b, c))

    print(
        f"The extremum of the quadratic curve occurs at t = {extremum_x:.15f} "
        f"with omega = {omega_min_quad_fit:.15f}. Implied omega(t=0) = {omega_at_t_zero:.15f}"
    )

    # Return the absolute value of the y-intercept as the estimated minimum omega.
    return float(omega_at_t_zero)


def perform_complex_fft(
    time: NDArray[np.float64], real: NDArray[np.float64], imag: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Perform a complex Fast Fourier Transform (FFT) on the input time, real, and imaginary data.

    Combine the real and imaginary parts into a complex signal, then apply
    the FFT to transform the signal from the time domain to the frequency domain.
    Calculate and return the corresponding frequencies.

    :param time: A NumPy array containing time values, assumed to be uniformly spaced.
    :param real: A NumPy array containing the real part of the signal.
    :param imag: A NumPy array containing the imaginary part of the signal.
    :return: A tuple containing two NumPy arrays:
             - frequencies (NDArray[np.float64]): The frequency values corresponding to the FFT output.
             - fft_data (NDArray[np.complex128]): The complex-valued FFT result.
    :raises ValueError: If the lengths of time, real, and imag arrays are not equal.

    DocTests:
    >>> time_test = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    >>> real_test = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64) # Roughly cos(pi/2 * t)
    >>> imag_test = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64) # Roughly sin(pi/2 * t)
    >>> freq, fft_data = perform_complex_fft(time_test, real_test, imag_test)
    >>> # Expected frequencies: [-0.25, 0.00, 0.25, 0.50] (Hz) from fftfreq(4, d=1.0)
    >>> np.round(freq, 2)
    array([-0.25,  0.  ,  0.25,  0.5 ])
    >>> # For a pure positive frequency sinusoid, only the corresponding positive frequency bin should be non-zero
    >>> # e.g. e^(i*2*pi*f*t). Here, f = 0.25 Hz.
    >>> np.round(fft_data, 5) # For a pure exp(i*pi/2*t), expect a peak at positive freq.
    array([ 0.+0.j,  0.+0.j,  4.+0.j,  0.+0.j])
    >>> time_const = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    >>> real_const = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    >>> imag_const = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    >>> freq_c, fft_data_c = perform_complex_fft(time_const, real_const, imag_const)
    >>> np.round(freq_c, 2)
    array([ 0.  ,  0.33, -0.33])
    >>> np.round(fft_data_c, 5) # Expect DC component (freq=0) to be non-zero (sum of signal)
    array([15.+0.j,  0.+0.j,  0.+0.j])
    """
    if not len(time) == len(real) == len(imag):
        raise ValueError("The lengths of time, real, and imag arrays must be equal.")

    # Combine the real and imaginary data into a single complex signal.
    complex_signal = real + 1j * imag

    # Perform the complex FFT on the combined signal.
    fft_data = np.fft.fft(complex_signal)

    # Calculate the frequency values corresponding to the FFT output.
    # `dt` is the uniform time step.
    dt = time[1] - time[0]
    n = len(time) # Number of samples.
    frequencies = np.fft.fftfreq(n, d=dt) # Frequencies in Hz.

    return frequencies, fft_data


def extract_min_omega_ell2_m2(
    extraction_radius: float,
    time_arr: NDArray[np.float64],
    mode_data: Dict[Tuple[int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]],
    output_folder_path: str, # Added this parameter as discussed
) -> float:
    """Extract phase, amplitude, and angular frequency data for the l=2, m=2 mode from psi4 wave data.

    This function specifically processes the (2, 2) mode data to calculate its
    cumulative phase, amplitude, and angular frequency (omega). It then saves
    these processed data to a file in the specified output folder. Finally, it
    fits a quadratic function to the extracted omega data within a specified time
    range to estimate the minimum omega value at t=0 (the y-intercept of the fit).

    :param extraction_radius: The extraction radius (r/M) used for data naming and
                              for defining the fitting interval.
    :param time_arr: A NumPy array of time values for the waveform data.
    :param mode_data: A dictionary containing the complex mode data, keyed by (ell, m) tuples.
    :param output_folder_path: The directory where the phase, amplitude, omega data file will be saved.
    :return: The estimated minimum omega value (absolute value of the y-intercept)
             from the quadratic fit to the (2, 2) mode's angular frequency data.
    :raises ValueError: If the (2,2) mode data is not found in `mode_data` or if
                        `compute_psi4_wave_phase_and_amplitude` or
                        `fit_quadratic_to_omega_and_find_minimum` raise it.
    :raises OSError: If there's an issue saving the output file.

    DocTests:
    >>> import tempfile
    >>> import os
    >>> # Create a temporary directory for output files
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Mock data for (2,2) mode
    ...     time_mock = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    ...     # Simulate a simple omega that decreases, so quadratic fit works
    ...     # omega = 0.0001*t^2 - 0.02*t + 1.0 (intercept is 1.0)
    ...     real_mock = np.array([np.cos(0.0001*t**3/3 - 0.01*t**2 + t) for t in time_mock], dtype=np.float64)
    ...     imag_mock = np.array([np.sin(0.0001*t**3/3 - 0.01*t**2 + t) for t in time_mock], dtype=np.float64)
    ...     mode_data_mock = {(2, 2): (real_mock, imag_mock)}
    ...     extraction_radius_mock = 100.0
    ...     # Call the function and get the result
    ...     min_omega = extract_min_omega_ell2_m2(extraction_radius_mock, time_mock, mode_data_mock, tmpdir)
    ...     # Check if the output file was created
    ...     output_file_path = os.path.join(tmpdir, 'Rpsi4_r0100.0_ell2_m2_phase_amp_omega.txt')
    ...     os.path.exists(output_file_path)
    True
    >>>     # Verify the content of the file (simplified check)
    >>>     with open(output_file_path, 'r') as f:
    ...         lines = f.readlines()
    ...         len(lines) > 1 # At least header and one data line
    True
    >>>     # Check the returned minimum omega value (it should be close to 1.0 from the mock omega)
    >>>     np.round(min_omega, 5)
    1.0
    """
    # Ensure (2,2) mode data exists.
    if (2, 2) not in mode_data:
        raise ValueError("Mode (2,2) data not found in provided mode_data dictionary.")

    real_ell2_m2, imag_ell2_m2 = mode_data[(2, 2)]

    (
        time_arr, # Re-obtained time array (should be same as input)
        cumulative_phase_ell2_m2,
        amplitude_ell2_m2,
        omega_ell2_m2, # Angular frequency
    ) = compute_psi4_wave_phase_and_amplitude(time_arr, real_ell2_m2, imag_ell2_m2)

    # Define the output filename for phase, amplitude, and omega data.
    phase_amp_omega_file = (
        f"Rpsi4_r{extraction_radius:06.1f}_ell2_m2_phase_amp_omega.txt"
    )
    # Construct the full path to the output file.
    paof_file_path = os.path.join(output_folder_path, phase_amp_omega_file)

    try:
        # Create output directory if it doesn't exist.
        os.makedirs(output_folder_path, exist_ok=True)
        with open(paof_file_path, mode="w", encoding="utf-8") as file:
            file.write("# Time    cumulative_phase    amplitude    omega\n")
            # Write data rows, formatting to 15 decimal places.
            for t, cp, a, o in zip(
                time_arr, cumulative_phase_ell2_m2, amplitude_ell2_m2, omega_ell2_m2
            ):
                file.write(f"{t:.15f} {cp:.15f} {a:.15f} {o:.15f}\n")
    except OSError as e:
        raise OSError(f"Error saving phase, amplitude, omega data to '{paof_file_path}': {e}") from e

    print(
        f"Phase, amplitude, omega data for l=m=2 have been saved to '{phase_amp_omega_file}'"
    )

    # Fit a quadratic curve to the omega data and find the estimated minimum omega at t=0.
    return fit_quadratic_to_omega_and_find_minimum(
        extraction_radius, time_arr, omega_ell2_m2
    )


def main() -> None:
    """Read gravitational wave data, process it, and save the output.

    This function serves as the entry point for the script. It parses
    command-line arguments for the path to the PSI4 data folder and the
    dimensionless extraction radius (r/M). It then reads the PSI4 data files,
    computes the minimum angular frequency from the (2,2) mode, performs
    Fast Fourier Transforms to calculate strain data, and finally saves
    the processed strain data and its second time derivative to files.
    """
    if len(sys.argv) != 3:
        # Replaced print(...) + sys.exit(...) with raise ValueError
        raise ValueError(
            """Please include path to psi4 folder data as well as the extraction radius of that data.
            Usage: python3 <script name> <path to psi4 folder> <extraction radius (r/M) (4 digits, e.g. 0100)>"""
        )
    psi4_folder_path: str = str(sys.argv[1])
    extraction_radius: float = float(sys.argv[2])
    generic_file_name: str = construct_generic_filename(extraction_radius)

    # Read the raw psi4 data files.
    # This reads all modes from l=2 to l=8, as defined internally in read_BHaH_psi4_files.
    try:
        time_arr, mode_data = read_BHaH_psi4_files(generic_file_name, psi4_folder_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        raise RuntimeError(f"Failed to read BHaH psi4 files: {e}") from e

    # Extract the minimum angular frequency (omega) from the (2,2) mode.
    # This involves phase/amplitude calculation and quadratic fitting.
    try:
        min_omega_ell2_m2 = extract_min_omega_ell2_m2(
            extraction_radius, time_arr, mode_data, psi4_folder_path
        )
    except (ValueError, OSError) as e:
        raise RuntimeError(f"Failed to extract minimum omega for (2,2) mode: {e}") from e

    # Initialize dictionaries to store the calculated strain data
    # and its second time derivative (ddot strain data) for all modes.
    strain_data: Dict[
        Tuple[int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = {}
    ddot_strain_data: Dict[
        Tuple[int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = {}

    # Loop over all modes (l from 2 to 8, and m from -l to l)
    # to perform Fixed-Frequency Integration (FFI) for strain calculation.
    for ell in range(2, 9):
        for m in range(-ell, ell + 1):
            # min_omega_m = np.fabs(m) * min_omega_ell2_m2 / 2.0 # Original comment; use min_omega_ell2_m2.
            min_omega = min_omega_ell2_m2  # The angular frequency of the l=m=2 mode at t=0 is used as the minimum physical omega, accounting for GW memory effects.

            # Get the real and imaginary parts of the current mode data.
            real_ell_m, imag_ell_m = mode_data[(ell, m)]

            # Combine them into a complex signal for FFT.
            complex_signal_mode = real_ell_m + 1j * imag_ell_m

            # Perform the Fast Fourier Transform on the complex signal.
            fft_result = np.fft.fft(complex_signal_mode)

            # Calculate the angular frequencies corresponding to the FFT result.
            omega_list = (
                np.fft.fftfreq(len(time_arr), time_arr[1] - time_arr[0]) * 2 * np.pi
            )

            # Apply the Fixed-Frequency Integration (FFI) filter in the frequency domain.
            # This is based on Eq. 27 in https://arxiv.org/abs/1006.1632.
            # It integrates twice by dividing by (i*omega)^2, with a special handling for low frequencies.
            for i, omega_val in enumerate(omega_list):
                if np.fabs(omega_val) <= min_omega:
                    # For frequencies below or equal to `min_omega`, use `min_omega` in the denominator.
                    # This prevents division by zero for omega=0 and stabilizes low-frequency integration.
                    if min_omega == 0.0: # Handle case where min_omega is exactly zero
                        fft_result[i] = 0.0
                    else:
                        fft_result[i] *= 1 / (1j * min_omega) ** 2
                else:
                    # For frequencies above `min_omega`, use the actual frequency.
                    if omega_val == 0.0: # This case should ideally be handled by the min_omega logic
                        fft_result[i] = 0.0
                    else:
                        fft_result[i] *= 1 / (1j * omega_val) ** 2

            # Perform the inverse FFT to transform back to the time domain, yielding the strain (h).
            second_integral_complex = np.fft.ifft(fft_result)

            # Separate the real and imaginary parts of the calculated strain.
            second_integral_real = np.real(second_integral_complex)
            second_integral_imag = np.imag(second_integral_complex)

            # Store the strain data for the current mode.
            strain_data[(ell, m)] = (second_integral_real, second_integral_imag)

            # Calculate the second time derivative of the calculated strain.
            # This is done to compare against the original Psi4 data as a validation check.
            second_derivative_real = compute_second_derivative_in_time(
                time_arr, second_integral_real
            )
            second_derivative_imag = compute_second_derivative_in_time(
                time_arr, second_integral_imag
            )
            ddot_strain_data[(ell, m)] = (
                second_derivative_real,
                second_derivative_imag,
            )

    # Save the processed strain data and its second derivative to files.
    for ell in range(2, 9):
        # Construct output filename for the strain data.
        strain_file = f"Rpsi4_r{extraction_radius:06.1f}_l{ell}_conv_to_strain.txt"
        strain_file_path = os.path.abspath(
            os.path.join(psi4_folder_path, strain_file)
        )
        try:
            with open(strain_file_path, mode="w", encoding="utf-8") as file:
                column = 1
                file.write(f"# column {column}: t-R_ext = [retarded time]\n")
                column += 1
                for m in range(-ell, ell + 1):
                    file.write(f"# column {column}: Re(h_{{l={ell},m={m}}}) * R_ext\n")
                    column += 1
                    file.write(f"# column {column}: Im(h_{{l={ell},m={m}}}) * R_ext\n")
                    column += 1
                for i, time_val in enumerate(time_arr):
                    out_str = str(time_val)
                    for m in range(-ell, ell + 1):
                        # Concatenate real and imaginary parts for the current mode.
                        out_str += (
                            f" {strain_data[(ell,m)][0][i]} {strain_data[(ell,m)][1][i]}"
                        )
                    file.write(out_str + "\n")
            print(f"Strain data for l={ell} saved to '{strain_file}'")
        except OSError as e:
            raise RuntimeError(f"Error saving strain data for l={ell}: {e}") from e


        # Construct output filename for the twice-integrated, twice-differentiated data (should approximate original Psi4).
        ddot_file = f"Rpsi4_r{extraction_radius:06.1f}_l{ell}_from_strain.txt"
        ddot_file_path = os.path.abspath(
            os.path.join(psi4_folder_path, ddot_file)
        )
        try:
            with open(ddot_file_path, mode="w", encoding="utf-8") as file:
                column = 1
                file.write(f"# column {column}: t-R_ext = [retarded time]\n")
                column += 1
                for m in range(-ell, ell + 1):
                    # Label for the second derivative of strain, approximating original Psi4.
                    file.write(f"# column {column}: Re(Psi4_{{l={ell},m={m}}}) * R_ext\n")
                    column += 1
                    file.write(f"# column {column}: Im(Psi4_{{l={ell},m={m}}}) * R_ext\n")
                    column += 1
                for i, time_val in enumerate(time_arr):
                    out_str = str(time_val)
                    for m in range(-ell, ell + 1):
                        # Concatenate real and imaginary parts of the ddot strain data.
                        out_str += f" {ddot_strain_data[(ell,m)][0][i]} {ddot_strain_data[(ell,m)][1][i]}"
                    file.write(out_str + "\n")
            print(f"Second derivative of strain data for l={ell} saved to '{ddot_file}'")
        except OSError as e:
            raise RuntimeError(f"Error saving second derivative of strain data for l={ell}: {e}") from e


if __name__ == "__main__":
    # First run doctests to ensure utility functions work as expected.
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    # Then run the main() function to execute the primary script logic.
    main()
