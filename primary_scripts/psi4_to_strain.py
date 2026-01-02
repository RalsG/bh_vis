import argparse
import os
import re
import sys
import glob
import logging
import numpy as np

# Configure basic logging
# INFO and DEBUG to stdout
# WARNING, ERROR, CRITICAL to stderr
# This provides a clean separation for typical operational output vs. issues.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the minimum level for the logger itself

# Handler for stdout (INFO and DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG) # Process messages from DEBUG up to INFO
stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING) # Filter out WARNING and above
stdout_formatter = logging.Formatter('%(levelname)s: %(message)s') # Simpler for INFO
stdout_handler.setFormatter(stdout_formatter)
logger.addHandler(stdout_handler)

# Handler for stderr (WARNING, ERROR, CRITICAL)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING) # Process messages from WARNING up
stderr_formatter = logging.Formatter('%(levelname)s: %(filename)s:%(lineno)d: %(message)s') # More detail for errors
stderr_handler.setFormatter(stderr_formatter)
logger.addHandler(stderr_handler)


def parse_args():
    """Parses command-line arguments and performs initial validation."""
    parser = argparse.ArgumentParser(
        description="Process time-domain Weyl scalar Psi4_lm data to gravitational wave strain h_lm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True,
                        help="Path to the directory containing input mp_psi4_l...asc files.")
    parser.add_argument("--output_dir", required=True,
                        help="Path to the directory where output h_lm files will be saved.")
    parser.add_argument("--R_ext", type=float, default=100.0,
                        help="Extraction radius for calculating retarded time t_ret = t - R_ext.")
    parser.add_argument("--ell_min", type=int, default=2,
                        help="Minimum l mode to process.")
    parser.add_argument("--ell_max", type=int, default=8,
                        help="Maximum l mode to process.")
    parser.add_argument("--interval", type=float, default=200.0,
                        help="Duration of the time interval (at the end of the signal) for the "
                             "phase fit of (l=2,m=2) mode to determine characteristic frequency.")
    parser.add_argument("--cutoff_factor", type=float, default=0.75,
                        help="Factor (0 < cutoff_factor <= 1) to multiply characteristic frequency "
                             "by to get freq_cutoff.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")
    if not (0 < args.cutoff_factor <= 1):
        parser.error("--cutoff_factor must be > 0 and <= 1.")
    if args.ell_min < 0: # l must be non-negative
        parser.error("--ell_min must be non-negative.")
    if args.ell_max < args.ell_min:
        parser.error("--ell_max must be greater than or equal to --ell_min.")
    if args.interval <= 0:
        parser.error("--interval for phase fit must be a positive duration.")

    return args

def load_psi4_data(input_dir, ell_min_arg, ell_max_arg, target_R_ext_from_arg):
    """
    Loads Psi4_lm data from specified directory.
    Filters modes based on ell_min_arg, ell_max_arg, and the extraction radius
    in the filename matching target_R_ext_from_arg.
    Assumes all time arrays are identical and uniformly sampled.

    Returns:
        time_array (np.array): The common time array. None if no data loaded.
        dt (float): The time step. None if no data loaded.
        psi4_data (dict): Dict mapping (l,m) tuples to complex Psi4_lm_scaled data arrays.
        active_l_values (set): Set of l values for which at least one m mode's data was found and processed.
    """
    psi4_data = {}
    time_array = None
    dt = None
    active_l_values = set()

    # Regex to parse filenames: e.g., mp_psi4_l2_m-2_r100.00.asc
    # It now captures the numerical part of the radius.
    filename_pattern = re.compile(r"mp_psi4_l(\d+)_m([-+]?\d+)_r([\d.]+)\.asc")
    
    glob_pattern = os.path.join(input_dir, "mp_psi4_l*_m*_r*.asc")
    potential_files = sorted(glob.glob(glob_pattern))

    if not potential_files:
        logger.warning(f"No files matching pattern '{glob_pattern}' found in '{input_dir}'.")
        return None, None, {}, set()

    logger.info(f"Found {len(potential_files)} potential Psi4 files in '{input_dir}'. Scanning...")
    processed_files_count = 0

    for filepath in potential_files:
        filename = os.path.basename(filepath)
        match = filename_pattern.fullmatch(filename)
        if not match:
            logger.debug(f"Skipping file with unrecognized name format: {filename}")
            continue

        l_val_str = match.group(1)
        m_val_str = match.group(2)
        radius_str_from_filename = match.group(3)

        try:
            l_val = int(l_val_str)
            m_val = int(m_val_str)
            radius_float_from_filename = float(radius_str_from_filename)
        except ValueError as e:
            logger.warning(f"Could not parse l, m, or radius from filename '{filename}' "
                           f"(l='{l_val_str}', m='{m_val_str}', radius='{radius_str_from_filename}'). Error: {e}. Skipping.")
            continue

        # --- New Filter: Check if filename radius matches target_R_ext_from_arg ---
        if not np.isclose(radius_float_from_filename, target_R_ext_from_arg):
            logger.debug(f"Skipping file '{filename}' with radius {radius_float_from_filename:.2f} "
                         f"in its name; target R_ext for filtering is {target_R_ext_from_arg:.2f}.")
            continue
        # --- End New Filter ---

        if not (ell_min_arg <= l_val <= ell_max_arg):
            logger.debug(f"Skipping file '{filename}' for l={l_val}, m={m_val} (l outside range [{ell_min_arg}, {ell_max_arg}]).")
            continue

        logger.debug(f"Processing file: {filename} for l={l_val}, m={m_val}, filename radius {radius_float_from_filename:.2f}")
        try:
            data = np.loadtxt(filepath, usecols=(0, 1, 2), comments="#", ndmin=2)

            if data.shape[0] == 0:
                logger.warning(f"File '{filename}' (l={l_val}, m={m_val}) contains no data rows. Skipping.")
                continue

            t_current = data[:, 0]
            re_psi4_scaled = data[:, 1]
            im_psi4_scaled = data[:, 2]

            if time_array is None:
                if len(t_current) < 2:
                    logger.error(f"Time array in '{filename}' (l={l_val}, m={m_val}) has {len(t_current)} points. "
                                 "At least 2 points are required to determine dt and perform FFT. Aborting.")
                    return None, None, {}, set()
                time_array = t_current
                dt_candidate = time_array[1] - time_array[0]

                if dt_candidate <= 1e-14:
                    logger.error(f"Time step dt ({dt_candidate:.2e}) determined from '{filename}' (l={l_val}, m={m_val}) "
                                 "is not positive or too small. Aborting.")
                    return None, None, {}, set()
                
                if not np.allclose(np.diff(time_array), dt_candidate):
                    logger.error(f"Time steps in the first processed file '{filename}' are not uniform. "
                                 "This script assumes uniformly sampled data. Aborting.")
                    return None, None, {}, set()
                dt = dt_candidate
            else:
                if len(t_current) != len(time_array) or not np.allclose(t_current, time_array):
                    logger.error(f"Time array in '{filename}' differs from first loaded. "
                                 "As per assumption, all time arrays must be identical. Aborting.")
                    return None, None, {}, set()

            psi4_data[(l_val, m_val)] = re_psi4_scaled + 1j * im_psi4_scaled
            active_l_values.add(l_val)
            processed_files_count += 1
            logger.info(f"Successfully loaded l={l_val}, m={m_val} from {filename} (matched R_ext={target_R_ext_from_arg:.2f})")

        except ValueError as e:
            logger.warning(f"Could not parse numeric data in '{filename}' (l={l_val}, m={m_val}). Error: {e}. Skipping.")
        except Exception as e:
            logger.warning(f"Unexpected error reading or processing '{filename}' (l={l_val}, m={m_val}). Error: {e}. Skipping.")

    logger.info(f"Scan complete. Processed {processed_files_count} files that matched target R_ext and (l_min, l_max) range.")

    if time_array is None:
        logger.warning("No valid Psi4 data files were loaded that met all criteria (including R_ext in filename).")

    return time_array, dt, psi4_data, active_l_values


def calculate_freq_cutoff(psi4_22_data, time_array, dt, interval_duration, cutoff_factor_arg):
    """
    Calculates the frequency cutoff based on the (l=2, m=2) mode's phase evolution.
    """
    if psi4_22_data is None:
        logger.warning("Psi4 (l=2, m=2) mode data not available for freq_cutoff calculation. Setting freq_cutoff = 0.0.")
        return 0.0
    if np.allclose(psi4_22_data, 0): # Use np.allclose for float comparison
        logger.warning("Psi4 (l=2, m=2) mode data is all zeros. Setting freq_cutoff = 0.0.")
        return 0.0
    
    num_points_total = len(time_array)
    if num_points_total < 3: # Quadratic fit (degree 2) needs at least 3 points
        logger.warning(f"Psi4 (l=2, m=2) mode has too few time points ({num_points_total} < 3) for phase fit. "
                       "Setting freq_cutoff = 0.0.")
        return 0.0

    try:
        phi_22 = np.unwrap(np.angle(psi4_22_data))

        # Determine number of points for the interval.
        # Add small epsilon to interval_duration/dt to handle floating point issues when rounding,
        # favouring slightly more points if interval_duration is an exact multiple of dt. Max with 1.
        num_points_for_interval = min(num_points_total, max(1, int(np.ceil(interval_duration / dt))))


        if num_points_for_interval < 3:
            logger.warning(f"Calculated interval duration ({interval_duration} with dt={dt:.2e}) results in {num_points_for_interval} points. "
                           "Minimum 3 required for quadratic fit. ")
            if num_points_total >=3:
                logger.warning("Attempting fit with all available signal points instead.")
                fit_start_idx = 0
                num_points_for_interval = num_points_total # use all points
            else: # num_points_total is also < 3
                logger.warning(f"Full signal also has insufficient points ({num_points_total}). Setting freq_cutoff = 0.0.")
                return 0.0
        else:
             fit_start_idx = num_points_total - num_points_for_interval
       
        t_segment_for_fit = time_array[fit_start_idx : fit_start_idx + num_points_for_interval]
        phi_segment_for_fit = phi_22[fit_start_idx : fit_start_idx + num_points_for_interval]
        t_prime_for_fit = t_segment_for_fit - t_segment_for_fit[0]

        coeffs = np.polyfit(t_prime_for_fit, phi_segment_for_fit, 2) # p(t') = a*t'^2 + b*t' + c
        a, b_coeff = coeffs[0], coeffs[1]

        # Determine min_freq: omega_fit evaluated at the midpoint of the *actual* fitting interval t_prime.
        # omega_fit(t') = 2*a*t' + b_coeff
        # Midpoint t'_mid = t_prime_for_fit[-1] / 2.0 (since t_prime_for_fit starts at 0)
        # min_freq_val = 2 * a * (t_prime_for_fit[-1] / 2.0) + b_coeff = a * t_prime_for_fit[-1] + b_coeff
        # If only 3 points, midpoint might be tricky, t'=0 (i.e. b_coeff) is also acceptable.
        # Using b_coeff (omega at start of interval) is simpler and specified as an option.
        # Let's use omega at t'=0 (start of fitting interval) as characteristic frequency.
        omega_char_from_fit = b_coeff

        freq_cutoff = np.abs(omega_char_from_fit * cutoff_factor_arg)

        actual_interval_duration_fitted = t_prime_for_fit[-1] - t_prime_for_fit[0]
        logger.info(f"Determined freq_cutoff = {freq_cutoff:.4e} rad/time_unit from (l=2,m=2) mode.")
        logger.info(f"  (Details: Raw characteristic omega (coeff b from fit at t'=0) = {omega_char_from_fit:.4e}, "
                    f"Fit using last {num_points_for_interval} points, actual duration used for fit = {actual_interval_duration_fitted:.2f}, "
                    f"Requested interval for selection = {interval_duration:.2f})")
        return freq_cutoff

    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"NumPy error during freq_cutoff polynomial fit for (l=2, m=2) mode: {e}. "
                       "Setting freq_cutoff = 0.0.")
        return 0.0
    except Exception as e:
        logger.warning(f"Unexpected error during freq_cutoff calculation for (l=2, m=2) mode: {e}. "
                       "Setting freq_cutoff = 0.0.")
        return 0.0


def compute_strain_mode(psi4_lm_scaled_data, N_points, dt, freq_cutoff):
    """
    Computes h_lm_scaled(t) from Psi4_lm_scaled(t) via FFT integration.
    freq_cutoff is an angular frequency, guaranteed non-negative.
    """
    psi4_lm_fft = np.fft.fft(psi4_lm_scaled_data)
    angular_freqs = np.fft.fftfreq(N_points, dt) * 2.0 * np.pi

    h_lm_fft_filtered = np.zeros_like(psi4_lm_fft, dtype=complex) # Ensures DC is 0 if not modified

    # Mask for non-zero frequencies (where division is applicable)
    nonzero_freq_mask = (angular_freqs != 0)
    
    # Denominators: (1j * omega_eff)^2 = -omega_eff^2
    # Initialize with -omega_k^2 for non-DC components
    denominators_at_nonzero_freqs = -np.square(angular_freqs[nonzero_freq_mask])

    # Apply freq_cutoff if it's practically non-zero
    # freq_cutoff is already guaranteed non-negative.
    if freq_cutoff > 1e-14: # Check against small epsilon
        # Identify non-zero frequencies where |omega_k| <= freq_cutoff
        # These will use -freq_cutoff^2 as denominator
        low_freq_sub_mask = (np.fabs(angular_freqs[nonzero_freq_mask]) <= freq_cutoff)
        denominators_at_nonzero_freqs[low_freq_sub_mask] = -freq_cutoff**2
    
    # Apply filter: h_lm_fft = Psi4_lm_fft / denominator for non-DC components
    h_lm_fft_filtered[nonzero_freq_mask] = psi4_lm_fft[nonzero_freq_mask] / denominators_at_nonzero_freqs
    
    # Inverse FFT to get time-domain scaled strain h_lm_scaled(t)
    h_lm_scaled_t = np.fft.ifft(h_lm_fft_filtered)
    return h_lm_scaled_t


def write_output_files(output_dir, h_data_scaled, time_common, R_ext_arg,
                       ell_min_arg, ell_max_arg, active_l_values):
    """
    Writes processed h_lm_scaled data to output files, one for each l-mode
    for which input data was found (i.e., l in active_l_values).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory '{output_dir}'. Error: {e}")
        # No sys.exit here, let main decide if this is fatal or if execution can end.
        raise # Re-raise to be caught by main

    retarded_time = time_common - R_ext_arg
    N_points = len(time_common)

    logger.info(f"Writing output files to: {output_dir}")

    # Format R_ext_arg for filename to avoid issues with too many decimals or float representation
    r_ext_str_for_filename = f"{R_ext_arg:.2f}".replace('.', 'p') # e.g., 100.00 -> 100p00, 75.50 -> 75p50

    if R_ext_arg == int(R_ext_arg):
        r_ext_str_for_filename = f"{int(R_ext_arg)}" # e.g. 100
    else:
        r_ext_str_for_filename = f"{R_ext_arg:.2f}".replace('.', 'p') # e.g. 75p50

    for l_val in range(ell_min_arg, ell_max_arg + 1):
        if l_val not in active_l_values:
            logger.info(f"No Psi4 input files were found or processed for any m mode with l={l_val}. "
                        f"Skipping output for h_l{l_val}.txt.")
            continue

        output_filename = os.path.join(output_dir, f"h_l{l_val}_r{r_ext_str_for_filename}.txt")
        
        header_parts = [
            f"# Gravitational wave strain h_l{l_val}_m modes",
            f"# Data is h_lm_scaled = h_lm * R_ext_actual, where R_ext_actual is from input data scaling.",
            f"# Retarded time t_ret = t_simulation - R_ext_arg, with R_ext_arg = {R_ext_arg:.6f}",
            f"# col 0: t_ret"
        ]
        
        data_columns_to_stack = [retarded_time]
        current_col_idx = 1

        for m_val in range(-l_val, l_val + 1):
            mode_key = (l_val, m_val)
            h_lm_mode_data = h_data_scaled.get(mode_key)

            if h_lm_mode_data is None:
                # This implies Psi4 input for (l_val, m_val) was missing or failed loading.
                logger.warning(f"Data for (l={l_val}, m={m_val}) was not available from input processing. "
                               f"Outputting zero columns for this mode in {os.path.basename(output_filename)}.")
                zeros_for_missing_mode = np.zeros(N_points)
                data_columns_to_stack.append(zeros_for_missing_mode) # Real part
                data_columns_to_stack.append(zeros_for_missing_mode) # Imaginary part
            else:
                if len(h_lm_mode_data) != N_points: # Should not happen if load_psi4_data is robust
                    logger.error(f"Strain data for (l={l_val},m={m_val}) has incorrect length! "
                                 f"Expected {N_points}, got {len(h_lm_mode_data)}. Filling with zeros.")
                    data_columns_to_stack.append(np.zeros(N_points))
                    data_columns_to_stack.append(np.zeros(N_points))
                else:
                    data_columns_to_stack.append(h_lm_mode_data.real)
                    data_columns_to_stack.append(h_lm_mode_data.imag)

            header_parts.append(f"# col {current_col_idx}: Re(h_{{l={l_val},m={m_val}}}_scaled)")
            current_col_idx += 1
            header_parts.append(f"# col {current_col_idx}: Im(h_{{l={l_val},m={m_val}}}_scaled)")
            current_col_idx += 1
            
        full_output_matrix = np.column_stack(data_columns_to_stack)
        full_header_str = "\n".join(header_parts)
        
        try:
            np.savetxt(output_filename, full_output_matrix, header=full_header_str, fmt='%.18e', comments='')
            logger.info(f"Successfully wrote: {output_filename}")
        except Exception as e:
            logger.error(f"Could not write output file '{output_filename}'. Error: {e}")


def main():
    args = parse_args()

    logger.info("--- Starting Psi4 to Strain Processing ---")
    logger.info(f"Input Directory:  {args.input_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Retarded Time R_ext: {args.R_ext}")
    logger.info(f"L-mode Range:     ell_min={args.ell_min}, ell_max={args.ell_max}")
    logger.info(f"Freq Cutoff Params: Interval duration for fit={args.interval}, Cutoff factor={args.cutoff_factor}")

    time_common, dt, psi4_data_all_modes, active_l_values = load_psi4_data(
        args.input_dir, args.ell_min, args.ell_max, args.R_ext
    )

    if time_common is None or dt is None:
        logger.error("No time-series data successfully loaded or dt could not be determined. "
                     "Cannot proceed with calculations. Aborting.")
        sys.exit(1)
    
    N_points = len(time_common)
    logger.info(f"Data Loading: Successfully loaded data for {len(psi4_data_all_modes)} (l,m) modes.")
    logger.info(f"  Time array: {N_points} points, dt = {dt:.4e} time_units.")
    
    if not psi4_data_all_modes and not active_l_values:
        logger.info("No Psi4 files were processed (either none found, none in l-range, or all failed). "
                    "No output will be generated.")
        logger.info("--- Processing Complete (No Data To Process) ---")
        sys.exit(0)
    
    # Determine frequency cutoff using (l=2,m=2) mode, if available
    # psi4_data_all_modes only contains modes within [ell_min, ell_max].
    psi4_22_for_cutoff = psi4_data_all_modes.get((2, 2))
    if psi4_22_for_cutoff is None and (args.ell_min <= 2 <= args.ell_max):
         logger.info("Psi4 (l=2, m=2) mode data was expected (l=2 in processing range) but not found/loaded. "
                     "freq_cutoff calculation will use fallback.")

    freq_cutoff = calculate_freq_cutoff(
        psi4_22_for_cutoff, time_common, dt, args.interval, args.cutoff_factor
    )
    
    h_data_scaled = {}
    logger.info(f"Strain Calculation: Processing {len(psi4_data_all_modes)} loaded Psi4 modes...")
    for mode_key, psi4_lm_data in psi4_data_all_modes.items():
        h_lm_scaled = compute_strain_mode(psi4_lm_data, N_points, dt, freq_cutoff)
        h_data_scaled[mode_key] = h_lm_scaled
    logger.info(f"Strain Calculation: Completed for {len(h_data_scaled)} modes.")

    try:
        write_output_files(
            args.output_dir, h_data_scaled, time_common, args.R_ext,
            args.ell_min, args.ell_max, active_l_values
        )
    except Exception as e: # Catch re-raised errors from write_output_files
        logger.critical(f"A critical error occurred during file writing: {e}. Aborting.")
        sys.exit(1)


    logger.info("--- Processing Complete ---")

if __name__ == "__main__":
    main()
