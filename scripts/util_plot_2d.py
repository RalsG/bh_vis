"""Plot 2D gravitational wave psi4 or strain data using Matplotlib.

Allow the user to visualize specific modes or the superimposed waveform
by providing system arguments for data directory, input type, and desired modes.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

import os
import sys
import random # Used for random color selection for plots.
import numpy as np
import matplotlib.pyplot as plt
import psi4_FFI_to_strain as util # Aliased for brevity

# --- Command-line argument parsing and initial setup ---

# Variables to store command-line argument values, initialized to default/empty states.
input_type = "" # Stores whether the input data is 'strain' or 'psi4'.
input_dir = ""  # Stores the directory path containing the gravitational wave data.
plot_full = False # A boolean flag; True if the full superimposed waveform should be plotted.

# List to store the (l, m) spherical harmonic modes to be plotted.
modes_to_plot: list[tuple[int, int]] = []

# Validate the number of command-line arguments.
# This script expects at least 3 arguments: script_name, input_directory, input_type.
# Additional arguments specify the modes to plot or the "full" waveform option.
if len(sys.argv) < 3:
    # Raise a ValueError if the required command-line arguments are not provided.
    raise ValueError(
        """Please include the following system arguments:
        python <name of this file> <directory for folder holding data> <input type of data (strain or psi4)> <mode index in (l,m) format (e.g. 2 2)> <as many modes as you want, or "full" to get superimposed waveform>"""
    )
else:
    # Assign command-line arguments to variables.
    input_dir = sys.argv[1]
    input_type = str(sys.argv[2]).lower()

    # Validate the input type.
    # The script only supports 'strain' or 'psi4' as input data types.
    if input_type not in ["strain", "psi4"]:
        # Raise a ValueError for an unsupported input type.
        raise ValueError("Invalid input type: please enter 'strain' or 'psi4'")

    # Process the remaining arguments to identify individual modes or the "full" waveform flag.
    plot_info_raw = sys.argv[3:]
    numerical_modes_raw = [] # A temporary list to collect integer arguments for (l, m) pairs.

    for item in plot_info_raw:
        if str(item).lower() == "full":
            plot_full = True
        else:
            try:
                # Attempt to convert the argument to an integer, expecting mode numbers.
                numerical_modes_raw.append(int(item))
            except ValueError as e:
                # Raise a ValueError if a mode argument is not an integer or 'full'.
                raise ValueError(f"Invalid mode argument: '{item}'. Must be an integer or 'full'.") from e

    # After parsing, check if the number of numerical arguments is even, as they are expected to be (l, m) pairs.
    if len(numerical_modes_raw) % 2 != 0:
        # Raise a ValueError if an odd number of mode arguments are provided.
        raise ValueError("Mode indices must be provided in (l, m) pairs. Found an odd number of numerical arguments.")

    # Construct (l, m) tuples from the parsed integer arguments.
    # Iterate in steps of 2 to create pairs.
    for i in range(0, len(numerical_modes_raw), 2):
        modes_to_plot.append((numerical_modes_raw[i], numerical_modes_raw[i+1]))

# Print the parsed plotting information to the console.
print(f"Modes to be plotted: {modes_to_plot}")
print(f"Plot full waveform: {plot_full}")

# Configure the utility module based on the input type and directory.
# Note: Modifying global variables of an imported module (`util`) can lead to side effects
# and is generally discouraged in larger applications.
util.INPUT_DIR = input_dir
if input_type == "strain":
    # If the user wants to plot strain data, adjust the file pattern for the utility function.
    # The FILE_PATTERN determines which files the util module reads.
    util.FILE_PATTERN = "_l[MODE=L]_conv_to_strain"

print(f"Input directory set to: {util.INPUT_DIR}")

# finds and stores data to plot, whether it's psi4 or strain
# Call the utility function to read and process the gravitational wave data.
# This function returns time data and complex-valued mode data.
try:
    time_data, modes_data = util.read_psi4_dir(util.INPUT_DIR, util.ELL_MAX)
except Exception as e: # Catch any exceptions from the utility module's file reading.
    raise OSError(f"Error reading data from {util.INPUT_DIR}: {e}") from e

# --- Plotting section ---

# create plots for inputted modes
# Iterate through each specified mode to plot its real part.
for current_mode in modes_to_plot:
    current_l = current_mode[0]
    current_m = current_mode[1]
    # Get the data for the current mode using the utility function.
    # The `modes_index` function translates (l, m) to an array index.
    y_plot = modes_data[util.modes_index(current_l, current_m)]
    # Randomly choose a color for the plot line from Matplotlib's 'tab10' colormap.
    color_choice = random.choice(plt.cm.tab10.colors)
    # Plot the real part of the mode data.
    plt.plot(
        time_data,
        y_plot.real,
        color=color_choice,
        alpha=0.5, # Set transparency (alpha) for overlapping plots.
        label=f"({current_l},{current_m})" # Use f-string for a clear plot label.
    )

# Plot the full superimposed waveform if the 'plot_full' flag is True.
if plot_full:
    # Initialize a NumPy array with zeros, having the same shape as `time_data`,
    # to accumulate the real parts of all modes.
    y_plot_sum = np.zeros_like(time_data, dtype=np.float64)
    # Iterate over all possible ell and m values (from ELL_MIN to ELL_MAX)
    # to sum up their real parts for the full waveform.
    for ell in range(util.ELL_MIN, util.ELL_MAX + 1):
        for m in range(-ell, ell + 1):
            y_plot_sum += modes_data[util.modes_index(ell, m)].real
    # Randomly choose a color for the full waveform plot.
    color_choice = random.choice(plt.cm.tab10.colors)
    # Plot the summed real part of the full waveform.
    plt.plot(
        time_data,
        y_plot_sum, # y_plot_sum is already a real-valued array.
        color=color_choice,
        alpha=0.5,
        label="Full Waveform" # Use a descriptive label for the legend.
    )

# Set the title of the plot. Capitalize the input type for better presentation.
plt.title(f"{input_type.capitalize()} vs. Time")
# Set the label for the y-axis. Capitalize the input type.
plt.ylabel(f"{input_type.capitalize()}")
# Set the label for the x-axis.
plt.xlabel("Time")
# Display the legend, showing labels for each plotted line.
plt.legend()
# Add a grid to the plot for better readability.
plt.grid(True)
# Display the plot.
plt.show()

if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
