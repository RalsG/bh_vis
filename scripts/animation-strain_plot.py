"""Create animations of gravitational wave strain data over time using Matplotlib.

Generate and save an animation of gravitational wave strain data, visualizing it
as a time-evolving plot. The script reads gravitational wave strain data,
processes it, and creates an animated plot that can be saved as a video file
or displayed directly.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import sys
from typing import List, Tuple, Any

# Check if the necessary arguments have been provided
if len(sys.argv) < 4:
    # Raise a ValueError if the required command-line arguments are not provided.
    raise ValueError("Usage: python script.py <output_directory> <save_video> <show_plot>")

# Get the output directory and save video option from command line arguments
output_directory: str = sys.argv[1]
save_video: bool = sys.argv[2].lower() == 'true'
show_plt: bool = sys.argv[3].lower() == 'true'

# Get a list of all files that begin with "Rpsi4"
# This path is hardcoded; adjust as needed for your environment.
input_files: List[str] = glob.glob("/home/guest/Documents/Users/Tyndale/bh_repos/bhah_waveform_analysis/r0100.0/many_modes/Rpsi4*")

def valid_line(line: str) -> bool:
    """Check if a line is a valid data line (i.e., not a comment).

    A line is considered valid if it does not start with '#'.

    :param line: The input string line to check.
    :return: True if the line is valid, False otherwise.

    DocTests:
    >>> valid_line("1.0 2.0 3.0")
    True
    >>> valid_line("# This is a comment")
    False
    >>> valid_line("   data")
    True
    >>> valid_line("")
    True
    """
    # Returns True if the line does not start with '#', indicating it's a data line.
    return not line.startswith("#")

def load_data(file_name: str) -> np.ndarray[Any, Any]:
    """Load numerical data from a file, filtering out comments and inconsistent rows.

    The function reads data from the specified file, skipping lines that start
    with '#'. It also ensures that all data rows have a consistent number of
    columns, dropping any rows that deviate. Duplicate rows are also removed.

    :param file_name: The path to the data file.
    :return: A NumPy array containing the loaded and processed data.
    :raises FileNotFoundError: If the specified file does not exist.

    DocTests:
    >>> import tempfile
    >>> import os
    >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
    ...     _ = tmp.write("# Header comment\\n")
    ...     _ = tmp.write("1.0 2.0 3.0\\n")
    ...     _ = tmp.write("4.0 5.0\\n") # Inconsistent column count
    ...     _ = tmp.write("6.0 7.0 8.0\\n")
    ...     _ = tmp.write("1.0 2.0 3.0\\n") # Duplicate row
    ...     temp_file_name = tmp.name
    >>> try:
    ...     loaded_data = load_data(temp_file_name)
    ...     expected_data = np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]])
    ...     np.array_equal(loaded_data, expected_data)
    ... finally:
    ...     os.remove(temp_file_name)
    True
    >>> # Test with a non-existent file
    >>> try:
    ...     load_data("non_existent_file.txt")
    ... except FileNotFoundError as e:
    ...     print(e)
    [Errno 2] No such file or directory: 'non_existent_file.txt'
    """
    # Loads data from a file, filtering out comment lines and ensuring consistent column counts.
    # It also removes duplicate rows to ensure uniqueness.
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            valid_lines = [line for line in f if valid_line(line)]
            data = [list(map(float, line.split())) for line in valid_lines]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: '{file_name}'") from e

    if not data: # Handle empty file or only comments
        return np.array([])

    # Determine expected number of columns based on the first row
    num_columns: int = len(data[0])

    # Filter out rows with a different number of columns
    data = [row for row in data if len(row) == num_columns]

    data_array: np.ndarray[Any, Any] = np.array(data)
    data_array = np.unique(data_array, axis=0) # Remove duplicate rows

    return data_array

# Load data from all files
# It is assumed that input_files[0] corresponds to time data
# and input_files[3] corresponds to strain data, based on original indexing.
data: List[np.ndarray[Any, Any]] = [load_data(input_file) for input_file in input_files]

# Gather strain data
# h_time contains the time values from the first data file.
h_time: np.ndarray[np.float64, Any] = data[0][:, 0]
# h_strain contains the raw strain data from the fourth data file (index 3), second column.
h_strain: np.ndarray[np.complex128, Any] = data[3][:, 1]
# strain_magnitude extracts the real part of the strain for plotting.
strain_magnitude: np.ndarray[np.float64, Any] = np.real(h_strain)

# plot the entire line initially
# Set up the Matplotlib figure and axes for the animation.
# The figsize is calculated to maintain specific pixel dimensions (772x80)
# at 80 DPI, potentially for compatibility with VisIt animations.
fig, ax = plt.subplots(figsize=(772/80, 80/80))
line, = ax.plot(h_time, strain_magnitude, color='lightgray') # Static background line
ani_line, = ax.plot([], [], color='blue') # Animated line, initially empty

# Configure axes to be invisible for a clean plot without borders.
ax.axis('off')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(h_time.min(), h_time.max()) # Set X-axis limits to time range
ax.set_ylim(strain_magnitude.min(), strain_magnitude.max()) # Set Y-axis limits to strain magnitude range

def update(i: int) -> Tuple[Any,]:
    """Update the animated line for each frame of the animation.

    This function is called by `FuncAnimation` for each frame to update the
    data of the animated line, showing the strain magnitude evolve over time.

    :param i: The current frame index, representing the number of data points to display.
    :return: A tuple containing the updated `ani_line` artist.
    """
    # Updates the animated line by setting data up to the current frame index 'i'.
    ani_line.set_data(h_time[:i], strain_magnitude[:i])
    return (ani_line,) # Return as a tuple, even if it's a single artist

# Create animation
# `blit=True` means only re-draw the parts that have changed, which can be faster.
ani: FuncAnimation = FuncAnimation(fig, update, frames=range(1, len(h_time)), blit=True)

# Save the animation as a video file if specified.
if save_video:
    Writer = writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(os.path.join(output_directory, 'animation.mp4'), writer=writer)

# Inform the user about the saved movie.
print(f"Movie successfully saved to {output_directory}")

# Display the plot window if specified.
if show_plt:
    plt.show()

if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
