"""Read black hole positional and velocity data.

Load data from a specified text file containing black hole positional and
velocity information. Filter out comment lines, then convert the data to a
NumPy array and sort it by the first column (time). Finally, print the
processed NumPy array to the console.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""
import os
import numpy as np
from numpy.typing import NDArray
import sys

# This path is relative to the current script's location.
# It attempts to go up two directories from the script's location and then into 'r100'.
file_name = "puncture_posns_vels_regridxyzU.txt"
file_path = os.path.abspath(os.path.join(__file__, '..', '..', "r100", file_name))

try:
    with open(file_path, mode="r", encoding="utf-8") as file:
        # Read all lines from the file and filter out lines that start with '#'.
        lines = [line for line in file.readlines() if not line.startswith("#")]
        # Convert the filtered lines into a NumPy array of float64.
        # Each line is split into components, which are then mapped to float64.
        data: NDArray[np.float64] = np.array(
                [list(map(np.float64, line.split())) for line in lines]
            )
        # Sort the entire array based on the values in the first column (index 0), which is assumed to be time.
        data = data[np.argsort(data[:, 0])]
    print(data)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at: {file_path}. Please ensure the path is correct.")
except ValueError as e:
    raise ValueError(f"Error parsing data in {file_name}: {e}. Ensure data is numeric and correctly formatted.")


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
