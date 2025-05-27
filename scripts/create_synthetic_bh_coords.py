"""Generate and plot synthetic data for a binary black hole inspiral.

Create a CSV file named 'bh_synthetic.csv' containing the simulated 3D
trajectories of two black holes in a decaying circular orbit. The orbital
separation decreases over time, modeled by an error function. After generating
the data, plot the X-Y projection of the trajectories using Matplotlib.

Authors: NLedge, Joey Perko
"""
import sys
import scipy
from math import pi, cos, sin
import matplotlib.pyplot as plt


def ERF(x: float, x0: float, w: float) -> float:
    """Calculate a smooth step function based on the error function (erf).

    This function provides a smooth transition from 1.0 to 0.0 (for negative w)
    or 0.0 to 1.0 (for positive w), centered around x0 with a transition
    width related to w.

    :param x: The input value.
    :param x0: The center of the transition.
    :param w: The width of the transition. A negative value inverts the step.
    :return: The value of the smooth step function, ranging from 0.0 to 1.0.

    DocTests:
    >>> ERF(1000.0, 1000.0, 10.0)
    0.5
    >>> round(ERF(990.0, 1000.0, 10.0), 5)
    0.07865
    >>> round(ERF(1010.0, 1000.0, 10.0), 5)
    0.92135
    >>> ERF(1000.0, 1000.0, -10.0)
    0.5
    >>> round(ERF(990.0, 1000.0, -10.0), 5)
    0.92135
    """
    # A smooth step function using the error function (erf).
    return 0.5 * (scipy.special.erf((x - x0) / w) + 1.0)

t_0 = 0
t_final = 2000
num_data_points = 6578
deltat = (t_final - t_0) / num_data_points
initial_seperation = 10
orbital_period = 225
omega = 2 * pi / orbital_period

bh_data = [[],[],[],[]]

with open("bh_synthetic.csv", "w", encoding="utf-8") as file:
    outstr = "time,BH1x,BH1y,BH1z,BH2x,BH2y,BH2z\n"
    file.write(outstr)
    for i in range(num_data_points):
        time = t_0 + deltat * i
        # The orbital separation decays over time, modeled by the ERF function.
        orbital_separation = ERF(time, 1000, -10)
        # BH1 coordinates
        BH1x = initial_seperation / 2 * orbital_separation * cos(omega * time) * (1 - i / num_data_points)
        BH1y = initial_seperation / 2 * orbital_separation * sin(omega * time) * (1 - i / num_data_points)
        BH1z = 0
        # BH2 coordinates
        BH2x = -initial_seperation / 2 * orbital_separation * cos(omega * time) * (1 - i / num_data_points)
        BH2y = -initial_seperation / 2 * orbital_separation * sin(omega * time) * (1 - i / num_data_points)
        BH2z = 0
        bh_data[0].append(BH1x)
        bh_data[1].append(BH1y)
        bh_data[2].append(BH2x)
        bh_data[3].append(BH2y)
        # Format the data as a CSV string.
        outstr = f"{time},{BH1x},{BH1y},{BH1z},{BH2x},{BH2y},{BH2z}\n"
        file.write(outstr)

plt.figure(figsize=(10, 6))
plt.scatter(bh_data[0], bh_data[1], label = "Black Hole 1")
plt.scatter(bh_data[2], bh_data[3], label = "Black Hole 2")
plt.xlabel("X-position")
plt.ylabel("Y-position")
plt.title("Black Hole Trajectories")
plt.legend()
plt.grid(True)

plt.show()

if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
