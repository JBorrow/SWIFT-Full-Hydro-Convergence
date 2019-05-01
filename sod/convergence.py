"""
Creates a convergence plot for the SodShock.

Describe the data that you have provided (i.e. all of the
sodshock_0001.hdf5 files in the directories, and their
associated numbers of threads for the timesteps files)
in the data.yml file.
"""


from analyticSolution import analytic

import yaml
import numpy as np
from scipy.interpolate import interp1d

from swiftsimio import load, SWIFTDataset

from typing import List, Dict, Callable, Tuple


def read_metadata(filename: str = "data.yml"):
    """
    Reads in the metadata from the data.yml file.
    """

    with open(filename, "r") as handle:
        data = dict(yaml.load(handle))

    return data["number_of_particles"], data["threads"]


def load_particle_data(number_of_particles: List) -> List[SWIFTDataset]:
    """
    Loads the particle data (as swiftsimio arrays).
    """

    return [load(f"{x}/sodShock_0001.hdf5") for x in number_of_particles]


def compute_analytic_solution(t: float) -> Tuple[Dict[str, Callable], List[float]]:
    """
    Computes the analytical solution at t, as a set of interpolated
    routines.
    """

    analytic_data, x_positions_for_convergence = analytic(t, return_x_positions_for_convergence=True)

    x = analytic_data["x"]

    smoothed = {
        k: interp1d(x, v, fill_value="extrapolate") for k, v in analytic_data.items() if k != "x"
    }

    return smoothed, x_positions_for_convergence


def chi_squared(observed, expected):
    """
    Computes the chi-squared statistic for a set of observed
    and expected values.
    """

    diff = observed - expected

    return np.sum((diff * diff) / (expected * expected))


if __name__ == "__main__":
    """
    Print out a number of statistics.
    """

    dx = 0.05

    number_of_particles, threads = read_metadata()
    particle_data = load_particle_data(number_of_particles)
    smoothed_analytic_solution, x_pos = compute_analytic_solution(
        t=particle_data[0].metadata.t.value
    )

    print(x_pos)

    observed = [dict(
        v=data.gas.velocities.value[:, 0],
        rho=data.gas.density.value,
        P=data.gas.pressure.value,
        u=data.gas.internal_energy.value,
        S=data.gas.entropy.value,
    ) for data in particle_data]

    for obs, data in zip(observed, particle_data):
        print(f"Dataset with {data.metadata.n_gas} particles\n")

        for point, x in enumerate(x_pos):
            print(f"Statistics at point x_{point+1}{point+2}\n")
            coordinates = data.gas.coordinates.value[:, 0]
            
            mask = np.logical_and(coordinates > x - dx, coordinates < x + dx)

            masked_coordinates = coordinates[mask]

            for statistic in obs.keys():
                value = chi_squared(
                    obs[statistic][mask], smoothed_analytic_solution[statistic](
                        masked_coordinates,
                    )
                )

                reduced = value / data.metadata.n_gas

                print(f"Chi Squared Statistic for {statistic} = {value:e}")
                print(f"Reduced Chi Squared for {statistic} = {reduced:e}")
            
            print("\n")

        print("\n\n")

    exit(0)
