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
from collections import namedtuple

from tqdm import tqdm

dx = 0.05

metadata = namedtuple("Metadata", ["num_part", "kernels", "schemes", "threads"])


def read_metadata(filename: str = "data.yml"):
    """
    Reads in the metadata from the data.yml file.
    """

    with open(filename, "r") as handle:
        data = dict(yaml.load(handle))

    return metadata(
        num_part=data["number_of_particles"],
        kernels=data["kernels"],
        schemes=data["schemes"],
        threads=data["threads"],
    )


def load_safe(filename):
    """
    Loads (saftely) the data. If not available, returns None.
    """

    try:
        return load(filename)
    except:
        return None


def load_particle_data(meta: metadata) -> Dict[str, Dict[str, Dict[str, SWIFTDataset]]]:
    """
    Loads the particle data (as swiftsimio objects), i.e. this does not yet actually read the data.
    """

    number_of_particles = meta.num_part
    schemes = meta.schemes
    kernels = meta.kernels

    return {
        num_part: {
            kernel: {
                scheme: load_safe(f"{num_part}/{kernel}/{scheme}/sodShock_0001.hdf5")
                for scheme in schemes
            }
            for kernel in kernels
        }
        for num_part in number_of_particles
    }


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


def L1_norm(observed, expected):
    """
    Returns the L1 norm per particle.
    """

    norm = np.sum(abs(observed - expected)) / len(observed)

    return float(norm)


def L2_norm(observed, expected):
    """
    Returns the L2 norm per particle.
    """

    norm = np.sum((observed - expected) ** 2) / len(observed)

    return float(norm)


def calculate_norms(particle_data: dict):
    """
    Calculates both the L1 and L2 norms for all particle quantities that we have
    available.
    """


    number_of_particles = list(particle_data.keys())
    kernels = list(particle_data[number_of_particles[0]].keys())
    schemes = list(particle_data[number_of_particles[0]][kernels[0]].keys())

    smoothed_analytic_solution, x_pos = compute_analytic_solution(
        t=particle_data[number_of_particles[0]][kernels[0]][schemes[0]].metadata.t.value
    )

    output = {}

    for num_part in tqdm(number_of_particles):
        output[num_part] = {}
        for kernel in kernels:
            output[num_part][kernel] = {}
            for scheme in schemes:
                this_data = particle_data[num_part][kernel][scheme]

                if this_data == None:
                    continue

                this_output = {}

                observed = dict(
                    v=this_data.gas.velocities.value[:, 0],
                    rho=this_data.gas.density.value,
                    P=this_data.gas.pressure.value,
                    u=this_data.gas.internal_energy.value,
                    S=this_data.gas.entropy.value,
                )


                masked_coordinates = coordinates[mask]

                for statistic in observed.keys():
                    L1 = []
                    L2 = []
                    for point, x in enumerate(x_pos):
                        # Select points close to the discontinuity
                        coordinates = this_data.gas.coordinates.value[:, 0]

                        mask = np.logical_and(
                            coordinates > x - dx, coordinates < x + dx)

                        L1.append(L1_norm(observed[statistic], smoothed_analytic_solution[statistic](
                            masked_coordinates,
                        )))
                        L2.append(L2_norm(observed[statistic], smoothed_analytic_solution[statistic](
                            masked_coordinates,
                        )))

                    this_output[statistic] = [L1, L2]
                    
                output[num_part][kernel][scheme] = this_output

    return output


if __name__ == "__main__":
    """
    Print out a number of statistics.
    """


    number_of_particles, threads = read_metadata()

    particle_data = load_particle_data(number_of_particles)
    smoothed_analytic_solution, x_pos = compute_analytic_solution(
        t=particle_data[16].metadata.t.value
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
