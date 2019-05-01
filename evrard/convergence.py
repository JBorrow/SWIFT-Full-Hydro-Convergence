"""
Creates a convergence plot for the Evrard collapse.
"""

from analyticSolution import smooth_analytic_same_api_as_swiftsimio

import yaml
import numpy as np
from scipy.interpolate import interp1d

from swiftsimio import load, SWIFTDataset

from typing import List, Dict, Callable, Tuple

from tqdm import tqdm

radius_range_to_calculate_in = [5e-2, 0.9]


def read_metadata(filename: str = "data.yml"):
    """
    Reads in the metadata from the data.yml file.
    """

    with open(filename, "r") as handle:
        data = dict(yaml.load(handle))

    return data["number_of_particles"], data["kernels"], data["schemes"], data["threads"]


def L1_norm(observed, expected):
    """
    Returns the L1 norm per particle.
    """

    norm = np.sum(abs(observed - expected)) / len(observed)

    return norm


def L2_norm(observed, expected):
    """
    Returns the L2 norm per particle.
    """

    norm = np.sum((observed - expected)**2) / len(observed)

    return norm


def load_safe(filename):
    """
    Loads (saftely) the data. If not available, returns None.
    """

    try:
        return load(filename)
    except:
        return None


def load_particle_data(number_of_particles: List, schemes: List, kernels: List) -> Dict[Dict[Dict[SWIFTDataset]]]:
    """
    Loads the particle data (as swiftsimio objects), i.e. this does not yet actually read the data.
    """

    return {np:
        {
            kernel: 
            {scheme: 
                load(f"{np}/{kernel}/{scheme}/evrard_0008.hdf5")
                for scheme in schemes
            } for kernel in kernels
        } for np in number_of_particles
    }


def calculate_norms(particle_data: dict):
    """
    Calculates both the L1 and L2 norms for all particle quantities that we have
    available.
    """

    x, analytic = smooth_analytic_same_api_as_swiftsimio(gas_gamma=5./3.)

    number_of_particles = list(particle_data.keys())
    kernels = list(particle_data[number_of_particles[0]].keys())
    schemes = list(particle_data[number_of_particles[0]][kernels[0]].keys())

    output = {}

    for np in tqdm(number_of_particles):
        output[np] = {}
        for kernel in kernels:
            output[np][kernel] = {}
            for scheme in schemes:
                this_data = particle_data[np][kernel][scheme]

                this_output = {}

                boxsize = this_data.metadata.boxsize[0].value
                coords = this_data.gas.coordinates.value - 0.5 * boxsize
                radii = np.sqrt(np.sum(coords*coords, axis=0))

                # Now need to mask the data to lie within the allowed region.
                mask = np.logical_and(radii > radius_range_to_calculate_in[0], radii < radius_range_to_calculate_in[1])
                coords = coords[mask]

                for property in analytic.keys():
                    if property == "velocity":
                        v = this_data.gas.velocities.value
                        v = np.sqrt(np.sum(v*v, axis=0))

                        analytic_velocity = analytic["velocity"](coords)

                        L1 = L1_norm(v, analytic_velocity)
                        L2 = L2_norm(v, analytic_velocity)

                        this_output["velocities"] = (L1, L2)
                    else:
                        y = getattr(this_data.gas, property).value
                        
                        analytic_y = analytic[property](coords)

                        L1 = L1_norm(y, analytic_y)
                        L2 = L2_norm(y, analytic_y)

                output[np][kernel][scheme] = this_output

    return output






