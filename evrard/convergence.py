"""
Creates a convergence plot for the Evrard collapse.
"""

from analyticSolution import smooth_analytic_same_api_as_swiftsimio

import yaml
import numpy as np
from scipy.interpolate import interp1d

from swiftsimio import load, SWIFTDataset

from typing import List, Dict, Callable, Tuple
from collections import namedtuple

from tqdm import tqdm

radius_range_to_calculate_in = [5e-2, 0.9]

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

    norm = np.sum((observed - expected) ** 2) / len(observed)

    return norm


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
                scheme: load_safe(f"{num_part}/{kernel}/{scheme}/evrard_0008.hdf5")
                for scheme in schemes
            }
            for kernel in kernels
        }
        for num_part in number_of_particles
    }


def calculate_norms(particle_data: dict):
    """
    Calculates both the L1 and L2 norms for all particle quantities that we have
    available.
    """

    x, analytic = smooth_analytic_same_api_as_swiftsimio(gas_gamma=5.0 / 3.0)

    number_of_particles = list(particle_data.keys())
    kernels = list(particle_data[number_of_particles[0]].keys())
    schemes = list(particle_data[number_of_particles[0]][kernels[0]].keys())

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

                boxsize = this_data.metadata.boxsize[0].value
                coords = this_data.gas.coordinates.value - 0.5 * boxsize
                radii = np.sqrt(np.sum(coords * coords, axis=1))

                # Now need to mask the data to lie within the allowed region.
                mask = np.logical_and(
                    radii > radius_range_to_calculate_in[0],
                    radii < radius_range_to_calculate_in[1],
                )
                radii = radii[mask]

                for property in analytic.keys():
                    if property == "velocities":
                        v = this_data.gas.velocities.value[mask]
                        v = np.sqrt(np.sum(v * v, axis=1))

                        analytic_velocity = analytic["velocities"](radii)

                        L1 = L1_norm(v, analytic_velocity)
                        L2 = L2_norm(v, analytic_velocity)

                        this_output["velocities"] = (L1, L2)
                    else:
                        y = getattr(this_data.gas, property).value[mask]

                        analytic_y = analytic[property](radii)

                        L1 = L1_norm(y, analytic_y)
                        L2 = L2_norm(y, analytic_y)

                output[num_part][kernel][scheme] = this_output

    return output


if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Script to make convergence DATA, not plot, for the Evrard Collapse."
    )

    parser.add_argument(
        "-m",
        "--metadata",
        help="Location of metadata file. Default: data.yml",
        type=str,
        required=False,
        default="data.yml",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename location. Default: norms.yml",
        type=str,
        required=False,
        default="norms.yml",
    )

    args = parser.parse_args()

    meta = read_metadata(args.metadata)
    particle_data = load_particle_data(meta)
    norms = calculate_norms(particle_data)

    with open(args.output, "w") as handle:
        yaml.dump(norms, handle)

    exit(0)
