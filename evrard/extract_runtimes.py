"""
Extracts the runtimes from the final runs.
"""

import numpy as np
import yaml

from tqdm import tqdm

from convergence import read_metadata

def get_runtime_from_timesteps_file(filename):
    """
    Gets the runtime from a given file. If this fails, it returns -1.
    """

    try:
        return float(np.sum(np.genfromtxt(filename, usecols=[-2]))) * 1e3
    except:
        return -1.0


def get_runtime_for_all(meta):
    """
    Gets the runtime for all runs.
    """

    number_of_particles = meta.num_part
    schemes = meta.schemes
    kernels = meta.kernels

    return {
        num_part: {
            kernel: {
                scheme: get_runtime_from_timesteps_file(
                    f"{num_part}/{kernel}/{scheme}/timesteps_{meta.threads[0]}"
                )
                for scheme in schemes
            }
            for kernel in kernels
        }
        for num_part in tqdm(number_of_particles)
    }

    
if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Script to grab the runtime for all simulations, for the Evrard Collapse."
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
        help="Output filename location. Default: runtimes.yml",
        type=str,
        required=False,
        default="runtimes.yml",
    )

    args = parser.parse_args()

    meta = read_metadata(args.metadata)
    data = get_runtime_for_all(meta)

    with open(args.output, "w") as handle:
        yaml.dump(data, handle)

    exit(0)
