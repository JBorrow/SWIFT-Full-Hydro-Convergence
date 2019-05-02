"""
Makes a plot for all quantities as a function of particle number.
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    plt.style.use("mnras_durham")
except:
    pass


particle_properties = [
    "density",
    "internal_energy",
    "velocities",
    "pressure"
]

particle_names = [
    r"Density $\rho$",
    r"Internal Energy $u$",
    r"Velocity $|v|$",
    r"Pressure $P$"
]

scheme_identifiers = [
    "gadget2",
    "minimal",
    "pressure-energy",
    "anarchy-pu",
    "gizmo-mfm",
    "gizmo-mfv"
]

scheme_names = [
    "Density-Entropy",
    "Density-Energy",
    "Pressure-Energy",
    "ANARCHY-PU",
    "GIZMO-MFM",
    "GIZMO-MFV"
]

def load_data(filename):
    with open(filename, "r") as handle:
        data = dict(yaml.load(handle))

    return data

def extract_from_data(data, scheme, kernel, property, norm):
    """
    Returns two lists, the number of particles, and the <n> norm for that number of
    particles, for a given scheme.
    """

    norms = []
    num_parts = []

    for key, value in data.items():
        try:
            # All datasets may not exist.
            norms.append(
                value[kernel][scheme][property][norm-1]
            )
            num_parts.append(key**3)

        except KeyError:
            pass

    return num_parts, norms


def filename(args):
    """
    Figures out what the filename should be.
    """

    if args.output == "DEFAULT":
        return f"{args.kernel}_L{args.norm}.pdf"
    else:
        return args.output


def make_plot(args):
    """
    Creates the plot based on the arguments.
    """

    data = load_data(args.input)

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.7, 6.7))

    ax = ax.flatten()

    for a, pprop, pname in zip(ax, particle_properties, particle_names):
        a.set_xscale("log", basex=2)
        a.set_yscale("log", basey=2)

        a.set_xlabel("Particle number")
        a.set_ylabel(f"L{args.norm} Norm for {pname}")

        for (c, sid), sname in zip(enumerate(scheme_identifiers), scheme_names) :
            this_data = extract_from_data(
                data,
                sid,
                args.kernel,
                pprop,
                args.norm
            )

            a.plot(*this_data, label=sname, ms=4, color=f"C{c}")

    ax[-1].legend()

    fig.tight_layout()
    fig.savefig(filename(args))

    return
    
    





if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Plots convergence as a function of particle number."
    )

    parser.add_argument(
        "-k",
        "--kernel",
        help="Kernel to use. Default: cubic-spline",
        required=False,
        type=str,
        default="cubic-spline"
    )

    parser.add_argument(
        "-n",
        "--norm",
        help="Norm to use. Default: 1",
        required=False,
        type=int,
        default=1
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename. Default: <kernel>_L<norm>.pdf",
        required=False,
        type=str,
        default="DEFAULT"
    )

    parser.add_argument(
        '-i',
        "--input",
        help="Input filename. Default: norms.yml",
        required=False,
        type=str,
        default="norms.yml"
    )

    args = parser.parse_args()

    make_plot(args)

    exit(0)
