"""
Makes a plot for all quantities as a function of particle number.
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml

from scipy.optimize import curve_fit

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

def extract_from_data(data, runtimes, scheme, kernel, property, norm):
    """
    Returns two lists, the number of particles, and the <n> norm for that number of
    particles, for a given scheme.
    """

    norms = []
    runtime = []

    for key, value in data.items():
        try:
            # All datasets may not exist.
            norms.append(
                value[kernel][scheme][property][norm-1]
            )
            runtime.append(
                runtimes[key][kernel][scheme]
            )

        except KeyError:
            pass

    return runtime, norms


def filename(args):
    """
    Figures out what the filename should be.
    """

    if args.output == "DEFAULT":
        return f"{args.kernel}_L{args.norm}_time.pdf"
    else:
        return args.output


def linear(x, m, c):
    return m * x + c


def fitted_line(x, y):
    """
    Return x, y for a fitted line for the input x, y, but in log-2 space.
    """

    x_log_2 = np.log2(np.array(x))
    y_log_2 = np.log2(np.array(y))

    popt, pcov = curve_fit(linear, x_log_2, y_log_2)

    new_x = np.linspace(x[0], x[-1], 100)
    new_y = 2**(linear(np.log2(new_x), *popt))

    return new_x, new_y


def make_plot(args):
    """
    Creates the plot based on the arguments.
    """

    data = load_data(args.input)
    runtimes = load_data(args.runtime)

    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(6.7, 6.7))

    ax = ax.flatten()

    for a, pprop, pname in zip(ax, particle_properties, particle_names):
        a.set_xscale("log", basex=2)
        a.set_yscale("log", basey=2)

        a.set_xlabel("Runtime [s]")
        a.set_ylabel(f"L{args.norm} Norm for {pname}")

        for (c, sid), sname in zip(enumerate(scheme_identifiers), scheme_names):
            this_data = extract_from_data(
                data,
                runtimes,
                sid,
                args.kernel,
                pprop,
                args.norm
            )

            fitted_data = fitted_line(*this_data)

            a.scatter(*this_data, color=f"C{c}", s=2)
            a.plot(*fitted_data, color=f"C{c}", label=sname)

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
        help="Output filename. Default: <kernel>_L<norm>_time.pdf",
        required=False,
        type=str,
        default="DEFAULT"
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Input filename. Default: norms.yml",
        required=False,
        type=str,
        default="norms.yml"
    )

    parser.add_argument(
        "-r",
        "--runtime",
        help="Runtime filename. Default: runtimes.yml",
        required=False,
        type=str,
        default="runtimes.yml"
    )

    args = parser.parse_args()

    make_plot(args)

    exit(0)
