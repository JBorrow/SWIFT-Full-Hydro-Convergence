"""
Makes a plot for all quantities as a function of particle number.
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml

from scipy.optimize import curve_fit

import matplotlib.ticker as mticker

try:
    plt.style.use("spheric_durham")
except:
    pass


particle_properties = ["velocities_phi"]

particle_names = [r"Azimuthal Velocity $v_\phi$"]

scheme_identifiers = [
    "gadget2",
    "minimal",
    "pressure-energy",
    "anarchy-pu",
    "gizmo-mfm",
    "gizmo-mfv",
]

scheme_names = [
    "",
    "Density-Energy",
    "Pressure-Energy",
    "ANARCHY-PU",
    "SPH-ALE, FM",
    "",
]

scheme_alphas = [0.5, 1.0, 1.0, 1.0, 1.0, 0.5]

scheme_colours = [0, 0, 1, 3, 2, 2]

highlight = 32


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
            norms.append(value[kernel][scheme][property][norm - 1])
            runtime.append(runtimes[key][kernel][scheme])

        except KeyError:
            pass

    return runtime, norms


def filename(args):
    """
    Figures out what the filename should be.
    """

    if args.output == "DEFAULT":
        return f"{args.kernel}_L{args.norm}_time_individual.pdf"
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
    new_y = 2 ** (linear(np.log2(new_x), *popt))

    return new_x, new_y


def make_plot(args):
    """
    Creates the plot based on the arguments.
    """

    data = load_data(args.input)
    runtimes = load_data(args.runtime)

    which_highlight = np.where(np.array([x for x in data.keys()]) == highlight)[0][0]

    fig, ax = plt.subplots(1)

    ax = [ax]

    for a, pprop, pname in zip(ax, particle_properties, particle_names):
        a.set_xscale("log", basex=10)
        a.set_yscale("log", basey=10)
        a.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        a.yaxis.set_major_formatter(mticker.ScalarFormatter())
        a.yaxis.set_minor_locator(mticker.AutoLocator())
        a.yaxis.set_major_locator(mticker.AutoLocator())

        a.set_xlabel("Runtime [s]")
        a.set_ylabel(f"L{args.norm} Norm for {pname}")

        for sid, sname, alpha, c in zip(
            scheme_identifiers, scheme_names, scheme_alphas, scheme_colours
        ):
            this_data = extract_from_data(
                data, runtimes, sid, args.kernel, pprop, args.norm
            )

            fitted_data = fitted_line(*this_data)

            a.scatter(*this_data, color=f"C{c}", s=2, alpha=alpha, zorder=alpha)
            a.scatter(
                this_data[0][which_highlight],
                this_data[1][which_highlight],
                marker="*",
                color=f"C{c}",
                s=30,
                alpha=alpha,
                zorder=alpha + 10,
                edgecolor="white",
                linewidth=0.2,
            )
            a.plot(*fitted_data, color=f"C{c}", label=sname, alpha=alpha, zorder=alpha)

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
        default="cubic-spline",
    )

    parser.add_argument(
        "-n",
        "--norm",
        help="Norm to use. Default: 1",
        required=False,
        type=int,
        default=1,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename. Default: <kernel>_L<norm>_time.pdf",
        required=False,
        type=str,
        default="DEFAULT",
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Input filename. Default: norms.yml",
        required=False,
        type=str,
        default="norms.yml",
    )

    parser.add_argument(
        "-r",
        "--runtime",
        help="Runtime filename. Default: runtimes.yml",
        required=False,
        type=str,
        default="runtimes.yml",
    )

    args = parser.parse_args()

    make_plot(args)

    exit(0)
