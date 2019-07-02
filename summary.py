"""
Creates a summary plot.
"""

import matplotlib
import yaml

# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# This bit is a bit nasty, get the analytic solutions
from gresho.analyticSolution import analytic as analytic_gresho
from evrard.analyticSolution import analytic as analytic_evrard
from sedov.analyticSolution import analytic as analytic_sedov

analytic_solutions = dict(
    sedov=analytic_sedov, evrard=analytic_evrard, gresho=analytic_gresho
)

from swiftsimio import load

quantities = ["P", "P", "v_phi"]
simulations = {
    "sedov": "Sedov-Taylor Blastwave",
    "evrard": "Evrard Collapse",
    "gresho": "Gresho-Chan Vortex",
}
log = dict(sedov=False, evrard=True, gresho=False)
xlims = dict(sedov=[0.15, 0.45], evrard=[-2, 0], gresho=[0.0, 0.6])
ylims = dict(sedov=[-0.325, 6.5], evrard=[-4, 4], gresho=[-0.05, 1.05])
n_bins = 25
xlabels = dict(sedov="Radius $r$", evrard="Radius $\\log_{10} r$", gresho="Radius $r$")
particle_counts = {"sedov": 32, "evrard": 32, "gresho": 32}
schemes = {"gizmo-mfm": "SPH-ALE", "anarchy-du": "Modern SPH"}
snapshots = dict(sedov=5, evrard=8, gresho=10)
kernel = "wendland-C2"

# TODO: Fix me later to use the real SPHERIC colours etc.
plt.style.use("spheric_newsletter")

# First, set up our figure
fig, ax = plt.subplots(2, 3, figsize=(6, 4), sharex="col", sharey="col")


def get_quantity_name(quantity_handle):
    if quantity_handle == "P":
        return "$P$"
    elif quantity_handle == "v_phi":
        return "$v_\\phi$"
    else:
        return ""


def get_quantity_property(quantity_handle):
    if quantity_handle == "P":
        return "pressure"
    elif quantity_handle == "v_phi":
        return "velocities_phi"
    else:
        return ""


def get_L1_norm(quantity_handle, simulation_handle, scheme_handle):
    particle_count = particle_counts[simulation_handle]
    quantity_property = get_quantity_property(quantity_handle)

    with open(f"{simulation_handle}/norms_du.yml", "r") as handle:
        data = yaml.load(handle)
        L1 = data[particle_count][kernel][scheme_handle][quantity_property][0]

    return f"L1: {L1:2.2f}"


def get_wallclock(quantity_handle, simulation_handle, scheme_handle):
    particle_count = particle_counts[simulation_handle]

    with open(f"{simulation_handle}/runtimes.yml", "r") as handle:
        data = yaml.load(handle)
        t = data[particle_count][kernel][scheme_handle] / (1000 * 1000)

    return f"{t:2.2f} s"


def get_filename(simulation_handle, scheme_handle):
    snapshot = snapshots[simulation_handle]
    return (
        f"{simulation_handle}/{scheme_handle}/{simulation_handle}_{snapshot:04d}.hdf5"
    )


def get_scatter_data(quantity_handle, simulation_handle, scheme_handle, wrap_function):
    quantity_property = get_quantity_property(quantity_handle)
    sim = load(get_filename(simulation_handle, scheme_handle))

    boxSize = sim.metadata.boxsize[0].value

    x = sim.gas.coordinates.value[:, 0] - boxSize / 2
    y = sim.gas.coordinates.value[:, 1] - boxSize / 2
    z = sim.gas.coordinates.value[:, 2] - boxSize / 2
    # For the gresho we just want r as the cylidrical
    if simulation_handle == "gresho":
        z = 0.0
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    try:
        property = getattr(sim.gas, quantity_property).value
    except AttributeError:
        if quantity_property == "velocities_phi":
            vel = sim.gas.velocities.value
            property = (-y * vel[:, 0] + x * vel[:, 1]) / r
        else:
            raise AttributeError(f"Unable to find {quantity_property} in {sim}")

    return wrap_function(r), wrap_function(property)


def get_binned_data(quantity_handle, simulation_handle, scheme_handle, wrap_function):
    x, y = get_scatter_data(
        quantity_handle, simulation_handle, scheme_handle, wrap_function
    )

    x_bin_edge = np.linspace(*get_xlim(quantity_handle, simulation_handle), n_bins)
    x_bin = 0.5 * (x_bin_edge[1:] + x_bin_edge[:-1])

    binned = stats.binned_statistic(x, y, statistic="mean", bins=x_bin_edge)[0]
    square_binned = stats.binned_statistic(x, y * y, statistic="mean", bins=x_bin_edge)[
        0
    ]
    sigma = np.sqrt(square_binned - binned * binned)

    return x_bin, binned, sigma


def get_analytic_data(quantity_handle, simulation_handle, scheme_handle, wrap_function):
    sim = load(get_filename(simulation_handle, scheme_handle))
    t = sim.metadata.time.value
    try:
        ref = analytic_solutions[simulation_handle](time=t)
    except TypeError:
        ref = analytic_solutions[simulation_handle]()

    return wrap_function(ref["x"]), wrap_function(ref[quantity_handle])


def get_xlim(quantity_handle, simulation_handle):
    return xlims[simulation_handle]


def get_ylim(quantity_handle, simulation_handle):
    return ylims[simulation_handle]


for (scheme_handle, scheme_name), axis_row in zip(schemes.items(), ax):
    for ((simulation_handle, simulation_name), quantity_handle, logy, axis) in zip(
        simulations.items(), quantities, log.values(), axis_row
    ):
        # Set the titles along the top row
        if scheme_handle == list(schemes.keys())[0]:
            axis.set_title(simulation_name)
        else:
            axis.set_xlabel(xlabels[simulation_handle])

        # Set up the fixed text for each plot
        offset = 0.05

        if logy:
            log_text = "$\\log_{10}$"
            wrap_function = np.log10
        else:
            log_text = ""
            wrap_function = lambda x: x

        axis.text(
            offset,
            offset,
            scheme_name,
            ha="left",
            va="bottom",
            transform=axis.transAxes,
        )

        axis.text(
            0.5,
            1.0 - offset,
            f"{log_text}{get_quantity_name(quantity_handle)}",
            ha="center",
            va="top",
            transform=axis.transAxes,
        )

        axis.text(
            offset,
            1.0 - offset,
            get_L1_norm(quantity_handle, simulation_handle, scheme_handle),
            ha="left",
            va="top",
            transform=axis.transAxes,
        )

        axis.text(
            1.0 - offset,
            1.0 - offset,
            get_wallclock(quantity_handle, simulation_handle, scheme_handle),
            ha="right",
            va="top",
            transform=axis.transAxes,
        )

        # Now we need to plot the actual data

        axis.plot(
            *get_scatter_data(
                quantity_handle, simulation_handle, scheme_handle, wrap_function
            ),
            ".",
            color="C1",
            ms=0.5,
            alpha=0.7,
            markeredgecolor="none",
            rasterized=True,
            zorder=1,
        )

        # Binned data
        axis.errorbar(
            *get_binned_data(
                quantity_handle, simulation_handle, scheme_handle, wrap_function
            ),
            fmt=".",
            ms=3.0,
            color="C3",
            lw=0.5,
            zorder=3,
        )

        # Exact solution
        axis.plot(
            *get_analytic_data(
                quantity_handle, simulation_handle, scheme_handle, wrap_function
            ),
            c="C0",
            ls="dashed",
            zorder=2,
            lw=1,
        )

        axis.set_xlim(get_xlim(quantity_handle, simulation_handle))
        axis.set_ylim(get_ylim(quantity_handle, simulation_handle))


fig.tight_layout(h_pad=0.1, w_pad=0.3)
fig.savefig("spheric_newsletter.pdf")
