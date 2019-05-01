###############################################################################
# This file is part of the ANARCHY paper.
# Copyright (c) 2016 Matthieu Schaller (matthieu.schaller@durham.ac.uk)
#               2019 Josh Borrow (joshua.boorrow@durham.ac.uk)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import sys
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from swiftsimio import load
from analyticSolution import analytic

snap = int(sys.argv[1])

sim = load(f"sedov_{snap:04d}.hdf5")

# Set up plotting stuff
try:
    plt.style.use("mnras_durham")
except:
    rcParams = {
        "font.serif": ["STIX", "Times New Roman", "Times"],
        "font.family": ["serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
    }
    plt.rcParams.update(rcParams)


# See analyticSolution for params.

# Read in the "solution" data and calculate those that don't exist.

ref = analytic(time=sim.metadata.time.value)
r_shock = ref.pop("r_shock")


def get_data_dump(metadata):
    """
    Gets a big data dump from the SWIFT metadata
    """

    try:
        viscosity = metadata.viscosity_info
    except:
        viscosity = "No info"

    try:
        diffusion = metadata.diffusion_info
    except:
        diffusion = "No info"

    output = (
        "$\\bf{SWIFT}$\n"
        + metadata.code_info
        + "\n\n"
        + "$\\bf{Compiler}$\n"
        + metadata.compiler_info
        + "\n\n"
        + "$\\bf{Hydrodynamics}$\n"
        + metadata.hydro_info
        + "\n\n"
        + "$\\bf{Viscosity}$\n"
        + viscosity
        + "\n\n"
        + "$\\bf{Diffusion}$\n"
        + diffusion
    )

    return output


# Read the simulation data
boxSize = sim.metadata.boxsize[0].value

x = sim.gas.coordinates.value[:, 0] - boxSize / 2
y = sim.gas.coordinates.value[:, 1] - boxSize / 2
z = sim.gas.coordinates.value[:, 2] - boxSize / 2
r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
vel = sim.gas.velocities.value
v_r = (x * vel[:, 0] + y * vel[:, 1] + z * vel[:, 2]) / r

data = dict(
    x=r,
    v_r=v_r,
    u=sim.gas.internal_energy.value,
    S=sim.gas.entropy.value,
    P=sim.gas.pressure.value,
    rho=sim.gas.density.value,
)

# Try to add on the viscosity and diffusion.
try:
    data["visc"] = sim.gas.viscosity.value
except:
    pass

try:
    data["diff"] = 100.0 * sim.gas.diffusion.value
except:
    pass

# Bin the data
x_bin_edge = np.linspace(0.15, 1.3 * r_shock, 50)
x_bin = 0.5 * (x_bin_edge[1:] + x_bin_edge[:-1])
binned = {
    k: stats.binned_statistic(data["x"], v, statistic="mean", bins=x_bin_edge)[0]
    for k, v in data.items()
}
square_binned = {
    k: stats.binned_statistic(data["x"], v ** 2, statistic="mean", bins=x_bin_edge)[0]
    for k, v in data.items()
}
sigma = {
    k: np.sqrt(v2 - v ** 2)
    for k, v2, v in zip(binned.keys(), square_binned.values(), binned.values())
}

# We only want to plot this for the region that we actually have data for, hence the masking.
mask = np.logical_and(ref["x"] < np.max(data["x"]), ref["x"] > np.min(data["x"]))
ref = {k: v[mask] for k, v in ref.items()}

# Now we can do the plotting.
fig, ax = plt.subplots(2, 3, figsize=(6.974, 6.974 * (2.0 / 3.0)))
ax = ax.flatten()

# These are stored in priority order
plot = dict(
    v_r="Radial Velocity ($v_r$)",
    u="Internal Energy ($u$)",
    rho=r"Density ($\rho$)",
    visc=r"Viscosity Coefficient ($\alpha_V$)",
    diff=r"100$\times$ Diffusion Coefficient ($\alpha_D$)",
    P="Pressure ($P$)",
    S="Entropy ($A$)",
)

log = dict(
    v_r=False, v_phi=False, u=False, S=False, P=False, rho=False, visc=False, diff=False
)
ylim = dict(
    diff=(0.0, None), visc=(0.0, None), u=(0, 100), v_r=(-0.05, None), rho=(-0.05, None)
)

current_axis = 0

for key, label in plot.items():
    if current_axis > 4:
        break
    else:
        axis = ax[current_axis]

    try:
        if log[key]:
            axis.semilogy()

        # Raw data
        axis.plot(
            data["x"],
            data[key],
            ".",
            color="C1",
            ms=0.5,
            alpha=0.5,
            markeredgecolor="none",
            rasterized=True,
            zorder=0,
        )
        # Binned data
        axis.errorbar(
            x_bin,
            binned[key],
            yerr=sigma[key],
            fmt=".",
            ms=3.0,
            color="C3",
            lw=0.5,
            zorder=2,
        )

        # Exact solution
        try:
            axis.plot(ref["x"], ref[key], c="C0", ls="dashed", zorder=1, lw=1)
        except KeyError:
            # No solution :(
            pass

        axis.set_xlabel("Radius ($r$)", labelpad=0)
        axis.set_ylabel(label, labelpad=0)

        axis.set_xlim(0.15, 1.3 * r_shock)

        try:
            axis.set_ylim(*ylim[key])
        except KeyError:
            # No worries pal
            pass

        current_axis += 1
    except KeyError:
        # Mustn't have that data!
        continue


info_axis = ax[-1]

info = get_data_dump(sim.metadata)

info_axis.text(
    0.5, 0.45, info, ha="center", va="center", fontsize=5, transform=info_axis.transAxes
)

info_axis.axis("off")


fig.tight_layout(pad=0.5)
fig.savefig("SedovBlast.pdf", dpi=300)
