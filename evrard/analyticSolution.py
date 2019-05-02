"""
Computes a smooth analytic solution for the Evrard collapse.
"""

import numpy as np
from scipy.interpolate import interp1d


def analytic(
        gas_gamma=5./3.
    ):
    reference = np.loadtxt("evrardCollapse3D_exact.txt")

    return dict(
        x=reference[:, 0],
        v=-reference[:, 2],
        rho=reference[:, 1],
        P=reference[:, 3],
        u=reference[:, 3] / reference[:, 1] / (gas_gamma - 1.0),
        S=reference[:, 3] / reference[:, 1] ** gas_gamma,
    )

def smooth_analytic(
        gas_gamma=5./3.
    ):

    reference = analytic(gas_gamma)

    smooth_reference = {}

    for key, value in reference.items():
        if key != "x":
            smooth_reference[key] = interp1d(
                reference["x"], value, fill_value="extrapolate"
            )
        else:
            smooth_reference[key] = reference[key]

    return smooth_reference

def smooth_analytic_same_api_as_swiftsimio(
        gas_gamma=5./3.
    ):

    smooth_reference = smooth_analytic(gas_gamma)

    output = dict(
        velocities=smooth_reference["v"],
        density=smooth_reference["rho"],
        pressure=smooth_reference["P"],
        internal_energy=smooth_reference["u"],
    )

    return smooth_reference["x"], output


