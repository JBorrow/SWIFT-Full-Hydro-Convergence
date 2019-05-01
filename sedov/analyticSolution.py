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

from numpy import (
    arange,
    zeros,
    log,
    ones,
    array,
    float64,
    outer,
    exp,
    pi,
    power,
    inner,
    diff,
    insert,
    size,
)
from scipy.special import gamma as Gamma


def analytic(
    time,
    rho_0=1.0,  # Background Density
    P_0=1.0e-6,  # Background Pressure
    E_0=1.0,  # Energy of the explosion
    gas_gamma=5.0 / 3.0,  # Gas polytropic index
):
    t = time

    def calc_a(g, nu=3):
        """ 
        exponents of the polynomials of the sedov solution
        g - the polytropic gamma
        nu - the dimension
        """
        a = [0] * 8

        a[0] = 2.0 / (nu + 2)
        a[2] = (1 - g) / (2 * (g - 1) + nu)
        a[3] = nu / (2 * (g - 1) + nu)
        a[5] = 2 / (g - 2)
        a[6] = g / (2 * (g - 1) + nu)

        a[1] = (((nu + 2) * g) / (2.0 + nu * (g - 1.0))) * (
            (2.0 * nu * (2.0 - g)) / (g * (nu + 2.0) ** 2) - a[2]
        )
        a[4] = a[1] * (nu + 2) / (2 - g)
        a[7] = (2 + nu * (g - 1)) * a[1] / (nu * (2 - g))
        return a

    def calc_beta(v, g, nu=3):
        """ 
        beta values for the sedov solution (coefficients of the polynomials of the similarity variables) 
        v - the similarity variable
        g - the polytropic gamma
        nu- the dimension
        """

        beta = (
            (nu + 2)
            * (g + 1)
            * array(
                (
                    0.25,
                    (g / (g - 1)) * 0.5,
                    -(2 + nu * (g - 1))
                    / 2.0
                    / ((nu + 2) * (g + 1) - 2 * (2 + nu * (g - 1))),
                    -0.5 / (g - 1),
                ),
                dtype=float64,
            )
        )

        beta = outer(beta, v)

        beta += (g + 1) * array(
            (
                0.0,
                -1.0 / (g - 1),
                (nu + 2) / ((nu + 2) * (g + 1) - 2.0 * (2 + nu * (g - 1))),
                1.0 / (g - 1),
            ),
            dtype=float64,
        ).reshape((4, 1))

        return beta

    def sedov(t, E0, rho0, g, n=1000, nu=3):
        """ 
        solve the sedov problem
        t - the time
        E0 - the initial energy
        rho0 - the initial density
        n - number of points (10000)
        nu - the dimension
        g - the polytropic gas gamma
        """
        # the similarity variable
        v_min = 2.0 / ((nu + 2) * g)
        v_max = 4.0 / ((nu + 2) * (g + 1))

        v = v_min + arange(n) * (v_max - v_min) / (n - 1.0)

        a = calc_a(g, nu)
        beta = calc_beta(v, g=g, nu=nu)
        lbeta = log(beta)

        r = exp(-a[0] * lbeta[0] - a[2] * lbeta[1] - a[1] * lbeta[2])
        rho = ((g + 1.0) / (g - 1.0)) * exp(
            a[3] * lbeta[1] + a[5] * lbeta[3] + a[4] * lbeta[2]
        )
        p = exp(
            nu * a[0] * lbeta[0] + (a[5] + 1) * lbeta[3] + (a[4] - 2 * a[1]) * lbeta[2]
        )
        u = beta[0] * r * 4.0 / ((g + 1) * (nu + 2))
        p *= 8.0 / ((g + 1) * (nu + 2) * (nu + 2))

        # we have to take extra care at v=v_min, since this can be a special point.
        # It is not a singularity, however, the gradients of our variables (wrt v) are.
        # r -> 0, u -> 0, rho -> 0, p-> constant

        u[0] = 0.0
        rho[0] = 0.0
        r[0] = 0.0
        p[0] = p[1]

        # volume of an n-sphere
        vol = (pi ** (nu / 2.0) / Gamma(nu / 2.0 + 1)) * power(r, nu)

        # note we choose to evaluate the integral in this way because the
        # volumes of the first few elements (i.e near v=vmin) are shrinking
        # very slowly, so we dramatically improve the error convergence by
        # finding the volumes exactly. This is most important for the
        # pressure integral, as this is on the order of the volume.

        # (dimensionless) energy of the model solution
        de = rho * u * u * 0.5 + p / (g - 1)
        # integrate (trapezium rule)
        q = inner(de[1:] + de[:-1], diff(vol)) * 0.5

        # the factor to convert to this particular problem
        fac = (q * (t ** nu) * rho0 / E0) ** (-1.0 / (nu + 2))

        # shock speed
        shock_speed = fac * (2.0 / (nu + 2))
        rho_s = ((g + 1) / (g - 1)) * rho0
        r_s = shock_speed * t * (nu + 2) / 2.0
        p_s = (2.0 * rho0 * shock_speed * shock_speed) / (g + 1)
        u_s = (2.0 * shock_speed) / (g + 1)

        r *= fac * t
        u *= fac
        p *= fac * fac * rho0
        rho *= rho0
        return r, p, rho, u, r_s, p_s, rho_s, u_s, shock_speed

    # The main properties of the solution
    r_s, P_s, rho_s, v_s, r_shock, _, _, _, _ = sedov(
        time, E_0, rho_0, gas_gamma, 1000, 3
    )

    # Append points for after the shock
    r_s = insert(r_s, size(r_s), [r_shock, r_shock * 1.5])
    rho_s = insert(rho_s, size(rho_s), [rho_0, rho_0])
    P_s = insert(P_s, size(P_s), [P_0, P_0])
    v_s = insert(v_s, size(v_s), [0, 0])

    # Additional arrays
    u_s = P_s / (rho_s * (gas_gamma - 1.0))  # internal energy
    s_s = P_s / rho_s ** gas_gamma  # entropic function

    return dict(x=r_s, P=P_s, v_r=v_s, u=u_s, rho=rho_s, S=s_s, r_shock=r_shock)
