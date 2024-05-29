import sympy as sp
from sympy.physics.vector import ReferenceFrame

import sys

sys.path.append("./src")

from exact_operators import (
    compute_all_elasticity,
    compute_all_darcy,
    compute_all_biot,
    make_as_lambda_elasticity,
    make_as_lambda_darcy,
    make_as_lambda_biot,
)


def exact_sol_elasticity(mu, lambda_):
    R = ReferenceFrame("R")
    x, y, _ = R.varlist

    # define the displacement
    u_x = sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y)
    u_y = u_x
    u = sp.Matrix([u_x, u_y, 0])

    # compute the stress and the source terms
    sigma, f_u = compute_all_elasticity(2, u, mu, lambda_, R)

    # lambdify the exact solution
    return make_as_lambda_elasticity(sigma, u, f_u, R)


def exact_sol_darcy():
    R = ReferenceFrame("R")
    x, y, z = R.varlist

    # define the pressure
    p = sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y)

    # define the flux and the source term
    q, f_q = compute_all_darcy(p, R)

    # lambdify the exact solution
    return make_as_lambda_darcy(q, p, f_q, R)


def exact_sol_biot(mu, lambda_, alpha, s0):
    R = ReferenceFrame("R")
    x, y, _ = R.varlist

    # define the displacement
    u_x = sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y)
    u_y = u_x
    u = sp.Matrix([u_x, u_y, 0])

    # define the pressure
    p = sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y)

    # compute the stress and the source terms
    sigma, q, f_u, f_q = compute_all_biot(2, u, p, mu, lambda_, alpha, s0, R)

    # lambdify the exact solution
    return make_as_lambda_biot(sigma, u, q, p, f_u, f_q, R)
