import sympy as sp


def scalar_gradient(p, R):
    return sp.Matrix([sp.diff(p, var) for var in R.varlist])


def vector_gradient(u, R):
    return sp.Matrix([sp.diff(u_i, var) for u_i in u for var in R.varlist]).reshape(
        3, 3
    )


def asym_T(r):
    return sp.Matrix([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def asym(sigma):
    asym_sigma = sigma - sigma.T
    return sp.Matrix([asym_sigma[2, 1], asym_sigma[0, 2], asym_sigma[1, 0]])


def vector_divergence(q, R):
    return sp.diff(q[0], R[0]) + sp.diff(q[1], R[1]) + sp.diff(q[2], R[2])


def matrix_divergence(sigma, R):
    x, y, z = R.varlist
    return sp.diff(sigma[:, 0], x) + sp.diff(sigma[:, 1], y) + sp.diff(sigma[:, 2], z)


def compute_all_elasticity(dim, u, mu, lambda_, R):
    if dim == 2:
        I = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    else:
        I = sp.Identity(3).as_explicit()

    # sigma
    eps = 0.5 * (vector_gradient(u, R) + vector_gradient(u, R).T)
    sigma = 2 * mu * eps + lambda_ * sp.trace(eps) * I

    # Compute the source term
    f_u = -matrix_divergence(sigma, R)

    return sigma, f_u


def compute_all_darcy(p, R):
    # q
    q = -scalar_gradient(p, R)

    # Compute the source term
    f_q = vector_divergence(q, R)

    return q, f_q


def compute_all_biot(dim, u, p, mu, lambda_, alpha, s0, R):

    if dim == 2:
        I = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    else:
        I = sp.Identity(3).as_explicit()

    # sigma
    eps = 0.5 * (vector_gradient(u, R) + vector_gradient(u, R).T)
    sigma = 2 * mu * eps + lambda_ * sp.trace(eps) * I - alpha * I * p

    q = -scalar_gradient(p, R)

    f_u = -matrix_divergence(sigma, R)
    f_q = vector_divergence(q, R) + alpha * vector_divergence(u, R) + s0 * p

    return sigma, q, f_u, f_q


def make_as_lambda_elasticity(sigma, u, f_u, R):
    x, y, z = R.varlist

    # lambdify the exact solution
    sigma_lamb = sp.lambdify([x, y, z], sigma)
    sigma_ex = lambda pt: sigma_lamb(*pt)

    u_lamb = sp.lambdify([x, y, z], u)
    u_ex = lambda pt: u_lamb(*pt)

    f_u_lamb = sp.lambdify([x, y, z], f_u)
    f_u_ex = lambda pt: f_u_lamb(*pt)

    return sigma_ex, u_ex, f_u_ex


def make_as_lambda_darcy(q, p, f_q, R):
    x, y, z = R.varlist

    # lambdify the exact solution
    q_lamb = sp.lambdify([x, y, z], q)
    q_ex = lambda pt: q_lamb(*pt)

    p_lamb = sp.lambdify([x, y, z], p)
    p_ex = lambda pt: p_lamb(*pt)

    f_q_lamb = sp.lambdify([x, y, z], f_q)
    f_q_ex = lambda pt: f_q_lamb(*pt)

    return q_ex, p_ex, f_q_ex


def make_as_lambda_biot(sigma, u, q, p, f_u, f_q, R):
    x, y, z = R.varlist

    # lambdify the exact solution
    sigma_lamb = sp.lambdify([x, y, z], sigma)
    sigma_ex = lambda pt: sigma_lamb(*pt)

    u_lamb = sp.lambdify([x, y, z], u)
    u_ex = lambda pt: u_lamb(*pt)

    q_lamb = sp.lambdify([x, y, z], q)
    q_ex = lambda pt: q_lamb(*pt)

    p_lamb = sp.lambdify([x, y, z], p)
    p_ex = lambda pt: p_lamb(*pt)

    f_u_lamb = sp.lambdify([x, y, z], f_u)
    f_u_ex = lambda pt: f_u_lamb(*pt)

    f_q_lamb = sp.lambdify([x, y, z], f_q)
    f_q_ex = lambda pt: f_q_lamb(*pt)

    return sigma_ex, u_ex, q_ex, p_ex, f_u_ex, f_q_ex
