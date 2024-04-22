import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


# useful for the stopping criteria in the iterative solvers
def compute_err(x, x_i, mass):
    diff = np.sqrt((x_i - x) @ mass @ (x_i - x))
    norm = np.sqrt(x_i @ mass @ x_i)
    return diff / norm if norm else diff


def main():
    mesh_size = 0.05
    delta_t = 0.1
    dim = 2

    # pts = np.array([[0, 3, 3, 2, 1, 0], [0, 0, 1, 1, 1, 1]])
    # sd = pg.grid_from_boundary_pts(pts, mesh_size, as_mdg=False)
    # num_pts = 200
    # sd = pg.VoronoiGrid(num_pts, seed=0)
    num_cells = 40
    sd = pp.CartGrid([num_cells] * 2, [1] * 2)
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    # data for the iterative solvers
    tol = 1e-4
    max_iter = 1e2

    key = "biot"

    # the physical parameters of the problem, assumed constant
    alpha = 1
    s0 = 0.01  # 1 0.1

    data = {}
    param = {"second_order_tensor": pp.SecondOrderTensor(np.ones(sd.num_cells))}
    pp.initialize_data(sd, data, key, param)
    data.update({"mu": 0.5, "lambda": 1})

    # selection of the boundary conditions
    bd_q = sd.tags["domain_boundary_faces"]
    bd_q[np.isclose(sd.face_centers[1, :], 1)] = False

    bd_u = sd.tags["domain_boundary_nodes"]
    bd_u[np.isclose(sd.nodes[1, :], 1)] = False
    bd_u = np.hstack([bd_u] * dim)

    top = np.isclose(sd.face_centers[1, :], 1)

    fun = lambda _: np.array([0, -1])

    # definition of the discretizations
    vec_vp1 = pg.VecVLagrange1(key)
    p0 = pg.PwConstants(key)
    mvem = pg.MVEM(key)

    # dofs for the different variables
    dof_u = sd.num_nodes * dim
    dof_q = sd.num_faces
    dof_p = sd.num_cells
    dofs = np.cumsum([dof_u, dof_q, dof_p])

    # construction of the block matrices
    mass_u = vec_vp1.assemble_mass_matrix(sd)
    mass_q = mvem.assemble_mass_matrix(sd)
    mass_p = p0.assemble_mass_matrix(sd)
    div_q = mass_p @ mvem.assemble_diff_matrix(sd)

    div_u = mass_p @ vec_vp1.assemble_div_matrix(sd)

    stiff_u = vec_vp1.assemble_stiff_matrix(sd, data)

    b = vec_vp1.assemble_nat_bc(sd, fun, top)

    # regroup of the right-hand side
    bd = np.hstack((bd_u, bd_q, np.zeros(dof_p, dtype=bool)))
    rhs = np.hstack((b, np.zeros(dof_q + dof_p)))

    beta = alpha**2 / (2 * (2 * data["mu"] / dim + data["lambda"]))

    # fmt: off
    # construction of the global problem
    spp = sps.bmat([[     stiff_u,             None,     -alpha * div_u.T],
                    [        None, delta_t * mass_q,   -delta_t * div_q.T],
                    [        None,  delta_t * div_q, (s0 + beta) * mass_p]])
    # fmt: on

    step = 0
    err = tol + 1

    # initialization of the solution
    u_i = np.zeros(dof_u)
    q_i = np.zeros(dof_q)
    p_i = np.zeros(dof_p)

    while err > tol and step < max_iter:
        # for a given u solve the flow problem
        rhs_i = rhs.copy()
        rhs_i[-dof_p:] = -alpha * div_u @ u_i + beta * mass_p @ p_i

        ls = pg.LinearSystem(spp, rhs_i)
        ls.flag_ess_bc(bd, np.zeros_like(bd))
        x = ls.solve()

        # split of the solution from the vector x
        u, q, p = np.split(x, dofs[:-1])

        # compute the stopping criteria
        step += 1
        err_u = compute_err(u_i, u, mass_u)
        err_q = compute_err(q_i, q, mass_q)
        err_p = compute_err(p_i, p, mass_p)

        u_i = u.copy()
        q_i = q.copy()
        p_i = p.copy()

        err = err_u + err_q + err_p

        print(step, err, err_u, err_q, err_p)


if __name__ == "__main__":
    main()
