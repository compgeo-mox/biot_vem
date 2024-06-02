import numpy as np
import os, sys
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

sys.path.append("./src")


from analytical_solutions import exact_sol_biot


def error(x, x_i, mass):
    norm = np.sqrt(x_i @ mass @ x_i)
    delta = x - x_i
    return np.sqrt(delta @ mass @ delta) / (norm if norm > 1e-10 else 1)


def main(sd, folder, file_name, mu, lambda_):
    sd.compute_geometry()
    key = "biot"

    delta_t = 1

    # data for the iterative solvers
    tol = 1e-4
    max_iter = 1e2

    # return the exact solution and related rhs
    alpha, s0 = 1, 1
    _, u_ex, q_ex, p_ex, f_u, f_q = exact_sol_biot(mu, lambda_, alpha, s0)

    perm = np.ones(sd.num_cells)
    data = {
        pp.PARAMETERS: {
            key: {
                "mu": mu,
                "lambda": lambda_,
                "second_order_tensor": pp.SecondOrderTensor(perm),
            }
        }
    }

    # definition of the discretizations
    vec_vp1 = pg.VecVLagrange1(key)
    p0 = pg.PwConstants(key)
    vrt0 = pg.MVEM(key)

    # dofs for the different variables
    dof_u = sd.num_nodes * sd.dim
    dof_q = sd.num_faces
    dof_p = sd.num_cells
    dofs = np.cumsum([dof_u, dof_q, dof_p])

    # construction of the block matrices
    mass_u = vec_vp1.assemble_mass_matrix(sd)
    mass_q = vrt0.assemble_mass_matrix(sd, data)
    mass_p = p0.assemble_mass_matrix(sd)
    div_q = mass_p @ vrt0.assemble_diff_matrix(sd)

    div_u = mass_p @ vec_vp1.assemble_div_matrix(sd)

    stiff_u = vec_vp1.assemble_stiff_matrix(sd, data)

    # selection of the boundary conditions
    bd_u = sd.tags["domain_boundary_nodes"]

    bd = np.zeros(dof_u + dof_q + dof_p, dtype=bool)
    bd[:dof_u] = np.hstack([bd_u] * sd.dim)

    u_ex_ = lambda x: u_ex(x)[: sd.dim].ravel()
    bd_val = np.hstack((vec_vp1.interpolate(sd, u_ex_), np.zeros(dof_q + dof_p)))

    # regroup of the right-hand side
    rhs = np.zeros(dof_u + dof_q + dof_p)
    rhs[:dof_u] += mass_u @ vec_vp1.interpolate(sd, lambda x: f_u(x)[: sd.dim].ravel())
    rhs[dof_u : dof_u + dof_q] += -vrt0.assemble_nat_bc(
        sd, p_ex, sd.tags["domain_boundary_faces"]
    )
    rhs[-dof_p:] += mass_p @ p0.interpolate(sd, f_q)

    beta = alpha**2 / (2 * (2 * mu / sd.dim + lambda_))

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
        rhs_i[-dof_p:] += -alpha * div_u @ u_i + beta * mass_p @ p_i

        ls = pg.LinearSystem(spp, rhs_i)
        ls.flag_ess_bc(bd, bd_val)
        x = ls.solve()

        # split of the solution from the vector x
        u, q, p = np.split(x, dofs[:-1])

        # compute the stopping criteria
        step += 1

        err = error(u_i, u, mass_u) + error(q_i, q, mass_q) + error(p_i, p, mass_p)

        u_i = u.copy()
        q_i = q.copy()
        p_i = p.copy()

    # compute the error
    err_u = vec_vp1.error_l2(sd, u, u_ex_)
    err_q = vrt0.error_l2(sd, q, q_ex)
    err_p = p0.error_l2(sd, p, p_ex)

    if folder is not None:
        # post process variables
        proj_q = vrt0.eval_at_cell_centers(sd)
        cell_q = (proj_q @ q).reshape((3, -1), order="F")
        cell_p = p0.eval_at_cell_centers(sd) @ p

        # we need to add the z component for the exporting
        u = np.hstack((u, np.zeros(sd.num_nodes))).reshape((3, -1))

        save = pp.Exporter(sd, file_name, folder_name=folder)
        save.write_vtu([("cell_q", cell_q), ("cell_p", cell_p)], data_pt=[("u", u)])

    # compute some of the indicators for the grid
    diams = sd.cell_diameters()
    return (
        err_u,
        err_q,
        err_p,
        np.amax(diams),
        np.amin(diams),
        np.average(diams),
        sd.num_cells,
        np.amax(sd.cell_volumes),
        np.amax(sd.face_areas),
        step,
    )


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))

    mus = [1, 10, 100, 1000]
    lambdas = [1, 10, 100, 1000]

    for mu in mus:
        for lambda_ in lambdas:
            # construct the einstein grids
            file_name = ["H2.svg", "H3.svg", "H4.svg", "H5.svg"]
            einstein = [
                pg.EinSteinGrid(os.path.join(folder, "../grids/", fn))
                for fn in file_name
            ]
            einstein_names = [fn[:-4] for fn in file_name]

            # construct the voronoi grids
            num_pts = [40, 80, 160, 320]
            seeds = [1, 0, 4, 12]
            voronoi = [
                pg.VoronoiGrid(num_pt, seed=seed)
                for num_pt, seed in zip(num_pts, seeds)
            ]
            voronoi_names = ["voro_" + str(num_pt) for num_pt in num_pts]

            # reference simplicial grids
            mesh_sizes = [0.1, 0.05, 0.025, 0.0125]
            simplex = [
                pg.unit_grid(2, mesh_size, as_mdg=False) for mesh_size in mesh_sizes
            ]
            simplex_names = ["simplex_" + str(mesh_size) for mesh_size in mesh_sizes]

            # do the computation for the normal grids
            sds = einstein + voronoi + simplex
            grid_names = einstein_names + voronoi_names + simplex_names
            err_1 = [
                main(sd, folder, grid_name, mu, lambda_)
                for sd, grid_name in zip(sds, grid_names)
            ]
            file_name_err = str(mu) + "_" + str(lambda_) + "_err_1.txt"
            np.savetxt(os.path.join(folder, file_name_err), err_1)

            # perform a regularization for the einstein grids
            einstein_reg1 = [
                pg.graph_laplace_regularization(sd, sliding=False) for sd in einstein
            ]
            einstein_names_reg1 = [name + "_reg1" for name in einstein_names]

            # do the computation for the einstein regularized grids
            sds = einstein_reg1
            grid_names = einstein_names_reg1
            err_2 = [
                main(sd, folder, grid_name, mu, lambda_)
                for sd, grid_name in zip(sds, grid_names)
            ]

            # perform a regularization for the voronoi grids
            [sd.compute_geometry() for sd in voronoi]
            voronoi_reg1 = [pg.graph_laplace_regularization(sd) for sd in voronoi]
            voronoi_names_reg1 = [name + "_reg1" for name in voronoi_names]

            voronoi_reg2 = [pg.graph_laplace_dual_regularization(sd) for sd in voronoi]
            voronoi_names_reg2 = [name + "_reg2" for name in voronoi_names]

            num_reg = 5
            voronoi_reg3 = [pg.lloyd_regularization(sd, num_reg) for sd in voronoi]
            voronoi_names_reg3 = [name + "_reg3" for name in voronoi_names]

            # do the computation for the voronoi regularized grids
            sds = voronoi_reg1 + voronoi_reg2 + voronoi_reg3
            grid_names = voronoi_names_reg1 + voronoi_names_reg2 + voronoi_names_reg3
            err_3 = [
                main(sd, folder, grid_name, mu, lambda_)
                for sd, grid_name in zip(sds, grid_names)
            ]

            err = err_2 + err_3
            file_name_err = str(mu) + "_" + str(lambda_) + "_err_2.txt"
            np.savetxt(os.path.join(folder, file_name_err), err)
