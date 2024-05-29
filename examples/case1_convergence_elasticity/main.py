import os, sys
import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp

sys.path.append("./src")

from analytical_solutions import exact_sol_elasticity


def main(sd, folder, file_name):
    sd.compute_geometry()
    key = "elasticity"

    # return the exact solution and related rhs
    mu, lambda_ = 0.5, 1
    _, u_ex, f_u = exact_sol_elasticity(mu, lambda_)

    # we define the data
    data = {pp.PARAMETERS: {key: {"mu": mu, "lambda": lambda_}}}

    ess = sd.tags["domain_boundary_nodes"]
    ess = np.hstack([ess] * sd.dim)

    # we define the discretization
    discr = pg.VecVLagrange1(key)

    u_ex_ = lambda x: u_ex(x)[: sd.dim].ravel()

    # we construct the matrix
    A = discr.assemble_stiff_matrix(sd, data)
    M = discr.assemble_mass_matrix(sd)
    b = M @ discr.interpolate(sd, lambda x: f_u(x)[: sd.dim].ravel())

    # solve the system
    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(ess, discr.interpolate(sd, u_ex_))
    u = ls.solve()

    # compute the error
    err = discr.error_l2(sd, u, u_ex_)

    if folder is not None:
        # we need to add the z component for the exporting
        u = np.hstack((u, np.zeros(sd.num_nodes))).reshape((3, -1))

        save = pp.Exporter(sd, file_name, folder_name=folder)
        save.write_vtu(data_pt=[("u", u)])

    # compute some of the indicators for the grid
    diams = sd.cell_diameters()
    A_0, _, _ = ls.reduce_system()
    eig_max = sps.linalg.eigsh(A_0, k=1, which="LM", return_eigenvectors=False)[0]
    eig_min = sps.linalg.eigsh(A_0, k=1, which="SM", return_eigenvectors=False)[0]

    return (
        err,
        np.amax(diams),
        np.amin(diams),
        np.average(diams),
        eig_max / eig_min,
        sd.num_cells,
        np.amax(sd.cell_volumes),
        np.amax(sd.face_areas),
    )


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))

    # construct the einstein grids
    file_name = ["H2.svg", "H3.svg", "H4.svg", "H5.svg"]
    einstein = [
        pg.EinSteinGrid(os.path.join(folder, "../grids/", fn)) for fn in file_name
    ]
    einstein_names = [fn[:-4] for fn in file_name]

    # construct the voronoi grids
    num_pts = [40, 80, 160, 320]
    seeds = [1, 0, 4, 12]
    voronoi = [
        pg.VoronoiGrid(num_pt, seed=seed) for num_pt, seed in zip(num_pts, seeds)
    ]
    voronoi_names = ["voro_" + str(num_pt) for num_pt in num_pts]

    # reference simplicial grids
    mesh_sizes = [0.1, 0.05, 0.025, 0.0125]
    simplex = [pg.unit_grid(2, mesh_size, as_mdg=False) for mesh_size in mesh_sizes]
    simplex_names = ["simplex_" + str(mesh_size) for mesh_size in mesh_sizes]

    # do the computation for the normal grids
    sds = einstein + voronoi + simplex
    grid_names = einstein_names + voronoi_names + simplex_names
    err_1 = [main(sd, folder, grid_name) for sd, grid_name in zip(sds, grid_names)]
    np.savetxt(os.path.join(folder, "err_1.txt"), err_1)

    # perform a regularization for the einstein grids
    einstein_reg1 = [
        pg.graph_laplace_regularization(sd, sliding=False) for sd in einstein
    ]
    einstein_names_reg1 = [name + "_reg1" for name in einstein_names]

    # do the computation for the einstein regularized grids
    sds = einstein_reg1
    grid_names = einstein_names_reg1
    err_2 = [main(sd, folder, grid_name) for sd, grid_name in zip(sds, grid_names)]

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
    err_3 = [main(sd, folder, grid_name) for sd, grid_name in zip(sds, grid_names)]

    err = err_2 + err_3
    np.savetxt(os.path.join(folder, "err_2.txt"), err)
