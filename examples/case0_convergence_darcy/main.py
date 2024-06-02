import os, sys
import numpy as np
import scipy.sparse as sps

import pygeon as pg
import porepy as pp

sys.path.append("./src")

from analytical_solutions import exact_sol_darcy


def main(sd, folder, file_name):
    sd.compute_geometry()
    key = "darcy"

    # return the exact solution and related rhs
    q_ex, p_ex, f_q = exact_sol_darcy()

    # we define the data
    perm = np.ones(sd.num_cells)
    data = {pp.PARAMETERS: {key: {"second_order_tensor": pp.SecondOrderTensor(perm)}}}

    vrt0 = pg.MVEM(key)
    p0 = pg.PwConstants(key)

    # construct the local matrices
    mass_rt0 = vrt0.assemble_mass_matrix(sd, data)
    mass_p0 = p0.assemble_mass_matrix(sd, data)

    div = mass_p0 @ vrt0.assemble_diff_matrix(sd)

    # get the degrees of freedom for each variable
    dof_p, dof_q = div.shape

    # assemble the saddle point problem
    spp = sps.bmat([[mass_rt0, -div.T], [div, None]], format="csc")

    # we construct the right hand side
    # assemble the right-hand side
    rhs = np.zeros(dof_q + dof_p)
    rhs[-dof_p:] = mass_p0 @ p0.interpolate(sd, f_q)
    rhs[:dof_q] = -vrt0.assemble_nat_bc(sd, p_ex, sd.tags["domain_boundary_faces"])

    # solve the system
    ls = pg.LinearSystem(spp, rhs)
    x = ls.solve()

    # extract the variables
    q, p = x[:dof_q], x[-dof_p:]

    # compute the error
    err_q = vrt0.error_l2(sd, q, q_ex)
    err_p = p0.error_l2(sd, p, p_ex)

    if folder is not None:
        # post process variables
        proj_q = vrt0.eval_at_cell_centers(sd)
        cell_q = (proj_q @ q).reshape((3, -1), order="F")
        cell_p = p0.eval_at_cell_centers(sd) @ p

        save = pp.Exporter(sd, file_name, folder_name=folder)
        save.write_vtu([("cell_q", cell_q), ("cell_p", cell_p)])

    # compute some of the indicators for the grid
    diams = sd.cell_diameters()

    return (
        err_q,
        err_p,
        np.amax(diams),
        np.amin(diams),
        np.average(diams),
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
