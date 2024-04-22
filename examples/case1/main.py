import numpy as np
import sys
import pygeon as pg
import porepy as pp


def u_ex_fct(pt):
    x, y, _ = pt
    return np.array([np.sin(x) * np.sin(y)] * 2)


def fun(pt):
    x, y, _ = pt
    val = -np.cos(x) * np.cos(y) + 2 * np.sin(x) * np.sin(y)
    return np.array([val] * 2)


def main(sd, folder=None, tol=1e-10):
    sd.nodes *= 2 * np.pi
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    # we define the data
    data = {"mu": 0.5, "lambda": 0.5}
    ess = sd.tags["domain_boundary_nodes"]
    ess = np.hstack([ess] * sd.dim)

    # we define the discretization
    key = "elasticity"
    discr = pg.VecVLagrange1(key)

    # we construct the matrix
    A = discr.assemble_stiff_matrix(sd, data)
    M = discr.assemble_mass_matrix(sd)
    b = discr.interpolate(sd, fun)

    # solve the system
    ls = pg.LinearSystem(A, M @ b)
    ls.flag_ess_bc(ess, np.zeros(discr.ndof(sd)))
    u = ls.solve()

    # compute the error
    u_ex = discr.interpolate(sd, u_ex_fct)
    norm_u = np.sqrt(u @ M @ u)
    err = np.sqrt((u - u_ex) @ M @ (u - u_ex)) / (norm_u if norm_u > tol else 1)

    if folder is not None:
        proj = discr.eval_at_cell_centers(sd)
        cell_u = proj @ u

        # we need to add the z component for the exporting
        cell_u = np.hstack((cell_u, np.zeros(sd.num_cells)))
        cell_u = cell_u.reshape((3, -1))

        save = pp.Exporter(sd, "sol", folder_name=folder)
        save.write_vtu([("cell_u", cell_u)])

    diams = sd.cell_diameters()
    return err, np.amax(diams), np.amin(diams), np.average(diams)


if __name__ == "__main__":
    folder = "examples/case1/"

    # num_pts = 100
    # sds = [pg.VoronoiGrid(num_pts, seed=0)]
    num_cells = 40
    sds = [pp.CartGrid([num_cells] * 2, [1] * 2)]
    # sds = [pg.unit_grid(2, 0.05, as_mdg=False)]
    # CartGrid

    err = [main(sd, folder) for sd in sds]
    print(err)
