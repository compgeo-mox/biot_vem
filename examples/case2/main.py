import numpy as np
import sys
import pygeon as pg
import porepy as pp


def main(sd, folder):
    pg.convert_from_pp(sd)
    sd.compute_geometry()

    data = {"mu": 0.5, "lambda": 0.5}

    bottom = np.hstack([np.isclose(sd.nodes[1, :], 0)] * sd.dim)
    top = np.isclose(sd.face_centers[1, :], 1)

    fun = lambda _: np.array([0, -1e-3])

    key = "elasticity"
    discr = pg.VecVLagrange1(key)

    # we construct the matrix
    A = discr.assemble_stiff_matrix(sd, data)
    b = discr.assemble_nat_bc(sd, fun, top)

    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(bottom, np.zeros(discr.ndof(sd)))
    u = ls.solve()

    proj = discr.eval_at_cell_centers(sd)
    cell_u = proj @ u

    # we need to add the z component for the exporting
    cell_u = np.hstack((cell_u, np.zeros(sd.num_cells)))
    cell_u = cell_u.reshape((3, -1))

    save = pp.Exporter(sd, "sol", folder_name=folder)
    save.write_vtu([("cell_u", cell_u)])


if __name__ == "__main__":
    folder = "examples/case1/"
    num_pts = 50

    # sds = [pg.VoronoiGrid(num_pts, seed=0)]
    sds = [pp.CartGrid([num_pts] * 2, [1] * 2)]

    err = [main(sd, folder) for sd in sds]
    print(err)
