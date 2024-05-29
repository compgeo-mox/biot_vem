import numpy as np
import os
import pygeon as pg
import porepy as pp


def main(sd, folder, file_name):
    key = "elasticity"

    pg.convert_from_pp(sd)
    sd.compute_geometry()

    # sd = pg.lloyd_relaxation(sd, 30)
    sd = pg.graph_laplace_regularization(sd)

    data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 1}}}

    bottom = np.hstack([np.isclose(sd.nodes[1, :], 0)] * sd.dim)
    top = np.isclose(sd.face_centers[1, :], 1)

    fun = lambda _: np.array([0, -1e-3])

    discr = pg.VecVLagrange1(key)

    # we construct the matrix
    A = discr.assemble_stiff_matrix(sd, data)
    b = discr.assemble_nat_bc(sd, fun, top)

    ls = pg.LinearSystem(A, b)
    ls.flag_ess_bc(bottom, np.zeros(discr.ndof(sd)))
    u = ls.solve()

    # we need to add the z component for the exporting
    u = np.hstack((u, np.zeros(sd.num_nodes))).reshape((3, -1))

    save = pp.Exporter(sd, file_name, folder_name=folder)
    save.write_vtu(data_pt=[("u", u)])


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    num_pts = 30

    sds = [pg.VoronoiGrid(num_pts, seed=0)]
    # sds = [pp.CartGrid([num_pts] * 2, [1] * 2)]

    file_name = "sol"
    err = [main(sd, folder, file_name) for sd in sds]
    print(err)
