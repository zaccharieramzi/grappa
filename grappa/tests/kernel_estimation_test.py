import numpy as np

from grappa.kernel_estimation import _geometry_kernel_estimation


def test_geometry_kernel_estimation():
    ac = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [3, 6, 3],
        [7, 2, -5],
    ])[None, :]
    expected_kernel = np.array([[1]*2])
    grappa_kernel = _geometry_kernel_estimation(
        ac,
        i_geom=0,
        ny=1,
        n_geometries=1,
        ncoils=1,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel)
