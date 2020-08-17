import numpy as np

from grappa.kernel_application import _geom_apply_kernel


def test_geom_apply_kernel():
    kspace = [
        [1, 0, 4],
        [2, 0, 5],
        [3, 0, 6],
    ]
    expected_kspace = [
        [1, 5, 4],
        [2, 7, 5],
        [3, 9, 6],
    ]
    grappa_kernel = [[1]*2]
    kspace = np.array(kspace)[None, :]
    expected_kspace = np.array(expected_kspace)[None, :]
    grappa_kernel = np.array(grappa_kernel)
    _geom_apply_kernel(
        kspace,
        grappa_kernel,
        0,
        1,
        1,
        1,
    )
    np.testing.assert_array_almost_equal(expected_kspace, kspace)
