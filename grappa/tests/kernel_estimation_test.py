import numpy as np
import pytest

from grappa.kernel_estimation import _geometry_kernel_estimation


@pytest.mark.parametrize('ac, ny, expected_kernel', [
    (  # first basic example with a kernel using only the 2 direct neighbours
        [
            [1, 2, 1],
            [2, 4, 2],
            [3, 6, 3],
            [7, 2, -5],
        ],
        1,
        [[1]*2],
    ),
    (  # second example, a bit more complex, still in single geometry
        [
            [1, 2, 1],
            [2, 4, 2],
            [3, 6, 3],
            [7, 2, -5],
            [0.1, 1.1, 1],
            [3.1, 5.1, 2],
            [3.1, 12.1, 9],
            [7.1, 3.1, -4],
        ],
        3,
        [[0, 0, 1, 1, 0, 0]],
    )
])
def test_geometry_kernel_estimation_singlecoil_simple(ac, ny, expected_kernel):
    ac = np.array(ac)[None, :]
    expected_kernel = np.array(expected_kernel)
    grappa_kernel = _geometry_kernel_estimation(
        ac,
        i_geom=0,
        ny=ny,
        n_geometries=1,
        ncoils=1,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel)

@pytest.mark.parametrize('ac, ny, i_geom, expected_kernel', [
    (  # first basic example with a kernel using only the 2 direct sampled neighbours
        [
            [1, 2, 1, 1],
            [2, 4, 2, 2],
            [3, 6, 3, 3],
            [7, 2, -5, -5],
        ],
        1,
        0,
        [[1]*2],
    ),
    (  # first basic example with a kernel using only the 2 direct sampled neighbours
        [
            [1, 2, 1, 1],
            [2, 4, 2, 2],
            [3, 6, 3, 3],
            [7, 2, -5, -5],
        ],
        1,
        1,
        [[0, 1]],
    ),
    (  # second example, a bit more complex
        [
            [1, 2, 1, 1],
            [2, 4, 2, 2],
            [3, 6, 3, 3],
            [7, 2, -5, -5],
            [0.1, 1.1, 1, 1],
            [3.1, 5.1, 2, 2],
            [3.1, 12.1, 9, 9],
            [7.1, 3.1, -4, -4],
        ],
        3,
        0,
        [[0, 0, 1, 1, 0, 0]],
    )
])
def test_geometry_kernel_estimation_singlecoil_double(ac, ny, i_geom, expected_kernel):
    ac = np.array(ac)[None, :]
    expected_kernel = np.array(expected_kernel)
    grappa_kernel = _geometry_kernel_estimation(
        ac,
        i_geom=i_geom,
        ny=ny,
        n_geometries=2,
        ncoils=1,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel)

@pytest.mark.parametrize('ac, ny, expected_kernel', [
    (  # first basic example with a kernel using only the 2 direct neighbours
       # from the first coil (either positively or negatively)
        [
            [
                [1, 2, 1],
                [2, 4, 2],
                [3, 6, 3],
                [7, 2, -5],
            ],
            [
                [0.001, -2, 0],
                [0.004, -4, 0],
                [-0.001, -6, 0],
                [0.002, -2, 0],
            ],
        ],
        1,
        [[1, 1, 0, 0], [-1, -1, 0, 0]],
    ),
])
def test_geometry_kernel_estimation_multicoil_simple(ac, ny, expected_kernel):
    ac = np.array(ac)
    expected_kernel = np.array(expected_kernel)
    grappa_kernel = _geometry_kernel_estimation(
        ac,
        i_geom=0,
        ny=ny,
        n_geometries=1,
        ncoils=2,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel)
