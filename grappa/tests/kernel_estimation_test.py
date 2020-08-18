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
    ),
    (  # third example to test the complex part of kernel estimation
        [
            [1, 1 + 1j, 1],
            [2, 2 + 2j, 2],
            [3, 3 + 3j, 3],
            [7, 7 - 5j, -5],
        ],
        1,
        [[1, 1j]],
    ),
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
        lamda=1e-8,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel, decimal=5)

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
        lamda=1e-8,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel, decimal=5)

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
                [0.001, -2, 0.0001],
                [0.004, -4, 0.002],
                [-0.001, -6, 0.0002],
                [0.002, -2, 0.001],
            ],
        ],
        1,
        [[1, 1, 0, 0], [-1, -1, 0, 0]],
    ),
    (  # second example, a bit more complex, still in single geometry
        [
            [
                [1, 2, 1],
                [2, 4, 2],
                [3, 6, 3],
                [7, 2, -5],
                [0.1, 1.1, 1],
                [3.1, 5.1, 2],
                [3.1, 12.1, 9],
                [7.1, 3.1, -4],
                [8, 3.5, -4.5],
                [18, 12, -6],
                [-3, 2, 5],
                [-4.2, 9, 13.2],
            ],
            [
                [0.01, -2, 0.0012],
                [0.02, -4, 0.0012],
                [0.001, -6, 0.0011],
                [-0.01, -2, 0.0013],
                [0.003, -1.1, 0.0014],
                [0.005, -5.1, 0.0013],
                [-0.02, -12.1, 0.0019],
                [0.007, -3.1, 0.0013],
                [0.002, -3.5, 0.001],
                [0.004, -12, 0.001],
                [0.00036, -2, 0.003],
                [-0.002, -9, -0.003],
            ],
        ],
        3,
        [
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
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
        lamda=1e-6,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel, decimal=3)


@pytest.mark.parametrize('ac, ny, i_geom, expected_kernel', [
    (  # first basic example with a kernel using only the 2 direct neighbours
       # from the first coil (either positively or negatively)
        [
            [
                [1, 2, 1, 1],
                [2, 4, 2, 2],
                [3, 6, 3, 3],
                [7, 2, -5, -5],
            ],
            [
                [0.001, -2, 0, 0],
                [0.004, -4, 0, 0],
                [-0.001, -6, 0, 0],
                [0.002, -2, 0, 0],
            ],
        ],
        1,
        0,
        [[1, 1, 0, 0], [-1, -1, 0, 0]],
    ),
])
def test_geometry_kernel_estimation_multicoil_double(ac, ny, i_geom, expected_kernel):
    ac = np.array(ac)
    expected_kernel = np.array(expected_kernel)
    grappa_kernel = _geometry_kernel_estimation(
        ac,
        i_geom=i_geom,
        ny=ny,
        n_geometries=2,
        ncoils=2,
    )
    np.testing.assert_array_almost_equal(expected_kernel, grappa_kernel, decimal=3)
