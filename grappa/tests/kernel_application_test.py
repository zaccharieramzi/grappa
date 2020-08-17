import numpy as np
import pytest

from grappa.kernel_application import _geom_apply_kernel, apply_kernel


@pytest.mark.parametrize('kspace, grappa_kernel, expected_kspace',[
    (
        [
            [1, 0, 4],
            [2, 0, 5],
            [3, 0, 6],
        ],
        [[1]*2],
        [
            [1, 5, 4],
            [2, 7, 5],
            [3, 9, 6],
        ],
    ),
    (
        [
            [1, 0, 4],
            [2, 0, 5],
            [3, 0, 6],
        ],
        [[1]*6],
        [
            [1, 0, 4],
            [2, 21, 5],
            [3, 0, 6],
        ],
    )
])
def test_geom_apply_kernel(kspace, grappa_kernel, expected_kspace):
    kspace = np.array(kspace)[None, :]
    expected_kspace = np.array(expected_kspace)[None, :]
    grappa_kernel = np.array(grappa_kernel)
    ny = grappa_kernel.size // 2
    _geom_apply_kernel(
        kspace,
        grappa_kernel,
        0,
        1,
        ny,
        1,
    )
    np.testing.assert_array_almost_equal(expected_kspace, kspace)

@pytest.mark.parametrize('kspace, grappa_kernel, expected_kspace',[
    (
        [
            [1, 0, 4],
            [2, 0, 5],
            [3, 0, 6],
        ],
        [[1]*2],
        [
            [1, 5, 4],
            [2, 7, 5],
            [3, 9, 6],
        ],
    ),
    (
        [
            [1, 0, 4],
            [2, 0, 5],
            [3, 0, 6],
        ],
        [[1]*6],
        [
            [1, 12, 4],
            [2, 21, 5],
            [3, 16, 6],
        ],
    )
])
def test_apply_kernel(kspace, grappa_kernel, expected_kspace):
    mask = np.array([1, 0, 1])[None, None, : ]
    kspace = np.array(kspace)[None, :]
    expected_kspace = np.array(expected_kspace)[None, :]
    grappa_kernel = np.array(grappa_kernel)
    grappa_kernels = [grappa_kernel]
    kspace = apply_kernel(kspace, mask, grappa_kernels)
    np.testing.assert_array_almost_equal(expected_kspace, kspace)
