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
@pytest.mark.parametrize('backend', ['tensorflow', 'numpy'])
def test_geom_apply_kernel(kspace, grappa_kernel, expected_kspace, backend):
    kspace = np.array(kspace)[None, :].astype(np.complex64)
    expected_kspace = np.array(expected_kspace)[None, :].astype(np.complex64)
    grappa_kernel = np.array(grappa_kernel).astype(np.complex64)
    ny = grappa_kernel.size // 2
    _geom_apply_kernel(
        kspace,
        grappa_kernel,
        0,
        1,
        ny,
        1,
        backend=backend,
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
@pytest.mark.parametrize('backend', ['tensorflow', 'numpy'])
def test_apply_kernel(kspace, grappa_kernel, expected_kspace, backend):
    mask = np.array([1, 0, 1])[None, None, : ].astype(np.complex64)
    kspace = np.array(kspace)[None, :].astype(np.complex64)
    expected_kspace = np.array(expected_kspace)[None, :].astype(np.complex64)
    grappa_kernel = np.array(grappa_kernel).astype(np.complex64)
    grappa_kernels = [grappa_kernel]
    kspace = apply_kernel(kspace, mask, grappa_kernels, backend=backend)
    np.testing.assert_array_almost_equal(expected_kspace, kspace)
