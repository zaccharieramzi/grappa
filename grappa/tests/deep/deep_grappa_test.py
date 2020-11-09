import numpy as np
import pytest

from grappa.deep.deep_grappa import DeepGRAPPA
from grappa.deep.linear import linear_deep_grappa_model
from grappa.deep import deep_grappa

def fake_autocalibration_signal(kspace, _mask, _af):
    return kspace

def fake_number_geometries(_mask):
    return 1

@pytest.mark.parametrize('kspace, ny, expected_kernel', [
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
def test_linear_deep_grappa(monkeypatch, kspace, ny, expected_kernel):
    monkeypatch.setattr(deep_grappa, "autocalibration_signal", fake_autocalibration_signal)
    monkeypatch.setattr(deep_grappa, "number_geometries", fake_number_geometries)
    linear_deep_grappa = DeepGRAPPA(linear_deep_grappa_model, ny=ny, n_epochs=600, lr=1)
    kspace = np.array(kspace)[None, :]
    expected_kernel = np.array(expected_kernel)
    linear_deep_grappa.calibrate_models(kspace)
    underlying_model = linear_deep_grappa.models[0].layers[0]
    deep_weights = underlying_model.denses['real'].get_weights()[0].astype(np.complex64)
    deep_weights += 1j * underlying_model.denses['imag'].get_weights()[0]
    np.testing.assert_array_almost_equal(expected_kernel, deep_weights.T, decimal=1)
