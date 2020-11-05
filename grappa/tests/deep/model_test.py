import numpy as np
import pytest

from grappa.deep.deep_grappa import DeepGRAPPA
from grappa.deep.model import DeepKSpaceFiller
from grappa.deep import deep_grappa

def fake_autocalibration_signal(kspace, _mask, _af):
    return kspace

def fake_number_geometries(_mask):
    return 1

kspace = [
    [1, 2, 1],
    [2, 4, 2],
    [3, 6, 3],
    [7, 2, -5],
]
kspace = np.array(kspace)[None, :]

@pytest.mark.parametrize('n_dense', [2, 3])
@pytest.mark.parametrize('instance_normalisation', [True, False])
@pytest.mark.parametrize('kernel_learning', [True, False])
@pytest.mark.parametrize('distance_from_center_feat', [True, False])
def test_nonlinear_deep_grappa(
        monkeypatch,
        n_dense,
        instance_normalisation,
        kernel_learning,
        distance_from_center_feat,
    ):
    monkeypatch.setattr(deep_grappa, "autocalibration_signal", fake_autocalibration_signal)
    monkeypatch.setattr(deep_grappa, "number_geometries", fake_number_geometries)
    grappa = DeepGRAPPA(
        DeepKSpaceFiller,
        ny=3,
        n_epochs=2,
        lr=1,
        n_dense=n_dense,
        instance_normalisation=instance_normalisation,
        kernel_learning=kernel_learning,
        distance_from_center_feat=distance_from_center_feat,
    )
    grappa.calibrate_models(kspace)
