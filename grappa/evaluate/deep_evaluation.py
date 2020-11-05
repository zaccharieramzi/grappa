from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tf_complex.dense import ComplexDense
from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder
from tqdm import tqdm

from grappa.deep.deep_grappa import DeepGRAPPA

def test_model(model_fun, model_kwargs, n_samples=50, **kwargs):
    ds = CartesianFastMRIDatasetBuilder(
        dataset='val',
        af=4,
        mask_mode='equidistant',
        multicoil=True,
        slice_random=False,
        prefetch=True,
        contrast='CORPD_FBK',
        repeat=False,
        n_samples=n_samples,
        scale_factor=1e6,
    )
    m = Metrics(METRIC_FUNCS)
    model_kwargs.update(kwargs)
    deep_grappa = DeepGRAPPA(model_fun, **model_kwargs)
    for (kspace, mask, _), image in tqdm(ds.preprocessed_ds.as_numpy_iterator(), total=n_samples):
        image_pred = deep_grappa.reconstruct(kspace[..., 0], mask)
        m.push(image[..., 0], image_pred)
    print(model_fun.__name__)
    print(model_kwargs)
    print(m)
    return METRIC_FUNCS, m


def deep_grappa_model(ncoils=15, n_dense=2, _distance_from_center_feat=False):
    dense_layers = [
        ComplexDense(ncoils, use_bias=False, activation='crelu')
        for _ in range(n_dense-1)
    ]
    dense_layers.append(
        ComplexDense(ncoils, use_bias=False, activation='linear')
    )
    model = Sequential(dense_layers)
    return model

if __name__ == '__main__':
    metrics = dict()
    params = [
        ('non_linear_distance', {'n_dense': 2, 'distance_from_center_feat': True}),
        ('non_linear', {'n_dense': 2, 'distance_from_center_feat': False}),
        ('linear', {'n_dense': 1}),
        ('deep_distance', {'n_dense': 3, 'distance_from_center_feat': True}),
        ('deep', {'n_dense': 3, 'distance_from_center_feat': False}),
    ]

    for name, param in params:
        m = test_model(deep_grappa_model, param)

        metrics[name] = (m.means(), m.stddevs())

    print(metrics)
