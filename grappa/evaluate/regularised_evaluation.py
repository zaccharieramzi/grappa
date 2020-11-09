from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from tf_fastmri_data.datasets.cartesian import CartesianFastMRIDatasetBuilder
from tqdm import tqdm

from grappa.reconstruction import fastmri_volume_reconstruction


n_samples = 50
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

lamda = 700
m = Metrics(METRIC_FUNCS)

for (kspace, mask, smaps), image in tqdm(ds.preprocessed_ds.as_numpy_iterator(), total=n_samples):
    image_pred = fastmri_volume_reconstruction(kspace[..., 0], mask, ny=3, lamda=lamda)
    m.push(image[..., 0], image_pred)

print(m.means(), m.stddevs())
