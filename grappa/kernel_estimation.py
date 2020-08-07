import numpy as np

from grappa.utils import cartesian_product, sources_from_targets, eval_at_positions, number_geometries


def kernel_estimation(kspace, mask=None, af=4, ny=3):
    """GRAPPA kernel estimation

    Arguments:
        - kspace (ndarray): the undersampled k-space, zero-filled. Its shape
            must be ncoils x readout x phase.
    """
    if mask is None:
        raise ValueError('For now mask has to be passed for kernel estimation')
    ac = _autocalibration_signal(kspace, mask, af)
    n_geometries = number_geometries(mask)
    ncoils = kspace.shape[0]
    grappa_kernels = [
        _geometry_kernel_estimation(ac, i_geom, ny, n_geometries, ncoils)
        for i_geom in range(n_geometries)
    ]
    return grappa_kernels

def _geometry_kernel_estimation(ac, i_geom, ny=3, n_geometries=4, ncoils=15):
    targets = cartesian_product(
        # readout dimension
        np.arange(ny // 2, ac.shape[1] - ny + (ny // 2)),
        # phase dimension
        np.arange(i_geom + 1, ac.shape[2] - n_geometries + i_geom),
    )
    target_values = [
        np.take(ac[c], np.ravel_multi_index(targets.T, ac[c].shape))
        for c in range(ncoils)
    ]
    target_values = np.array(target_values)
    # to refactor: for a given target position we always know the source
    # positions. This will be used for application.
    sources = sources_from_targets(
        targets,
        i_geom,
        n_geometries,
        ny,
        ncoils,
    )
    source_values = eval_at_positions(ac, sources)
    source_values = np.array(source_values)
    inverted_sources = np.linalg.pinv(source_values)
    grappa_kernel = target_values @ inverted_sources
    return grappa_kernel


def _autocalibration_signal(kspace, mask, af=4):
    center_fraction = (32//af) / 100
    num_low_freqs = int(np.round(mask.shape[-1] * center_fraction))
    ac_center = mask.shape[-1] // 2
    ac_slice = slice(ac_center - num_low_freqs//2, ac_center + num_low_freqs//2)
    ac = kspace[..., ac_slice]
    return ac
