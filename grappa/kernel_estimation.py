import numpy as np
import tensorflow as tf

from grappa.utils import cartesian_product, sources_from_targets, eval_at_positions, number_geometries


def kernel_estimation(kspace, mask=None, af=4, ny=3, lamda=1e-6, fastmri=True, backend='tensorflow'):
    """GRAPPA kernel estimation

    Arguments:
        - kspace (ndarray): the undersampled k-space, zero-filled. Its shape
            must be ncoils x readout x phase.
    """
    if mask is None:
        raise ValueError('For now mask has to be passed for kernel estimation')
    ac = autocalibration_signal(kspace, mask, af, fastmri=fastmri)
    n_geometries = number_geometries(mask)
    ncoils = kspace.shape[0]
    grappa_kernels = [
        _geometry_kernel_estimation(ac, i_geom, ny, n_geometries, ncoils, lamda=lamda, backend=backend)
        for i_geom in range(n_geometries)
    ]
    return grappa_kernels

def _geometry_kernel_estimation(ac, i_geom, ny=3, n_geometries=4, ncoils=15, lamda=1e-6, backend='tensorflow'):
    target_values, source_values = list_target_source_values_for_estimation(
        ac=ac,
        i_geom=i_geom,
        ny=ny,
        n_geometries=n_geometries,
        ncoils=ncoils,
    )
    # taken from
    # https://users.fmrib.ox.ac.uk/~mchiew/Teaching.html
    regularizer = lamda*np.linalg.norm(source_values)*np.eye(source_values.shape[0])
    source_values_conj_t = source_values.conj().T
    regularized_inverted_sources = np.linalg.pinv(source_values @ source_values_conj_t + regularizer)
    if backend == 'tensorflow':
        source_values_conj_t = tf.constant(source_values_conj_t)
        target_values = tf.constant(target_values)
        regularized_inverted_sources = tf.constant(regularized_inverted_sources)
        grappa_kernel = target_values @ source_values_conj_t @ regularized_inverted_sources
        grappa_kernel = grappa_kernel.numpy()
    else:
        grappa_kernel = target_values @ source_values_conj_t @ regularized_inverted_sources
    return grappa_kernel

def list_target_source_values_for_estimation(ac, i_geom, ny, n_geometries, ncoils):
    targets = cartesian_product(
        # readout dimension
        np.arange(ny // 2, ac.shape[1] - ny + (ny // 2) + 1),
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
    return target_values, source_values

def _autocalibration_signal_fastmri(kspace, mask, af=4):
    center_fraction = (32//af) / 100
    num_low_freqs = int(np.round(mask.shape[-1] * center_fraction))
    ac_center = mask.shape[-1] // 2
    ac_slice = slice(ac_center - num_low_freqs//2, ac_center + num_low_freqs//2)
    ac = kspace[..., ac_slice]
    return ac

def _autocalibration_indices(mask):
    mask_int = np.array(mask).astype(np.int)
    mask_diff = np.abs(mask_int[1:] - mask_int[:-1])
    ac_indexes = np.where(np.logical_and(
        mask_diff == 0,
        mask_int[:-1] == 1,
    ))[0]
    ac_inf = min(ac_indexes)
    ac_sup = max(ac_indexes) + 1
    return ac_inf, ac_sup

def _autocalibration_signal_general(kspace, mask):
    ac_inf, ac_sup = _autocalibration_indices(mask)
    ac = kspace[..., ac_inf:ac_sup+1]
    return ac

def autocalibration_signal(kspace, mask, af=4, fastmri=True):
    if fastmri:
        return _autocalibration_signal_fastmri(kspace, mask, af=af)
    else:
        return _autocalibration_signal_general(kspace, mask)
