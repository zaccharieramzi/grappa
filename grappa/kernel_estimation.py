import numpy as np

def kernel_estimation(kspace, mask=None, af=4, ny=3):
    """GRAPPA kernel estimation

    Arguments:
        - kspace (ndarray): the undersampled k-space, zero-filled. Its shape
            must be ncoils x readout x phase.
    """
    if mask is None:
        raise ValueError('For now mask has to be passed for kernel estimation')
    ac = _autocalibration_signal(kspace, mask, af)
    n_geometries = _number_geometries(mask)
    ncoils = kspace.shape[0]
    grappa_kernels = [
        _geometry_kernel_estimation(ac, i_geom, ny, n_geometries, ncoils)
        for i_geom in range(n_geometries)
    ]
    return grappa_kernels

def _geometry_kernel_estimation(ac, i_geom, ny=3, n_geometries=4, ncoils=15):
    targets = _cartesian_product(
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
    sources = list()
    for sc in range(ncoils):
        for delta_si in range(-(ny//2), ny//2 + 1):
            for delta_sj in [-(i_geom+1), n_geometries - i_geom]:
                source = targets + np.array([delta_si, delta_sj])
                source = np.concatenate([
                    np.ones((targets.shape[0], 1), dtype=int) * sc,
                    source
                ], axis=-1)
                sources.append(source)
    source_values = [
        np.take(ac, np.ravel_multi_index(source.T, ac.shape))
        for source in sources
    ]
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

def _number_geometries(mask):
    # the number of geometries basically corresponds to the spacing between
    # the sampled lines out of the autocalibration area.
    # get the number of geometries
    sampled_lines = np.where(np.squeeze(mask))[0]
    first_1 = sampled_lines[0]
    second_1 = sampled_lines[1]
    n_geometries = second_1 - first_1 - 1
    return n_geometries

def _cartesian_product(*arrays):
    # taken from https://stackoverflow.com/a/11146645/4332585
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
