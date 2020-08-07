import numpy as np

from grappa.utils import cartesian_product, sources_from_targets, eval_at_positions


def apply_kernel(kspace, mask, grappa_kernels):
    if mask is None:
        raise ValueError('For now mask has to be passed for kernel estimation')
    ncoils = kspace.shape[0]
    ny = grappa_kernels[0].shape[1] // (ncoils * 2)
    pad_right, pad_left = _padding_for_kspace(mask)
    kspace_padded = np.pad(
        kspace,
        [
            (0, 0),
            (ny//2, ny//2),
            (pad_left, pad_right),
        ]
    )
    for i_geom, grappa_kernel in enumerate(grappa_kernels):
        _geom_apply_kernel()

def _geom_apply_kernel(kspace, grappa_kernel, i_geom, spacing=4, ny=3, ncoils=15):
    targets = cartesian_product(
        # readout dimension
        np.arange(ny // 2, kspace.shape[1] - ny + (ny // 2)),
        # phase dimension
        # 1 is for the first dimension
        np.arange(i_geom + 1, kspace.shape[2] - spacing + i_geom, spacing+1),
    )
    sources = sources_from_targets(
        targets,
        i_geom,
        spacing,
        ny,
        ncoils,
    )
    source_values = eval_at_positions(kspace, sources)
    target_values = grappa_kernel @ source_values
    for c in range(ncoils):
        np.put(
            kspace[c],
            np.ravel_multi_index(targets.T, kspace[c].shape),
            target_values[c],
        )



def _padding_for_kspace(mask):
    sampled_lines = np.where(np.squeeze(mask))[0]
    first_1 = sampled_lines[0]
    second_1 = sampled_lines[1]
    last_1 = sampled_lines[-1]
    n_phase = mask.size

    spacing = second_1 - first_1 - 1

    if first_1 == 0:
        pad_left = 0
    else:
        pad_left = spacing - first_1 + 1

    if last_1 == n_phase - 1:
        pad_right = 0
    else:
        # we want last_1 + spacing + 1 = (n_phase-1) + pad_right
        pad_right = last_1 + spacing - (n_phase - 1) + 1


    return pad_right, pad_left
