import numpy as np

from grappa.kernel_estimation import kernel_estimation
from grappa.kernel_application import apply_kernel


def ifft2(kspace):
    image = np.fft.fftshift(np.fft.ifft2(kspace, norm='ortho', axes=(-2, -1)))
    return image

def rss(kspace):
    image = ifft2(kspace)
    image_rss = np.linalg.norm(image, axis=0)
    return image_rss

def crop(image, output_shape=None):
    """fastMRI-style cropping
    """
    if output_shape is None:
        output_shape = (320, 320)
    output_shape = np.array(output_shape)
    current_shape = np.array(image.shape)
    to_crop = (current_shape - output_shape)//2
    to_crop_left = to_crop
    to_crop_right = - to_crop
    zero_crop = np.where(to_crop == 0)[0]
    to_crop_right[zero_crop] = current_shape[zero_crop]
    image_cropped = image[
        to_crop_left[0]:to_crop_right[0],
        to_crop_left[1]:to_crop_right[1],
    ]
    return image_cropped

def slice_reconstruction(
        kspace,
        mask,
        ny=3,
        output_shape=None,
        lamda=1e-6,
        fastmri=True,
        backend='tensorflow',
    ):
    grappa_kernels = kernel_estimation(
        kspace,
        mask,
        ny=ny,
        lamda=lamda,
        fastmri=fastmri,
        backend=backend,
    )
    filled_kspace = apply_kernel(kspace, mask, grappa_kernels, backend=backend)
    reco_grappa = rss(filled_kspace)
    reco_grappa_cropped = crop(reco_grappa, output_shape=output_shape)
    return reco_grappa_cropped

def fastmri_volume_reconstruction(
        kspace,
        mask,
        ny=3,
        output_shape=None,
        lamda=1e-6,
        backend='tensorflow',
    ):
    reco_slices = list()
    for kspace_slice, mask_slice in zip(kspace, mask):
        reco_slice = slice_reconstruction(
            kspace_slice,
            mask_slice,
            ny=ny,
            output_shape=output_shape,
            lamda=lamda,
            fastmri=True,
            backend=backend
        )
        reco_slices.append(reco_slice)
    reco_slices = np.array(reco_slices)
    return reco_slices
