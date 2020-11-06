import time

import numpy as np

from grappa.reconstruction import slice_reconstruction


# Benchmark params
R = 4
dim = 256
ncoils = 15
num_grappa = 20

# Data generation
kspace = np.random.normal(size=[ncoils, dim, dim]).astype(np.complex64)
mask = np.array([
    i%R == 0 for i in range(dim)
])
mask[dim//2-12:dim//2+12] = True
kspace_undersampled = kspace * mask[None, None, :].astype(np.complex64)

for backend in ['numpy', 'tensorflow']:
    # warm-up
    for i in range(5):
        slice_reconstruction(
            kspace_undersampled,
            mask,
            backend=backend,
            fastmri=False,
            ny=3,
        )

    # real computation
    start_time = time.perf_counter()
    for i in range(num_grappa):
        slice_reconstruction(
            kspace_undersampled,
            mask,
            backend=backend,
            fastmri=False,
            ny=3,
        )
    end_time = time.perf_counter()
    avg_time = (end_time-start_time) / num_grappa
    print('{backend} backend takes {avg_time}s')
