import numpy as np
import tensorflow as tf


def cartesian_product(*arrays):
    # taken from https://stackoverflow.com/a/11146645/4332585
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def sources_from_targets(targets, i_geom, n_geometries=4, ny=3, ncoils=15):
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
    return sources

def eval_at_positions(values, indices_list):
    evaluations = [
        np.take(values, np.ravel_multi_index(indices.T, values.shape))
        for indices in indices_list
    ]
    evaluations = np.array(evaluations)
    return evaluations

def number_geometries(mask):
    # the number of geometries basically corresponds to the spacing between
    # the sampled lines out of the autocalibration area.
    # get the number of geometries
    sampled_lines = np.where(np.squeeze(mask))[0]
    first_1 = sampled_lines[0]
    second_1 = sampled_lines[1]
    n_geometries = second_1 - first_1 - 1
    return n_geometries

def pinv(a, rcond=None):
    """Taken from
    https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/ops/linalg/linalg_impl.py
    """
    dtype = a.dtype.as_numpy_dtype

    if rcond is None:
        def get_dim_size(dim):
            dim_val = a.shape[dim]
            if dim_val is not None:
                return dim_val
            return tf.shape(a)[dim]

        num_rows = get_dim_size(-2)
        num_cols = get_dim_size(-1)
        if isinstance(num_rows, int) and isinstance(num_cols, int):
            max_rows_cols = float(max(num_rows, num_cols))
        else:
            max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
        rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = tf.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is Hermitian then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,  # Sigma
        left_singular_vectors,  # U
        right_singular_vectors,  # V
    ] = tf.linalg.svd(
        a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = tf.cast(rcond, dtype=singular_values.dtype) * tf.reduce_max(singular_values, axis=-1)
    singular_values = tf.where(
        singular_values > cutoff[..., None], singular_values,
        np.array(np.inf, dtype))

    # By the definition of the SVD, `a == u @ s @ v^H`, and the pseudo-inverse
    # is defined as `pinv(a) == v @ inv(s) @ u^H`.
    a_pinv = tf.matmul(
        right_singular_vectors / tf.cast(singular_values[..., None, :], dtype=dtype),
        left_singular_vectors,
        adjoint_b=True)

    if a.shape is not None and a.shape.rank is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv
