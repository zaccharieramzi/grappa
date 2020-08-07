import numpy as np

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
