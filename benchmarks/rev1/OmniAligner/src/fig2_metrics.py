"""Original Figure 2 categorical scores, with bounded-memory CLC evaluation.

The notebook displays this complete implementation. No expression matrix enters
these scores. CLC preserves the historical directional 1-NN fallback and 2.5%
neighbourhood definition; its name in the original source was CLC-MNN.
"""
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from numba import njit


def ari(reference_image, moving_image):
    """Categorical raster ARI, including the background pixels."""
    return float(adjusted_rand_score(reference_image.ravel(), moving_image.ravel()))


def nmi(reference_image, moving_image):
    """Categorical raster NMI, sklearn arithmetic normalization."""
    return float(normalized_mutual_info_score(reference_image.ravel(), moving_image.ravel()))


@njit(cache=True)
def _membership_fraction(mapped_neighbors, target_neighbors):
    """Exact set membership using a reusable integer mark array."""
    n, k = mapped_neighbors.shape
    upper = max(np.max(mapped_neighbors), np.max(target_neighbors)) + 1
    marks = np.zeros(upper, dtype=np.int64)
    result = np.zeros(n)
    for row in range(n):
        stamp = row + 1
        for col in range(k):
            marks[target_neighbors[row, col]] = stamp
        count = 0
        for col in range(k):
            if marks[mapped_neighbors[row, col]] == stamp:
                count += 1
        result[row] = count / k
    return result


def clc(labels_reference, labels_moving, coords_reference, coords_moving, batch_size=256):
    """Directional original CLC, computed in bounded-memory blocks.

    Original MNN code keeps a mutual match when present and falls back to the
    source-to-target 1-NN otherwise. Both branches choose the same target index,
    so the final mapping is the single directional nearest-neighbour query.
    Each source point contributes label agreement times the fraction of its
    mapped neighbours belonging to its target point's own neighbourhood.
    Neighbourhoods include the point itself, exactly as in the original code.
    """
    n1, n2 = len(coords_reference), len(coords_moving)
    k = max(1, int((n1 + n2) / 2 * 2.5 / 100))
    if k > min(n1, n2):
        raise ValueError('Historical neighbourhood exceeds one section size.')
    tree1, tree2 = KDTree(coords_reference), KDTree(coords_moving)
    target_index = tree2.query(coords_reference, k=1)[1]
    agreement = labels_reference == labels_moving[target_index]
    contributions = np.zeros(n1, dtype=np.float64)
    active = np.flatnonzero(agreement)
    for start in range(0, len(active), batch_size):
        rows = active[start:start + batch_size]
        source_neighbors = tree1.query(coords_reference[rows], k=k)[1].reshape(len(rows), k)
        target_neighbors = tree2.query(coords_moving[target_index[rows]], k=k)[1].reshape(len(rows), k)
        mapped_neighbors = target_index[source_neighbors]
        contributions[rows] = _membership_fraction(mapped_neighbors, target_neighbors)
    return float(contributions.mean())


def clc_from_neighbors(labels_reference, labels_moving, graph, batch_size=256):
    """Recalculate CLC on frozen evaluation neighbours, including tie choices."""
    target_index = graph['target_index']
    agreement = labels_reference == labels_moving[target_index]
    contributions = np.zeros(len(labels_reference), dtype=np.float64)
    rows_active = np.flatnonzero(agreement)
    target_neighbors = graph['target_neighbors']
    source_neighbors = graph['source_neighbors']
    for start in range(0, len(rows_active), batch_size):
        rows = rows_active[start:start + batch_size]
        mapped_neighbors = target_index[source_neighbors[rows]]
        sorted_neighbors = target_neighbors[target_index[rows]]
        contributions[rows] = _membership_fraction(mapped_neighbors, sorted_neighbors)
    return float(contributions.mean())


def score_method(npz_path, neighbor_folder=None):
    """Yield one row per adjacent pair; inputs retain original cell ordering."""
    with np.load(npz_path, allow_pickle=False) as data:
        layers = np.unique(data['layers'])
        labels, coords, raster = data['labels'], data['coords'], data['raster']
        if len(layers) != len(raster):
            raise ValueError('Raster/coordinate layer counts differ.')
        for pair, (left, right) in enumerate(zip(layers[:-1], layers[1:])):
            reference, moving = data['layers'] == left, data['layers'] == right
            score = clc(labels[reference], labels[moving], coords[reference], coords[moving])
            result = {'pair': pair,
                      'ari': ari(raster[pair], raster[pair + 1]),
                      'nmi': nmi(raster[pair], raster[pair + 1]),
                      'CLC': score}
            if neighbor_folder is not None:
                with np.load(neighbor_folder / f'{pair}.npz') as graph:
                    result['coordinate_CLC'] = score
                    result['CLC'] = clc_from_neighbors(labels[reference], labels[moving], graph)
            yield result
