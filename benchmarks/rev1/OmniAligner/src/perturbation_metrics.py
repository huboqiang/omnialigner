"""Visible SABench metric formulas for the released perturbation NPZ replay.

The notebook embeds these functions directly. Original expression dtypes,
lexicographic grid ordering, duplicate gene arguments and undefined scores are
preserved. The functions do not perform registration or checkpoint selection.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


def shared_grid(reference_xy, moving_xy):
    """Original 10 x 10 overlap bins, half-open boundaries and density rule."""
    xy = (reference_xy, moving_xy)
    low = np.maximum(reference_xy.min(0), moving_xy.min(0))
    high = np.minimum(reference_xy.max(0), moving_xy.max(0))
    boxes = []
    for i in range(10):
        for j in range(10):
            x_start = low[0] + i * (high[0] - low[0]) / 10
            y_start = low[1] + j * (high[1] - low[1]) / 10
            boxes.append((x_start, x_start + (high[0] - low[0]) / 10,
                          y_start, y_start + (high[1] - low[1]) / 10))

    def inside(points, box):
        x_start, x_end, y_start, y_end = box
        return ((points[:, 0] >= x_start) & (points[:, 0] < x_end)
                & (points[:, 1] >= y_start) & (points[:, 1] < y_end))

    masks = [[inside(points, box) for box in boxes] for points in xy]
    counts = [[mask.sum() for mask in side] for side in masks]
    threshold = min(np.mean(counts[0]), np.mean(counts[1])) / 2
    labels = [np.full(len(points), None, dtype=object) for points in xy]
    counter = 1
    for index in range(100):
        keep = counts[0][index] >= threshold and counts[1][index] >= threshold
        label = f'Grid_{counter}' if keep else 'NonOverlap'
        for side in range(2):
            labels[side][masks[side][index]] = label
        if keep:
            counter += 1
    supported = [set(values[pd.notna(values)]) - {'NonOverlap'} for values in labels]
    return dict(xy=xy, grid_labels=labels, supported_grids=supported)


def grid_expression(X, coords):
    """One gene's original dtype and the original pandas grouped mean."""
    means = []
    for expression, labels in zip(X, coords['grid_labels']):
        frame = pd.DataFrame({'expression': expression, 'grid_label': labels})
        frame = frame[(frame.grid_label != 'NonOverlap') & frame.grid_label.notnull()]
        means.append(frame.groupby('grid_label').mean().sort_index()['expression'].to_numpy())
    return tuple(means)


def PCC(X, coords):
    a, b = X
    return 0.0 if a.size < 2 or b.size < 2 else pearsonr(a, b).statistic


def CosSim(X, coords):
    a, b = X
    return 0.0 if a.size < 2 or b.size < 2 else a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def SSIM(X, coords):
    a, b = X
    if a.size < 2 or b.size < 2:
        return 0.0
    a, b = a / a.max(), b / b.max()
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = np.sqrt(((a - mu_a) ** 2).mean())
    sigma_b = np.sqrt(((b - mu_b) ** 2).mean())
    covariance = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    C3 = C2 / 2
    luminance = (2 * mu_a * mu_b + C1) / (mu_a ** 2 + mu_b ** 2 + C1)
    contrast = (2 * sigma_a * sigma_b + C2) / (sigma_a ** 2 + sigma_b ** 2 + C2)
    structure = (covariance + C3) / (sigma_a * sigma_b + C3)
    return luminance * contrast * structure


def MI(X, coords):
    a, b = X
    if a.size < 2 or b.size < 2:
        return 0.0
    np.random.seed(0)
    return mutual_info_regression(a.reshape(-1, 1), b)[0]


def MSE(X, coords):
    a, b = X
    return 100.0 if a.size < 2 or b.size < 2 else np.mean(np.square(a - b))


def Region(labels, coords):
    """Moving-to-reference label agreement within the closed overlap box."""
    reference_xy, moving_xy = coords['xy']
    low = np.maximum(reference_xy.min(0), moving_xy.min(0))
    high = np.minimum(reference_xy.max(0), moving_xy.max(0))
    keep = [((points >= low) & (points <= high)).all(1)
            for points in (reference_xy, moving_xy)]
    ref, mov = reference_xy[keep[0]], moving_xy[keep[1]]
    reference_labels, moving_labels = [np.asarray(side)[mask]
                                     for side, mask in zip(labels, keep)]
    _, neighbors = cKDTree(ref).query(mov)
    if len(neighbors) == 0:
        return 0.0
    matches = sum(reference_labels[target] == moving_labels[index]
                  for index, target in enumerate(neighbors)
                  if target < len(reference_labels) and index < len(moving_labels))
    return matches / len(neighbors)
