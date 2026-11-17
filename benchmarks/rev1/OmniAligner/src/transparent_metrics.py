"""Explicit versions of the frozen SABench metrics, also shown in the notebooks.

X contains the reference and moving expression vectors for one gene.
coords contains their XY arrays and the shared 10 x 10 grid assignments.
Region instead receives the two regional-label vectors.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
from sklearn.feature_selection import mutual_info_regression


def shared_grid(reference_xy, moving_xy):
    """Use the original overlap box, half-open bins and native density threshold."""
    xy = (reference_xy, moving_xy)
    x_min = max(np.min(reference_xy[:, 0]), np.min(moving_xy[:, 0]))
    x_max = min(np.max(reference_xy[:, 0]), np.max(moving_xy[:, 0]))
    y_min = max(np.min(reference_xy[:, 1]), np.min(moving_xy[:, 1]))
    y_max = min(np.max(reference_xy[:, 1]), np.max(moving_xy[:, 1]))
    bins = []
    for i in range(10):
        for j in range(10):
            x_start = x_min + i * (x_max - x_min) / 10
            y_start = y_min + j * (y_max - y_min) / 10
            x_end = x_start + (x_max - x_min) / 10
            y_end = y_start + (y_max - y_min) / 10
            bins.append((x_start, x_end, y_start, y_end))

    def inside(points, box):
        x_start, x_end, y_start, y_end = box
        return ((points[:, 0] >= x_start) & (points[:, 0] < x_end)
                & (points[:, 1] >= y_start) & (points[:, 1] < y_end))

    counts = [[np.sum(inside(points, box)) for box in bins] for points in xy]
    threshold = min(np.mean(counts[0]), np.mean(counts[1])) / 2
    labels = [np.full(len(points), None, dtype=object) for points in xy]
    counter = 1
    for index, box in enumerate(bins):
        keep = counts[0][index] >= threshold and counts[1][index] >= threshold
        name = f"Grid_{counter}" if keep else "NonOverlap"
        for points, values in zip(xy, labels):
            values[inside(points, box)] = name
        if keep:
            counter += 1
    return {"xy": xy, "grid_labels": labels}


def grid_expression(X, coords):
    """Mean expression per retained bin, sorted by the native string grid label."""
    means = []
    for expression, labels in zip(X, coords["grid_labels"]):
        frame = pd.DataFrame({"expression": expression, "grid_label": labels})
        frame = frame[(frame.grid_label != "NonOverlap") & frame.grid_label.notnull()]
        means.append(frame.groupby("grid_label").mean().sort_index()["expression"].to_numpy())
    return means


def PCC(X, coords):
    a, b = grid_expression(X, coords)
    return 0.0 if a.size < 2 or b.size < 2 else pearsonr(a, b).statistic


def CosSim(X, coords):
    a, b = grid_expression(X, coords)
    return 0.0 if a.size < 2 or b.size < 2 else a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def SSIM(X, coords):
    a, b = grid_expression(X, coords)
    if a.size < 2 or b.size < 2:
        return 0.0
    # Native convention: separately divide each gene's grid vector by its maximum.
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
    a, b = grid_expression(X, coords)
    if a.size < 2 or b.size < 2:
        return 0.0
    # Reset the seed as in the frozen scorer, including its jitter convention.
    np.random.seed(0)
    return mutual_info_regression(a.reshape(-1, 1), b)[0]


def MSE(X, coords):
    a, b = grid_expression(X, coords)
    if a.size < 2 or b.size < 2:
        return 100.0  # Native fallback, excluded from eligible table rankings.
    return np.mean(np.square(a - b))


def Region(labels, coords):
    """Moving-to-reference nearest-neighbor label agreement in the overlap box."""
    reference_xy, moving_xy = coords["xy"]
    x_min = max(np.min(reference_xy[:, 0]), np.min(moving_xy[:, 0]))
    x_max = min(np.max(reference_xy[:, 0]), np.max(moving_xy[:, 0]))
    y_min = max(np.min(reference_xy[:, 1]), np.min(moving_xy[:, 1]))
    y_max = min(np.max(reference_xy[:, 1]), np.max(moving_xy[:, 1]))
    keep = [((points[:, 0] >= x_min) & (points[:, 0] <= x_max)
             & (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
            for points in (reference_xy, moving_xy)]
    reference_xy, moving_xy = reference_xy[keep[0]], moving_xy[keep[1]]
    reference_labels = np.asarray(labels[0])[keep[0]]
    moving_labels = np.asarray(labels[1])[keep[1]]
    _, neighbors = cKDTree(reference_xy).query(moving_xy)
    if len(neighbors) == 0:
        return 0.0
    matches = sum(reference_labels[target] == moving_labels[index]
                  for index, target in enumerate(neighbors)
                  if target < len(reference_labels) and index < len(moving_labels))
    return matches / len(neighbors)
