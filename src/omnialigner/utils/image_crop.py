import numpy as np


def find_strongest_continuous_signal(signal, threshold, min_length):
    """
    Find the strongest continuous region above threshold in a 1D signal.

    :param signal: 1D signal array.
    :param threshold: Signal threshold.
    :param min_length: Minimum length of continuous region.
    :return: Start and end indices of the strongest continuous region.
    """
    above_threshold = signal > threshold
    starts = np.where(np.diff(np.concatenate(([False], above_threshold, [False]))))[0][::2]
    ends = np.where(np.diff(np.concatenate(([False], above_threshold, [False]))))[0][1::2]

    valid_regions = [(start, end) for start, end in zip(starts, ends) if end - start >= min_length]
    if not valid_regions:
        return 0, len(signal)

    strongest_region = max(valid_regions, key=lambda region: np.mean(signal[region[0]:region[1]]))
    return strongest_region

def crop_noise_area(mat, threshold=0.1, min_length=10):
    """
    Crop noise areas from an image matrix.

    :param mat: Input image matrix, NxM.
    :param threshold: Threshold for determining valid data regions.
    :param min_length: Minimum length of continuous valid data region.
    :return: Cropped image matrix.
    """
    row_means = np.mean(mat, axis=1)
    col_means = np.mean(mat, axis=0)
    H, W = mat.shape[0], mat.shape[1]
    row_start, row_end = find_strongest_continuous_signal(row_means, threshold, min_length)
    col_start, col_end = find_strongest_continuous_signal(col_means, threshold, min_length)
    
    np_coords = np.array([row_start, col_start, H-row_end, W-col_end])
    
    return mat[row_start:row_end, col_start:col_end], np_coords

