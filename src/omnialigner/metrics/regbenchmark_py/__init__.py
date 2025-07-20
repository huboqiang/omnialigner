from .tre import regbenchmark_PairwiseTRE, regbenchmark_PairwiseTRE_more_points
from .rtre import regbenchmark_AccumulatedTRE_linearfit, regbenchmark_AccumulatedTRE_cumulative
from .utils import kpt_pairs_to_fiducialpoints

import numpy as np
from scipy.io import savemat

def save_fiducialpoints_to_mat(fiducialpoints, filename='fiducialpoints.mat'):
    """
    Save fiducial points from Python (NumPy arrays) to a MATLAB .mat file.
    
    Parameters:
    ----------
    fiducialpoints : list of np.ndarray
        A list where each element is an Nx2 or Nx4 NumPy array representing
        fiducial points for each image or image pair.
        - Nx2 array: [Y, X] coordinates (global type)
        - Nx4 array: [Y1, X1, Y2, X2] coordinates (pairwise type)
    
    filename : str, optional
        The name of the output .mat file. Defaults to 'fiducialpoints.mat'.
    
    Returns:
    -------
    None
        Saves the fiducial points to the specified .mat file.
    """
    # Verify that fiducialpoints is a list of NumPy arrays
    if not isinstance(fiducialpoints, list):
        raise TypeError("fiducialpoints must be a list of NumPy arrays.")
    
    for idx, fp in enumerate(fiducialpoints):
        if not isinstance(fp, np.ndarray):
            raise TypeError(f"Element {idx} in fiducialpoints is not a NumPy array.")
        if fp.ndim != 2:
            raise ValueError(f"Element {idx} in fiducialpoints must be a 2D array.")
        if fp.shape[1] not in [2, 4]:
            raise ValueError(f"Element {idx} in fiducialpoints must have 2 or 4 columns.")
    
    # Convert the list of NumPy arrays to a MATLAB cell array
    fiducialpoints_matlab = np.empty((len(fiducialpoints), 1), dtype=object)
    for i, fp in enumerate(fiducialpoints):
        fiducialpoints_matlab[i, 0] = fp
    
    # Create a dictionary to save
    data_to_save = {'fiducialpoints': fiducialpoints_matlab}
    
    # Save to a .mat file
    savemat(filename, data_to_save)


def benchmark_kpts(l_kpt_pairs_nonrigid, pixelsize=0.46, slicethickness=5):
    fiducialpoints = kpt_pairs_to_fiducialpoints(l_kpt_pairs_nonrigid, n_kpt_used=4)

    #save_fiducialpoints_to_mat(fiducialpoints, "fiducialpoints.mat")
    TRE_pairwise = pixelsize*regbenchmark_PairwiseTRE(fiducialpoints)

    if len(fiducialpoints[0][1]) == 2:
        # print("regbenchmark_AccumulatedTRE_linearfit")
        [TRE_accumulated,fittedpoints] = regbenchmark_AccumulatedTRE_linearfit(fiducialpoints,slicethickness/pixelsize)
        TRE_accumulated = pixelsize*TRE_accumulated
    
    elif len(fiducialpoints[0][1]) == 4:
        # print("regbenchmark_AccumulatedTRE_cumulative")
        [TRE_accumulated,TRE_accumulated_vectors] = regbenchmark_AccumulatedTRE_cumulative(fiducialpoints)
        TRE_accumulated = pixelsize*TRE_accumulated
        TRE_accumulated_vectors = pixelsize*TRE_accumulated_vectors

    return TRE_pairwise, TRE_accumulated




def benchmark_kpts_more(l_kpt_pairs_nonrigid, pixelsize=0.46, slicethickness=5):
    fiducialpoints = kpt_pairs_to_fiducialpoints(l_kpt_pairs_nonrigid)

    #save_fiducialpoints_to_mat(fiducialpoints, "fiducialpoints.mat")
    TRE_pairwise = regbenchmark_PairwiseTRE_more_points(fiducialpoints, pixelsize=pixelsize)

    # print("regbenchmark_AccumulatedTRE_cumulative")
    [TRE_accumulated,TRE_accumulated_vectors] = regbenchmark_AccumulatedTRE_cumulative(fiducialpoints)
    TRE_accumulated = pixelsize*TRE_accumulated
    TRE_accumulated_vectors = pixelsize*TRE_accumulated_vectors
    return np.array([ x.mean() for x in TRE_pairwise]), TRE_accumulated
