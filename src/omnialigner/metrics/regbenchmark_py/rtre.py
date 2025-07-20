import numpy as np

def regbenchmark_AccumulatedTRE_cumulative(fiducialpoints):
    """
    Calculate the cumulative Target Registration Error (TRE) and corresponding error vectors.

    Args:
    ----------
    fiducialpoints : list of np.ndarray
        A list of M elements, where each element is an N x 4 array.
        Each row in the array contains coordinates [Y1, X1, Y2, X2] of a single point 
        in image i and image i+1.

    Returns:
    -------
    TRE : np.ndarray
        A 1D array of length M containing the cumulative TRE values (in pixels) 
        for each image.
    
    TREvectors : np.ndarray
        An M x 2 array containing the resulting error vectors for each image pair.
        Each row contains the error components in [Y, X] directions.
    """
    
    numpairs = len(fiducialpoints)
    TRE = np.zeros(numpairs)
    TREvectors = np.zeros((numpairs, 2))
    
    cumerror = np.array([0.0, 0.0])
    
    for i in range(numpairs):
        Yerrors = fiducialpoints[i][:, 2] - fiducialpoints[i][:, 0]
        Xerrors = fiducialpoints[i][:, 3] - fiducialpoints[i][:, 1]
        
        TREvectors[i, 0] = np.nanmean(Yerrors)
        TREvectors[i, 1] = np.nanmean(Xerrors)
        cumerror += TREvectors[i, :]
        TRE[i] = np.sqrt(cumerror[0]**2 + cumerror[1]**2)
    
    return TRE, TREvectors

import numpy as np
from scipy import stats

def regbenchmark_AccumulatedTRE_linearfit(fiducialpoints, Zscaling):
    """
    Compute Accumulated Target Registration Errors (TRE) relative to linear fits.

    Args:
    ----------
    fiducialpoints : list of np.ndarray
        A list containing M elements, each corresponding to an image.
        Each element is an N x 2 array of fiducial point locations.
        Each row contains the [Y, X] coordinates of a single point.
        Points retain their correspondence across images.

    Zscaling : float
        A scalar specifying the interval of z-planes relative to the in-plane resolution.
        For example, for a pixel size of 1 µm and a section thickness of 10 µm,
        Zscaling should be 10.

    Returns:
    -------
    TRE : np.ndarray
        An M x N array containing the Accumulated Target Registration Error (ATRE)
        values for M images and N fiducial points in pixels.

    fittedpoints : list of np.ndarray
        A list containing M elements, each corresponding to an image.
        Each element is an N x 2 array of point locations on the fitted line.
    """
    # Determine the number of images and the number of fiducial points per image
    num_images = len(fiducialpoints)
    if num_images == 0:
        raise ValueError("The fiducialpoints list is empty.")

    num_fiducials = fiducialpoints[0].shape[0]

    # Initialize the result matrices
    fittedpoints = [np.empty((num_fiducials, 2)) for _ in range(num_images)]
    TRE = np.zeros((num_images, num_fiducials))

    # Iterate through each fiducial point
    for j in range(num_fiducials):
        # Initialize arrays to hold Y, X, and Z coordinates across all images
        pointstofitY = np.zeros(num_images)
        pointstofitX = np.zeros(num_images)
        pointstofitZ = np.zeros(num_images)

        for i in range(num_images):
            pointstofitY[i] = fiducialpoints[i][j, 0]  # Y-coordinate
            pointstofitX[i] = fiducialpoints[i][j, 1]  # X-coordinate
            pointstofitZ[i] = i * Zscaling             # Z-coordinate

        # Identify and exclude NaN points (missing fiducial points)
        nanpoints = np.isnan(pointstofitY)
        valid_indices = ~nanpoints

        # Extract non-NaN coordinates
        valid_X = pointstofitX[valid_indices]
        valid_Y = pointstofitY[valid_indices]
        valid_Z = pointstofitZ[valid_indices]

        # Check if there are enough points to perform fitting
        if len(valid_Z) < 2:
            # Not enough points to fit a line; assign NaN
            for i in range(num_images):
                fittedpoints[i][j, :] = [np.nan, np.nan]
                TRE[i, j] = np.nan
            continue

        # Compute the centroid of the valid data points
        P = np.array([
            np.mean(valid_X),
            np.mean(valid_Y),
            np.mean(valid_Z)
        ])

        # Perform linear least squares fitting for X and Y separately against Z
        # Fit X = P1 + t * V1
        slope_X, intercept_X, _, _, _ = stats.linregress(valid_Z, valid_X)
        # Fit Y = P2 + t * V2
        slope_Y, intercept_Y, _, _, _ = stats.linregress(valid_Z, valid_Y)

        # Construct the direction vector V
        V = np.array([slope_X, slope_Y, 1.0])

        # Iterate through each image to compute fitted points and TRE
        for i in range(num_images):
            if nanpoints[i]:
                # If the original point is missing, set fitted point and TRE to NaN
                fittedpoints[i][j, :] = [np.nan, np.nan]
                TRE[i, j] = np.nan
            else:
                # Solve for parameter t using the known Z coordinate
                t = (pointstofitZ[i] - P[2]) / V[2]
                # Compute the fitted Y-coordinate
                Y2 = P[1] + t * V[1]
                # Compute the fitted X-coordinate
                X2 = P[0] + t * V[0]
                # Assign the fitted point
                fittedpoints[i][j, :] = [Y2, X2]
                # Retrieve the actual coordinates
                Y1 = fiducialpoints[i][j, 0]
                X1 = fiducialpoints[i][j, 1]
                # Calculate the Euclidean distance between actual and fitted points
                TRE[i, j] = np.sqrt((Y2 - Y1)**2 + (X2 - X1)**2)

    return TRE, fittedpoints
