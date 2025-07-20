
import numpy as np

def regbenchmark_PairwiseTRE(fiducialpoints):
    """
    Calculate the Pairwise TRE (TRE). 4 keypoints for raw code.
    
    Args:
    ----------
    fiducialpoints : list of np.ndarray
        A list of M elements, each corresponding to a set of fiducial points.
        Each element can be one of the following formats:
        
        - N x 2 : [Y, X] coordinates.
            - This format represents "global" type fiducial points, where all images share the same fiducial points.
        - N x 4 : Each row contains the coordinates of a fiducial point in two images [Y1, X1, Y2, X2].
            - This format represents "pairwise" type fiducial points, where each image pair has its own independent fiducial points.
    
    Returns:
    -------
    TRE : np.ndarray
        An M x N array containing the TRE values for M image pairs and N fiducial points.
        - M: The number of image pairs.
        - N: The number of fiducial points per image pair.
    """
    
    if not fiducialpoints:
        raise ValueError("fiducialpoints 列表为空。")
    
    first_point = fiducialpoints[0]
    if first_point.shape[1] == 2:
        fiducialtype = 'global'
    elif first_point.shape[1] == 4:
        fiducialtype = 'pairwise'
    else:
        raise ValueError("无效的标志点矩阵！每个标志点应为 N x 2 或 N x 4 矩阵。")
    
    # 确定图像对的数量
    if fiducialtype == 'global':
        numpairs = len(fiducialpoints) - 1
        if numpairs < 1:
            raise ValueError("对于 'global' 类型，fiducialpoints 列表的长度应至少为 2。")
    else:  # 'pairwise'
        numpairs = len(fiducialpoints)
    
    # 确定每对图像的标志点数量
    numfiducials = fiducialpoints[0].shape[0]
    
    # 初始化结果数组
    TRE = np.zeros((numpairs, numfiducials))
    
    # 遍历每对图像
    for i in range(numpairs):
        for j in range(numfiducials):
            if fiducialtype == 'global':
                # 获取当前标志点在两幅图像中的坐标
                Y1, X1 = fiducialpoints[i][j, 0], fiducialpoints[i][j, 1]
                Y2, X2 = fiducialpoints[i + 1][j, 0], fiducialpoints[i + 1][j, 1]
            else:  # 'pairwise'
                # 获取当前标志点在两幅图像中的坐标
                Y1, X1, Y2, X2 = fiducialpoints[i][j, 0], fiducialpoints[i][j, 1], fiducialpoints[i][j, 2], fiducialpoints[i][j, 3]
            
            # 计算两个标志点之间的欧几里得距离
            TRE[i, j] = np.sqrt((Y2 - Y1)**2 + (X2 - X1)**2)
    
    return TRE


def regbenchmark_PairwiseTRE_more_points(fiducialpoints, pixelsize=0.46):
    """
    Calculate the Pairwise TRE for regbenchmark_AccumulatedTRE_cumulative, revised for > 4 keypoint pairs(TRE)。
    
    Args:
    ----------
    fiducialpoints : list of np.ndarray
        A list of M elements, each corresponding to a set of fiducial points.
        Each element can be one of the following formats:
        
        - N x 2 : [Y, X] coordinates.
            - This format represents "global" type fiducial points, where all images share the same fiducial points.
        - N x 4 : Each row contains the coordinates of a fiducial point in two images [Y1, X1, Y2, X2].
            - This format represents "pairwise" type fiducial points, where each image pair has its own independent fiducial points.
    
    Returns:
    -------
    TRE : np.ndarray
        An M x N array containing the TRE values for M image pairs and N fiducial points.
        - M: The number of image pairs.
        - N: The number of fiducial points per image pair.
    """
    numpairs = len(fiducialpoints) - 1
    TRE = []
    for i in range(numpairs):
        numfiducials = fiducialpoints[i].shape[0]
        local_TRE = []
        for j in range(numfiducials):
            Y1, X1, Y2, X2 = fiducialpoints[i][j, 0], fiducialpoints[i][j, 1], fiducialpoints[i][j, 2], fiducialpoints[i][j, 3]
            local_TRE.append(np.sqrt((Y2 - Y1)**2 + (X2 - X1)**2))

        TRE.append(pixelsize*np.array(local_TRE))

    return TRE
