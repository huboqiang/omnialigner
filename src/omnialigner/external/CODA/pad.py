import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import mode
from skimage.color import rgb2gray

def pad_im_both2(im, sz, ext=0, fillval=None):
    """
    Pads histological images with whitespace to make files the same size for use in a nonlinear image registration algorithm.

    参数:
    - im: 输入图像 (NumPy 数组)
    - sz: 目标大小 (tuple 或 list，如 (height, width))
    - ext: 额外填充 (int, 默认为 0)
    - fillval: 填充值 (int 或 tuple, 如果为 None，则根据图像计算)

    返回:
    - im_padded: 填充后的图像 (NumPy 数组)
    """
    if fillval is None:
        if im.ndim == 3:
            fillval = tuple(mode(im, axis=(0, 1))[0][0].astype(int))
        else:
            fillval = int(mode(im, axis=None)[0][0])
    
    if im.ndim == 3 and isinstance(fillval, int):
        fillval = (fillval, fillval, fillval)

    szim = np.array([sz[0] - im.shape[0], sz[1] - im.shape[1]])
    szA = np.floor_divide(szim, 2)
    szB = szim - szA + ext
    szA = szA + ext

    if im.ndim == 3:
        pad_width_pre = ((szA[0], szB[0]), (szA[1], szB[1]), (0, 0))
        im_padded = np.pad(im, pad_width_pre, mode='constant', constant_values=((fillval[0], fillval[0]),
                                                                                  (fillval[1], fillval[1]),
                                                                                  (fillval[2], fillval[2])))
    else:
        pad_width_pre = ((szA[0], szB[0]), (szA[1], szB[1]))
        im_padded = np.pad(im, pad_width_pre, mode='constant', constant_values=fillval)
    
    return im_padded

def preprocessing(im, TA, szz, padall, IHC):
    """
    Performs pre-processing of histological images for use in a nonlinear image registration algorithm.

    参数:
    - im: 输入图像 (NumPy 数组)
    - TA: 相关数组 (NumPy 数组)
    - szz: 目标大小 (tuple 或 list，如 (height, width))
    - padall: 填充参数 (如果为空，则不进行填充)
    - IHC: 标志 (int)

    返回:
    - im: 处理后的图像 (NumPy 数组)
    - impg: 处理后的灰度图像 (NumPy 数组)
    - TA: 更新后的 TA (NumPy 数组)
    - fillval: 填充值 (int 或 tuple)
    """
    fillval = [0, 0, 0] if im.ndim == 3 else 0
    if IHC == 5:
        fillval = [0, 0, 0] if im.ndim == 3 else 0
    elif IHC == 2:
        fillval = [241, 241, 241] if im.ndim == 3 else 241
    
    if padall:
        im = pad_im_both2(im, szz, padall, fillval)
        if TA.shape[0] != im.shape[0] or TA.shape[1] != im.shape[1]:
            TA = pad_im_both2(TA, szz, padall, 0)
    
    TA = TA > 0
    
    if im.dtype != np.uint8:
        im = (255 * (im / im.max())).astype(np.uint8)
    
    if IHC == 2:
        impg = rgb2gray(im).astype(np.uint8) * 255  # rgb2gray 返回 0-1 的浮点数
        TA = np.ones(im.shape[:2], dtype=bool)
    elif IHC == 5:
        impg = cv2.bitwise_not(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
    else:
        if im.ndim == 3:
            imp = im.copy()
            imp[~TA] = 255
            impg = cv2.bitwise_not(cv2.cvtColor(imp, cv2.COLOR_RGB2GRAY))
        else:
            imp = im.copy()
            imp[~TA] = 255
            impg = cv2.bitwise_not(imp)
    
    impg = ndimage.gaussian_filter(impg, sigma=2)
    
    return im, impg, TA, fillval

def mode_multidimensional(a, axis=None):
    """
    计算多维数组的众数，模仿 MATLAB 的 mode 函数行为。

    参数:
    - a: 输入数组
    - axis: 计算众数的轴

    返回:
    - mode_result: 众数
    - count: 众数的计数
    """
    m, c = mode(a, axis=axis)
    return m, c

