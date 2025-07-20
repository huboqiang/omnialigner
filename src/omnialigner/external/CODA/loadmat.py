import cv2
import pandas as pd
from scipy.ndimage import map_coordinates, zoom
from scipy.interpolate import griddata
import matlab.engine
import numpy as np
from omnialigner.external.CODA.pad import pad_im_both2, preprocessing
import matplotlib.pyplot as plt

def calculate_landmarks_offset(landmarks, szz, raw_size, padall=250):
    H_raw, W_raw = raw_size
    szim = np.array([szz[0] - H_raw, szz[1] - W_raw])
    szA = np.floor_divide(szim, 2)
    # szA = np.zeros_like(szim)
    szA = szA + padall
    landmarks_padded = landmarks + np.array([szA[1], szA[0]])
    return landmarks_padded


def invert_D(D, skk=10, skk2=3):
    """
    Inverts the displacement field D similar to the MATLAB code.
    
    Parameters:
        D (np.ndarray): Input displacement field with shape (rows, cols, 2).
        skk (int): Subsampling factor for scattered interpolation (default: 10).
        skk2 (int): Subsampling factor for evaluation grid (default: 3).
        
    Returns:
        Dnew (np.ndarray): Inverted displacement field with same shape as D.
    """
    rows, cols = D.shape[0], D.shape[1]
    
    # Create meshgrid with MATLAB-style (1-indexed) coordinates
    xx, yy = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    
    # Calculate new positions:
    xnew = xx + D[:, :, 0]
    ynew = yy + D[:, :, 1]
    
    # Flatten arrays for interpolation
    D1 = D[:, :, 0].flatten()
    D2 = D[:, :, 1].flatten()
    xnew2 = xnew.flatten()
    ynew2 = ynew.flatten()
    
    # Subsample the data (MATLAB: 1:skk:end is equivalent to Python slicing with step)
    inds = np.arange(0, len(D1), skk)
    points_sub = np.vstack((xnew2[inds], ynew2[inds])).T
    values1_sub = D1[inds]
    values2_sub = D2[inds]
    
    # Create evaluation grid (using 1-indexed positions)
    grid_x, grid_y = np.meshgrid(np.arange(1, cols + 1, skk2),
                                 np.arange(1, rows + 1, skk2))
    
    # Perform scattered data interpolation using linear method
    D1_interp = griddata(points_sub, values1_sub, (grid_x, grid_y), method='linear')
    D2_interp = griddata(points_sub, values2_sub, (grid_x, grid_y), method='linear')
    
    # Multiply by -1 to get the inverted field
    D1_interp = -D1_interp
    D2_interp = -D2_interp
    
    # Resize the interpolated fields to the original image size
    # Compute zoom factors from the size of the evaluation grid to the original size
    zoom_factor_row = rows / grid_x.shape[0]
    zoom_factor_col = cols / grid_x.shape[1]
    D1_resized = zoom(D1_interp, (zoom_factor_row, zoom_factor_col), order=1)
    D2_resized = zoom(D2_interp, (zoom_factor_row, zoom_factor_col), order=1)
    
    # Crop if necessary in case of rounding issues in zoom
    D1_resized = D1_resized[:rows, :cols]
    D2_resized = D2_resized[:rows, :cols]
    
    # Create the output array
    Dnew = np.zeros_like(D)
    Dnew[:, :, 0] = D1_resized
    Dnew[:, :, 1] = D2_resized
    
    # Replace any NaN values with zero
    Dnew = np.nan_to_num(Dnew)
    
    return Dnew


def register_landmarks(landmarks_M, np_cent, np_tform, f, np_D, szz, raw_size, padall=250):
    """
    基于标记点从图像 F 到图像 M 进行图像注册。

    参数:
    - landmarks_M: 目标图像 M 的标记点，形状为 (N, 2) 的 NumPy 数组。
    - np_cent: 中心点 (tuple 或 list，如 (cent_x, cent_y))，见 register_global_im
    - np_tform: 仿射变换矩阵 (3x3 NumPy 数组)，见 register_global_im
    - f: 翻转标志 (int, 如果为1则垂直翻转标记点)
    - np_D: 形变场 (NumPy 数组，形状为 (H, W, 2))，见 imwarp_map_coordinates
    - szz: 目标图像尺寸 (tuple, 如 (H, W))
    - raw_size: 原始图像尺寸 (tuple, 如 (H, W))
    - padall: 填充像素数 (int, 默认250)


    返回:
    - landmarks_M_to_F: 变换后的标记点，形状为 (N, 2) 的 NumPy 数组
    """

    # 1. Apply padding and szz for landmarks_M
    # 假设 padall 和 szz 是已知的，并且与图像的填充和尺寸调整一致
    # 例如，padall 是填充的像素数，szz 是目标尺寸 (height, width)
    # 这里需要根据实际情况调整 padall 和 szz 的值

    # 应用填充：将标记点的坐标根据 padall 进行偏移
    H, W = np_D.shape[:2]
    np_D = invert_D(np_D)

    landmarks_padded = calculate_landmarks_offset(landmarks_M, szz, raw_size, padall)
    if f == 1:
        landmarks_padded[:, 1] = H - landmarks_padded[:, 1] - 1

    # 2. Apply affine transformation to landmarks_M considering np_cent
    N = landmarks_padded.shape[0]
    ones = np.ones((N, 1))
    landmarks_homog = np.hstack([landmarks_padded, ones])  # 形状 (N, 3)

    # 应用仿射变换矩阵
    T_translate_to_origin = np.array([
        [1, 0, -np_cent[0]],
        [0, 1, -np_cent[1]],
        [0, 0, 1]
    ])

    T_translate_back = np.array([
        [1, 0, np_cent[0]],
        [0, 1, np_cent[1]],
        [0, 0, 1]
    ])

    T_total = T_translate_back @ np_tform @ T_translate_to_origin
    # affine_matrix = T_total[:2, :]

    
    landmarks_transformed_homog = (T_total @ landmarks_homog.T).T  # 形状 (N, 3)

    # 转换回非齐次坐标
    landmarks_transformed = landmarks_transformed_homog[:, :2] / landmarks_transformed_homog[:, 2, np.newaxis]

    # 3. Apply deformation field np_D to landmarks_transformed
    # 提取 y 和 x 坐标
    y_coords = landmarks_transformed[:, 1]
    x_coords = landmarks_transformed[:, 0]


    # 使用 map_coordinates 插值获取偏移量
    # 注意：np_D[..., 0] 表示 y 方向的偏移，np_D[..., 1] 表示 x 方向的偏移
    y_coords = np.clip(y_coords.astype(np.int32), 0, np_D.shape[0]-1)
    x_coords = np.clip(x_coords.astype(np.int32), 0, np_D.shape[1]-1)
    
    D_y = np_D[y_coords, x_coords, 1]
    D_x = np_D[y_coords, x_coords, 0]

    # 应用偏移量
    # landmarks_M_to_F = landmarks_transformed + np.vstack([D_y, D_x]).T  # 形状 (N, 2)
    landmarks_M_to_F = landmarks_transformed + np.vstack([D_x, D_y]).T  # 形状 (N, 2)

    # 4. 如果需要，应用垂直翻转
    # if f == 1:

    #     landmarks_M_to_F[:, 1] = H - landmarks_M_to_F[:, 1] - 1

    return landmarks_M_to_F


def imwarp_map_coordinates(image, D, fillval=0):
    """
    Apply a deformation field to an image using scipy.ndimage.map_coordinates.

    Parameters:
    - image: Input image as a NumPy array (H, W) or (H, W, C).
    - D: Deformation field as a NumPy array with shape (H, W, 2).
         D[y, x, 0] is the y-coordinate in the input image.
         D[y, x, 1] is the x-coordinate in the input image.
    - fillval: Fill value for areas outside the input image.

    Returns:
    - warped_image: Transformed image as a NumPy array.
    """
    if D.shape[:2] != image.shape[:2]:
        raise ValueError("Deformation field D must have the same height and width as the image.")

    if D.shape[2] != 2:
        raise ValueError("Deformation field D must have 2 channels for y and x coordinates.")

    # Extract the y and x coordinates from the deformation field
    # y_coords, x_coords = D[..., 0], D[..., 1]
    H, W = image.shape[:2]
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 应用偏移量
    y_new = y + D[..., 1]
    x_new = x + D[..., 0]
    
    # 确保坐标在有效范围内
    y_new = np.clip(y_new, 0, H - 1)
    x_new = np.clip(x_new, 0, W - 1)
    
    # 堆叠坐标并展平
    coordinates = np.vstack([y_new.ravel(), x_new.ravel()])

    # Stack coordinates for map_coordinates
    # map_coordinates expects the first dimension to be y and the second to be x
    # coordinates = np.vstack([y_coords.ravel(), x_coords.ravel()])

    # Initialize the output image
    if image.ndim == 3:
        channels = image.shape[2]
        warped_image = np.zeros_like(image)
        for c in range(channels):
            # Apply map_coordinates for each channel
            warped_channel = map_coordinates(
                image[..., c],
                coordinates,
                order=0,                # nearest neighbor
                mode='constant',
                cval=fillval
            )

            warped_image[..., c] = warped_channel.reshape([H, W])
    else:
        # Single-channel image
        warped_image = map_coordinates(
            image,
            coordinates,
            order=0,                    # nearest neighbor
            mode='constant',
            cval=fillval
        ).reshape([H, W])

    return warped_image

def register_global_im(im, tform, cent, f=0, fillval=0):
    """
    注册图像，应用仿射变换。

    参数:
    - im: 输入图像 (NumPy 数组)
    - tform: 仿射变换矩阵 (3x3 NumPy 数组)
    - cent: 中心点 (tuple 或 list，如 (cent_x, cent_y))
    - f: 翻转标志 (int, 如果为1则垂直翻转图像)
    - fillval: 填充值 (用于填充变换后图像中的空白区域)

    返回:
    - imG: 注册后的图像 (NumPy 数组)
    """
    # 确保 tform 是 3x3 矩阵
    if tform.shape != (3, 3):
        raise ValueError("tform 必须是 3x3 矩阵")

    # 1. 设置旋转参考点，通过调整仿射矩阵来实现中心化
    T_translate_to_origin = np.array([
        [1, 0, -cent[0]],
        [0, 1, -cent[1]],
        [0, 0, 1]
    ])

    T_translate_back = np.array([
        [1, 0, cent[0]],
        [0, 1, cent[1]],
        [0, 0, 1]
    ])

    T_total = T_translate_back @ tform @ T_translate_to_origin
    affine_matrix = T_total[:2, :]

    # 2. 图像翻转
    if f == 1:
        im = np.flipud(im)

    # 3. 应用仿射变换
    height, width = im.shape[:2]

    # 处理填充值，确保与图像通道数匹配
    if len(im.shape) == 3:
        # 多通道图像，例如 RGB
        if isinstance(fillval, (int, float)):
            fillval = (fillval, fillval, fillval)
    else:
        # 单通道图像，例如灰度
        if isinstance(fillval, (tuple, list)):
            fillval = fillval[0]

    imG = cv2.warpAffine(
        im,
        affine_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fillval
    )

    return imG



def load_matlab_params(path_tform, path_D):

    eng = matlab.engine.start_matlab()
    eng.eval(f"""load('{path_tform}');""", nargout=0)
    cent = eng.eval("cent")
    np_cent = np.array(cent._data)[0]

    f = int(eng.eval("f"))
    T_matlab = eng.eval("tform.T")
    np_tform = np.array(T_matlab._data).reshape(3, 3)
    print("tform:", np_tform)

    eng.eval(f"""load('{path_D}');""", nargout=0)
    D_matlab = eng.workspace['D']
    
    # 获取 D 的尺寸
    H = D_matlab.size[0]
    W = D_matlab.size[1]
    C = D_matlab.size[2]
    
    if C != 2:
        raise ValueError("D 的第三维必须为 2，表示 y 和 x 坐标。")
    
    # 将 MATLAB 的 double 类型矩阵转换为 NumPy 数组
    # MATLAB 的数据是列优先的，因此需要使用 order='F' 进行重塑
    D_np_flat = np.array(D_matlab._data).reshape(D_matlab.size, order='F')
    
    # 重塑为 (H, W, 2)
    np_D = D_np_flat.reshape((H, W, 2), order='F')
    
    # 关闭 MATLAB 引擎
    eng.quit()


    return np_cent, np_tform, f, np_D


if __name__ == "__main__":
    img_M_raw = cv2.imread(path_img)
    img_F_raw = cv2.imread(path_img_F)
    szz = (max(img_M_raw.shape[0], img_F_raw.shape[0]), max(img_M_raw.shape[1], img_F_raw.shape[1]))
    img_M, impg, TA, fillval = preprocessing(img_M_raw, img_M_raw, szz=szz, padall=250, IHC=0)
    img_F, impg_F, TA_F, fillval_F = preprocessing(img_F_raw, img_F_raw, szz=szz, padall=250, IHC=0)

    np_cent, np_tform, f, np_D = load_matlab_params(path_tform=path_tform, path_D=path_D)
    print(np_cent, np_tform, f, np_D.shape)

    img_reg = register_global_im(img_M, np_tform, np_cent, f)
    img_reg_affine = img_reg.copy()
    img_reg_affine[:img_F.shape[0], :img_F.shape[1], 2] = img_F[:, :, 0]
    img_reg_affine[:, :, 1] = 255

    np_D = np_D.reshape(img_M.shape[0], img_M.shape[1], 2)
    img_reg_disp = imwarp_map_coordinates(img_reg, np_D)
    img_reg_disp[:img_F.shape[0], :img_F.shape[1], 2] = img_F[:, :, 0]
    img_reg_disp[:, :, 1] = 255

    cv2.imwrite(r"./S7_reg.jpg", img_reg_affine)
    cv2.imwrite(r"./S7_reg_map.jpg", img_reg_disp)


    image_F_coda = cv2.imread(f"{root_dir}/2/8/fix stain/Hchannel/registered/elastic registration/S1.tif")
    image_M_coda = cv2.imread(f"{root_dir}/2/8/fix stain/Hchannel/registered/elastic registration/S7.tif")
    image_M_coda_affine = cv2.imread(f"{root_dir}/2/8/fix stain/Hchannel/registered/S7.tif")

    df_lm = pd.read_csv(f"{root_dir}/2/landmarks.csv")
    df_lm = df_lm / 8

    df_lm_M = df_lm.values[:, :2]
    df_lm_F = df_lm.values[:, 2:4]

    df_lm_M_to_F = register_landmarks(df_lm_M, np_cent, np_tform, f, np_D, szz, raw_size=img_M_raw.shape[:2], padall=250)
    df_lm_F_pad = calculate_landmarks_offset(df_lm_F, szz, img_F_raw.shape[:2], padall=250)

    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 3, 1)
    ovlp_F = img_F_raw.copy()
    ax.imshow(ovlp_F)
    ax.scatter(df_lm_F[:, 0], df_lm_F[:, 1], c='r')

    ax = fig.add_subplot(1, 3, 2)
    ovlp_M = img_M_raw.copy()
    ax.imshow(ovlp_M)
    ax.scatter(df_lm_M[:, 0], df_lm_M[:, 1], c='r')

    ax = fig.add_subplot(1, 3, 3)
    ovlp_M = image_M_coda.copy()
    ovlp_M[:, :, 1] = 255
    ovlp_M[:, :, 2] = img_reg_disp[:, :, 0]
    ax.imshow(ovlp_M)
    ax.scatter(df_lm_F_pad[:, 0], df_lm_F_pad[:, 1], c='g', s=10)
    ax.scatter(df_lm_M_to_F[:, 0], df_lm_M_to_F[:, 1], c='r', s=4)

    fig.savefig(r"./S7_ovlp.jpg")