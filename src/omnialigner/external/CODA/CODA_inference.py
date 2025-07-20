import os
import glob

import torch
import cv2
from tqdm import tqdm
import pandas as pd
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

from omnialigner.external.CODA.pad import preprocessing
from omnialigner.external.CODA.loadmat import register_global_im, imwarp_map_coordinates, register_landmarks, calculate_landmarks_offset

Np_kpts_N_yx_raw = np.ndarray


def convert_mat_to_np(dataset, sample_id, name_M, zoom_level, szz, padall=250, format_file=".png"):
    root_dir = f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/{sample_id}/"
    path_tform = f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/save_warps/{name_M}.mat"
    path_D = f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/save_warps/D/{name_M}.mat"
    
    path_img = f"{root_dir}/{zoom_level}/{name_M}.{format_file}"
    img_M_raw = cv2.imread(path_img)
    
    img_M, impg, TA, fillval = preprocessing(img_M_raw, img_M_raw, szz=szz, padall=padall, IHC=0)
    try:
        if os.path.isfile(path_tform):
            np_cent, np_tform, f, np_D = load_CODA(path_tform=path_tform, path_D=path_D)
        else:
            np_cent = np.array([0, 0]).astype(np.float32)
            np_tform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
            f = 0
            np_D = np.zeros([img_M.shape[0], img_M.shape[1], 2]).astype(np.float32)
    except:
        np_cent = np.array([0, 0]).astype(np.float32)
        np_tform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
        f = 0
        np_D = np.zeros([img_M.shape[0], img_M.shape[1], 2]).astype(np.float32)

    dict_pt = {
        "np_cent": np_cent,
        "np_tform": np_tform,
        "f": f,
        "np_D": np_D,
    }
    os.makedirs(f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/pth/{sample_id}/", exist_ok=True)
    torch.save(dict_pt, f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/pth/{sample_id}/{name_M}.{zoom_level}.pth", pickle_protocol=4, _use_new_zipfile_serialization=False)


def calculate_tre(
        source_landmarks: Np_kpts_N_yx_raw, 
        target_landmarks: Np_kpts_N_yx_raw
    ) -> float:
    """
    Calculate TRE between source and target landmarks
    Args:
        source_landmarks:  Np_kpts_N_yx_raw 
        target_landmarks:  Np_kpts_N_yx_raw 
    Returns:
        tre:  float
    """
    n = min(source_landmarks.shape[0], target_landmarks.shape[0])
    tre = np.sqrt(np.square(source_landmarks[0:n, 0] - target_landmarks[0:n, 0]) +\
                  np.square(source_landmarks[0:n, 1] - target_landmarks[0:n, 1]))
    return tre

def calculate_rtre(
        source_landmarks: Np_kpts_N_yx_raw, 
        target_landmarks: Np_kpts_N_yx_raw, 
        image_diagonal: float
    ) -> float:
    """
    Calculate R-TRE between source and target landmarks
    Args:
        source_landmarks:  Np_kpts_N_yx_raw 
        target_landmarks:  Np_kpts_N_yx_raw 
        image_diagonal:    float
    Returns:
        rtre:  float
    """
    tre = calculate_tre(source_landmarks, target_landmarks)
    rtre = tre / image_diagonal
    return rtre



def load_CODA(path_tform, path_D):

    eng = matlab.engine.start_matlab()
    eng.eval(f"""load('{path_tform}');""", nargout=0)
    cent = eng.eval("cent")
    np_cent = np.array(cent._data)[0]

    f = int(eng.eval("f"))
    T_matlab = eng.eval("tform.T")
    np_tform = np.array(T_matlab._data).reshape(3, 3)
    # print("tform:", np_tform)

    eng.eval(f"""load('{path_D}');""", nargout=0)
    D_matlab = eng.workspace['D']

    H = D_matlab.size[0]
    W = D_matlab.size[1]
    C = D_matlab.size[2]
    
    if C != 2:
        raise ValueError("D 的第三维必须为 2，表示 y 和 x 坐标。")
    
    D_np_flat = np.array(D_matlab._data).reshape(D_matlab.size, order='F')
    np_D = D_np_flat.reshape((H, W, 2), order='F')
    eng.quit()


    return np_cent, np_tform, f, np_D

def plot_landmarks(img_F_raw, img_M_raw, image_F_coda, image_M_coda, img_reg_disp, df_lm_F, df_lm_M, df_lm_M_to_F, df_lm_F_pad, padall=250):
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 3, 1)
    ovlp_F = img_F_raw.copy()
    ax.imshow(ovlp_F)
    ax.scatter(df_lm_F[:, 0], df_lm_F[:, 1], c='g')

    ax = fig.add_subplot(1, 3, 2)
    ovlp_M = img_M_raw.copy()
    ax.imshow(ovlp_M)
    ax.scatter(df_lm_M[:, 0], df_lm_M[:, 1], c='r')

    ax = fig.add_subplot(1, 3, 3)
    ovlp_M = image_F_coda.copy()
    ovlp_M[:, :, 1] = 255
    ovlp_M[:, :, 2] = img_reg_disp[:, :, 0]

    ovlp_M = ovlp_M[padall:-padall, padall:-padall, :]
    df_lm_F_pad = df_lm_F_pad - padall
    df_lm_M_to_F = df_lm_M_to_F - padall
    ax.imshow(ovlp_M)
    ax.scatter(df_lm_F_pad[:, 0], df_lm_F_pad[:, 1], c='g', s=10)
    ax.scatter(df_lm_M_to_F[:, 0], df_lm_M_to_F[:, 1], c='r', s=4)
    
    diag_ = np.power(ovlp_M.shape[0]**2 + ovlp_M.shape[1]**2, 0.5)
    rtre = calculate_rtre(df_lm_F_pad, df_lm_M_to_F, diag_)
    median_rtre = np.nanmedian(rtre)
    ax.set_title(f"{median_rtre:.2e}")
    return fig, median_rtre


def process_2D_CODA(dataset, sample_id, zoom_level, padall=250):
    # if os.path.isfile(f"./ovlp_results/{dataset}/{dataset}-{sample_id}-{zoom_level}_ovlp.jpg"):
    #     return
    
    df_samples = pd.read_csv(f"./{dataset}.csv", sep="\t")
    df_samples["sample_id"] = [ str(x) for x in  df_samples["sample_id"] ]
    df_line = df_samples[df_samples["sample_id"]==sample_id].iloc[0]

    name_M = df_line["name_M"]
    name_F = df_line["name_F"]
    root_dir = f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/{sample_id}/"
    path_tform = f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/save_warps/{name_M}.mat"
    path_D = f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/save_warps/D/{name_M}.mat"
    path_img_F = f"{root_dir}/{zoom_level}/{name_F}.jpg"
    path_img = f"{root_dir}/{zoom_level}/{name_M}.jpg"

    img_M_raw = cv2.imread(path_img)
    img_F_raw = cv2.imread(path_img_F)    
    szz = (max(img_M_raw.shape[0], img_F_raw.shape[0]), max(img_M_raw.shape[1], img_F_raw.shape[1]))
    img_M, impg, TA, fillval = preprocessing(img_M_raw, img_M_raw, szz=szz, padall=padall, IHC=0)
    img_F, impg_F, TA_F, fillval_F = preprocessing(img_F_raw, img_F_raw, szz=szz, padall=padall, IHC=0)

    try:
        np_cent, np_tform, f, np_D = load_CODA(path_tform=path_tform, path_D=path_D)
    except:
        np_cent = np.array([0, 0])
        np_tform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        f = 0
        np_D = np.zeros([img_M.shape[0], img_M.shape[1], 2])

    img_reg = register_global_im(img_M, np_tform, np_cent, f)
    img_reg_affine = img_reg.copy()
    img_reg_affine[:img_F.shape[0], :img_F.shape[1], 2] = img_F[:, :, 0]
    img_reg_affine[:, :, 1] = 255

    np_D = np_D.reshape(img_M.shape[0], img_M.shape[1], 2)
    
    img_reg_disp = imwarp_map_coordinates(img_reg, np_D)
    img_reg_disp[:img_F.shape[0], :img_F.shape[1], 2] = img_F[:, :, 0]
    img_reg_disp[:, :, 1] = 255

    # cv2.imwrite(f"./{name_M}_reg.jpg", img_reg_affine)
    # cv2.imwrite(f"./{name_M}_reg_map.jpg", img_reg_disp)


    image_F_coda = cv2.imread(f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/{name_F}.tif")
    # image_M_coda_affine = cv2.imread(f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/{name_M}.tif")

    image_M_coda = cv2.imread(f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/{name_M}.tif")
    df_lm = pd.read_csv(f"{root_dir}/landmarks.csv")
    df_lm = df_lm / zoom_level

    df_lm_M = df_lm.values[:, :2]
    df_lm_F = df_lm.values[:, 2:4]

    df_lm_M_to_F = register_landmarks(df_lm_M, np_cent, np_tform, f, np_D, szz, raw_size=img_M_raw.shape[:2], padall=padall)
    df_lm_F_pad = calculate_landmarks_offset(df_lm_F, szz, img_F_raw.shape[:2], padall=padall)


    fig, mRTRE = plot_landmarks(img_F_raw, img_M_raw, image_F_coda, image_M_coda, img_reg_disp, df_lm_F, df_lm_M, df_lm_M_to_F, df_lm_F_pad, padall=padall)
    print(dataset, sample_id, zoom_level, mRTRE)
    with open(f"./ovlp_results/csv/{dataset}-{sample_id}-{zoom_level}_ovlp.csv", "w") as f_out:
        f_out.write(f"{dataset},{sample_id},{zoom_level},{mRTRE:.4e}\n")
    
    os.makedirs(f"./ovlp_results/{dataset}/", exist_ok=True)
    fig.savefig(f"./ovlp_results/{dataset}/{dataset}-{sample_id}-{zoom_level}_ovlp.jpg")
    return None



def CODA_one_sample(dataset, sample_id, name_M, zoom_level, l_kpts, szz, padall=250, format_file=".png"):
    root_dir = f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/{sample_id}/"
    path_tform = f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/save_warps/{name_M}.mat"
    path_D = f"{root_dir}/{zoom_level}/fix stain/Hchannel/registered/elastic registration/save_warps/D/{name_M}.mat"
    
    path_img = f"{root_dir}/{zoom_level}/{name_M}.{format_file}"
    img_M_raw = cv2.imread(path_img)
    
    img_M, impg, TA, fillval = preprocessing(img_M_raw, img_M_raw, szz=szz, padall=padall, IHC=0)
    try:
        np_cent, np_tform, f, np_D = load_CODA(path_tform=path_tform, path_D=path_D)
    except:
        np_cent = np.array([0, 0]).astype(np.float32)
        np_tform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
        f = 0
        np_D = np.zeros([img_M.shape[0], img_M.shape[1], 2]).astype(np.float32)
    
    
    img_reg = register_global_im(img_M, np_tform, np_cent, f)
    np_D = np_D.reshape(img_M.shape[0], img_M.shape[1], 2)
    np_D_0 = np.zeros_like(np_D)
    img_reg_disp_0 = imwarp_map_coordinates(img_reg, np_D_0)
    img_reg_disp = imwarp_map_coordinates(img_reg, np_D)
    
    
    l_df_out = [np.array([]), np.array([])]
    for i_kpt, df_lm_M in enumerate(l_kpts):
        if df_lm_M is None:
            continue
        
        df_lm_M_to_F = register_landmarks(df_lm_M, np_cent, np_tform, f, np_D, szz, raw_size=img_M_raw.shape[:2], padall=padall)
        l_df_out[i_kpt] = df_lm_M_to_F

    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_M_raw)
    if l_kpts[0] is not None:
        ax.scatter(l_kpts[0][:, 0], l_kpts[0][:, 1], c='g')
    if len(l_kpts) > 1 and l_kpts[1] is not None:
        ax.scatter(l_kpts[1][:, 0], l_kpts[1][:, 1], c='r')
    

    ax = fig.add_subplot(1, 3, 2)
    ovlp_F = img_reg_disp.copy()
    os.makedirs(f"./ovlp_results/{dataset}/nonrigid", exist_ok=True)
    cv2.imwrite(f"./ovlp_results/{dataset}/nonrigid/{sample_id}_{name_M}-{zoom_level}.png", img_reg_disp)
    ax.imshow(ovlp_F)
    if len(l_df_out[0]) > 0:
        ax.scatter(l_df_out[0][:, 0], l_df_out[0][:, 1], c='g')
    if len(l_df_out) > 1 and len(l_df_out[1]) > 0:
        ax.scatter(l_df_out[1][:, 0], l_df_out[1][:, 1], c='r')
    
    os.makedirs(f"./ovlp_results/{dataset}/", exist_ok=True)
    fig.savefig(f"./ovlp_results/{dataset}/{dataset}-{sample_id}_{name_M}-{zoom_level}_ovlp3d.jpg")
    
    
    return img_reg_disp, l_df_out
    



def process_3D_CODA(dataset, sample_id, zoom_level, padall=250):
    
    root_dir = f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/{sample_id}/"
    l_names = sorted(glob.glob(f"{root_dir}/{zoom_level}/*tif") + glob.glob(f"{root_dir}/{zoom_level}/*jpg") + glob.glob(f"{root_dir}/{zoom_level}/*png"))#[12:]
    max_h, max_w = -1, -1
    for file_img in tqdm(l_names):
        img_M_raw = cv2.imread(file_img)
        max_h = max(max_h, img_M_raw.shape[0])
        max_w = max(max_w, img_M_raw.shape[1])
        
    szz = (max_h, max_w)
    l_kpt_pairs = torch.load(f"./l_kpt_pairs_{sample_id}.pth")#[12:]
    
    for idx, file_raw in tqdm(enumerate(l_names)):
        name_M = ".".join( os.path.basename(file_raw).split(".")[0:-1] )
        format_file = os.path.basename(file_raw).split(".")[-1]
        if len(l_kpt_pairs[0]) > 1:
            l_kpts = [None, None]
            if idx > 0 and len(l_kpt_pairs[idx-1][1]) > 0:
                l_kpts[0]  = l_kpt_pairs[idx-1][1] / zoom_level
            if idx < len(l_kpt_pairs) and len(l_kpt_pairs[idx][0]) > 0:
                l_kpts[1] = l_kpt_pairs[idx][0] / zoom_level
        else:
            l_kpts = [None]
            l_kpts[0]  = l_kpt_pairs[idx][0] / zoom_level
        
        img_reg_disp, l_df_out = CODA_one_sample(dataset, sample_id, name_M, zoom_level, l_kpts, szz, padall=250, format_file=format_file)
        if len(l_kpt_pairs[0]) > 1:
            if idx > 0:
                l_kpt_pairs[idx-1][1] = l_df_out[0] * zoom_level
            if idx < len(l_kpt_pairs):
                l_kpt_pairs[idx][0] = l_df_out[1] * zoom_level
        else:
            l_kpt_pairs[idx][0] = l_df_out[0] * zoom_level

    torch.save(l_kpt_pairs, f"./l_kpt_pairs_{sample_id}-{zoom_level}_coda3d.pth", _use_new_zipfile_serialization=False)
    return l_kpt_pairs


def process_2D_CODA_convertonly(dataset, sample_id, zoom_level, padall=250):
    df_samples = pd.read_csv(f"./{dataset}.csv", sep="\t")
    df_samples["sample_id"] = [ str(x) for x in  df_samples["sample_id"] ]
    df_line = df_samples[df_samples["sample_id"]==sample_id].iloc[0]

    name_M = df_line["name_M"]
    name_F = df_line["name_F"]
    root_dir = f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/{sample_id}/"
    path_img_F = f"{root_dir}/{zoom_level}/{name_F}.jpg"
    path_img = f"{root_dir}/{zoom_level}/{name_M}.jpg"
    l_names = [path_img_F, path_img]
    max_h, max_w = -1, -1
    for file_img in tqdm(l_names):
        img_M_raw = cv2.imread(file_img)
        max_h = max(max_h, img_M_raw.shape[0])
        max_w = max(max_w, img_M_raw.shape[1])
    
    szz = (max_h, max_w)
    for idx, file_raw in tqdm(enumerate(l_names)):
        name_M = ".".join( os.path.basename(file_raw).split(".")[0:-1] )
        format_file = os.path.basename(file_raw).split(".")[-1]
        convert_mat_to_np(dataset, sample_id, name_M, zoom_level, szz, padall=padall, format_file=format_file)



def process_3D_CODA_convertonly(dataset, sample_id, zoom_level, padall=250):
    root_dir = f"/cluster/home/bqhu_jh/projects/CODA/{dataset}/{sample_id}/"
    l_names = sorted(glob.glob(f"{root_dir}/{zoom_level}/*tif") + glob.glob(f"{root_dir}/{zoom_level}/*jpg") + glob.glob(f"{root_dir}/{zoom_level}/*png"))#[12:]
    max_h, max_w = -1, -1
    for file_img in tqdm(l_names):
        img_M_raw = cv2.imread(file_img)
        max_h = max(max_h, img_M_raw.shape[0])
        max_w = max(max_w, img_M_raw.shape[1])
        
    szz = (max_h, max_w)
    for idx, file_raw in tqdm(enumerate(l_names)):
        name_M = ".".join( os.path.basename(file_raw).split(".")[0:-1] )
        format_file = os.path.basename(file_raw).split(".")[-1]
        convert_mat_to_np(dataset, sample_id, name_M, zoom_level, szz, padall=padall, format_file=format_file)



def convert_only():
    padall = 250
    dict_samp = {
        "ANHIR" : 230,
        "Acrobat" : 100,
    }
    dict_zoom = {
        "ANHIR" : [4, 8, 40],
        "Acrobat" : [8, 16, 32],
    }
    dataset = "ANHIR"
    for sample_id in tqdm([ str(x) for x in range(dict_samp[dataset]) ]):
        l_zoom_level = dict_zoom[dataset]
        for zoom_level in l_zoom_level:
            process_2D_CODA_convertonly(dataset, sample_id, zoom_level, padall)

    dataset = "Acrobat"
    for sample_id in tqdm([ str(x) for x in range(16, dict_samp[dataset]) ]):
        l_zoom_level = dict_zoom[dataset]
        for zoom_level in l_zoom_level:
            process_2D_CODA_convertonly(dataset, sample_id, zoom_level, padall)

    for zoom_level in [40, 8, 4]:
        process_3D_CODA_convertonly("fair", "liver", zoom_level, 250)
        process_3D_CODA_convertonly("fair", "prostate", zoom_level, 250)
        process_3D_CODA_convertonly("panlab", "pdac", zoom_level, 250)



if __name__ == "__main__":
    convert_only()
    dataset = "ANHIR"
    sample_id = "2"
    zoom_level = 4
    padall = 250
    
    # # process_2D_CODA(dataset, sample_id, zoom_level, padall)
    # dict_samp = {
    #     "ANHIR" : 230,
    #     "Acrobat" : 100,
    # }
    # dict_zoom = {
    #     "ANHIR" : [4, 8, 40],
    #     "Acrobat" : [8, 16, 32],
    # }
    # for sample_id in tqdm([ str(x) for x in range(dict_samp[dataset]) ]):
    #     l_zoom_level = dict_zoom[dataset]
    #     for zoom_level in l_zoom_level:
    #         process_2D_CODA(dataset, sample_id, zoom_level, padall)

    # dataset = "Acrobat"
    # for sample_id in tqdm([ str(x) for x in range(16, dict_samp[dataset]) ]):
    #     l_zoom_level = dict_zoom[dataset]
    #     for zoom_level in l_zoom_level:
    #         process_2D_CODA(dataset, sample_id, zoom_level, padall)


    # for zoom_level in [40, 8, 4]:
    #     process_3D_CODA("fair", "liver", zoom_level, 250)
    #     process_3D_CODA("fair", "prostate", zoom_level, 250)
    #     process_3D_CODA("panlab", "pdac", zoom_level, 250)