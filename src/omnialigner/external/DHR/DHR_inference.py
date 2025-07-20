import os
import json
from multiprocessing import Pool

import pyvips
import cv2
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm



import omnialigner as om
from omnialigner.utils.point_transform import warp_landmark_grid_faiss
from deeperhistreg.dhr_input_output.dhr_loaders import displacement_loader as df_loader



def preprocess_kpt_DHR(kpt_point, ratio=0.2, initial_resample_ratio=1.0, pad=None):
    if pad is None:
        pad = [[0, 0], [0, 0]]
        
    kpt_point_ = (kpt_point*ratio + np.array([pad[1][0], pad[0][0]])) / initial_resample_ratio
    return kpt_point_

def post_kpt_DHR(kpt_point_, ratio=0.2, initial_resample_ratio=1.0, pad=None):
    if pad is None:
        pad = [[0, 0], [0, 0]]

    kpt_point = (kpt_point_*initial_resample_ratio - np.array([pad[1][0], pad[0][0]])) / ratio
    return kpt_point



def __process_Acrobat(group_id):
    df_Acrobat = pd.read_csv("../benchmark/Acrobat/ACROBAT_validation_annotated_kps.csv", index_col=0).dropna()
    l_kpt_pairs = torch.load(f"/cluster/home/bqhu_jh/projects/omni/config/Acrobat/l_kpt_pairs_Acrobat.pth")

    json_path = f"/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/Acrobat/{group_id:03d}/postprocessing_params.json"
    disp_path = f"/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/Acrobat/{group_id:03d}/displacement_field.mha"

    df_sub = df_Acrobat[df_Acrobat["anon_id"] == group_id]
    type_IHC = df_sub.iloc[0]["ihc_antibody"]
    source_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/Acrobat/01.ome_tiff/val_{group_id}/{group_id}_{type_IHC}_val.ome.tiff" 
    target_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/Acrobat/01.ome_tiff/val_{group_id}/{group_id}_HE_val.ome.tiff"

    img_F = None
    img_M = pyvips.Image.new_from_file(source_path, page=3).numpy()
    tensor_M = om.tl.im2tensor(img_M)
    tensor_M = F.interpolate(tensor_M, scale_factor=8)

    ## scale to img_F // 5
    with open(json_path, "r") as f_json:
        post_process_params = json.load(f_json)

    np_pad_size = np.array(post_process_params["pad_1"]).ravel()
    tensor_M = F.pad(F.interpolate(tensor_M, scale_factor=post_process_params["source_resample_ratio"]), [np_pad_size[2], np_pad_size[3], np_pad_size[0], np_pad_size[1]])

    l_lms = []
    ratio = post_process_params.get("initial_resample_ratio", 1.0)
    lm_M = preprocess_kpt_DHR(l_kpt_pairs[group_id][1], ratio=post_process_params["source_resample_ratio"], initial_resample_ratio=ratio, pad=post_process_params["pad_1"])
    lm_F = preprocess_kpt_DHR(l_kpt_pairs[group_id][0], ratio=post_process_params["target_resample_ratio"], initial_resample_ratio=ratio, pad=post_process_params["pad_2"])
    l_lms.append(lm_M)

    displacement_field = df_loader.DisplacementFieldLoader().load(disp_path)
    grid = om.tl.disp2grid(displacement_field)
    tensor_M_moved = F.grid_sample(tensor_M, grid, mode="nearest")
    # print(grid.shape, tensor_M_moved.shape, tensor_M.shape)
    _, _, H, W = tensor_M_moved.shape
    # l_lms_new = [lm_F]
    l_lms_new = [ l_kpt_pairs[group_id][0] ]
    for lm in l_lms:
        tensor_lm = torch.from_numpy(lm / np.array([W, H])).float()
        kpt_tensor = warp_landmark_grid_faiss(tensor_lm, grid)[:, 0, :]
        kpt_tensor[:, 0] = kpt_tensor[:, 0] * W
        kpt_tensor[:, 1] = kpt_tensor[:, 1] * H
        kpt_tensor_ = post_kpt_DHR(kpt_tensor, ratio=post_process_params["target_resample_ratio"], initial_resample_ratio=ratio, pad=post_process_params["pad_2"])
        l_lms_new.append(kpt_tensor_)

    return img_F, om.tl.tensor2im(tensor_M_moved), l_lms_new

def process_Acrobat():
    l_kpt_Acrobat = []
    for group_id in tqdm(df_Acrobat["anon_id"].unique()):
        img_F, img_moved, kpt_pair = __process_Acrobat(group_id)
        l_kpt_Acrobat.append(kpt_pair)

    # torch.save(l_kpt_Acrobat, "/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/Acrobat/l_kpt_pairs.pth")



def __process_ANHIR(idx):
    def parse_line(line):
        group = line["Source image"].split("/")[0]
        sample_F = line["Target image"].split("/")[-1].split(".")[0]
        sample_M = line["Source image"].split("/")[-1].split(".")[0]
        return pd.Series([group, sample_F, sample_M, line["Target landmarks"], line["Source landmarks"], line["status"]])

    df_ANHIR = pd.read_csv("/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/ANHIR-2019/dataset_medium.csv", index_col=0)
    l_kpt_pairs = torch.load("/cluster/home/bqhu_jh/projects/omni/config/ANHIR/l_kpt_pairs_ANHIR.pth")

    df_ANHIR = df_ANHIR.query(f"status == 'training'")
    df_ANHIR = df_ANHIR.apply(parse_line, axis=1)
    df_ANHIR.columns = ["group", "sample_F", "sample_M", "lm_F", "lm_M", "status"]
    df_ANHIR = df_ANHIR.query(f" status == 'training'")
    df_ANHIR["i_line"] = df_ANHIR.index
    df_ANHIR['group_id'] = [ str(x) for x in range(len(df_ANHIR))]

    json_path = f"/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/ANHIR/{idx:03d}/postprocessing_params.json"
    disp_path = f"/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/ANHIR/{idx:03d}/displacement_field.mha"

    df_sub = df_ANHIR[df_ANHIR["group_id"] == str(idx)]
    sample_F = df_sub.iloc[0]["sample_F"]
    sample_M = df_sub.iloc[0]["sample_M"]
    group = df_sub.iloc[0]["group"]
    target_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/ANHIR/01.ome_tiff/{group}/{sample_F}.ome.tiff" 
    source_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/ANHIR/01.ome_tiff/{group}/{sample_M}.ome.tiff" 

    img_F = cv2.imread(target_path)
    img_M = cv2.imread(source_path)
    tensor_M = om.tl.im2tensor(img_M)

    ## scale to img_F // 5
    with open(json_path, "r") as f_json:
        post_process_params = json.load(f_json)


    np_pad_size = np.array(post_process_params["pad_1"]).ravel()
    tensor_M = F.pad(F.interpolate(tensor_M, scale_factor=post_process_params["source_resample_ratio"]), [np_pad_size[2], np_pad_size[3], np_pad_size[0], np_pad_size[1]])

    l_lms = []
    ratio = post_process_params.get("initial_resample_ratio", 1.0)
    lm_M = preprocess_kpt_DHR(l_kpt_pairs[idx][1], ratio=post_process_params["source_resample_ratio"], initial_resample_ratio=ratio, pad=post_process_params["pad_1"])
    lm_F = preprocess_kpt_DHR(l_kpt_pairs[idx][0], ratio=post_process_params["target_resample_ratio"], initial_resample_ratio=ratio, pad=post_process_params["pad_2"])
    l_lms.append(lm_M)

    displacement_field = df_loader.DisplacementFieldLoader().load(disp_path)
    grid = om.tl.disp2grid(displacement_field)
    tensor_M_moved = F.grid_sample(tensor_M, grid, mode="nearest")
    _, _, H, W = tensor_M_moved.shape
    l_lms_new = [ l_kpt_pairs[idx][0] ]
    for lm in l_lms:
        tensor_lm = torch.from_numpy(lm / np.array([W, H])).float()
        kpt_tensor = warp_landmark_grid_faiss(tensor_lm, grid)[:, 0, :]
        kpt_tensor[:, 0] = kpt_tensor[:, 0] * W
        kpt_tensor[:, 1] = kpt_tensor[:, 1] * H
        kpt_tensor_ = post_kpt_DHR(kpt_tensor, ratio=post_process_params["target_resample_ratio"], initial_resample_ratio=ratio, pad=post_process_params["pad_2"])
        l_lms_new.append(kpt_tensor_)


    return img_F, om.tl.tensor2im(tensor_M_moved), l_lms_new


def process_ANHIR():
    l_kpt_ANHIR = []
    for i in tqdm(range(230)):
        img_F, img_moved, kpt_pair = __process_ANHIR(i, "ANHIR")
        l_kpt_ANHIR.append(kpt_pair)
        
    # torch.save(l_kpt_ANHIR, "/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/ANHIR/l_kpt_pairs.pth")


def __process_fair(idx, tissue):    
    project = "panlab"
    if tissue == "pdac":
        project = "panlab"
        l_files = sorted(glob("/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/SupplementaryData_A_comparison_of_reconstruction_algorithms_for_3D_histology/Data_to_IDA/CODA/panlab/pdac/4/*jpg"))

    else:
        project = "fair"
        l_files = sorted(glob(f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/{project}/{tissue}/*tif"))
        
    l_kpt_pairs = torch.load(f"/cluster/home/bqhu_jh/projects/omni/config/{project}/l_kpt_pairs_{tissue}.pth")
    l_kpt_pairs_ = l_kpt_pairs.copy()
    if len(l_kpt_pairs[0]) > 1:
        l_kpt_pairs_ = [ [[], l_kpt_pairs[0][0]] ]
        for i in range(1, len(l_kpt_pairs)):
            l_kpt_pairs_.append( [l_kpt_pairs[i-1][1], l_kpt_pairs[i][0]] )

        l_kpt_pairs_.append([l_kpt_pairs[-1][1], []])
    

    json_path = f"/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/{tissue}/{idx:03d}/postprocessing_params.json"
    disp_path = f"/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/{tissue}/{idx:03d}/displacement_field.mha"

    # target_path = l_files[idx-1]
    source_path = l_files[idx]

    # img_F = cv2.imread(target_path)
    img_M = cv2.imread(source_path)
    tensor_M = om.tl.im2tensor(img_M)

    ## scale to img_F // 5
    with open(json_path, "r") as f_json:
        post_process_params = json.load(f_json)


    np_pad_size = np.array(post_process_params["pad_1"]).ravel()
    tensor_M = F.pad(F.interpolate(tensor_M, scale_factor=post_process_params["source_resample_ratio"]), [np_pad_size[2], np_pad_size[3], np_pad_size[0], np_pad_size[1]])

    l_lms = []
    for kpt_p in l_kpt_pairs_[idx]:
        if len(kpt_p) == 0:
            l_lms.append(np.array([]))
            continue
        
        lm = post_process_params["source_resample_ratio"]*kpt_p + np.array([post_process_params["pad_1"][0][0], post_process_params["pad_1"][1][0]])
        l_lms.append(lm)

    displacement_field = df_loader.DisplacementFieldLoader().load(disp_path)
    grid = om.tl.disp2grid(displacement_field)
    tensor_M_moved = F.grid_sample(tensor_M, grid, mode="nearest")

    _, _, H, W = tensor_M.shape
    l_lms_new = []
    for lm in l_lms:
        if len(lm) == 0:
            l_lms_new.append([])
            continue

        tensor_lm = torch.from_numpy(lm / np.array([W, H])).float()
        kpt_tensor = warp_landmark_grid_faiss(tensor_lm, grid)
        l_lms_new.append(kpt_tensor[:, 0, :])

    return om.tl.tensor2im(tensor_M_moved), l_lms_new


def process_fair(tissue = "liver"):    
    project = "panlab"
    if tissue == "pdac":
        project = "panlab"
        l_files = sorted(glob("/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/SupplementaryData_A_comparison_of_reconstruction_algorithms_for_3D_histology/Data_to_IDA/CODA/panlab/pdac/4/*jpg"))

    else:
        project = "fair"
        l_files = sorted(glob(f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/{project}/{tissue}/*tif"))
        
    l_kpt_pairs = torch.load(f"/cluster/home/bqhu_jh/projects/omni/config/{project}/l_kpt_pairs_{tissue}.pth")

    def process_wrapper(args):
        idx, tissue = args
        return __process_fair(idx, tissue)

    with Pool(32) as pool:
        results = list(tqdm(pool.imap(process_wrapper, [(idx, tissue) for idx in range(1, len(l_files))]), total=len(l_files)-1))

    img_F_raw = cv2.imread(l_files[0])
    H, W, _ = img_F_raw.shape
    img_F = cv2.resize(img_F_raw, (W//5, H//5))
    l_imgs = [ img_F ]
    l_kpts_new = [  [ torch.from_numpy(lm / np.array([W, H])).float() if len(lm)>0 else [] for lm in l_kpt_pairs_[0] ] ]
    for img, kpt_pair in results:
        l_imgs.append(img)
        l_kpts_new.append(kpt_pair)

    
    l_kpts_new_ = l_kpts_new.copy()
    if len(l_kpt_pairs[0]) > 1:
        l_kpts_new_ = []
        for i in range(1, len(l_kpts_new)):
            l_kpts_new_.append( [l_kpts_new[i-1][1], l_kpts_new[i][0]] )
        

    l_kpts_new_scaled = []
    for kpt_pair in l_kpts_new_:
        l_pair = []
        for kpt in kpt_pair:
            if len(kpt) > 0:
                kpt = kpt * np.array([W*5, H*5])
            
            l_pair.append(kpt)
        
        l_kpts_new_scaled.append(l_pair)

    torch.save(l_kpts_new_scaled, f"/cluster/home/bqhu_jh/projects/omni/benchmark/results/deeperhistreg/{tissue}/l_kpt_pairs_{tissue}.pth", _use_new_zipfile_serialization=False)



if __name__ == "__main__":
    process_fair(tissue="liver")
    process_fair(tissue="prostate")
    process_fair(tissue="pdac")
    
    process_ANHIR()
    process_Acrobat()
