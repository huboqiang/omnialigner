import os
import pandas as pd
import torch

from omnialigner.datasets import generate_WSI
from omnialigner.utils.image_pad import pad_tensors
from omnialigner.utils.image_viz import tensor2im, im2tensor


selected_project = "ANHIR"
file_dataset = "/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/ANHIR-2019/dataset_medium.csv"
root_dir = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen"

df = pd.read_csv("/cluster/home/bqhu_jh/projects/scGaussian3dGen/scripts/02.dino_hipt_work.sh", sep="\s+", header=None)[[2,3,4]]
df.columns = ["project", "group", "sample"]

pca_cutoff = {
    "ANHIR": "0.7",
    "Acrobat": "0.7",
    "panlab": "0.9",
    "CRC": "0.95"
}

def read_scaled_landmark(file_name):
    if os.path.isfile(file_name):
        return pd.read_csv(file_name, index_col=[0]).values
    
    return None


def load_landmark_ANHIR(idx):
    parse_path = "/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/ANHIR-2019/"
    df_dataset = pd.read_csv(file_dataset, index_col=[0])
    df_line = df_dataset.loc[idx]
    source_lm = f"{parse_path}/landmarks/{df_line['Source landmarks']}"
    target_lm = f"{parse_path}/landmarks/{df_line['Target landmarks']}"
    source_scaled_lm = source_lm[0:-4] + ".scaled.csv"
    target_scaled_lm = target_lm[0:-4] + ".scaled.csv"
    l_landmarks = [ read_scaled_landmark(x) for x in [source_scaled_lm, target_scaled_lm] ]

    selected_group = df_line['Source landmarks'].split("/")[0]
    selected_sample_1 = df_line['Source landmarks'].split("/")[-1][0:-4]
    selected_sample_2 = df_line['Target landmarks'].split("/")[-1][0:-4]

    return l_landmarks, selected_group, selected_sample_1, selected_sample_2


def load_data(idx, group="ANHIR", is_PCA=True, zoom_level=-1):
    if group == "ANHIR":
        l_landmarks, selected_group, selected_sample_1, selected_sample_2 = load_landmark_ANHIR(idx)
    else:
        print(f"group {group} benchmark is not supported!!!")
        return 
    
    if is_PCA:
        cos_sim_cutoff = pca_cutoff[selected_project]
        uploaded_file1 = f"{root_dir}/analysis/{selected_project}/03.dino_PCA_feats/{selected_group}/{cos_sim_cutoff}/{selected_sample_1}.PCA.pt"
        uploaded_file2 = f"{root_dir}/analysis/{selected_project}/03.dino_PCA_feats/{selected_group}/{cos_sim_cutoff}/{selected_sample_2}.PCA.pt"
        source_PCA_tensor = torch.load(uploaded_file1)
        target_PCA_tensor = torch.load(uploaded_file2)
        img_pairs = [source_PCA_tensor, target_PCA_tensor]
    else:
        uploaded_file1 = f"{root_dir}/analysis/{selected_project}/01.ome_tiff/{selected_group}/{selected_sample_1}.ome.tiff"
        uploaded_file2 = f"{root_dir}/analysis/{selected_project}/01.ome_tiff/{selected_group}/{selected_sample_2}.ome.tiff"

        image_M = generate_WSI(selected_project)(uploaded_file1, zoom_level=zoom_level).compute()
        image_F = generate_WSI(selected_project)(uploaded_file2, zoom_level=zoom_level).compute()
        img_pairs = [im2tensor(image_M), im2tensor(image_F)]

    print(f"Reading: {uploaded_file1}, {uploaded_file2}")
    l_paddled_tensors, l_paddled_landmarks, padded_sizes, l_ratio = pad_tensors(img_pairs, l_landmarks=l_landmarks, same_hw=True)
    image_M, image_F =  tensor2im(l_paddled_tensors[0]), tensor2im(l_paddled_tensors[1])
    return image_M, image_F, l_paddled_landmarks
