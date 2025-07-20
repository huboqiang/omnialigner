import os
from glob import glob
import pandas as pd

dirname = "/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/SupplementaryData_A_comparison_of_reconstruction_algorithms_for_3D_histology/Data_to_IDA/"

project = "fair"
groups = ["liver", "prostate"]

l_dfs = []
for group in groups:
    l_files = sorted(glob(os.path.join(dirname, group, "*.tif")))

    df = pd.DataFrame({
        "project": project,
        "group": group,
        "sample": [ os.path.splitext(os.path.basename(file))[0] for file in l_files],
        "group_idx": range(len(l_files)),
        "type": "HE"
    })
    l_dfs.append(df)

df = pd.concat(l_dfs).reset_index(drop=True)
df.to_csv("./data_fair.csv")