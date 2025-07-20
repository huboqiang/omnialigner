import sys
import os
from glob import glob
from typing import Union
import pathlib
from tqdm import tqdm
import torch
import pandas as pd

import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid
### Run Registration ###

def run_deeperhistreg(source_path, target_path, output_path):
    ### Define Inputs/Outputs ###
    source_path = pathlib.Path(source_path)
    target_path = pathlib.Path(target_path)
    output_path = pathlib.Path(output_path)

    ### Define Params ###
    registration_params : dict = default_initial_nonrigid()
    registration_params['loading_params']['loader'] = 'pil' # For .jpg or .png formats
    save_displacement_field : bool = True # Whether to save the displacement field (e.g. for further landmarks/segmentation warping)
    copy_target : bool = True # Whether to copy the target (e.g. to simplify the further analysis
    delete_temporary_results : bool = False # Whether to keep the temporary results
    case_name : str = f"{source_path.stem}_{target_path.stem}" # Used only if the temporary_path is important, otherwise - provide whatever
    temporary_path : Union[str, pathlib.Path] = output_path / f"{pathlib.Path(source_path).stem}_{pathlib.Path(target_path).stem}_TEMP" # Will use default if set to None

    ### Create Config ###
    config = dict()
    config['source_path'] = source_path
    config['target_path'] = target_path
    config['output_path'] = output_path
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['save_displacement_field'] = save_displacement_field
    config['copy_target'] = copy_target
    config['delete_temporary_results'] = delete_temporary_results
    config['temporary_path'] = temporary_path

    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    config['registration_parameters']["device"] = dev
    config['registration_parameters']['initial_registration_params']["device"] = dev
    config['registration_parameters']['nonrigid_registration_params']["device"] = dev
    
    ### Run Registration ###
    deeperhistreg.run_registration(**config)

def process_fair():
    for tissue, n_samples in zip(["liver", "prostate"], [48, 261]):
        for idx in tqdm(range(1, n_samples)):
            source_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/fair/{tissue}/{idx+1:03d}.tif"
            target_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/fair/{tissue}/{idx:03d}.tif"
            if idx > 1:
                target_path = f"./results/deeperhistreg/{tissue}/{idx-1:03d}/warped_source.tiff"
            
            output_path = f"./results/deeperhistreg/{tissue}/{idx:03d}/"
            os.makedirs(output_path, exist_ok=True)
            run_deeperhistreg(source_path, target_path, output_path)

def process_pdac():
    tissue = "pdac"
    l_files = sorted(glob("/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/SupplementaryData_A_comparison_of_reconstruction_algorithms_for_3D_histology/Data_to_IDA/CODA/panlab/pdac/4/*jpg"))
    for idx in tqdm(range(1, len(l_files))):
        source_path = l_files[idx]
        target_path = l_files[idx-1]
        if idx > 1:
            target_path = f"./deeperhistreg/{tissue}/{idx-1:03d}/warped_source.tiff"

        output_path = f"./deeperhistreg/{tissue}/{idx:03d}/"
        os.makedirs(output_path, exist_ok=True)
        run_deeperhistreg(source_path, target_path, output_path)

def process_ANHIR():
    def parse_line(line):
        group = line["Source image"].split("/")[0]
        sample_F = line["Target image"].split("/")[-1].split(".")[0]
        sample_M = line["Source image"].split("/")[-1].split(".")[0]
        return pd.Series([group, sample_F, sample_M, line["Target landmarks"], line["Source landmarks"], line["status"]])


    df_ANHIR = pd.read_csv("/cluster/home/bqhu_jh/projects/scGaussian3dGen/data/ANHIR-2019/dataset_medium.csv", index_col=0)
    df_ANHIR = df_ANHIR.query(f"status == 'training'")
    df_ANHIR = df_ANHIR.apply(parse_line, axis=1)
    df_ANHIR.columns = ["group", "sample_F", "sample_M", "lm_F", "lm_M", "status"]
    df_ANHIR = df_ANHIR.query(f" status == 'training'")
    df_ANHIR["i_line"] = df_ANHIR.index
    df_ANHIR['group_id'] = [ str(x) for x in range(len(df_ANHIR))]

    for group_id in tqdm(range(len(df_ANHIR))):
        df_sub = df_ANHIR[df_ANHIR["group_id"] == str(group_id)]
        sample_F = df_sub.iloc[0]["sample_F"]
        sample_M = df_sub.iloc[0]["sample_M"]
        group = df_sub.iloc[0]["group"]
        source_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/ANHIR/01.ome_tiff/{group}/{sample_M}.ome.tiff" 
        target_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/ANHIR/01.ome_tiff/{group}/{sample_F}.ome.tiff" 
        output_path = f"./deeperhistreg/ANHIR/{group_id:03d}/"
        os.makedirs(output_path, exist_ok=True)
        run_deeperhistreg(source_path, target_path, output_path)


def process_Acrobat():
    df_Acrobat = pd.read_csv("./Acrobat/ACROBAT_validation_annotated_kps.csv", index_col=0).dropna()
    df_Acrobat
    for group_id in tqdm(df_Acrobat["anon_id"].unique()):
        df_sub = df_Acrobat[df_Acrobat["anon_id"] == group_id]
        type_IHC = df_sub.iloc[0]["ihc_antibody"]
        output_path = f"./deeperhistreg/Acrobat/{group_id:03d}/"
        if os.path.isdir(output_path):
            continue

        source_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/Acrobat/01.ome_tiff/val_{group_id}/{group_id}_{type_IHC}_val.ome.tiff" 
        target_path = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/Acrobat/01.ome_tiff/val_{group_id}/{group_id}_HE_val.ome.tiff" 
        os.makedirs(output_path, exist_ok=True)
        run_deeperhistreg(source_path, target_path, output_path)



if __name__ == "__main__":
    # process_fair()
    # process_pdac()
    # process_ANHIR()
    # process_Acrobat()
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    output_path = sys.argv[3]
    run_deeperhistreg(source_path, target_path, output_path)

     # sample=ALL-TLS_121; for dtype in P1 P2 P3 P4;do python  /cluster/home/bqhu_jh/projects/omni/src/omnialigner/external/DHR/DHR.py /cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/DHR/${sample}/gray/${dtype}.gray.ome.tiff /cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/DHR//${sample}/gray/HE.gray.ome.tiff   /cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/DHR//${sample}/;done
