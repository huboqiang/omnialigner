""" Registration of whole slide images (WSI) using higher resolution images

This example shows how to register the slides using higher resolution images.
An initial rigid transform is found using low resolition images, but the
`MicroRigidRegistrar` can be used to update that transform using feature matches
found in higher resoltion images. This can be followed up by the high resolution
non-rigid registration (i.e. micro-registration).

"""

import pickle
from tqdm import tqdm
import omnialigner.external.VALIS.run_valis as run_valis
import pyvips
import numpy as np
import torch
from omnialigner.external.VALIS.run_valis import warp_tools

def valis_inference(file_kpts, file_valis_pickle, file_result):
    
    l_kpt_pairs = torch.load(file_kpts)
    l_kpt_pairs_ = l_kpt_pairs.copy()
    if len(l_kpt_pairs[0]) > 1:
        l_kpt_pairs_ = [ [[], l_kpt_pairs[0][0]] ]
        for i in range(1, len(l_kpt_pairs)):
            l_kpt_pairs_.append( [l_kpt_pairs[i-1][1], l_kpt_pairs[i][0]] )

        l_kpt_pairs_.append([l_kpt_pairs[-1][1], []])
        

    with open(file_valis_pickle, "rb") as f:
        registrar = pickle.load(f)
        l_ids = sorted(registrar.slide_dict)
        for idx, id in tqdm(enumerate(l_ids)):
            valis_obj = registrar.slide_dict[id]
            np_tform = valis_obj.M
            np_D = valis_obj.bk_dxdy if isinstance(valis_obj.bk_dxdy, np.ndarray) else warp_tools.vips2numpy(valis_obj.bk_dxdy)
            
            
            image_moved = valis_obj.warp_slide(level=0)
            if isinstance(image_moved, pyvips.Image):
                image_moved = warp_tools.vips2numpy(image_moved)

            kpt_pairs = l_kpt_pairs_[idx]
            l_kpt_out = []
            for np_kpts in kpt_pairs:
                if len(np_kpts) == 0:
                    l_kpt_out.append([])
                    continue

                np_kpts_moved = valis_obj.warp_xy(np_kpts)
                l_kpt_out.append(np_kpts_moved)

            dict_out = {"tform": np_tform, "D": np_D, "image_moved": image_moved, "np_kpts_moved": l_kpt_out}
            torch.save(dict_out, file_result, pickle_protocol=4, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    tissue = "pdac_he"
    dir_root = "/mnt"  # /cluster/home/bqhu_jh/projects
    results_dst_dir = f"{dir_root}/scGaussian3dGen/data/SupplementaryData_A_comparison_of_reconstruction_algorithms_for_3D_histology/Data_to_IDA/{tissue}_valis/"
    
    file_kpts = f"{dir_root}/omni/config/fair/l_kpt_pairs_{tissue}.pth"
    file_valis_pickle = f"{results_dst_dir}/{tissue}/data/{tissue}_registrar.pickle"
    file_result = f"{dir_root}/omni/benchmark/results/valis/{tissue}/{id}.pth"
    valis_inference(file_kpts, file_valis_pickle, file_result)

    # singularity exec --cleanenv --nv -B /cluster/home/bqhu_jh/sig/:/cluster/home/bqhu_jh/ -B /cluster/home/bqhu_jh/projects/:/mnt /cluster/home/bqhu_jh/projects/scGaussian3dGen/data/z.sif_images/valis-wsi_tifffile-2024-03-10-13720e46e7dc.sif bash -c "python /mnt/omni/src/omnialigner/external/VALIS/valis_inference.py "
