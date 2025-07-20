
import sys
import time
import os
os.environ["OPENBLAS_NUM_THREADS"] = '128'
import numpy as np
from valis import registration
from valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration

def valis(slide_src_dir, results_dst_dir, micro_reg_fraction=0.25):
    # Perform high resolution rigid registration using the MicroRigidRegistrar
    start = time.time()
    registrar = registration.Valis(slide_src_dir, results_dst_dir, micro_rigid_registrar_cls=MicroRigidRegistrar, imgs_ordered=True)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
    img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
    min_max_size = np.min([np.max(d) for d in img_dims])
    img_areas = [np.multiply(*d) for d in img_dims]
    max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
    micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)
    micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)


    stop = time.time()
    elapsed = stop - start
    print(f"regisration time is {elapsed/60} minutes")

    # We can also plot the high resolution matches using `Valis.draw_matches`:
    matches_dst_dir = os.path.join(registrar.dst_dir, "hi_rez_matches")
    registrar.draw_matches(matches_dst_dir)

if __name__ == "__main__":
    slide_src_dir   = sys.argv[1]
    results_dst_dir = sys.argv[2]
    micro_reg_fraction = 0.25 # Fraction full resolution used for non-rigid registration
    valis(slide_src_dir=slide_src_dir, results_dst_dir=results_dst_dir, micro_reg_fraction=micro_reg_fraction)

    # singularity exec --cleanenv --nv -B /cluster/home/bqhu_jh/sig/:/cluster/home/bqhu_jh/ -B /cluster/home/bqhu_jh/projects/:/mnt /cluster/home/bqhu_jh/projects/scGaussian3dGen/data/z.sif_images/valis-wsi_tifffile-2024-03-10-13720e46e7dc.sif bash -c "python /mnt/omni/src/omnialigner/external/VALIS/run_valis.py /mnt/scGaussian3dGen/analysis/panlab/v1/01.ome_tiff/ALL-TLS_121/ /mnt/scGaussian3dGen/analysis/panlab/valis/ALL-TLS_121"