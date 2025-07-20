import os
# os.chdir("/Users/bqhu/Documents/projects/pdac3d/omni")
import omnialigner as om
from omnialigner.align.stack import init_default_detector
from omnialigner.plotting.matplotlib_init import plt

detector = om.kp.init_default_detector()
omni_dataset = om.Omni3D(config_file="./configs/pdac/config.yaml")
# omni_dataset.root_dir = "/Users/bqhu/Documents/projects/pdac3d/omni"
omni_dataset.load_3d_NCHW("RAW")
l_images, l_kpts_pairs, fig = om.align.stack(omni_dataset, detector=detector)

plt.show()