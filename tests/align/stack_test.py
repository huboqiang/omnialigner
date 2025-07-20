import os
os.chdir("~/projects/pdac3d/omni")
from omni_3D import Omni3D
from align.stack import stack, init_default_detector
from plotting.image_viz import tensor2im
from plotting.keypoint_viz import plot_kpt_pairs
from plotting.matplotlib_init import plt
import matplotlib
matplotlib.use("Agg")

detector = init_default_detector()
omni_dataset = Omni3D(config_file="./configs/pdac/config.yaml")
omni_dataset.root_dir = "~/projects/pdac3d/omni"
# omni_dataset.plt_row_col = [3, 3]
# omni_dataset.plt_figsize = (10, 10)
# l_images, l_kpts_pairs = stack(omni_dataset, l_layers=range(30, 39), detector=detector)
# omni_dataset.load_NCHW("RAW")
l_images, l_kpts_pairs = stack(omni_dataset, detector=detector)

n_col, n_row = omni_dataset.plt_row_col
fig = plt.figure(figsize=omni_dataset.plt_figsize)
for i_layer in range(len(l_images)):
    ax = fig.add_subplot(n_col, n_row, i_layer+1)
    ax.imshow(tensor2im(l_images[i_layer]))

    plot_kpt_pairs(
        image=tensor2im(l_images[i_layer]), 
        title=f"{i_layer}",
        kpts1=l_kpts_pairs[i_layer][0], 
        kpts0=l_kpts_pairs[i_layer][1],
        ax=ax
    )

fig.savefig("test_stack.png")