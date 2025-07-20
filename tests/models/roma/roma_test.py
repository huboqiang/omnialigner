import os

import torch
import cv2
import omnialigner as om

from omnialigner.models.grid_2d.roma_module import RomaModule

Tensor_image_NCHW = torch.FloatTensor
plt = om.pl.plt


def viz_roma(raw_im1, raw_im2, out_img_BA) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(raw_im1)
    ax.set_title("raw_A")
    ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(raw_im2)
    ax.set_title("raw_B")
    ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(om.pl.tensor2im(out_img_BA[0:1]))
    ax.set_title("B_to_A")
    ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(om.pl.tensor2im(out_img_BA[1:2]))
    ax.set_title("A_to_B")
    ax.set_axis_off()
    return fig
    
def match_roma(raw_im1, raw_im2) -> plt.Figure:
    device = torch.device('cpu')
    size_coarse = 560
    roma_model = RomaModule(device=device, size_coarse=size_coarse, size_upsample=(1024, 1024), finest_scale=1)
    H, W = 1024, 1024
    x1 = om.pl.im2tensor(cv2.resize(raw_im1, (size_coarse, size_coarse)))
    x2 = om.pl.im2tensor(cv2.resize(raw_im2, (size_coarse, size_coarse)))
    x1 = x1.to(device)
    x2 = x2.to(device)
    out_img_A2B_B2A = roma_model.forward_image(x1, x2)
    fig = viz_roma(raw_im1, raw_im2, out_img_A2B_B2A)
    return fig

if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(__file__))
    im1_path = os.path.join(curr_path, "roma_kpt_label.png")
    im2_path = os.path.join(curr_path, "roma_kpt_input.png")
    raw_im1 = cv2.imread(im1_path)
    raw_im2 = cv2.imread(im2_path)
    fig = match_roma(raw_im1, raw_im2)
    fig.savefig(os.path.join(curr_path, "out.png"))
    plt.show()