from typing import Tuple
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omnialigner.utils.point_transform import padded_landmarks_to_raw, raw_landmarks_to_scaled, scaled_landmarks_to_raw, raw_landmarks_to_padded
from omnialigner.utils.image_viz import get_bg_tensor
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy

transform = transforms.Compose([
    transforms.ToTensor()
])

def scale_image_ratio(image_height, image_width, max_height, max_width):
    """Calculate scaling ratio to fit image within maximum dimensions while preserving aspect ratio.

    Args:
        image_height (int): Original image height.
        image_width (int): Original image width.
        max_height (int): Maximum allowed height.
        max_width (int): Maximum allowed width.

    Returns:
        tuple: A tuple containing:
            - int: Scaled image height
            - int: Scaled image width
            - float: Scaling ratio used
    """
    r_height = image_height/max_height
    r_width  = image_width /max_width
    ratio = max(r_height, r_width)
    
    image_height_, image_width_ = int(image_height/ratio), int(image_width/ratio)
    image_height = min(image_height, image_height_)
    image_width = min(image_width, image_width_)
    return image_height, image_width, ratio


def center_pad_with_flank(
        tensor: Tensor_image_NCHW, 
        max_size: tuple[int, int], 
        pt_bg: Tensor_image_NCHW = None, 
        mode: str = "bilinear", 
        align_corners: bool = True
    ) -> tuple[Tensor_image_NCHW, tuple[int, int, int, int], float]:
    """Center pad an image tensor with background values and optional resizing.

    This function first resizes the input tensor if it exceeds the maximum dimensions,
    then adds padding to center the image within the target size. The padding is filled
    with background values derived from the image borders or provided explicitly.

    Args:
        tensor (Tensor_image_NCHW): Input tensor of shape [N, C, H, W].
        max_size (tuple[int, int]): Maximum dimensions as [max_height, max_width].
        pt_bg (Tensor_image_NCHW, optional): Background value tensor of shape [N, C, 1, 1].
            If None, computed from image borders. Defaults to None.
        mode (str, optional): Interpolation mode for resizing. Defaults to "bilinear".
        align_corners (bool, optional): Whether to align corners in interpolation. 
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - Tensor_image_NCHW: Padded tensor of shape [N, C, max_height, max_width]
            - tuple[int, int, int, int]: Padding sizes (left, right, up, down)
            - float: Scaling ratio used if resizing was applied
    """
    image_height = tensor.size(2)
    image_width = tensor.size(3)
    max_height, max_width = max_size[0], max_size[1]
    ratio = 1.
    if max_height < image_height or max_width < image_width:
        if mode=="nearest":
            align_corners=None
        
        image_height, image_width, ratio = scale_image_ratio(image_height, image_width, max_height, max_width)
        tensor = F.interpolate(tensor, size=[image_height, image_width], mode=mode, align_corners=align_corners)

    pad_up = (max_height - image_height) // 2
    pad_left = (max_width - image_width) // 2
    
    pad_down = max_height - image_height - pad_up
    pad_right = max_width - image_width - pad_left
    pad_size = (pad_left, pad_right, pad_up, pad_down)
    
    m = torch.nn.ConstantPad2d(pad_size, value=0)
    padded_tensor = m(tensor)
    _, C, H, W = padded_tensor.size()
    
    if pt_bg is None:
        pt_bg = torch.mean(torch.stack([
            tensor[:, :, 0:1, :].mean((0, 2, 3)), tensor[:, :, :, 0:1].mean((0, 2, 3)),
            tensor[:, :, -1:, :].mean((0, 2, 3)), tensor[:, :, :, -1:].mean((0, 2, 3))
        ]), axis=0).view(1, C, 1, 1)
    
    padded_tensor[:, :, 0:pad_up, :] = pt_bg.expand(1, C, pad_up, max_width)
    padded_tensor[:, :, :, 0:pad_left ] = pt_bg.expand(1, C, max_height, pad_left)
    padded_tensor[:, :, max_height-pad_down:, :] = pt_bg.expand(1, C, pad_down, max_width)
    padded_tensor[:, :, :,   max_width-pad_right:] = pt_bg.expand(1, C, max_height, pad_right)
    return padded_tensor, pad_size, ratio
    
def pad_tensors(
        tensors: list[Tensor_image_NCHW], 
        l_landmarks: list[Tensor_kpts_N_xy], 
        max_size: tuple[int, int] = None, 
        same_hw: bool = False, 
        pt_bg: Tensor_image_NCHW = None, 
        mode: str = "bilinear", 
        align_corners: bool = True
    ) -> tuple[
        list[Tensor_image_NCHW], 
        list[Tensor_kpts_N_xy],
        list[tuple[int, int, int, int]], 
        list[float]
    ]:
    """Pad and optionally resize a batch of images while transforming their landmarks.

    This function processes a batch of images and their associated landmarks through the following steps:
    1. Resize images if they exceed maximum dimensions
    2. Add padding to center the images
    3. Transform landmarks to match the new image dimensions

    The transformation flow for landmarks is:
    tensor -> resize(image, (r*h, r*w)) -> padded_surroundings(centered) -> padded_tensor
    scaled_landmark -> raw_landmark -> r*landmark -> padded_landmark -> scaled_padded_landmark

    Args:
        tensors (list[Tensor_image_NCHW]): List of image tensors [N, C, H, W], scaled to (0,1).
        l_landmarks (list[Tensor_kpts_N_xy]): List of landmark coordinates [x/W, y/H].
        max_size (tuple[int, int], optional): Maximum dimensions [height, width]. 
            If None, uses maximum dimensions from input tensors. Defaults to None.
        same_hw (bool, optional): If True, forces output height and width to be equal.
            Defaults to False.
        pt_bg (Tensor_image_NCHW, optional): Background values for padding.
            Defaults to None.
        mode (str, optional): Interpolation mode for resizing. Defaults to "bilinear".
        align_corners (bool, optional): Whether to align corners in interpolation.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - list[Tensor_image_NCHW]: Padded image tensors
            - list[Tensor_kpts_N_xy]: Transformed landmarks in normalized coordinates
            - list[tuple[int, int, int, int]]: Padding sizes for each image
            - list[float]: Scaling ratios applied to each image
    """
    if max_size is not None and len(max_size) > 1:
        max_height, max_width = max_size[0:2]
    else:
        max_height = max(tensor.size(2) for tensor in tensors)
        max_width = max(tensor.size(3) for tensor in tensors)
    
    if same_hw:
        hw = max(max_height, max_width)
        max_height = hw
        max_width = hw
    
    padded_tensors = []
    landmark_pads = []
    padded_sizes = []
    l_ratio = []
    if pt_bg is None:
        is_gray = tensors[0].shape[0] == 1
        pt_bg = get_bg_tensor(N=len(tensors), is_gray=is_gray)

    for idx,tensor in enumerate(tensors):
        padded_tensor, pad_size, ratio = center_pad_with_flank(tensor, max_size=[max_height, max_width], pt_bg=pt_bg[idx:idx+1], mode=mode, align_corners=align_corners)
        padded_tensors.append(padded_tensor)
        if l_landmarks[idx] is None:
            landmark_pads.append(None)
            padded_sizes.append(pad_size)
            l_ratio.append(ratio)
            continue
        
        landmark = l_landmarks[idx].clone()
        landmark_raw = scaled_landmarks_to_raw(landmark, tensor[0, 0])
        landmark_pad = raw_landmarks_to_padded(landmark_raw, ratio=ratio, padded_size=pad_size)
        landmark_pad_scaled = raw_landmarks_to_scaled(landmark_pad, padded_tensor[0, 0])
        landmark_pads.append(landmark_pad_scaled)
        padded_sizes.append(pad_size)
        l_ratio.append(ratio)
    return padded_tensors, landmark_pads, padded_sizes, l_ratio


def scaled_padded_landmark_to_raw(landmark_pad_scaled, image, ratio, pad_size):
    """Convert scaled padded landmarks back to raw coordinates.

    The transformation flow is:
    raw_landmark <- r*landmark <- padded_landmark <- scaled_padded_landmark

    Args:
        landmark_pad_scaled (Tensor_kpts_N_xy): Normalized coordinates [(r*x+pad_left)/W_, (r*y+pad_up)/H_].
        image (Tensor_image_NCHW): Reference image tensor of shape [H_, W_].
        ratio (float): Scaling ratio used in the original transformation.
        pad_size (tuple[int, int, int, int]): Padding sizes [top, left, height, width].

    Returns:
        Tensor_kpts_N_xy: Raw landmark coordinates [x, y].
    """
    landmark_pad = scaled_landmarks_to_raw(landmark_pad_scaled, image)
    landmark_raw = padded_landmarks_to_raw(landmark_pad, ratio=ratio, padded_size=pad_size)
    return landmark_raw


def pad_image_to_target(tensor_image: Tensor_image_NCHW, target_size: Tuple[int, int]) -> Tensor_image_NCHW:
    """Resize an image tensor to target dimensions by cropping or zero-padding.

    Args:
        tensor_image (Tensor_image_NCHW): Input image tensor.
        target_size (Tuple[int, int]): Target dimensions as (height, width).

    Returns:
        Tensor_image_NCHW: Resized image tensor, either cropped or padded to match
            target dimensions.
    """
    target_h, target_w = target_size
    _, _, height, width = tensor_image.shape
    if height > target_h or width > target_w:
        tensor_image = tensor_image[:, :, :target_h, :target_w]
        return tensor_image
    
    pad_h = target_h - height
    pad_w = target_w - width
    tensor_image = F.pad(tensor_image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return tensor_image
