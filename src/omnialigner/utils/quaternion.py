"""
This script is derived from the repository at:
https://github.com/facebookresearch/pytorch3d/releases/tag/V0.7.8


Original code is adapted under the BSD-license provided by the repository, with modifications for specific project requirements.
As the install is difficult for users, only used functions are moved here. For more information, please refer to the original repository.

from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_multiply, quaternion_apply
from pytorch3d.transforms import quaternion_to_matrix
"""
import torch



def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def quaternion_to_angle_flip(q_combined):
    """
    Converts a quaternion to an angle and flip indicators.

    Parameters:
    q_combined (torch.Tensor): A quaternion representing the combined rotation.

    Returns:
    tuple: A tuple containing:
        - angle (float): The rotation angle in degrees, within the range [-180, 180].
        - flip (list): A list of two integers [fx, fy] indicating the flip status along x and y axes.
                       0 means no flip, 1 means flip.
    """
    M = quaternion_to_matrix(q_combined)

    k_x_1 = torch.sqrt(1 - torch.pow(M[0, 1], 2)) / M[0, 0]
    k_x_2 = torch.sqrt(1 - torch.pow(M[1, 0], 2)) / M[1, 1]
    angle = torch.arcsin(torch.clamp(M[1, 0], min=-1, max=1)) / torch.pi * 180
    
    fy = 0 if k_x_1 > 0 else 1
    fx = 0 if k_x_2 > 0 else 1
    if fy:
        angle = -angle

    if fx and fy:
        fx = 0
        fy = 0
        angle = (angle + 180 + 180) % 360 - 180
        
    
    return angle.item(), [fx, fy]


def combine_image_transformations(angle=0, flip=[0, 0]):
    """
    Combines angle and flip into a single quaternion.

    Parameters:
    angle (float): The rotation angle in degrees, within the range [-180, 180].
    flip (list): A list of two integers [fx, fy] indicating the flip status along x and y axes.
                0 means no flip, 1 means flip.

    Returns:
    torch.Tensor: A quaternion representing the combined transformation.
    """
    angle = angle / 180 * torch.pi

    horizontal_flip_axis_angle = torch.tensor([0, flip[1] * torch.pi, 0])
    vertical_flip_axis_angle = torch.tensor([flip[0] * torch.pi, 0, 0])
    rotation_axis_angle = torch.tensor([0, 0, angle])

    q_horizontal = axis_angle_to_quaternion(horizontal_flip_axis_angle)
    q_vertical = axis_angle_to_quaternion(vertical_flip_axis_angle)
    q_rotation = axis_angle_to_quaternion(rotation_axis_angle)

    q_combined = quaternion_multiply(q_vertical, q_horizontal)
    q_combined = quaternion_multiply(q_rotation, q_combined)

    return q_combined


