import unittest
import cv2
import numpy as np
import sys
import torch

from omnialigner.utils.field_transform import calculate_M_from_theta, tfrs_to_grid_M, calculate_theta_from_M
from omnialigner.utils.image_transform import apply_theta_grid
from omnialigner.utils.point_transform import scaled_landmarks_to_raw, transform_keypoints, transfer_landmarks_inv


class TestPadTensors(unittest.TestCase):
    def test_grid_transfer_with_raw_landmarks(self):
        transformed_image = torch.from_numpy(np.ones([1, 3, 540, 480]))
        l_landmarks_scaled = [
            torch.FloatTensor([
                [0.42670645, 0.12985634],
                [0.48041775, 0.07380936],
                [0.49682954, 0.06078834],
                [0.63036181, 0.07494162]
            ])
        ]
        tensor_kpt_source_raw = scaled_landmarks_to_raw(l_landmarks_scaled[0], transformed_image[0, 0, :, :])

        tensor_tfrs_0 = torch.FloatTensor([-0.5843,  0.0680, -0.0448,  0.0637, -0.0109])
        grid_M = tfrs_to_grid_M(tensor_tfrs_0)
        
        grid = torch.nn.functional.affine_grid(grid_M.unsqueeze(0), transformed_image.size())
        
        
        tensor_M = torch.eye(3)
        tensor_M[0:2, :] = grid_M
        param = calculate_M_from_theta(torch.inverse(tensor_M), transformed_image.shape[2], transformed_image.shape[3])
        transformed_source_points = transform_keypoints(tensor_kpt_source_raw, param)

        transformed_target_points_inv = transfer_landmarks_inv(transformed_source_points, grid)

        self.assertTrue( torch.abs(tensor_kpt_source_raw-transformed_target_points_inv).mean() < 0.3)
        
    def test_grid_transfer_with_raw_landmarks_pytorch(self):
        transformed_image = torch.ones([1, 3, 540, 480])
        l_landmarks_scaled = [
            torch.tensor([
                [0.42670645, 0.12985634],
                [0.48041775, 0.07380936],
                [0.49682954, 0.06078834],
                [0.63036181, 0.07494162]
            ], dtype=torch.float32)
        ]
        tensor_tfrs_0 = torch.nn.Parameter(torch.FloatTensor([-0.5843,  0.0680, -0.0448,  0.0637, -0.0109]))

        landmark_source_raw = scaled_landmarks_to_raw(l_landmarks_scaled[0], transformed_image[0, 0, :, :])
        grid_M = tfrs_to_grid_M(tensor_tfrs_0)
        tensor_M = torch.eye(3)
        tensor_M[0:2, :] = grid_M
        param = calculate_M_from_theta(torch.inverse(tensor_M), transformed_image.shape[2], transformed_image.shape[3])
        transformed_source_points = transform_keypoints(landmark_source_raw, param)

        landmark_source_raw = scaled_landmarks_to_raw(l_landmarks_scaled[0], transformed_image[0, 0, :, :])
        grid_M = tfrs_to_grid_M(tensor_tfrs_0)
        tensor_M = torch.eye(3)
        tensor_M[0:2, :] = grid_M
        param = calculate_M_from_theta(torch.inverse(tensor_M).detach(), transformed_image.shape[2], transformed_image.shape[3])
        tensor_transformed_source_points = transform_keypoints(landmark_source_raw, param)
        self.assertTrue(torch.allclose(tensor_transformed_source_points, transformed_source_points, 1e-4))
        

    def test_grid_M_and_inv(self):
        img_raw = np.zeros([1800, 1300, 3], dtype=np.uint8)
        
        img_raw[300:1400, 500:650, :] = 255
        image_tensor = torch.from_numpy(np.transpose(img_raw, axes=[2,0,1]).astype(np.float32)).unsqueeze(0)
        tensor_tfrs_0 = torch.from_numpy(np.array([-0.5203,  0.2253, -0.0221, -0.0770,  0.0269], dtype=np.float32))
        grid_M = tfrs_to_grid_M(tensor_tfrs_0).detach()
        # image_tensor = apply_theta_grid(image_tensor, grid_M)


        transformed_image = apply_theta_grid(image_tensor, grid_M)
        tensor_M = torch.eye(3)
        tensor_M[0:2, :] = grid_M
        param = calculate_M_from_theta(torch.inverse(tensor_M), transformed_image.shape[2], transformed_image.shape[3])
        grid_M_inv = calculate_theta_from_M(param[0:2, :], img_raw.shape[0], img_raw.shape[1])

        transformed_image_inv = apply_theta_grid(transformed_image, grid_M_inv)
        np_plot = np.zeros([transformed_image.shape[2], transformed_image.shape[3], 3], dtype=np.uint8)
        np_plot[:,:,0] = image_tensor[0,0,:,:]
        np_plot[:,:,1] = transformed_image[0,0,:,:]
        np_plot[:,:,2] = transformed_image_inv[0,0,:,:]
        cv2.imwrite("./test.png", np_plot)
        self.assertLessEqual(torch.mean(torch.abs(image_tensor-transformed_image_inv)), 1.0)




def _example_point_transform():
    # Example usage
    theta = np.array([[0.5, 0.2, 0.1], 
                    [0.3, 0.6, 0.4]])
    w = 640  # width
    h = 480  # height

    param = calculate_M_from_theta(theta, h, w)

    print("Calculated param matrix:")
    print(param)

    theta = calculate_theta_from_M(param, h, w)
    print("Calculated theta matrix:")
    print(theta)


def _example_affine_trans(np_points, image_tensor, grid_M):
    transformed_image = apply_theta_grid(image_tensor, grid_M)

    tensor_M = torch.eye(3)
    tensor_M[0:2, :] = grid_M

    param = calculate_M_from_theta(torch.inverse(tensor_M), transformed_image.shape[2], transformed_image.shape[3])
    transformed_points = transform_keypoints(np_points, param)
    return transformed_image, transformed_points


if __name__ == '__main__':
    unittest.main()