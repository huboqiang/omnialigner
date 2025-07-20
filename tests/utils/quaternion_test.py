import unittest
import torch
from omnialigner.utils.quaternion import combine_image_transformations, quaternion_to_angle_flip, quaternion_apply, quaternion_multiply


def apply_quaternion_transform_with_center(points, quaternion):
    points[:, :2] -= 0.5
    points = quaternion_apply(quaternion, points)
    points[:, :2] += 0.5
    return points


def apply_manual_transformations(points, angle=0, flip=[0, 0]):
    points[:, :2] -= 0.5

    if flip[1]:
        points[:, 0] = -points[:, 0]
    if flip[0]:
        points[:, 1] = -points[:, 1]

    angle = torch.tensor(angle / 180 * torch.pi)
    rotation_matrix = torch.tensor([
        [torch.cos(angle), -torch.sin(angle)],
        [torch.sin(angle), torch.cos(angle)]
    ])
    
    points[:, :2] = torch.matmul(points[:, :2], rotation_matrix.T)
    points[:, :2] += 0.5
    
    return points

class TestQuaternion(unittest.TestCase):
    def test_quaternion_points(self):
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        angle = 60
        flip = [0, 0]
        q_combined = combine_image_transformations(angle=angle, flip=flip)
        points_quaternion_transformed = apply_quaternion_transform_with_center(points.clone(), q_combined)
        points_manual_transformed = apply_manual_transformations(points.clone(), angle=angle, flip=flip)
        test_result = torch.allclose(points_quaternion_transformed, points_manual_transformed, atol=1e-5)
        self.assertTrue(test_result, "Quaternion transformation does not match manual transformation")

    def test_quaternion_to_angle_flip(self):
        l_angle_flips = [
            [  40, [0, 0]],[ -40, [0, 0]],[  40, [1, 0]],[ -40, [1, 0]],[  40, [0, 1]],[ -40, [0, 1]],[  40, [1, 1]],[ -40, [1, 1]],
            [ 140, [0, 0]],[-140, [0, 0]],[ 140, [1, 0]],[-140, [1, 0]],[ 140, [0, 1]],[-140, [0, 1]],[ 140, [1, 1]],[-140, [1, 1]],
        ]
        l_angle_flips_expected = [
            [40.0, [0, 0]],[-40.0, [0, 0]],[40.0, [1, 0]],[-40.0, [1, 0]],[40.0, [0, 1]],[-40.0, [0, 1]],[-140.0, [0, 0]], [140.0, [0, 0]], 
            [140.0, [0, 0]],[-140.0, [0, 0]],[-40.0, [0, 1]],[40.0, [0, 1]],[-40.0, [1, 0]],[40.0, [1, 0]],[-40.0, [0, 0]],[40.0, [0, 0]],
        ]

        for res, expected in zip(l_angle_flips, l_angle_flips_expected):
            angle = res[0]
            flip = res[1]
            q_combined = combine_image_transformations(angle=angle, flip=flip)
            angle_, flip_ = quaternion_to_angle_flip(q_combined)

            self.assertAlmostEqual(expected[0], angle_, delta=1e-5, msg=f"Angle mismatch: expected {expected[0]}, got {angle_}")
            self.assertEqual(expected[1], flip_, f"Flip mismatch: expected {expected[1]}, got {flip_}")

        
    def test_flip_angle_multiple(self):
        l_input = [
            [[15, [0,0]], [30, [0,0]]],
            [[0, [1,1]], [0, [1,0]]],
        ]

        l_res = [
            [45, [0,0]],
            [ 0, [0,1]],
        ]
        
        for r1, expected in zip(l_input, l_res):
            q_combined = combine_image_transformations(angle=0, flip=[0, 0])
            for res in r1:
                a1_, f1_ = res[0], res[1]
                q_combined_ = combine_image_transformations(angle=a1_, flip=f1_)
                q_combined = quaternion_multiply(q_combined, q_combined_)

            a1, a2 = quaternion_to_angle_flip(q_combined)
            self.assertAlmostEqual(expected[0], a1, delta=1e-5, msg=f"Angle mismatch: expected {expected[0]}, got {a1}")
            self.assertAlmostEqual(expected[1], a2, delta=1e-5, msg=f"Angle mismatch: expected {expected[1]}, got {a2}")


if __name__ == '__main__':
    unittest.main()