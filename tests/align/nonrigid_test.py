import unittest

import cv2
import torch
import yaml
import omnialigner as om
from omnialigner.align.nonrigid import calculate_dense_keypoints_pairs

class TestAlignNonrigid(unittest.TestCase):
    def test_calculate_dense_keypoints_pairs(self):
        group_id = "ALL-TLS_100"
        with open("/cluster/home/bqhu_jh/projects/panlab/code/wujiangchao/ALLTLS/config/panlab/config_pdac.yaml", 'r') as f:
            template_string = f.read()
            config_info = yaml.load(template_string, Loader=yaml.FullLoader)

        config_info["datasets"]["group"] = f"{group_id}"

        detector_roma = om.kp.init_detector("roma")
        detector_roma.set_upsample_res([1024, 1024])
    
        om_data = om.Omni3D(config_info=config_info)
        out = calculate_dense_keypoints_pairs(om_data, i_layer=0, detector_roma=detector_roma, tag="gray", overwrite_cache=True)
        print(out)

if __name__ == "__main__":
    unittest.main()