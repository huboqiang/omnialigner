import unittest
import numpy as np
import torch
from omnialigner.keypoints import KanGridDatasets


class TestFieldToGrids(unittest.TestCase):
    def test_dataset_reorder_by_dist(self):
        # Initialize Dense2DSpatialTransformer and Dense2DSpatialTransformerV2
        ds = KanGridDatasets(
            image_F=torch.rand(10, 10),
            image_M=torch.rand(10, 10),
            mkpts_F=np.random.random_sample([500, 2]),
            mkpts_M=np.random.random_sample([500, 2]),
            index_matches=np.random.permutation(500)[0:100]
        )
        index_matches = np.array(ds.dataset["index_matches"])
        mkpts_M = ds.dataset['test_input'].detach().numpy()
        mkpts_F = ds.dataset['test_label'].detach().numpy()
        d1 = np.sum(np.abs(mkpts_M-mkpts_F))
        ds.reorder_by_dist()
        index_matches = np.array(ds.dataset["index_matches"])
        mkpts_M = ds.dataset['test_input'].detach().numpy()
        mkpts_F = ds.dataset['test_label'].detach().numpy()
        d2 = np.sum(np.abs(mkpts_M-mkpts_F))
        self.assertLessEqual(d1-d2, 1e-5)

        d2_ = np.sum(np.abs(mkpts_M-mkpts_F), axis=1)
        order_loss = np.sum(d2_-np.sort(d2_))
        self.assertLessEqual(order_loss, 1e-5)

    def test_dataset_update_indices(self):
        # Initialize Dense2DSpatialTransformer and Dense2DSpatialTransformerV2
        ds = KanGridDatasets(
            image_F=torch.rand(10, 10),
            image_M=torch.rand(10, 10),
            mkpts_F=np.random.random_sample([500, 2]),
            mkpts_M=np.random.random_sample([500, 2]),
            index_matches=np.random.permutation(500)[0:100]
        )
        ds.reorder_by_dist()
    
        dist_good = ds.calculate_kpt_dists(indices=ds.dataset["index_matches"])
        dist_next = 1.5*dist_good.mean()

        dist_all = ds.calculate_kpt_dists()
        new_indices = np.arange(len(dist_all))[dist_all < dist_next]
        ds.update_indices(new_indices)
        self.assertListEqual(list(ds.dataset["index_matches"]), list(new_indices))
        self.assertEqual(ds.dataset["train_input"].shape[0], len(new_indices))

if __name__ == '__main__':
    unittest.main()