import unittest
import torch
import dask.array as da
from omnialigner.utils.tiles import merge_from_overlapped_tiles

class TestPadTensors(unittest.TestCase):
    def test_merge_from_overlapped_tiles(self):
        # 1) Non-overlapping tiles 2×2 each inside a 4×4 canvas
        tiles1 = [[0,0,2,2], [0,2,2,4]]
        disp1  = [torch.ones((1,2,2,2), dtype=torch.float32),
                2*torch.ones((1,2,2,2), dtype=torch.float32)]

        expect1 = torch.tensor([[[[1., 1.],
          [1., 1.],
          [2., 2.],
          [2., 2.]],

         [[1., 1.],
          [1., 1.],
          [2., 2.],
          [2., 2.]]]])
        # 2) Two identical 2×2 tiles fully overlapping, expect average = (1+3)/2 = 2
        tiles2 = [[0,0,2,2], [0,0,2,2]]
        disp2  = [torch.ones((2,2,2), dtype=torch.float32),
                3*torch.ones((2,2,2), dtype=torch.float32)]
        expect2 = torch.tensor([[[[2., 2.],
          [2., 2.]],
         [[2., 2.],
          [2., 2.]]],


        [[[2., 2.],
          [2., 2.]],
         [[2., 2.],
          [2., 2.]]]])

        # 3) Partial overlap: 3×3 canvas. Tile-A covers entire canvas with 1,
        #    Tile-B covers bottom-right 2×2 with value 5.
        tiles3 = [[0,0,3,3], [1,1,3,3]]
        disp3  = [torch.ones((3,3,2), dtype=torch.float32),
                5*torch.ones((2,2,2), dtype=torch.float32)]

        expect3 = torch.tensor([[[[1., 1.],
          [1., 1.],
          [1., 1.]],

         [[1., 1.],
          [3., 3.],
          [3., 3.]],

         [[1., 1.],
          [3., 3.],
          [3., 3.]]],


        [[[1., 1.],
          [1., 1.],
          [1., 1.]],

         [[1., 1.],
          [3., 3.],
          [3., 3.]],

         [[1., 1.],
          [3., 3.],
          [3., 3.]]],


        [[[1., 1.],
          [1., 1.],
          [1., 1.]],

         [[1., 1.],
          [3., 3.],
          [3., 3.]],

         [[1., 1.],
          [3., 3.],
          [3., 3.]]]])
        l_inputs = [ (tiles1, disp1), (tiles2, disp2), (tiles3, disp3) ]
        l_expected = [expect1, expect2, expect3]
        for args, out_expect in zip(l_inputs, l_expected):
            out = merge_from_overlapped_tiles(*args)
            epi = torch.mean(out-out_expect)
            self.assertLess(epi, 1e-3)

        l_inputs = [ (tiles1, [da.from_array(x.numpy()) for x in disp1] ), 
                     (tiles2, [da.from_array(x.numpy()) for x in disp2] ),
                     (tiles3, [da.from_array(x.numpy()) for x in disp3] ) ]
        l_expected = [da.from_array(expect1.numpy()), da.from_array(expect2.numpy()), da.from_array(expect3.numpy())]
        for args, out_expect in zip(l_inputs, l_expected):
            out = merge_from_overlapped_tiles(*args)
            epi = da.mean(out-out_expect).compute()
            self.assertLess(epi, 1e-3)
    
if __name__ == "__main__":
    unittest.main()
