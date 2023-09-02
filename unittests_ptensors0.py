from unittest import TestCase

import torch
from objects import TransferData1, atomspack
from ptensors0 import transfer0_0
# linmaps
class transfer0(TestCase):
    def test_overlapping_triangles_single(self):
        a = [[0,1,2]]
        b = [[2,3,4],[1,2,3],[3,4,5]]
        
        a = atomspack.from_list(a)
        b = atomspack.from_list(b)
        transfer_map = TransferData1.from_atomspacks(a,b)
        
        x = torch.tensor([1]).unsqueeze(-1)

        y_expected = torch.tensor([
            [1],[1],[0]
        ])
        y = transfer0_0(x,transfer_map)
        self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
if __name__ == '__main__':
    import unittest
    unittest.main()