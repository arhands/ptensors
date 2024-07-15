from unittest import TestCase

import torch
from ptensors.objects1 import atomspack1
# linmaps
class atomspack(TestCase):
    def test_overlapping_simple(self):
        b = [[0,1],[1,2],[1,3],[3,4]]
        a = [[0,1,2],[1,2,3]]
        # b = [[0,1],[1,2]]
        # a = [[0,1]]
        
        a = atomspack1.from_list(a)
        b = atomspack1.from_list(b)
        overlaps = b.overlaps1(a,True)
        expected_overlaps = torch.tensor([
            [0,1,2,2,3,3,4,5],
            [0,1,1,3,2,4,3,5]
        ],dtype=torch.int64)
        print(overlaps)
        print(expected_overlaps)
        self.assertTrue(torch.allclose(overlaps,expected_overlaps),f"\n{overlaps} != \n{expected_overlaps}")
if __name__ == '__main__':
    import unittest
    unittest.main()