from unittest import TestCase

import torch
from objects1 import TransferData1, atomspack1
from ptensors1 import transfer0_1, transfer1_0, transfer1_1, linmaps0_1, linmaps1_0, linmaps1_1

class transfer0_1_test(TestCase):
    def test_overlapping_triangles_single(self):
        a = [[0,1,2]]
        b = [[2,3,4],[1,2,3],[3,4,5]]
        
        a = atomspack1.from_list(a)
        b = atomspack1.from_list(b)
        transfer_map = TransferData1.from_atomspacks(a,b,False)
        
        x = torch.tensor([1]).unsqueeze(-1)

        y_expected = torch.tensor([
            [1,1],[0,1],[0,1],
            [1,1],[1,1],[0,1],
            [0,0],[0,0],[0,0],
        ])
        y = transfer0_1(x,transfer_map)
        self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
    def test_overlapping_triangles_multiple(self):
        a = [[0,1,2],[2,3,4]]
        b = [[1,2,5]]
        
        a = atomspack1.from_list(a)
        b = atomspack1.from_list(b)
        transfer_map = TransferData1.from_atomspacks(a,b,False)
        
        x = torch.tensor([1,10]).unsqueeze(-1)

        y_expected_1 = torch.tensor([
            [1,1],[1,1],[0,1],
        ])
        y_expected_2 = torch.tensor([
            [0,10],[10,10],[0,10],
        ])
        y_expected = y_expected_1 + y_expected_2
        reduce = 'sum'
        with self.subTest(f'reduce = {reduce}'):
            y = transfer0_1(x,transfer_map,reduce=reduce)
            self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
        y_expected = torch.tensor([
            [1/1,(10 + 1)/2],[(10 + 1)/2,(10 + 1)/2],[0,(10 + 1)/2],
        ])
        reduce = 'mean'
        with self.subTest(f'reduce = {reduce}'):
            y = transfer0_1(x.float(),transfer_map,reduce=reduce)
            self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")

class transfer1_0_test(TestCase):
    def test_overlapping_triangles_single(self):
        a = [[0,1,2]]
        b = [[2,3,4],[1,2,3],[3,4,5]]
        
        a = atomspack1.from_list(a)
        b = atomspack1.from_list(b)
        transfer_map = TransferData1.from_atomspacks(a,b,False)
        
        x = torch.tensor([1,10,100]).unsqueeze(-1)

        y_expected = torch.tensor([
            [100,111],
            [110,111],
            [0,0],
        ])
        y = transfer1_0(x,transfer_map)
        self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
        # TODO: add reduction tests.
    def test_overlapping_triangles_multiple(self):
        a = [[0,1,2],[2,3,4]]
        b = [[1,2,5]]
        
        a = atomspack1.from_list(a)
        b = atomspack1.from_list(b)
        transfer_map = TransferData1.from_atomspacks(a,b,False)
        
        x = torch.tensor([
            1,2,4,
            16,32,64]).unsqueeze(-1)

        y_expected_1 = torch.tensor([
            [2+4,1+2+4],
        ])
        y_expected_2 = torch.tensor([
            [16,16+32+64],
        ])
        y_expected = y_expected_1 + y_expected_2
        y = transfer1_0(x,transfer_map)
        self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")

class transfer1_1_test(TestCase):
    def test_overlapping_triangles_single(self):
        a = [[0,1,2]]
        b = [[2,3,4],[1,2,3],[3,4,5]]
        
        a = atomspack1.from_list(a)
        b = atomspack1.from_list(b)
        transfer_map = TransferData1.from_atomspacks(a,b,False)
        
        x = torch.tensor([1,10,100]).unsqueeze(-1)
        
        y_expected = torch.tensor([
            # id, local->local, local->global, global->local, global->global 
            [100,100,100,111,111], # 2
            [  0,  0,100,  0,111], # 3
            [  0,  0,100,  0,111], # 4

            [ 10,110,110,111,111], # 1
            [100,110,110,111,111], # 2
            [  0,  0,110,  0,111], # 3

            [  0,  0,  0,  0,  0], # 3
            [  0,  0,  0,  0,  0], # 4
            [  0,  0,  0,  0,  0], # 5
        ])
        # id, local->local, global->local, local->global, global->global 
        y_expected[:,[2,3]] = y_expected[:,[3,2]]
        y = transfer1_1(x,transfer_map)
        self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
        # TODO: add reduction tests.
    def test_overlapping_triangles_multiple(self):
        a = [[0,1,2],[2,3,4]]
        b = [[1,2,5]]
        
        a = atomspack1.from_list(a)
        b = atomspack1.from_list(b)
        transfer_map = TransferData1.from_atomspacks(a,b,False)
        
        x = torch.tensor([
            1,10,100,
            1000,10000,100000
            ]).unsqueeze(-1)
        
        y_expected_1 = torch.tensor([
            # id, local->local, global->local, local->global, global->global 
            [ 10,110,111,110,111], # 1
            [100,110,111,110,111], # 2
            [  0,  0,  0,110,111], # 5
        ])
        y_expected_2 = torch.tensor([
            # id, local->local, global->local, local->global, global->global 
            [    0,    0,      0, 1000,111000], # 1
            [ 1000, 1000, 111000, 1000,111000], # 2
            [    0,    0,      0, 1000,111000], # 5
        ])
        y_expected = y_expected_1 + y_expected_2
        # id, local->local, global->local, local->global, global->global 
        y = transfer1_1(x,transfer_map)
        self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
        # TODO: add reduction tests.
    # def test_overlapping_triangles_multiple(self):
    #     a = [[0,1,2],[2,3,4]]
    #     b = [[1,2,5]]
        
    #     a = atomspack.from_list(a)
    #     b = atomspack.from_list(b)
    #     transfer_map = TransferData1.from_atomspacks(a,b)
        
    #     x = torch.tensor([
    #         1,2,4,
    #         16,32,64]).unsqueeze(-1)

    #     y_expected_1 = torch.tensor([
    #         [2+4,1+2+4],
    #     ])
    #     y_expected_2 = torch.tensor([
    #         [16,16+32+64],
    #     ])
    #     y_expected = y_expected_1 + y_expected_2
    #     y = transfer1_0(x,transfer_map)
    #     self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")

class linmaps(TestCase):
    def test_linmaps0_1(self):
        a = [[2,3,4],[1,2,3]]
        
        a = atomspack1.from_list(a)
        
        x = torch.tensor([1,10]).unsqueeze(-1)

        y_expected = torch.tensor([
            [1],[1],[1],
            [10],[10],[10],
        ])
        y = linmaps0_1(x,a)
        self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
    def test_linmaps1_0(self):
        a = [[2,3,4],[1,2,3,7]]
        
        a = atomspack1.from_list(a)
        
        x = torch.tensor([
            1,2,4,
            16,32,64,128]).unsqueeze(-1)

        y_expected = torch.tensor([
            [1 + 2 + 4],
            [16 + 32 + 64 + 128],
        ])
        reduce = 'sum'
        with self.subTest(f'reduce = {reduce}'):
            y = linmaps1_0(x,a,reduce)
            self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
        
        y_expected = torch.tensor([
            [(1 + 2 + 4)/3],
            [(16 + 32 + 64 + 128)/4],
        ])
        reduce = 'mean'
        with self.subTest(f'reduce = {reduce}'):
            y = linmaps1_0(x.float(),a,reduce)
            self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
    
    def test_linmaps1_1(self):
        a = [[2,3],[1,2,5]]
        
        a = atomspack1.from_list(a)
        
        x = torch.tensor([
            1,2,
            8,16,32]).unsqueeze(-1)

        y_expected = torch.tensor([
            [1,3],[2,3],
            [8,56],[16,56],[32,56],
        ])
        reduce = 'sum'
        with self.subTest(f'reduce = {reduce}'):
            y = linmaps1_1(x,a,reduce)
            self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
        
        y_expected = torch.tensor([
            [1,3/2],[2,3/2],
            [8,56/3],[16,56/3],[32,56/3],
        ])
        reduce = 'mean'
        with self.subTest(f'reduce = {reduce}'):
            y = linmaps1_1(x.float(),a,reduce)
            self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
    # def test_linmaps0_2(self):
    #     a = [[2,3,4],[1,2,3]]
        
    #     a = atomspack.from_list(a)
        
    #     x = torch.tensor([1,10]).unsqueeze(-1)

    #     y_expected = torch.tensor([
    #         [1,1],[1,0],[1,0],
    #         [1,0],[1,1],[1,0],
    #         [1,0],[1,0],[1,1],

    #         [10,10],[10, 0],[10, 0],
    #         [10, 0],[10,10],[10, 0],
    #         [10, 0],[10, 0],[10,10],
    #     ])
    #     y = linmaps0_2(x,a)
    #     self.assertTrue(torch.allclose(y_expected,y),f"\n{y_expected} != \n{y}")
if __name__ == '__main__':
    import unittest
    unittest.main()