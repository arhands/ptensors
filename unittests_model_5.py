from typing import Callable, Union
from unittest import TestCase

import torch
from torch import Tensor
from objects import TransferData1, atomspack1, MultiScaleData_2
from ptensors1 import linmaps1_1, transfer0_1, transfer1_0, transfer0_1_bi_msg
from transforms import get_pre_transform, get_transform
from model import Net, ModelLayer, get_node_encoder, get_edge_encoder
from torch_geometric.datasets import ZINC
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter 

class batching(TestCase):
    @torch.no_grad()
    def test_equivariance(self):
        ds = ZINC('data/ZINC_base',True,'val')
        tf1 = get_pre_transform('ZINC',False,include_cycles2cycles=True)
        tf2 = get_transform('ZINC')
        sub_batch_size = 4
        num_sub_batches = 3
        graphs_raw : list[Data] = [ds[i] for i in range(sub_batch_size*num_sub_batches)] #type: ignore
        graphs : list[MultiScaleData_2] = [tf2(tf1(g.clone())) for g in graphs_raw] #type: ignore
        small_batches = [Batch.from_data_list([graphs[i*sub_batch_size + j] for j in range(sub_batch_size)]) for i in range(num_sub_batches)] #type: ignore
        batch = Batch.from_data_list(graphs) #type: ignore

        with self.subTest('batch reorder'):
            for num_layers in [0,1,4,10]:
                with self.subTest(num_layers=num_layers):
                    model = Net(128,num_layers,0,'ZINC','sum',0.00001,0.1,'sum',True).eval()
                    ind_preds = torch.cat([model(g).flatten() for g in small_batches])
                    # ind_preds = torch.cat([ind_preds.flatten(),torch.cat([torch.arange(sub_batch_size) + i * sub_batch_size for i in range(num_sub_batches)])],-1)
                    batch_preds = model(batch).flatten()
                    # batch_preds = torch.cat([batch_preds.flatten(),torch.arange(sub_batch_size*num_sub_batches)],-1)

                    self.assertTrue(torch.allclose(ind_preds,batch_preds),f"\n{ind_preds} != \n{batch_preds}")
        def permute_nodes(data):
            perm = torch.randperm(data.num_nodes)
            data.edge_index = perm[data.edge_index]
            data.x[perm] = data.x.clone()
            return data
        graphs_p = [tf2(tf1(permute_nodes(g.clone()))) for g in graphs_raw]
        batch_p = Batch.from_data_list(graphs_p) #type: ignore
        for num_layers in [0,1,4,10]:
            with self.subTest('reorder nodes',num_layers=num_layers):
                model = Net(128,num_layers,0,'ZINC','sum',0.00001,0.1,'sum',True).eval()
                # ind_preds = torch.cat([ind_preds.flatten(),torch.cat([torch.arange(sub_batch_size) + i * sub_batch_size for i in range(num_sub_batches)])],-1)
                batch_preds = model(batch).flatten()
                batch_preds_p = model(batch_p).flatten()
                # batch_preds = torch.cat([batch_preds.flatten(),torch.arange(sub_batch_size*num_sub_batches)],-1)

                self.assertTrue(torch.allclose(batch_preds_p,batch_preds),f"\n{batch_preds_p} != \n{batch_preds}")
if __name__ == '__main__':
    import unittest
    unittest.main()