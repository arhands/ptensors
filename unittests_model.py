from unittest import TestCase

import torch
from objects import MultiScaleData_2, TransferData1, atomspack1
from ptensors0 import transfer0_0
from transforms import PreprocessTransform
from model import Net, ModelLayer, SplitLayer0_1_complex, get_node_encoder, get_edge_encoder, CycleEmbedding1, SplitLayer
from torch_geometric.datasets import ZINC
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

class batching(TestCase):
    @torch.no_grad()
    def test_batching(self):
        ds = ZINC('data/ZINC_base',True,'val')
        tf = PreprocessTransform()
        sub_batch_size = 3
        num_sub_batches = 3
        graphs : list[MultiScaleData] = [tf(ds[i]) for i in range(sub_batch_size*num_sub_batches)] #type: ignore
        small_batches = [Batch.from_data_list([graphs[i*sub_batch_size + j] for j in range(sub_batch_size)]) for i in range(num_sub_batches)] #type: ignore
        batch = Batch.from_data_list(graphs) #type: ignore

        with self.subTest('full model'):
            for num_layers in [0,4]:
                with self.subTest(num_layers=num_layers):
                    model = Net(128,num_layers,0,'ZINC',False,'sum').eval()
                    ind_preds = torch.cat([model(g).flatten() for g in small_batches])
                    # ind_preds = torch.cat([ind_preds.flatten(),torch.cat([torch.arange(sub_batch_size) + i * sub_batch_size for i in range(num_sub_batches)])],-1)
                    batch_preds = model(batch).flatten()
                    # batch_preds = torch.cat([batch_preds.flatten(),torch.arange(sub_batch_size*num_sub_batches)],-1)

                    self.assertTrue(torch.allclose(ind_preds,batch_preds),f"\n{ind_preds} != \n{batch_preds}")
        

        with self.subTest('isolated layer'):
            hidden_dim = 1
            split_layer = ModelLayer(hidden_dim,0.).eval()
            node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
            cycle_encoder = CycleEmbedding1(hidden_dim,'ZINC').eval()
            def forward(data : MultiScaleData_2):
                node_rep = node_encoder(data.x)
                edge_rep = edge_encoder(data.edge_attr)
                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                return split_layer(node_rep,edge_rep,cycle_rep,data,data.get_edge2cycle())
            
            ind_preds = [forward(g) for g in small_batches]
            ind_preds = [torch.cat([p[k] for p in ind_preds]) for k in range(3)]
            # ind_preds = torch.cat([ind_preds.flatten(),torch.cat([torch.arange(sub_batch_size) + i * sub_batch_size for i in range(num_sub_batches)])],-1)
            batch_preds = forward(batch)
            for i in range(3):
                rep_name = ['node','edge','cycle'][i]
                with self.subTest(rep = rep_name):
                    ind_pred = ind_preds[i]
                    batch_pred = batch_preds[i]
                    self.assertTrue(torch.allclose(ind_pred,batch_pred),f"\n{ind_pred} != \n{batch_pred}")


        with self.subTest('split layer (node-edge)'):
            hidden_dim = 1
            split_layer = SplitLayer(hidden_dim).eval()
            node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
            # cycle_encoder = CycleEmbedding(hidden_dim).eval()

            def forward(data : MultiScaleData_2):
                node_rep = node_encoder(data.x)
                edge_rep = edge_encoder(data.edge_attr)
                return split_layer(node_rep,edge_rep,data.node2edge_index)
            
            ind_preds = [forward(g) for g in small_batches]
            ind_preds = [torch.cat([p[k] for p in ind_preds]) for k in range(2)]
            batch_preds = forward(batch)
            for name, ind_pred, batch_pred in zip(['node','edge'],ind_preds,batch_preds):
                ind_pred = ind_pred.flatten()
                batch_pred = batch_pred.flatten()
                with self.subTest(rep = name):
                    self.assertTrue(torch.allclose(ind_pred,batch_pred),f"\n{ind_pred} != \n{batch_pred}")
        

        with self.subTest('edge cycle split layer'):
            hidden_dim = 1
            split_layer = SplitLayer0_1_complex(hidden_dim).eval()
            # node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
            cycle_encoder = CycleEmbedding1(hidden_dim,'ZINC').eval()

            def forward(data : MultiScaleData_2):
                edge_rep = edge_encoder(data.edge_attr)
                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                return split_layer(edge_rep,cycle_rep,data.get_edge2cycle())
            
            ind_preds = [forward(g) for g in small_batches]
            ind_preds = [torch.cat([p[k] for p in ind_preds]) for k in range(2)]
            batch_preds = forward(batch)
            for name, ind_pred, batch_pred in zip(['edge','cycle'],ind_preds,batch_preds):
                ind_pred = ind_pred.flatten()
                batch_pred = batch_pred.flatten()
                with self.subTest(rep = name):
                    self.assertTrue(torch.allclose(ind_pred,batch_pred),f"\n{ind_pred} != \n{batch_pred}")
        
        with self.subTest('pooling'):
            hidden_dim = 1
            node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
            cycle_encoder = CycleEmbedding1(hidden_dim,'ZINC').eval()
            def forward(data : MultiScaleData_2):
                node_rep = node_encoder(data.x)
                edge_rep = edge_encoder(data.edge_attr)
                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                return [
                    global_add_pool(node_rep,data.batch,size=data.num_graphs),
                    global_add_pool(edge_rep,data.edge_batch,size=data.num_graphs),
                    global_add_pool(cycle_rep,data.cycle_batch,size=data.num_graphs)
                ]
            ind_preds = [forward(g) for g in small_batches]
            ind_preds = [torch.cat([p[k] for p in ind_preds]) for k in range(3)]
            batch_preds = forward(batch)
            for name, ind_pred, batch_pred in zip(['node','edge','cycle'],ind_preds,batch_preds):
                with self.subTest(rep_name=name):
                    self.assertTrue(torch.allclose(ind_pred,batch_pred),f"\n{ind_pred} != \n{batch_pred}")
if __name__ == '__main__':
    import unittest
    unittest.main()