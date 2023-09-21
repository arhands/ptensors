from typing import Callable, Union
from unittest import TestCase

import torch
from torch import Tensor
from objects import TransferData1, atomspack1, MultiScaleData_2
from ptensors1 import linmaps1_1, transfer0_1, transfer1_0, transfer0_1_bi_msg
from transforms import PreprocessTransform as PreprocessTransform
from model_4 import Net, ModelLayer, get_node_encoder, get_edge_encoder, CycleEmbedding, SplitLayer, EdgeCycleSplitLayer
from torch_geometric.datasets import ZINC
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter 

def _no_x_transfer0_1_bi_msg_inv(data: TransferData1, y: Tensor, reduce: Union[list[str],str]='sum'):
    r"""
        for transfering from a ptensors0 to a ptensors1
    """
    if isinstance(reduce,str):
        reduce = [reduce]*2
    # TODO: add intersection invariant map from y!!!!
    y_inv = scatter(y,data.target.domain_indicator,0,dim_size=data.target.num_domains,reduce=reduce[0])
    y_inv = y_inv[data.domain_map_edge_index[1]]

    msg_inv = y_inv

    out_inv = scatter(msg_inv,data.domain_map_edge_index[1],0,dim_size=data.target.num_domains,reduce=reduce[1])
    # out_inv = out_inv[data.target.domain_indicator]

    return out_inv

def _linear_split_layer_lift(edge_rep: Tensor, cycle_rep: Tensor, edge2cycle: TransferData1, bi_msg_inv: bool, bi_msg_int_int: bool, bi_msg_int_inv: bool, aggr_no_x: bool) -> Tensor:
    def inv_msg_encoder(x: Tensor, y: Tensor) -> Tensor:
        y_int, y_inv = y[:,:cycle_rep.size(1)], y[:,cycle_rep.size(1):]
        agg = torch.zeros_like(x)
        if bi_msg_inv:
            agg = y_int + y_inv
        if not aggr_no_x:
            agg = agg + x
        # y = x + y if bi_msg_inv else x
        return torch.cat([
            x,
            y_int + y_inv
        ],-1)
    def int_msg_encoder(x: Tensor, y: Tensor) -> Tensor:
        y_int, y_inv = y[:,:cycle_rep.size(1)], y[:,cycle_rep.size(1):]
        agg = torch.zeros_like(x)
        if bi_msg_int_int:
            agg = y_int
        if bi_msg_int_inv:
            agg = agg + y_inv
        if not aggr_no_x:
            agg = agg + x
        return torch.cat([
            x,
            agg
        ],-1)
    
    return transfer0_1_bi_msg(edge_rep,edge2cycle,int_msg_encoder,inv_msg_encoder,cycle_rep)



def _linear_split_layer(edge_rep: Tensor, cycle_rep: Tensor, edge2cycle: TransferData1, bi_msg_inv: bool, bi_msg_int_int: bool, bi_msg_int_inv: bool, agg_no_x: bool) -> tuple[Tensor,Tensor]:
    cycle2edge = edge2cycle.reverse()

    cat_cycle = _linear_split_layer_lift(edge_rep,cycle_rep,edge2cycle,bi_msg_inv, bi_msg_int_int, bi_msg_int_inv, agg_no_x)
    lvl_aggr_cycle, lift_aggr = cat_cycle[:,edge_rep.size(1):], cat_cycle[:,:edge_rep.size(1)]
    lvl_aggr = transfer1_0(lvl_aggr_cycle,cycle2edge)

    edge_out = edge_rep + lvl_aggr

    cycle_out = cycle_rep + lift_aggr

    return edge_out, cycle_out

class batching(TestCase):
    @torch.no_grad()
    def test_batching(self):
        ds = ZINC('data/ZINC_base',True,'val')
        tf = PreprocessTransform()
        sub_batch_size = 4
        num_sub_batches = 3
        graphs : list[MultiScaleData_2] = [tf(ds[i]) for i in range(sub_batch_size*num_sub_batches)] #type: ignore
        small_batches = [Batch.from_data_list([graphs[i*sub_batch_size + j] for j in range(sub_batch_size)]) for i in range(num_sub_batches)] #type: ignore
        batch = Batch.from_data_list(graphs) #type: ignore

        with self.subTest('full model'):
            for num_layers in [0,1,4,10]:
                with self.subTest(num_layers=num_layers):
                    model = Net(128,num_layers,0,'ZINC',False).eval()
                    ind_preds = torch.cat([model(g).flatten() for g in small_batches])
                    # ind_preds = torch.cat([ind_preds.flatten(),torch.cat([torch.arange(sub_batch_size) + i * sub_batch_size for i in range(num_sub_batches)])],-1)
                    batch_preds = model(batch).flatten()
                    # batch_preds = torch.cat([batch_preds.flatten(),torch.arange(sub_batch_size*num_sub_batches)],-1)

                    self.assertTrue(torch.allclose(ind_preds,batch_preds),f"\n{ind_preds} != \n{batch_preds}")
        

        with self.subTest('isolated layer'):
            hidden_dim = 1
            split_layer = ModelLayer(hidden_dim).eval()
            node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
            cycle_encoder = CycleEmbedding(hidden_dim).eval()
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
                    ind_pred = ind_preds[i].flatten()
                    batch_pred = batch_preds[i].flatten()
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
            split_layer = EdgeCycleSplitLayer(hidden_dim).eval()
            # node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
            cycle_encoder = CycleEmbedding(hidden_dim).eval()

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
                    self.assertTrue(torch.allclose(ind_pred,batch_pred),f"\n{ind_pred} != \n{batch_pred}, {(ind_pred - batch_pred).abs().max()}")
        
        for bi_msg_inv in [False,True]:
            for bi_msg_int_int in [False,True]:
                for bi_msg_int_inv in [False,True]:
                    for agg_no_x in [False,True]:
                        with self.subTest('edge cycle split layer (linear)',bi_msg_inv=bi_msg_inv,bi_msg_int_int=bi_msg_int_int,bi_msg_int_inv=bi_msg_int_inv,agg_no_x=agg_no_x):
                            hidden_dim = 1
                            # node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
                            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
                            cycle_encoder = CycleEmbedding(hidden_dim).eval()

                            def forward(data : MultiScaleData_2):
                                edge_rep = edge_encoder(data.edge_attr)
                                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                                return _linear_split_layer(edge_rep,cycle_rep,data.get_edge2cycle(),bi_msg_inv,bi_msg_int_int,bi_msg_int_inv,agg_no_x)
                            
                            ind_preds = [forward(g) for g in small_batches]
                            ind_preds = [torch.cat([p[k] for p in ind_preds]) for k in range(2)]
                            batch_preds = forward(batch)
                            for name, ind_pred, batch_pred in zip(['edge','cycle'],ind_preds,batch_preds):
                                ind_pred = ind_pred.flatten()
                                batch_pred = batch_pred.flatten()
                                with self.subTest(rep = name):
                                    self.assertTrue(torch.allclose(ind_pred,batch_pred),f"{(ind_pred - batch_pred).abs().max()}")
                                    # self.assertTrue(torch.allclose(ind_pred,batch_pred),f"\n{ind_pred} != \n{batch_pred}, {(ind_pred - batch_pred).abs().max()}")


        for bi_msg_inv in [False,True]:
            for bi_msg_int_int in [False,True]:
                for bi_msg_int_inv in [False,True]:
                    for agg_no_x in [False,True]:
                        with self.subTest('lift transfer (edge-cycle, linear)',bi_msg_inv=bi_msg_inv,bi_msg_int_int=bi_msg_int_int,bi_msg_int_inv=bi_msg_int_inv,agg_no_x=agg_no_x):
                            hidden_dim = 1
                            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
                            cycle_encoder = CycleEmbedding(hidden_dim).eval()
                            
                            def forward(data : MultiScaleData_2):
                                edge_rep = edge_encoder(data.edge_attr)
                                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                                cycle_rep = _linear_split_layer_lift(edge_rep,cycle_rep,data.get_edge2cycle(),bi_msg_inv,bi_msg_int_int,bi_msg_int_inv,agg_no_x)
                                return cycle_rep
                            
                            ind_pred = [forward(g) for g in small_batches]
                            ind_pred = torch.cat(ind_pred).flatten()
                            batch_pred = forward(batch).flatten()
                            self.assertTrue(torch.allclose(ind_pred,batch_pred),f"{(ind_pred - batch_pred).abs().max()}")
                            # self.assertTrue(torch.allclose(ind_preds,batch_preds),f"\n{ind_preds} != \n{batch_preds}")


        with self.subTest('cycle linmaps'):
            hidden_dim = 1
            cycle_encoder = CycleEmbedding(hidden_dim).eval()

            def forward(data : MultiScaleData_2):
                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                return linmaps1_1(cycle_rep,data.get_edge2cycle().target)
            
            ind_preds = [forward(g) for g in small_batches]
            ind_pred = torch.cat(ind_preds).flatten()
            batch_pred = forward(batch).flatten()
            self.assertTrue(torch.allclose(ind_pred,batch_pred),f"{(ind_pred - batch_pred).abs().max()}")
        
        with self.subTest('bi msg self transfer'):
            hidden_dim = 1
            cycle_encoder = CycleEmbedding(hidden_dim).eval()

            def forward(data : MultiScaleData_2):
                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                return _no_x_transfer0_1_bi_msg_inv(data.get_edge2cycle(),cycle_rep)
            
            ind_preds = [forward(g) for g in small_batches]
            ind_pred = torch.cat(ind_preds).flatten()
            batch_pred = forward(batch).flatten()
            self.assertTrue(torch.allclose(ind_pred,batch_pred),f"{(ind_pred - batch_pred).abs().max()}")


        with self.subTest('cycle2edge transfer'):
            hidden_dim = 1
            cycle_encoder = CycleEmbedding(hidden_dim).eval()

            
            def forward(data : MultiScaleData_2):
                cycle_rep = cycle_encoder(data.x,(data.cycle_atoms,data.cycle_domain_indicator))
                edge_rep = transfer1_0(cycle_rep,data.get_edge2cycle().reverse())
                return edge_rep
            
            ind_preds = [forward(g) for g in small_batches]
            ind_preds = torch.cat(ind_preds).flatten()
            batch_preds = forward(batch).flatten()
            self.assertTrue(torch.allclose(ind_preds,batch_preds),f"\n{ind_preds} != \n{batch_preds}")

        with self.subTest('pooling'):
            hidden_dim = 1
            node_encoder = get_node_encoder(hidden_dim,'ZINC').eval()
            edge_encoder = get_edge_encoder(hidden_dim,'ZINC').eval()
            cycle_encoder = CycleEmbedding(hidden_dim).eval()
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