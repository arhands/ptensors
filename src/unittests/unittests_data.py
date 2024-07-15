from unittest import TestCase, main

import torch
import torch_geometric
from objects1 import TransferData1, TransferData0, atomspack1
from objects2 import TransferData2, atomspack2
from torch_geometric.utils import segment
from torch_geometric.data import Data, Batch
from data import FancyDataObject
from ptensors.ptensors2 import transfer2_2_minimal, linmaps0_2, linmaps2_0
from ptensors.ptensors1 import linmaps0_1, linmaps1_0, transfer1_1
from data_transforms import GeneratePtensObject, AddNodes, AddEdges, AddChordlessCycles, AddTransferMap, PtensObjects

class batching(TestCase):
  @torch.no_grad()
  def test_edges_1(self):
    A = FancyDataObject(torch.tensor([1,10]).unsqueeze(-1),
             torch.tensor([[],[]],dtype=torch.int64))
    ap: atomspack1 = atomspack1.from_list([[1,0],[0,1]])
    source = 'single_edge'
    A.set_atomspack(ap,source)
    B = FancyDataObject(torch.tensor([100,1000]).unsqueeze(-1),
             torch.tensor([[],[]],dtype=torch.int64))
    B.set_atomspack(ap,source)
    def forward(G: FancyDataObject, reduce: str = 'sum'):
      res = G.get_ptens_params()
      aps = res[0][source]
      y = linmaps0_1(G.x,aps)
      z = linmaps1_0(y,aps,reduce)
      return z
    C: FancyDataObject = Batch.from_data_list([A,B])#type: ignore
    D: FancyDataObject = Batch.from_data_list([B,A])#type: ignore
    for reduce in ['sum','mean','min','max']:
      with self.subTest(reduce=reduce):
        y1 = forward(C,reduce)
        y2 = forward(D,reduce)
        y2p = torch.cat([y2[D.batch == 1],y2[D.batch == 0]],0)
        self.assertTrue((y1 == y2p).all())
  @torch.no_grad()
  def test_edges_transfer_1(self):
    A = FancyDataObject(torch.tensor([1,10]).unsqueeze(-1),
             torch.tensor([[],[]],dtype=torch.int64))
    aps: atomspack1 = atomspack1.from_list([[0],[1]])
    apt: atomspack1 = atomspack1.from_list([[1,0]])
    tf: TransferData1 = TransferData1.from_atomspacks(aps,apt,False)
    # aps2: atomspack1 = atomspack1.from_list([[0],[1],[2],[3]])
    # apt2: atomspack1 = atomspack1.from_list([[1,0],[3,2]])
    # tf2: TransferData1 = TransferData1.from_atomspacks(aps2,apt2,False)
    source = 'nodes'
    target = 'edge'
    A.set_atomspack(aps,source)
    A.set_atomspack(apt,target)
    A.set_transfer_maps(source,target,tf)
    B = FancyDataObject(torch.tensor([100,1000]).unsqueeze(-1),
             torch.tensor([[],[]],dtype=torch.int64))
    B.set_atomspack(aps,source)
    B.set_atomspack(apt,target)
    B.set_transfer_maps(source,target,tf)
    def forward(G: FancyDataObject, reduce: str = 'sum'):
      res = G.get_ptens_params()
      aps = res[0][source]
      apt = res[0][target]
      tf = res[3][(source,target)]
      y = linmaps0_1(G.x,aps)
      z: torch.Tensor = transfer1_1(y,tf)#type: ignore
      # print("node_map_edge_index:\n",tf.node_map_edge_index)
      # print("intersect_indicator:\n",tf.intersect_indicator)
      a = linmaps1_0(z,apt,reduce)
      return a
    C: FancyDataObject = Batch.from_data_list([A,B])#type: ignore
    D: FancyDataObject = Batch.from_data_list([B,A])#type: ignore
    for reduce in ['sum','mean','min','max']:
      with self.subTest(reduce=reduce):
        y1 = forward(C,reduce)
        y2 = forward(D,reduce)
        y2p = y2.flip(0)
        self.assertTrue((y1 == y2p).all(),f"\n{y1} !=\n{y2p}")
  @torch.no_grad()
  def test_mult_edges_transfer_1(self):
    A = FancyDataObject(torch.tensor([1,10,100]).unsqueeze(-1),
             torch.tensor([[],[]],dtype=torch.int64))
    aps: atomspack1 = atomspack1.from_list([[0],[1],[2]])
    apt: atomspack1 = atomspack1.from_list([[1,0],[0,2]])
    tf: TransferData1 = TransferData1.from_atomspacks(aps,apt,False)
    source = 'nodes'
    target = 'edge'
    A.set_atomspack(aps,source)
    A.set_atomspack(apt,target)
    A.set_transfer_maps(source,target,tf)
    B = FancyDataObject(torch.tensor([[1000*10**i] for i in range(6)]),
             torch.tensor([[],[]],dtype=torch.int64))
    aps: atomspack1 = atomspack1.from_list([[0],[1],[2],[3],[4],[5]])
    apt: atomspack1 = atomspack1.from_list([[0,2],[1,2],[3,4]])
    tf: TransferData1 = TransferData1.from_atomspacks(aps,apt,False)
    
    B.set_atomspack(aps,source)
    B.set_atomspack(apt,target)
    B.set_transfer_maps(source,target,tf)
    def forward(G: FancyDataObject, reduce: str = 'sum'):
      res = G.get_ptens_params()
      aps = res[0][source]
      # apt = res[0][target]
      self.assertEqual(len(G.x),len(aps.domain_indicator))
      self.assertEqual(len(G.x),len(aps.atoms))
      tf = res[3][(source,target)]
      y = linmaps0_1(G.x,aps)
      z: torch.Tensor = transfer1_1(y,tf)#type: ignore
      # print("node_map_edge_index:\n",tf.node_map_edge_index)
      # print("intersect_indicator:\n",tf.intersect_indicator)
      # a = linmaps1_0(z,apt,reduce)
      # return a
      return z
    C: FancyDataObject = Batch.from_data_list([A,B])#type: ignore
    D: FancyDataObject = Batch.from_data_list([B,A])#type: ignore
    # for reduce in ['sum']:
    for reduce in ['sum','mean','min','max']:
      with self.subTest(reduce=reduce):
        y1 = forward(C,reduce)
        y2 = forward(D,reduce)
        y2p = torch.cat([
          y2[6:],
          y2[:6]
        ])
        self.assertTrue((y1 == y2p).all(),f"\n{y1} !=\n{y2p}")
  @torch.no_grad()
  def test_mult_edges_transfer_2(self):
    A = FancyDataObject(torch.tensor([1,10,100]).unsqueeze(-1),
             torch.tensor([[],[]],dtype=torch.int64))
    aps: atomspack2 = atomspack2.from_list([[0],[1],[2]])
    apt: atomspack2 = atomspack2.from_list([[1,0],[0,2]])
    tf: TransferData2 = TransferData2.from_atomspacks(aps,apt,False)
    source = 'nodes'
    target = 'edge'
    A.set_atomspack(aps,source)
    A.set_atomspack(apt,target)
    A.set_transfer_maps(source,target,tf)
    B = FancyDataObject(torch.tensor([[1000*10**i] for i in range(6)]),
             torch.tensor([[],[]],dtype=torch.int64))
    aps: atomspack2 = atomspack2.from_list([[0],[1],[2],[3],[4],[5]])
    apt: atomspack2 = atomspack2.from_list([[0,2],[1,2],[3,4]])
    tf: TransferData2 = TransferData2.from_atomspacks(aps,apt,False)
    aps2: atomspack2 = atomspack2.from_list([[i] for i in range(9)])
    apt2: atomspack2 = atomspack2.from_list([[1,0],[0,2],[3,5],[4,5],[6,7]])
    tf2: TransferData2 = TransferData2.from_atomspacks(aps2,apt2,False)

    B.set_atomspack(aps,source)
    B.set_atomspack(apt,target)
    B.set_transfer_maps(source,target,tf)
    C: FancyDataObject = Batch.from_data_list([A,B])#type: ignore
    res = C.get_ptens_params()
    aps = res[1][source]
    for ap_name, ap2 in [(source,aps2),(target,apt2)]:
      ap: atomspack2 = res[1][ap_name]
      with self.subTest(ap_name=ap_name):
        # self.assertTrue((ap.atoms == ap2.atoms).all(),f"\n{ap.atoms} !=\n{ap2.atoms}")
        self.assertTrue((ap.col_indicator == ap2.col_indicator).all(),f"\n{ap.col_indicator} !=\n{ap2.col_indicator}")
        self.assertTrue((ap.row_indicator == ap2.row_indicator).all(),f"\n{ap.row_indicator} !=\n{ap2.row_indicator}")
        self.assertTrue((ap.diag_idx == ap2.diag_idx).all(),f"\n{ap.diag_idx} !=\n{ap2.diag_idx}")
        # self.assertTrue((ap.domain_indicator == ap2.domain_indicator).all(),f"\n{ap.domain_indicator} !=\n{ap2.domain_indicator}")
        # self.assertEqual(ap.num_domains,ap2.num_domains)
        self.assertTrue((ap.transpose_indicator == ap2.transpose_indicator).all(),f"\n{ap.transpose_indicator} !=\n{ap2.transpose_indicator}")
    tf = res[4][(source,target)]
    self.assertTrue((tf.domain_map_edge_index == tf2.domain_map_edge_index).all(),f"\n{tf.domain_map_edge_index} !=\n{tf2.domain_map_edge_index}")
    self.assertTrue((tf.ij_indicator == tf2.ij_indicator).all(),f"\n{tf.ij_indicator} !=\n{tf2.ij_indicator}")
    self.assertTrue((tf.node_pair_map == tf2.node_pair_map).all(),f"\n{tf.node_pair_map} !=\n{tf2.node_pair_map}")
  @torch.no_grad()
  def test_mult_edges_transfer_2_nonrev(self):
    A = FancyDataObject(torch.tensor([1,10,100]).unsqueeze(-1),
             torch.tensor([[],[]],dtype=torch.int64))
    aps: atomspack2 = atomspack2.from_list([[0],[1],[2]])
    apt: atomspack2 = atomspack2.from_list([[1,0],[0,2]])
    tf: TransferData2 = TransferData2.from_atomspacks(aps,apt,False)
    source = 'nodes'
    target = 'edge'
    A.set_atomspack(aps,source)
    A.set_atomspack(apt,target)
    A.set_transfer_maps(source,target,tf)

    B = FancyDataObject(torch.tensor([[1000*10**i] for i in range(6)]),
             torch.tensor([[],[]],dtype=torch.int64))
    aps: atomspack2 = atomspack2.from_list([[0],[1],[2],[3],[4],[5]])
    apt: atomspack2 = atomspack2.from_list([[0,2],[1,2],[3,4]])
    tf: TransferData2 = TransferData2.from_atomspacks(aps,apt,False)
    B.set_atomspack(aps,source)
    B.set_atomspack(apt,target)
    B.set_transfer_maps(source,target,tf)

    C: FancyDataObject = Batch.from_data_list([A,B])#type: ignore
    D: FancyDataObject = Batch.from_data_list([B,A])#type: ignore
    # for reduce in ['sum']:
    def forward(G: FancyDataObject, reduce: str = 'sum'):
      res = G.get_ptens_params()
      aps: atomspack2 = res[1][source]
      # apt = res[0][target]
      tf = res[4][(source,target)]
      y = linmaps0_2(G.x,aps)
      z: torch.Tensor = transfer2_2_minimal(y,tf,reduce)#type: ignore
      return z
    for reduce in ['sum','mean','min','max']:
      with self.subTest(reduce=reduce):
        y1 = forward(C,reduce)
        y2 = forward(D,reduce)
        y2p = torch.cat([
          y2[12:],
          y2[:12]
        ])
        self.assertTrue((y1 == y2p).all(),f"\n{y1} !=\n{y2p}")
  def test_zinc(self):
    from torch_geometric.datasets import ZINC
    dataset = ZINC('./data/ZINC_base',True)
    G1 = dataset[0]
    G2 = dataset[1]
    transform = GeneratePtensObject([
      AddNodes(2),
      AddChordlessCycles(2),
    ],[
      AddTransferMap('nodes','cycles',2,False)
    ])
    # G1.x = torch.arange(G1.num_nodes).unsqueeze(-1).float().broadcast_to(G1.num_nodes,256) - G1.num_nodes/2
    # G2.x = torch.arange(G2.num_nodes).unsqueeze(-1).float().broadcast_to(G2.num_nodes,256) - G2.num_nodes/2
    A: FancyDataObject = transform(G1)
    B: FancyDataObject = transform(G2)
    assert A.get_ptens_params()[1]['cycles'].num_domains > 0
    assert B.get_ptens_params()[1]['cycles'].num_domains > 0

    C: FancyDataObject = Batch.from_data_list([A,B])#type: ignore
    D: FancyDataObject = Batch.from_data_list([B,A])#type: ignore
    # for reduce in ['sum']:
    nc = 256
    mlp2 = torch.nn.Sequential(
      torch.nn.LazyLinear(nc),
      torch.nn.ReLU(True),
      # torch.nn.LazyLinear(nc),
    )
    emb = torch.nn.Embedding(22,nc)
    def forward(G: FancyDataObject, reduce: str = 'sum'):
      ptobjs = PtensObjects.from_fancy_data(G)
      aps: atomspack2 = ptobjs[('nodes',2)]
      apt: atomspack2 = ptobjs[('cycles',2)]
      # apt = res[0][target]
      tf: TransferData2 = ptobjs[(('nodes','cycles'),2)]
      y = emb(G.x.flatten())
      # y = G.x
      y = linmaps0_2(y,aps)
      y: torch.Tensor = transfer2_2_minimal(y,tf,reduce)#type: ignore
      # y = mlp1(y)
      y = linmaps2_0(y,apt)
      y = mlp2(y)
      # batch = torch.empty(len(apt.raw_num_domains) + 1,dtype=torch.int64,device=y.device)#type: ignore
      # batch[0] = 0
      # batch[1:] = apt.raw_num_domains.cumsum(0)#type: ignore
      # y = segment(y,batch,reduce)
      return y, apt.raw_num_domains
    # for reduce in ['sum','mean','min','max']:
    for reduce in ['sum']:
      with self.subTest(reduce=reduce):
        y1, _ = forward(C,reduce)
        y2, num_domains = forward(D,reduce)
        # y2p = y2.flip(0)
        y2p = torch.cat([
          y2[num_domains[0]:],
          y2[:num_domains[0]]
        ])
        
        self.assertTrue(torch.allclose(y1,y2p),f"\n{y1} !=\n{y2p}")

if __name__ == '__main__':
  main()