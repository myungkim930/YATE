from data import YATE_loaddata

data_dir = "/storage/store3/work/mkim/gitlab/YATE/data/YAGO3_NO.pkl"
YATE_data = YATE_loaddata(data_dir)

from graphlet_construction import make_batch

idx_cen = [10234,12345]
num_hops = 1
d = make_batch(idx_cen = idx_cen, num_hops=num_hops, main_data=YATE_data)







g = Graphlet(YATE_data)
idx = int(11351)
data = g.make_graphlet(cen_ent=idx, num_hops=1)

aug = Augment(max_nodes=100, n_pos=1, per_pos=0.8, n_neg=1, per_neg=0.05)
data_total = aug.generate(data, main_data=YATE_data)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ent_list = YATE_data.edgelist_total[1,:].unique()

# Train
n_batch = 10
num_hops = 1
n_epoch = 100
idx_data = DataLoader(ent_list, batch_size=64, shuffle=True)
for _ in n_epoch:
    for x in idx_data:
    
from torch import Tensor
from typing import List, Union

idx_cen = torch.tensor([2616, 1354])
idx_cen.tolist()

def make_batch(idx_cen: Union[int, List[int], Tensor], num_hops: int, main_data):
    
    if isinstance(idx_cen, Tensor):
        idx_cen =idx_cen.tolist()
    elif isinstance(idx_cen, Tensor):
        idx_cen = [idx_cen]
        
    data = []

    for g_idx in range(len(idx_cen)):
        
        data_temp = g.make_graphlet(cen_ent=idx_cen[g_idx], num_hops=num_hops)
        data_total_temp = aug.generate(data_temp, main_data=main_data)
        
        data.append(data_total_temp)
        
    d_batch = next(iter(DataLoader(data, batch_size=len(idx_cen))))

    return d_batch

d = make_batch(idx_cen = idx_cen, num_hops=num_hops, main_data=YATE_data)






idx = [2616, 1354]

idx = 246543

ent_list = YATE_data.edgelist_total[1,:]
no_list = YATE_data.edgelist_total[0,:]
YATE_data.ent2idx[297876]
YATE_data.rel2idx[32]

temp = YATE_data.ent2idx[[ent_list==38108]]

temp = YATE_data.edgelist_total[:, (ent_list==38108).nonzero().squeeze()]
YATE_data.edgetype_total[(ent_list==38108).nonzero().squeeze()]
(ent_list==246543).nonzero().squeeze().size()
(no_list==246543).nonzero().squeeze().size()



[
246543
38108

from torch_geometric.loader import DataLoader
loader1 = DataLoader(data_total, batch_size=len(data_total))
batch1 = next(iter(loader1))

from models import YATE_Attention as YA
from models import YATE_Encode as YE
yate = YE(input_dim=300, emb_dim = 300, output_dim=2, num_heads=2, num_layers = 5)
# yate = YA(input_dim=300, output_dim=300, num_heads=2)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch1 = batch1.to(device)
yate = yate.to(device)

out_x, out_e = yate(batch1.x, batch1.edge_index, batch1.edge_attr)
out_x = yate(batch1.x, batch1.edge_index, batch1.edge_attr)

out_x.device
num_input = batch1.x.size()[1]
num_edges = batch1.edge_index.size()[1]



import torch

Z = torch.zeros((batch1.num_edges, batch1.num_node_features), device = batch1.x.device)
for i in range(batch1.num_edges):
    Z[i, :] = torch.mul(batch1.edge_attr[i, :], batch1.x[batch1.edge_index[1, i], :])

Z1 = torch.mul(batch1.edge_attr, batch1.x[batch1.edge_index[1, :]])

torch.unique(torch.eq(Z, Z1))








Z.is_cuda

out_x.is_cuda
out_e.is_cuda

from models import YATE_Z
a = YATE_Z(x=batch1.x, edge_index=batch1.edge_index, edge_feat=batch1.edge_attr)
batch1.is_cuda
a.is_cuda
next(yate.parameters()).is_cuda

yate.is_cuda
batch1.x.is_cuda
batch1.edge_attr.is_cuda








from models import YATE_enc_layer as YE
yate = YE.YATE_Encode(
    input_dim=300, emb_dim=300, output_dim=300, num_heads=2, num_layers=5
)
out_x = yate(batch.x, batch.edge_index, batch.edge_attr)

data_temp = data_total[0:2]

from graphlet_construction import feature_extract_lm

temp = feature_extract_lm(main_data=YATE_data, node_idx=1222176)

gent_names = YATE_data.ent2idx[torch.tensor(1222176)]

import numpy as np
import torch

x = [
    YATE_data.x_model.get_sentence_vector(
        x.replace("_", " ").replace("<", "").replace(">", "").lower()
    )
    for x in gent_names
]
temp3 = np.array(x)
temp3 = torch.tensor(temp3,)
temp3 = temp3[1,:]

YATE_data.ent2idx[torch.tensor(1222176)]

word = '<Devon>'
temp1 = YATE_data.x_model.get_sentence_vector(
                word.replace("_", " ").replace("<", "").replace(">", "").lower()
            )

temp1 = np.array(temp1)
temp1 = torch.tensor(temp1,)
temp2 = data.x[1]

grel_names = YATE_data.rel2idx[data.edge_type]





import torch
print(torch.__version__)
print(torch.version.cuda)
device = torch.device("cuda")

batch.is_cuda

print(device)


.to(dev)



data = data.to(dev)
yate = yate.to(device)



import torch

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data.x.is_cuda


# yate_att = YE.YATE_Block(in_features, output_dim = 100, num_heads = 2, concat = True)
# out_x, out_e = yate_att(x, edge_index, edge_feat)


# node_idx = torch.unique(edge_list).type(torch.LongTensor)
# nb_nodes = node_idx.size()[0]


# edge_list = torch.hstack((edge_list, torch.unique(edge_list).repeat(2, 1)))
# edge_type = torch.hstack((edge_type, torch.zeros(1, nb_nodes)))[0]
# edge_type = edge_type.type(torch.LongTensor)

# nb_edges = edge_list.size()[1]

# x = x_total[node_idx, :]
# edge_feat = edge_feat_total[edge_type, :]
# edge_index = edge_list


YATE_data.x_model
x_neg_replace = feature_extract_lm(main_data=YATE_data, node_idx=1537801)
x_neg_replace1 = temp.x[1, :]
n_neg = 10
per_neg = 0.2


g_pos = gen_pos(n_pos=5, per_pos=0.8, data=data)

g_neg = gen_neg(n_neg=10, per_neg=0.2, data=data, main_data=YATE_data)

from graphlet_construction import subgraph, feature_extract_lm
import numpy as np
import math
import torch
from torch_geometric.data import Data


q = torch.rand(10, 5)
k = torch.rand(12, 5)
v = torch.rand(12, 5)

num_nodes = q.size()[0]
num_edges = k.size()[0]
num_emb = q.size()[1]

att_logit = torch.zeros((num_nodes, num_nodes))

edge_index = torch.tensor([[0,1,2,6,7,9,3,8,6,7,5,0],[1,2,3,4,5,6,7,6,4,6,6,7]])

for i in range(num_edges):

    att_logit[edge_index[0, i], edge_index[1, i]] = torch.matmul(
        q[edge_index[0, i], :], k[i, :]
    )

q[edge_index[0, :], :].size(), k[i, :]





def YATE_Att_Calc(edge_index: Adj, query: Tensor, key: Tensor, value: Tensor):

    num_nodes = query.size()[0]
    num_edges = key.size()[0]
    num_emb = query.size()[1]

    att_logit = torch.zeros((num_nodes, num_nodes))

    for i in range(num_edges):

        att_logit[edge_index[0, i], edge_index[1, i]] = torch.matmul(
            query[edge_index[0, i], :], key[i, :]
        )

    att_logit = att_logit / math.sqrt(num_emb)
    zero_vec = -9e15 * torch.ones_like(att_logit)
    att_logit = torch.where(att_logit != 0, att_logit, zero_vec)
    attention = F.softmax(att_logit, dim=1)

    output = torch.zeros(num_nodes, num_nodes, num_emb)

    for i in range(num_edges):

        output[edge_index[0, i], edge_index[1, i], :] = (
            attention[edge_index[0, i], edge_index[1, i]] * value[i, :]
        )

    output = output.sum(dim=1)

    return output, attention

