"""
Functions that can be utilized in YATE GNN model

"""

# Python
import math

# Pytorch
import torch
from torch import Tensor
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.typing import Adj, PairTensor

#########

# YATE - Obtaining Z tensor (element-wise multiplication of node and edge features)
def YATE_Z(x: Tensor, edge_index: Adj, edge_feat: Adj):

    num_input = x.size()[1]
    num_edges = edge_index.size()[1]

    Z = torch.zeros((num_edges, num_input))

    for i in range(num_edges):

        Z[i, :] = torch.mul(edge_feat[i, :], x[edge_index[1, i], :])

    return Z


# YATE - Attention Calculation
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
