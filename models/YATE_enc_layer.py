"""
YATE encoder block that contains attention layer.

"""

# Python
import math
from typing import Tuple, Union

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# PyTorch Geometric
from torch_geometric.typing import Adj, PairTensor

##########

# YATE - Obtaining Z tensor (Replace)
def YATE_Z(x: Tensor, edge_index: Adj, edge_feat: Adj):

    num_input = x.size()[1]
    num_edges = edge_index.size()[1]

    Z = torch.zeros((num_edges, num_input), device=x.device)

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


# YATE - Attention Layer
class YATE_Attention(nn.Module):
    def __init__(
        self,
        input_dim: Union[int, Tuple[int, int]],
        output_dim: int,
        num_heads: int = 1,
        concat: bool = True,
    ):
        super(YATE_Attention, self).__init__()

        if concat:
            assert output_dim % num_heads == 0
            self.lin_query = nn.Linear(input_dim, num_heads * output_dim // num_heads)
            self.lin_key = nn.Linear(input_dim, num_heads * output_dim // num_heads)
            self.lin_value = nn.Linear(input_dim, num_heads * output_dim // num_heads)
        else:
            self.lin_query = nn.Linear(input_dim, num_heads * output_dim)
            self.lin_key = nn.Linear(input_dim, num_heads * output_dim)
            self.lin_value = nn.Linear(input_dim, num_heads * output_dim)

        self.lin_edge = nn.Linear(input_dim, output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat = concat

        self.reset_parameters()

    def reset_parameters(self):

        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_feat: Adj,
        return_attention=False,
    ):

        # Z = YATE_Z(x, edge_index, edge_feat)
        Z = torch.mul(edge_feat, x[edge_index[1, :]])

        if self.concat:
            H, C = self.num_heads, self.output_dim // self.num_heads
        else:
            H, C = self.num_heads, self.output_dim

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(Z).view(-1, H, C)
        value = self.lin_value(Z).view(-1, H, C)

        output = torch.zeros((x.size()[0], H, C), device=x.device)
        attention = torch.zeros((x.size()[0], H, x.size()[0]))

        for head in range(H):

            Q, K, V = query[:, head, :], key[:, head, :], value[:, head, :]
            O, A = YATE_Att_Calc(edge_index, Q, K, V)
            output[:, head, :] = O
            attention[:, head, :] = A

        if self.concat:
            output = output.view(-1, self.output_dim)
        else:
            output = output.mean(dim=1)

        edge = self.lin_edge(Z)

        if return_attention:
            return output, edge, attention
        else:
            return output, edge


class YATE_Block(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, concat=True, dropout=0.1):

        super().__init__()

        # Graph Attention Layer
        self.g_attn = YATE_Attention(input_dim, input_dim, num_heads, concat)

        # Two-layer MLP
        self.linear_net_x = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, input_dim),
        )

        self.linear_net_e = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_feat: Adj,
    ):

        # Attention part
        attn_out_x, attn_out_e = self.g_attn(x, edge_index, edge_feat)

        x = x + attn_out_x
        x = self.norm1(x)

        edge_feat = edge_feat + attn_out_e
        edge_feat = self.norm1(edge_feat)

        # MLP part
        linear_out_x = self.linear_net_x(x)
        x = x + linear_out_x
        x = self.norm2(x)

        linear_out_e = self.linear_net_e(edge_feat)
        edge_feat = edge_feat + linear_out_e
        edge_feat = self.norm2(edge_feat)

        return x, edge_feat


class YATE_Encode(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, num_layers, **block_args):

        super(YATE_Encode, self).__init__()

        self.layers = [
            YATE_Block(input_dim, emb_dim, **block_args) for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(self.layers)
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index, edge_feat):

        # edge_index_undireced, edge_feat to undirected, index of new needed

        for l in self.layers:
            x, edge_feat = l(x, edge_index, edge_feat)

        # Transformer with Directed Graph (Readout layer)

        # 3. Apply a final classifier (need indexing)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        # Final layer for the task that can be replaced for later usage

        return x
