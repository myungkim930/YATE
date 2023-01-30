"""
YATE Transformer layer that contains the attention mechanism

"""
# Python
from typing import Tuple, Union

# Pytorch
import torch
from torch import Tensor
import torch.nn as nn

# PyTorch Geometric
from torch_geometric.typing import Adj, PairTensor

# YATE utils
from .YATE_utils import YATE_Z, YATE_Att_Calc

#########

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

        Z = YATE_Z(x, edge_index, edge_feat)

        if self.concat:
            H, C = self.num_heads, self.output_dim // self.num_heads
        else:
            H, C = self.num_heads, self.output_dim

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(Z).view(-1, H, C)
        value = self.lin_value(Z).view(-1, H, C)

        output = torch.zeros((x.size()[0], H, C))
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


# YATE - Transformer Block
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
