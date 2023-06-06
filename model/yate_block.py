"""
YATE neural network blocks used for pretrain/finetune.

"""

# Python
import math
from typing import Tuple

# PyTorch
import torch
import torch.nn as nn

from torch import Tensor

# PyTorch Geometric
from torch_geometric.utils import softmax
from torch_scatter import scatter

##################################################


## YATE - Attention and output calculation
def yate_attention(
    edge_index: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tuple[Tensor, Tensor]:
    attention = torch.sum(torch.mul(query[edge_index[0], :], key), dim=1) / math.sqrt(
        query.size(1)
    )
    attention = softmax(attention, edge_index[0])
    src = torch.mul(attention, value.t()).t()
    output = scatter(src, edge_index[0], dim=0, reduce="sum")
    return output, attention


## YATE - output calculation with multi-head (message passing)
def yate_multihead(
    edge_index: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int = 1,
    concat: bool = True,
):
    if concat:
        H, C = num_heads, query.size(1) // num_heads
        for i in range(H):
            O, A = yate_attention(
                edge_index,
                query[:, i * C : (i + 1) * C],
                key[:, i * C : (i + 1) * C],
                value[:, i * C : (i + 1) * C],
            )
            if i == 0:
                output, attention = O, A
            else:
                output = torch.cat((output, O), dim=1)
                attention = torch.cat((attention, A), dim=0)
    else:
        H, C = num_heads, query.size(1)
        for i in range(H):
            O, A = yate_attention(
                edge_index,
                query[:, i * C : (i + 1) * C],
                key[:, i * C : (i + 1) * C],
                value[:, i * C : (i + 1) * C],
            )
            if i == 0:
                output, attention = O, A
            else:
                output = torch.cat((output, O), dim=0)
                attention = torch.cat((attention, A), dim=0)
        output = output / H
        attention = attention / H
    return output, attention


## YATE - Attention Layer
class YATE_Attention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        read_out: bool = False,
    ):
        super(YATE_Attention, self).__init__()

        if concat:
            assert output_dim % num_heads == 0
            self.lin_query = nn.Linear(input_dim, output_dim, bias=False)
            self.lin_key = nn.Linear(input_dim, output_dim, bias=False)
            self.lin_value = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.lin_query = nn.Linear(input_dim, num_heads * output_dim, bias=False)
            self.lin_key = nn.Linear(input_dim, num_heads * output_dim, bias=False)
            self.lin_value = nn.Linear(input_dim, num_heads * output_dim, bias=False)

        if read_out == False:
            self.lin_edge = nn.Linear(input_dim, output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat = concat
        self.readout = read_out

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if self.readout == False:
            self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        return_attention: bool = False,
    ):
        Z = torch.mul(edge_attr, x[edge_index[1]])

        query = self.lin_query(x)
        key = self.lin_key(Z)
        value = self.lin_value(Z)

        output, attention = yate_multihead(
            edge_index=edge_index,
            query=query,
            key=key,
            value=value,
            num_heads=self.num_heads,
            concat=self.concat,
        )

        if self.readout == False:
            edge_attr = self.lin_edge(edge_attr)

        if return_attention:
            return output, edge_attr, attention
        else:
            return output, edge_attr


## YATE - single encoding block
class YATE_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout: float = 0.1,
        read_out: bool = False,
    ):
        super().__init__()

        # Graph Attention Layer
        self.g_attn = YATE_Attention(
            input_dim, input_dim, num_heads, concat, read_out=read_out
        )

        # Two-layer MLP + Layers to apply in between the main layers for x and edges
        self.linear_net_x = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim),
        )
        self.norm1_x = nn.LayerNorm(input_dim)
        self.norm2_x = nn.LayerNorm(input_dim)

        self.read_out = read_out
        if self.read_out == False:
            self.linear_net_e = nn.Sequential(
                nn.Linear(input_dim, ff_dim),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(ff_dim, input_dim),
            )
            self.norm1_e = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ):
        # Attention part
        attn_out_x, attn_out_e = self.g_attn(x, edge_index, edge_attr)

        # MLP part - Node
        x = x + self.dropout(attn_out_x)
        x = self.norm1_x(x)

        linear_out_x = self.linear_net_x(x)
        x = x + self.dropout(linear_out_x)
        x = self.norm2_x(x)

        # MLP part - Edge
        if self.read_out == False:
            edge_attr = self.linear_net_e(attn_out_e)
            edge_attr = edge_attr + self.dropout(edge_attr)
            edge_attr = self.norm1_e(edge_attr)
            return x, edge_attr
        else:
            return x


## YATE - contrast block
class YATE_Contrast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        x = nn.functional.normalize(x, dim=1)

        # Cosine similarity
        # x = 1 - (torch.cdist(x, x) / 2)

        # RBF kernel (Gaussian similarity)
        sig = torch.median(torch.cdist(x, x))
        x = torch.exp(-(torch.cdist(x, x) / (2 * sig)))

        return x


## YATE - finetune base block


class YATE_Base(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        num_layers: int,
        **block_args
    ):
        super(YATE_Base, self).__init__()

        self.initial_x = nn.Sequential(
            nn.Linear(input_dim_x, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.initial_e = nn.Sequential(
            nn.Linear(input_dim_e, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList(
            [YATE_Block(input_dim=hidden_dim, **block_args) for _ in range(num_layers)]
        )

        self.read_out_block = YATE_Block(
            input_dim=hidden_dim, read_out=True, **block_args
        )

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        # Initial layer for the node/edge features
        x = self.initial_x(x)
        edge_attr = self.initial_e(edge_attr)

        for l in self.layers:
            x, edge_attr = l(x, edge_index, edge_attr)

        x = self.read_out_block(x, edge_index, edge_attr)

        if return_attention:
            attention_maps = []
            for l in self.layers:
                _, _, attention = l.g_attn(x, edge_index, edge_attr, return_attention)
                attention_maps.append(attention)
            return x, attention_maps
        elif return_attention == False:
            return x
