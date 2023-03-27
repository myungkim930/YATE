"""
YATE gnn model with defined layers.

"""

# Python
import math
from typing import Optional

# PyTorch
import torch
import torch.nn as nn

from torch import Tensor

# PyTorch Geometric
from torch_geometric.utils import softmax
from torch_scatter import scatter

#########################
## Necessary functions ##
#########################

## YATE - Attention calculation
def yate_att_calc(
    edge_index: Tensor, query: Tensor, key: Tensor, head_idx: Optional[Tensor] = None
) -> Tensor:
    num_emb = query.size(1)
    attention = torch.sum(torch.mul(query[edge_index[0], :], key), dim=1) / math.sqrt(
        num_emb
    )
    if head_idx is not None:
        head_, tail_ = edge_index[0], edge_index[1]
        node_mask = head_.new_empty(edge_index.max().item() + 1, dtype=torch.bool)
        node_mask.fill_(False)
        node_mask[head_idx] = True
        edge_mask_head = node_mask[head_]
        edge_mask_tail = node_mask[tail_]
        attention[edge_mask_head.nonzero()] = 1
        attention[edge_mask_tail.nonzero()] = -9e15
    attention = softmax(attention, edge_index[0])
    return attention


## YATE - output calculation with attention (message passing)
def yate_att_output(edge_index: Tensor, attention: Tensor, value: Tensor) -> Tensor:
    output = scatter(
        torch.mul(attention, value.t()).t(), edge_index[0, :], dim=0, reduce="sum"
    )
    return output


## YATE - output calculation with multi-head (message passing)
def yate_multihead(
    edge_index: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    head_idx: Optional[Tensor] = None,
    num_heads: int = 1,
    concat: bool = True,
):
    if concat:
        H, C = num_heads, query.size(1) // num_heads
        for i in range(H):
            A = yate_att_calc(
                edge_index,
                query[:, i * C : (i + 1) * C],
                key[:, i * C : (i + 1) * C],
                head_idx,
            )
            O = yate_att_output(edge_index, A, value[:, i * C : (i + 1) * C])
            if i == 0:
                output = O
                attention = A
            else:
                output = torch.cat((output, O), dim=1)
                attention = torch.cat((attention, A), dim=0)
    else:
        H, C = num_heads, query.size(1)
        for i in range(H):
            A = yate_att_calc(
                edge_index,
                query[:, i * C : (i + 1) * C],
                key[:, i * C : (i + 1) * C],
                head_idx,
            )
            if i == 0:
                attention = A
            else:
                attention = torch.cat((attention, A), dim=0)
        attention = attention / H
        output = yate_att_output(edge_index, attention, value)

    return output, attention


#################
## YATE Blocks ##
#################

## YATE - Attention Layer
class YATE_Attention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 1,
        concat: bool = True,
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
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        head_idx: Optional[Tensor] = None,
        return_attention: bool = False,
    ):

        Z = torch.mul(edge_attr, x[edge_index[1, :]])

        query = self.lin_query(x)
        key = self.lin_key(Z)
        value = self.lin_value(Z)
        edge_attr = self.lin_edge(Z)

        output, attention = yate_multihead(
            edge_index=edge_index,
            query=query,
            key=key,
            value=value,
            head_idx=head_idx,
            num_heads=self.num_heads,
            concat=self.concat,
        )

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
    ):

        super().__init__()

        # Graph Attention Layer
        self.g_attn = YATE_Attention(input_dim, input_dim, num_heads, concat)

        # Two-layer MLP + Layers to apply in between the main layers for x and edges
        self.linear_net_x = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, input_dim),
        )

        self.linear_net_e = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, input_dim),
        )

        self.norm1_e = nn.LayerNorm(input_dim)
        self.norm2_e = nn.LayerNorm(input_dim)

        self.norm1_x = nn.LayerNorm(input_dim)
        self.norm2_x = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        head_idx: Optional[Tensor] = None,
    ):

        # Attention part
        attn_out_x, attn_out_e = self.g_attn(x, edge_index, edge_attr, head_idx)

        # MLP part - Node
        x = x + attn_out_x
        x = self.norm1_x(x)

        linear_out_x = self.linear_net_x(x)
        x = x + linear_out_x
        x = self.norm2_x(x)

        # MLP part - Edge
        edge_attr = edge_attr + attn_out_e
        edge_attr = self.norm1_e(edge_attr)

        linear_out_e = self.linear_net_e(edge_attr)
        edge_attr = edge_attr + linear_out_e
        edge_attr = self.norm2_e(edge_attr)

        return x, edge_attr


## YATE - contrast block
class YATE_Constrast(nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, x: Tensor):

        norm = torch.norm(x, p=2, dim=1)
        x = x / norm.unsqueeze(1)
        sig = torch.median(torch.cdist(x, x)) * 2
        x = torch.exp(-(torch.cdist(x, x) / sig))

        return x


#########################
## YATE Encoding Block ##
#########################


## YATE - encoding block with several layers and the final classification layer
class YATE_Encode(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_class_dim: int,
        node_class_dim: int,
        num_layers: int,
        contrast: bool = False,
        **block_args
    ):

        super(YATE_Encode, self).__init__()

        self.initial_attention = YATE_Block(input_dim=hidden_dim, **block_args)

        self.layers = nn.ModuleList(
            [
                YATE_Block(input_dim=hidden_dim, **block_args)
                for _ in range(num_layers - 1)
            ]
        )

        self.classifier_edge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_class_dim),
        )

        if contrast:
            self.classifier_node = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(4 * hidden_dim, hidden_dim),
                YATE_Constrast(),
            )
        else:
            self.classifier_node = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, node_class_dim),
            )

    def forward(self, input, return_attention=False):

        # Define the appropriate variables
        x, edge_index, edge_attr, idx_perturb, head_idx = (
            input.x,
            input.edge_index,
            input.edge_attr,
            input.idx_perturb,
            input.head_idx,
        )

        x, edge_attr = self.initial_attention(x, edge_index, edge_attr, head_idx)

        for l in self.layers:
            x, edge_attr = l(x, edge_index, edge_attr)

        if return_attention:
            attention_maps = []
            for l in self.layers:
                _, _, attention = l.g_attn(x, edge_index, edge_attr, return_attention)
                attention_maps.append(attention)

        # Extract representations of central entities and perturbed edges
        x = x[head_idx, :]
        x = self.classifier_node(x)

        if self.training:
            edge_attr = edge_attr[idx_perturb, :]

        edge_attr = self.classifier_edge(edge_attr)

        if return_attention:
            return x, edge_attr, attention_maps
        elif return_attention == False:
            return x, edge_attr

        # Initial layer for the node/edge features
        # x = self.initial_x(x)
        # edge_attr = self.initial_e(edge_attr)


# self.initial_x = nn.Sequential(
#     nn.Linear(input_dim_x, hidden_dim),
#     nn.Dropout(),
#     nn.ReLU(inplace=True),
#     nn.LayerNorm(hidden_dim),
# )

# self.initial_e = nn.Sequential(
#     nn.Linear(input_dim_e, hidden_dim),
#     nn.Dropout(),
#     nn.ReLU(inplace=True),
#     nn.LayerNorm(hidden_dim),
# )

# if contrast:
#     self.contrast_node = nn.Sequential(
#         nn.Linear(hidden_dim, 4 * hidden_dim),
#         nn.ReLU(inplace=True),
#         nn.Linear(4 * hidden_dim, hidden_dim),
#         YATE_Constrast(),
#     )
#     self.contrast = contrast
