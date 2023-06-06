"""
YATE gnn model with defined layers.

"""

# PyTorch
import torch.nn as nn

from .yate_block import YATE_Contrast, YATE_Base


##################################################
class YATE_Pretrain(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        num_layers: int,
        **block_args
    ):
        super(YATE_Pretrain, self).__init__()

        self.ft_base = YATE_Base(
            input_dim_x=input_dim_x,
            input_dim_e=input_dim_e,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **block_args
        )

        self.pretrain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            YATE_Contrast(),
        )

    def forward(self, input):
        x, edge_index, edge_attr, head_idx = (
            input.x,
            input.edge_index,
            input.edge_attr,
            input.head_idx,
        )

        x = self.ft_base(x, edge_index, edge_attr)
        x = x[head_idx, :]
        x = self.pretrain_classifier(x)

        return x


# ## YATE - encoding block with several layers and the final classification layer
# class YATE_Pretrain(nn.Module):
#     def __init__(
#         self,
#         input_dim_x: int,
#         input_dim_e: int,
#         hidden_dim: int,
#         num_layers: int,
#         **block_args
#     ):
#         super(YATE_Pretrain, self).__init__()

#         self.initial_x = nn.Sequential(
#             nn.Linear(input_dim_x, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#         )

#         self.initial_e = nn.Sequential(
#             nn.Linear(input_dim_e, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#         )

#         self.layers = nn.ModuleList(
#             [YATE_Block(input_dim=hidden_dim, **block_args) for _ in range(num_layers)]
#         )

#         self.read_out_block = YATE_Block(
#             input_dim=hidden_dim, read_out=True, **block_args
#         )

#         self.classifier_node = nn.Sequential(
#             nn.Linear(hidden_dim, 4 * hidden_dim),
#             nn.GELU(),
#             nn.Linear(4 * hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim, elementwise_affine=False),
#             YATE_Contrast(),
#         )

#     def forward(self, input, return_attention=False):
#         # Define the appropriate variables
#         x, edge_index, edge_attr, head_idx = (
#             input.x,
#             input.edge_index,
#             input.edge_attr,
#             input.head_idx,
#         )

#         # Initial layer for the node/edge features
#         x = self.initial_x(x)
#         edge_attr = self.initial_e(edge_attr)

#         for l in self.layers:
#             x, edge_attr = l(x, edge_index, edge_attr)

#         x = self.read_out_block(x, edge_index, edge_attr)

#         if return_attention:
#             attention_maps = []
#             for l in self.layers:
#                 _, _, attention = l.g_attn(x, edge_index, edge_attr, return_attention)
#                 attention_maps.append(attention)

#         # Extract representations of central entities and perturbed edges
#         x = x[head_idx, :]
#         x = self.classifier_node(x)

#         if return_attention:
#             return x, attention_maps
#         elif return_attention == False:
#             return x
