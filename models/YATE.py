"""
The total YATE GNN model

"""

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

#########

from .YATE_T_layer import YATE_Block

##
class YATE_GNN(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, num_layers, **block_args):

        super(YATE_GNN, self).__init__()

        self.linear_net_initial_e = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim),
        )

        self.T_layers = nn.ModuleList(
            [YATE_Block(input_dim, emb_dim, **block_args) for _ in range(num_layers)]
        )

        self.lin = nn.Linear(input_dim, output_dim)
        self.norm1 = nn.LayerNorm(input_dim)

    def forward(self, x, edge_index, edge_feat):

        self.linear_net_initial_e(edge_feat)
        edge_feat = self.norm1(edge_feat)

        for l in self.T_layers:
            x, edge_feat = l(x, edge_index, edge_feat)

        # 3. Apply a final MLP
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
