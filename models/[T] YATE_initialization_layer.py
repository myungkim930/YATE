"""
YATE initialization layer that brings together:
generation of positive and negative samples / fastext for entities and relations / numerical values

"""


# PyTorch
import torch
import torch.nn as nn

##########

class YATE_Initial(nn.Module):
    def __init__(
        self,
        
):

        super(YATE_Initial, self).__init__()
        
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

        
        
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_feat: Adj,
    ):
        