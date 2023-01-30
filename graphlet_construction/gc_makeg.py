"""
Graphlet construction module

"""

# Python


# Pytorch
import torch
from torch_geometric.data import Data

# Graphlet
from .gc_utils import k_hop_subgraph, feature_extract_lm, add_self_loops

# Graphlet class to construct a graphlet of a given entity
class Graphlet:
    def __init__(self, main_data):

        super(Graphlet, self).__init__()

        self.main_data = main_data
        self.edgelist_total = main_data.edgelist_total
        self.edgetype_total = main_data.edgetype_total
        self.x_model = main_data.x_model
        self.ent2idx = main_data.ent2idx
        self.rel2idx = main_data.rel2idx

    def make_graphlet(
        self,
        cen_ent: int,
        num_hops: int,
    ):

        _, edgelist_new, edgetype_new, mapping = k_hop_subgraph(
            edge_index=self.edgelist_total,
            node_idx=cen_ent,
            num_hops=num_hops,
            edge_type=self.edgetype_total,
        )

        edgelist_new, edgetype_new = add_self_loops(
            edge_index=edgelist_new, edge_type=edgetype_new
        )

        x, edge_feat = feature_extract_lm(
            main_data=self.main_data, node_idx=mapping[0, :], edge_type=edgetype_new
        )

        data = Data(
            x=x,
            edge_index=edgelist_new,
            edge_type=edgetype_new,
            edge_attr=edge_feat,
            y=1,
            g_idx=cen_ent,
            mapping=torch.transpose(mapping, 0, 1),
        )

        return data
