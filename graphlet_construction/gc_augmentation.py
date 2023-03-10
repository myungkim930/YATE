"""
Graph augmentation for generating pos/neg graphlets.

"""

# Python
import numpy as np
import math

# Pytorch
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Graphlet
from .gc_utils import subgraph, feature_extract_lm

## Function to create positives
def gen_pos(n_pos, per_pos, data):

    g_pos = []
    check_dup = []
    nb_mark = 0
    n_neighbor = data.num_nodes - 1

    if math.comb(n_neighbor, int(n_neighbor * per_pos)) < n_pos:
        crit_pos = math.comb(n_neighbor, int(n_neighbor * per_pos))
    else:
        crit_pos = n_pos

    while nb_mark < crit_pos:

        idx_pos = list(
            np.random.choice(
                data.mapping[1:, 1],
                replace=False,
                size=int(n_neighbor * per_pos),
            )
        )

        if set(idx_pos) in check_dup:
            continue
        else:
            check_dup.append(set(idx_pos))

        idx_pos.insert(0, 0)

        if len(idx_pos) == 1:
            break

        edge_index, edge_mask, mask_p = subgraph(idx_pos, data.edge_index)

        x_pos = data.x[mask_p[0, :], :]
        edge_feat_pos = data.edge_attr[edge_mask]
        edge_type_pos = data.edge_type[edge_mask]

        mapping_pos = torch.vstack((data.mapping[mask_p[0, :], 0], mask_p[1, :]))

        pos_data = Data(
            x=x_pos,
            edge_index=edge_index,
            edge_attr=edge_feat_pos,
            edge_type=edge_type_pos,
            y=1,
            g_idx=data.mapping[0, 0],
            mapping=torch.transpose(mapping_pos, 0, 1),
        )

        g_pos.append(pos_data)
        nb_mark += 1

    return g_pos


## Function to create negatives
def gen_neg(n_neg, per_neg, data, main_data, neg_set=None):

    g_neg = []
    n_neighbor = data.num_nodes - 1
    nb_mark = 0

    if neg_set is None:
        neg_set = np.setdiff1d(
            range(main_data.edgelist_total.max().item() + 1), data.mapping[:, 0]
        )

    idx_neg = np.random.choice(
        neg_set, (n_neg, math.ceil(n_neighbor * per_neg)), replace=False
    )

    while nb_mark < n_neg:

        idx_replace = np.random.choice(
            range(1, n_neighbor + 1),
            size=math.ceil(n_neighbor * per_neg),
            replace=False,
        )

        mapping_neg = data.mapping.clone()
        mapping_neg[idx_replace, 0] = torch.tensor(idx_neg[nb_mark, :])

        x_neg_replace = feature_extract_lm(
            main_data=main_data, node_idx=idx_neg[nb_mark, :]
        )
        x_neg = data.x.clone()
        x_neg[idx_replace, :] = x_neg_replace

        neg_data = Data(
            x=x_neg,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_type=data.edge_type,
            y=0,
            g_idx=data.mapping[0, 0],
            mapping=mapping_neg,
        )

        g_neg.append(neg_data)
        nb_mark += 1

    return g_neg


## Graph augmentation class
class Augment:
    def __init__(
        self,
        max_nodes: int = 100,
        n_pos: int = 10,
        per_pos: float = 0.8,
        n_neg: int = 10,
        per_neg: float = 0.05,
    ):

        super(Augment, self).__init__()

        self.max_nodes = max_nodes
        self.n_pos = n_pos
        self.per_pos = per_pos
        self.n_neg = n_neg
        self.per_neg = per_neg

    def generate(self, data, main_data):

        if data.num_nodes > self.max_nodes:
            g_temp = gen_pos(1, 100 / data.num_nodes, data)
            data = g_temp[0]

        g_p = gen_pos(self.n_pos, self.per_pos, data)
        g_p.insert(0, data)

        g_n = []

        neg_set = np.setdiff1d(
            range(main_data.edgelist_total.max().item() + 1), data.mapping[:, 0]
        )

        for i in range(len(g_p)):

            g_n += gen_neg(self.n_neg, self.per_neg, g_p[i], main_data, neg_set=neg_set)

        g_p = g_p + g_n

        batch = next(iter(DataLoader(g_p, batch_size=len(g_p))))
        batch = Data(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_type=batch.edge_type,
            y=batch.y,
            g_idx=batch.g_idx,
            mapping=batch.mapping,
            head_idx=batch.ptr[: batch.y.size()[0]],
        )

        return batch
