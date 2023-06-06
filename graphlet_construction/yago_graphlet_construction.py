# Python
import numpy as np
import math
from typing import List, Union

# Pytorch
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import Batch

# Graphlet
from .gc_utils import (
    k_hop_subgraph,
    subgraph,
    feature_extract_lm,
    to_undirected,
)


# Graphlet class to construct a graphlet of a given entity
class Graphlet:
    def __init__(self, main_data, num_hops: int = 1, max_nodes: int = 100):
        super(Graphlet, self).__init__()

        self.main_data = main_data
        self.edge_index = main_data.edge_index
        self.edge_type = main_data.edge_type
        self.x_model = main_data.x_model
        self.ent2idx = main_data.ent2idx
        self.rel2idx = main_data.rel2idx

        self.num_hops = num_hops
        self.max_nodes = max_nodes

        self.perturb_methods = ["mask_edge", "mask_node", "neg_edge", "neg_node"]
        self.perturb_graphlet = dict(
            {
                "mask_edge": _perturb_mask_edge,
                "mask_node": _perturb_mask_node,
                "neg_edge": _perturb_replace_edge,
                "neg_node": _perturb_replcae_node,
            }
        )

    def make_batch(
        self,
        cen_idx: Union[int, List[int], Tensor],
        aggregate: bool,
        n_perturb: int = 0,
        per_keep: float = 0.9,
    ):
        if isinstance(cen_idx, Tensor):
            cen_idx = cen_idx.tolist()
        if isinstance(cen_idx, int):
            cen_idx = [cen_idx]

        # Obtain the of entities and edge_types in the batch (reduced set)
        head_ = self.edge_index[0, :]
        tail_ = self.edge_index[1, :]

        node_mask = head_.new_empty(self.edge_index.max().item() + 1, dtype=torch.bool)
        node_mask.fill_(False)

        subset = cen_idx

        for _ in range(self.num_hops):
            node_mask[subset] = True
            reduce_mask = node_mask[head_]
            subset = tail_[reduce_mask].unique()

        self.edge_index_reduced = self.edge_index[:, reduce_mask]
        self.edge_type_reduced = self.edge_type[reduce_mask]

        # Obtain the list of data with original and perturbed graphs
        data_total = []
        self.neg_node_set = self.edge_index_reduced.unique()
        for g_idx in range(len(cen_idx)):
            # Obtain the original graph
            data_original_ = self._make_graphlet(node_idx=cen_idx[g_idx])
            data_original_.idx_perturb = torch.tensor([-1])

            # Get indices for edge_type prediction
            data_perturb_ = [
                _perturb_truncate_node(
                    data=data_original_, per_keep=per_keep, keep_name=True
                )
                for _ in range(n_perturb)
            ]

            # Obtain the perturbed graphs
            data_total = data_total + [data_original_] + data_perturb_

        if aggregate:
            # Form the batch with obtained graphlets
            makebatch = Batch()
            data_batch_temp = makebatch.from_data_list(
                data_total, follow_batch=["edge_index"]
            )
            data_batch = Data(
                x=data_batch_temp.x,
                edge_index=data_batch_temp.edge_index,
                edge_type=data_batch_temp.edge_type,
                edge_attr=data_batch_temp.edge_attr,
                g_idx=data_batch_temp.g_idx,
                head_idx=data_batch_temp.ptr[:-1],
            )

            return data_batch
        else:
            return data_total

    def _make_graphlet(
        self,
        node_idx: Union[int, List[int], Tensor],
    ):
        if isinstance(node_idx, Tensor):
            node_idx = int(node_idx)
        elif isinstance(node_idx, List):
            node_idx = node_idx[0]

        edge_index, edge_type, mapping = k_hop_subgraph(
            edge_index=self.edge_index_reduced,
            node_idx=node_idx,
            max_nodes=self.max_nodes,
            num_hops=self.num_hops,
            edge_type=self.edge_type_reduced,
        )

        x, edge_feat = feature_extract_lm(
            main_data=self.main_data,
            node_idx=mapping[0, 1:],
            edge_type=edge_type,
            control_center=True,
        )

        Z = torch.mul(edge_feat, x[edge_index[1]])
        x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

        edge_index, edge_type, edge_feat, _ = to_undirected(
            edge_index,
            edge_type,
            edge_feat,
        )

        data_out = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_feat,
            g_idx=node_idx,
            y=torch.tensor([1]),
            flag_perturb=torch.tensor([0]),
            mapping=torch.transpose(mapping, 0, 1),
        )

        return data_out


def _perturb_truncate_node(data, per_keep: float, keep_name: bool = True):
    data_perturb = data.clone()
    idx_map = data.edge_index[1, (data.edge_index[0] == 0)]
    idx_map, _ = torch.sort(idx_map)
    idx_keep = torch.zeros(1, dtype=torch.long)
    idx_keep_ = torch.ones(idx_map.size(0), dtype=bool)
    if keep_name:
        idx_keep = torch.hstack((idx_keep, idx_map[-1]))
        idx_keep_[-1] = False
    idx_keep_ = idx_map[idx_keep_.nonzero().view(-1)]
    if idx_keep_.size(0) > 1:
        num_keep = torch.randint(
            math.floor(per_keep * idx_keep_.size(0)), idx_keep_.size(0), (1,)
        ).item()
        # num_keep = math.floor(per_keep * idx_keep_.size(0)) + 1
        idx_keep_ = torch.tensor(np.random.choice(idx_keep_, num_keep, replace=False))
        idx_keep = torch.hstack((idx_keep, idx_keep_))
    edge_index, edge_mask, mask_ = subgraph(idx_keep, data_perturb.edge_index)
    data_perturb.edge_index = edge_index
    data_perturb.edge_type = data_perturb.edge_type[edge_mask]
    data_perturb.edge_attr = data_perturb.edge_attr[edge_mask, :]
    data_perturb.x = data_perturb.x[mask_[1, :]]
    Z = torch.mul(data_perturb.edge_attr, data_perturb.x[data_perturb.edge_index[1]])
    data_perturb.x[0, :] = Z[(data_perturb.edge_index[0] == 0), :].mean(dim=0)
    return data_perturb


def _perturb_mask_edge(data, idx_perturb, **kwargs):
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb
    data_perturb.edge_attr[idx_perturb, :] = torch.ones(
        idx_perturb.size(0), data_perturb.edge_attr.size(1)
    )
    Z = torch.mul(data_perturb.edge_attr, data_perturb.x[data_perturb.edge_index[1]])
    data_perturb.x[0, :] = Z[(data_perturb.edge_index[0] == 0), :].mean(dim=0)
    data_perturb.y = torch.tensor([1])
    data_perturb.flag_perturb = torch.tensor([1])
    return data_perturb


def _perturb_mask_node(data, idx_perturb, **kwargs):
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb
    tail_idx_ = data_perturb.edge_index[1, idx_perturb].unique()
    data_perturb.x[tail_idx_, :] = torch.ones(tail_idx_.size(0), data.x.size(1))
    mask_keep_ = torch.zeros(data_perturb.edge_index.size(1), dtype=bool)
    mask_keep_[(data_perturb.edge_index[0] == 0)] = True
    mask_keep_[idx_perturb] = False
    idx_keep = data_perturb.edge_index[1, mask_keep_]
    data_perturb.x[0, :] = data_perturb.x[idx_keep, :].mean(dim=0)
    data_perturb.y = torch.tensor([1])
    data_perturb.flag_perturb = torch.tensor([1])
    return data_perturb


def _perturb_replace_edge(main_data, data, idx_perturb, **kwargs):
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb
    neg_edge_ = np.setdiff1d(
        torch.arange(1, len(main_data.rel2idx)), data.edge_type[idx_perturb]
    )
    neg_edge_type = np.random.choice(neg_edge_, idx_perturb.size(0), replace=True)
    data_perturb.edge_attr[idx_perturb, :] = feature_extract_lm(
        main_data, edge_type=neg_edge_type
    )
    data_perturb.y = torch.tensor([0])
    data_perturb.flag_perturb = torch.tensor([1])
    return data_perturb


def _perturb_replcae_node(main_data, data, neg_node_set, idx_perturb):
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb
    neg_node_ = np.setdiff1d(neg_node_set, data.mapping[:, 0])
    neg_node_idx = np.random.choice(neg_node_, idx_perturb.size(0), replace=True)
    data_perturb.x[data_perturb.edge_index[1, idx_perturb], :] = feature_extract_lm(
        main_data,
        node_idx=neg_node_idx,
    )
    data_perturb.y = torch.tensor([0])
    data_perturb.flag_perturb = torch.tensor([1])
    return data_perturb
