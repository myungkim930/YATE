# Python
import numpy as np
import math
import random
from typing import List, Union

# Pytorch
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import Batch

# Graphlet
from .gc_utils import (
    k_hop_subgraph,
    feature_extract_lm,
    add_self_loops,
    to_undirected,
)

# Graphlet class to construct a graphlet of a given entity
class Graphlet:
    def __init__(self, main_data, num_hops: int = 1, max_nodes=None):

        super(Graphlet, self).__init__()

        self.main_data = main_data
        self.edge_index = main_data.edge_index
        self.edge_type = main_data.edge_type
        self.x_model = main_data.x_model
        self.ent2idx = main_data.ent2idx
        self.rel2idx = main_data.rel2idx

        self.num_hops = num_hops
        self.max_nodes = max_nodes

        self.perturb_methods = ["mask_edge", "neg_edge", "neg_node"]
        self.perturb_graphlet = dict(
            {
                "mask_edge": _perturb_mask_edge,
                "neg_edge": _perturb_replace_edge,
                "neg_node": _perturb_replcae_node,
            }
        )

    def make_batch(
        self,
        cen_idx: Union[int, List[int], Tensor],
        n_perturb_mask: int = 0,
        n_perturb_replace: int = 0,
        per_perturb: float = 1,
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
            per_perturb_ = torch.linspace(0.05, per_perturb, int(per_perturb / 0.05))[
                np.random.randint(int(per_perturb / 0.05))
            ].item()
            per_perturb_ = round(per_perturb_, 2)
            idx_perturb_ = (data_original_.edge_index[0] == 0) & (
                data_original_.edge_index[0] + data_original_.edge_index[1] != 0
            )
            idx_perturb_ = idx_perturb_.nonzero().view(-1)
            n_perturb = math.ceil(per_perturb_ * idx_perturb_.size(0))
            idx_perturb_ = idx_perturb_[
                torch.randperm(idx_perturb_.size(0))[0:n_perturb]
            ]

            # Methods for perturbation
            method_ = [self.perturb_methods[0]] * n_perturb_mask + random.choices(
                self.perturb_methods[1:], k=n_perturb_replace
            )

            # Obtain the perturbed graphs
            data_perturb_ = [
                self.perturb_graphlet[method_[i]](
                    main_data=self.main_data,
                    data=data_original_,
                    neg_node_set=self.neg_node_set,
                    idx_perturb=idx_perturb_,
                )
                for i in range(n_perturb_mask + n_perturb_replace)
            ]
            # Obtain the perturbed graphs
            data_total = data_total + [data_original_] + data_perturb_

        # Form the batch with obtained graphlets
        makebatch = Batch()
        data_batch_temp = makebatch.from_data_list(
            data_total, follow_batch=["edge_index", "idx_perturb"]
        )
        for i in range(data_batch_temp.idx_perturb.size(0)):
            if data_batch_temp.idx_perturb[i] < 0:
                continue
            else:
                data_batch_temp.idx_perturb[i] = (
                    data_batch_temp.idx_perturb[i]
                    + data_batch_temp.edge_index_ptr[
                        data_batch_temp.idx_perturb_batch[i]
                    ]
                )

        if (data_batch_temp.idx_perturb > -1).nonzero().size(0) > 0:
            idx_perturb_ = data_batch_temp.idx_perturb[data_batch_temp.idx_perturb > -1]
        else:
            idx_perturb_ = None

        edge_index, edge_type, edge_attr, idx_perturb = to_undirected(
            data_batch_temp.edge_index,
            data_batch_temp.edge_type,
            data_batch_temp.edge_attr,
            idx_perturb=idx_perturb_,
        )

        data_batch = Data(
            x=data_batch_temp.x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            g_idx=data_batch_temp.g_idx,
            y=data_batch_temp.y,
            flag_perturb=data_batch_temp.flag_perturb,
            idx_perturb=idx_perturb,
            head_idx=data_batch_temp.ptr[:-1],
        )

        return data_batch

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

        edge_index, edge_feat, edge_type = add_self_loops(
            edge_index=edge_index,
            edge_feat=edge_feat,
            edge_type=edge_type,
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


def _perturb_mask_edge(data, idx_perturb, **kwargs):
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb
    data_perturb.edge_attr[idx_perturb, :] = torch.ones(
        idx_perturb.size(0), data_perturb.edge_attr.size(1)
    )
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


# def _perturb_mask_node(data, idx_perturb, n_perturb, **kwargs):
#     idx_perturb_ = idx_perturb[torch.randperm(idx_perturb.size(0))[0:n_perturb]]
#     data_perturb = data.clone()
#     data_perturb.idx_perturb = idx_perturb_
#     tail_idx_ = data_perturb.edge_index[1, idx_perturb_].unique()
#     data_perturb.x[tail_idx_, :] = -9e15 * torch.ones(tail_idx_.size(0), data.x.size(1))
#     data_perturb.y = torch.tensor([1])
#     data_perturb.flag_perturb = torch.tensor([1])
#     return data_perturb


# def _perturb_truncate_node(data, per_pos: float, keep_name: bool = False):
#     per_pos_ = torch.linspace(0.05, per_pos, int(per_pos / 0.05))[
#         np.random.randint(int(per_pos / 0.05))
#     ].item()
#     per_pos_ = round(per_pos_, 2)
#     data_pos = data.clone()
#     idx_map = data.edge_index[1, (data.edge_index[0] == 0)]
#     idx_map, _ = torch.sort(idx_map)
#     # idx_map = data_pos.mapping[:, 1].view(-1)
#     idx_keep_ = torch.ones(idx_map.size(0), dtype=bool)
#     if keep_name:
#         idx_keep = idx_map[[0, -1]]
#         idx_keep_[[0, -1]] = False
#     else:
#         idx_keep = idx_map[0]
#         idx_keep_[[0]] = False
#     idx_keep_ = idx_map[idx_keep_.nonzero().view(-1)]
#     num_keep = math.floor(per_pos_ * idx_keep_.size(0)) + 1
#     idx_keep_ = torch.tensor(np.random.choice(idx_keep_, num_keep, replace=False))
#     idx_keep = torch.hstack((idx_keep, idx_keep_))
#     edge_index, edge_mask, mask_ = subgraph(idx_keep, data_pos.edge_index)
#     data_pos.edge_index = edge_index
#     data_pos.edge_type = data_pos.edge_type[edge_mask]
#     data_pos.x = data_pos.x[mask_[1, :]]
#     return data_pos


# def _idx_perturb_mask(idx_perturb_, n_candidate, n_perturb):
#     return idx_perturb_[torch.randperm(n_candidate)[0:n_perturb]]


# def _idx_perturb_neg(idx_perturb_, n_candidate, n_perturb):
#     return idx_perturb_[torch.randperm(n_candidate)[0 : (n_candidate - n_perturb)]]


# data_pos_ = [
#     _perturb_truncate_node(data_original_, per_pos, keep_name=True)
#     for _ in range(n_pos)
# ]
#######

# self.neg_edge_set = self.edge_type_reduced.unique()

# def _perturb_graphlet(
#     self,
#     data,
#     method: str,
#     per_perturb: float,
# ):

#     # Obtain indexes for perturbation
#     idx_perturb_ = (data.edge_index[0, :] == 0).nonzero().view(-1)
#     n_candidate = idx_perturb_.size(0)
#     n_perturb = math.ceil(per_perturb * n_candidate)
#     idx_perturb_ = idx_perturb_[torch.randperm(n_candidate)[0:n_perturb]]
#     data_perturb = data.clone()
#     data_perturb.idx_perturb = idx_perturb_.clone()

#     # Obtain Data class for the perturbed graphlet
#     if method == "mask_edge":
#         data_perturb.edge_attr[idx_perturb_, :] = torch.ones(
#             n_perturb, data_perturb.edge_attr.size(1)
#         )
#         data_perturb.y = torch.tensor([1])
#     elif method == "mask_node":
#         tail_idx_ = data_perturb.edge_index[1, idx_perturb_].unique()
#         data_perturb.x[tail_idx_, :] = -9e15 * torch.ones(
#             tail_idx_.size(0), data_perturb.x.size(1)
#         )
#         data_perturb.y = torch.tensor([1])
#     elif method == "neg_edge":
#         neg_edge_ = np.setdiff1d(
#             torch.arange(len(self.rel2idx)), data_perturb.edge_type[idx_perturb_]
#         )
#         neg_edge_type = np.random.choice(neg_edge_, n_perturb, replace=True)
#         data_perturb.edge_attr[idx_perturb_, :] = feature_extract_lm(
#             self.main_data, edge_type=neg_edge_type
#         )
#         data_perturb.y = torch.tensor([0])
#     elif method == "neg_node":
#         neg_node_ = np.setdiff1d(self.neg_node_set, data_perturb.mapping[:, 0])
#         neg_node_idx = np.random.choice(neg_node_, n_perturb, replace=True)
#         data_perturb.x[
#             data_perturb.edge_index[1, idx_perturb_], :
#         ] = feature_extract_lm(
#             self.main_data, node_idx=neg_node_idx, exclude_center=False
#         )
#         data_perturb.y = torch.tensor([0])
#     else:
#         print("error")

#     return data_perturb
# data_perturb_mask = [
#     self._perturb_graphlet(
#         data=data_original_,
#         method=perturb_methods[np.random.randint(0, 2)],
#         per_perturb=per_perturb,
#     )
#     for _ in range(n_perturb_mask)
# ]
# data_perturb_replace = [
#     self._perturb_graphlet(
#         data=data_original_,
#         method=perturb_methods[np.random.randint(2, 4)],
#         per_perturb=per_perturb,
#     )
#     for _ in range(n_perturb_replace)
# ]
# data_ = data_perturb_mask + data_perturb_replace
# data_.insert(0, data_original_)

# idx_keep = np.random.choice(
#     mapping[1, 1 : edge_index.max().item()],
#     self.max_nodes - 2,
#     replace=False,
# )
# idx_keep = torch.tensor(idx_keep)
# idx_keep = torch.hstack((idx_keep, mapping[1, [0, -1]]))
# idx_keep, _ = torch.sort(idx_keep)
