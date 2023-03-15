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
    subgraph,
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
        per_perturb: float = 0.3,
        n_perturb_mask: int = 1,
        n_perturb_replace: int = 1,
    ):

        if isinstance(cen_idx, Tensor):
            cen_idx = cen_idx.tolist()
        if isinstance(cen_idx, int):
            cen_idx = [cen_idx]

        # Obtain the of entities and edge_types in the batch
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
            data_original_ = self._make_graphlet(node_idx=cen_idx[g_idx])
            data_original_.idx_perturb = torch.tensor([-1])
            method_ = random.choices(
                self.perturb_methods[:2], k=n_perturb_mask
            ) + random.choices(self.perturb_methods[2:], k=n_perturb_replace)
            idx_perturb_ = (data_original_.edge_index[0, :] == 0).nonzero().view(-1)
            n_perturb = math.ceil(per_perturb * idx_perturb_.size(0))
            data_perturb_ = [
                self.perturb_graphlet[method_[i]](
                    main_data=self.main_data,
                    data=data_original_,
                    neg_node_set=self.neg_node_set,
                    idx_perturb=idx_perturb_,
                    n_perturb=n_perturb,
                )
                for i in range(n_perturb_mask + n_perturb_replace)
            ]
            data_perturb_.insert(0, data_original_)
            data_total = data_total + data_perturb_

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

        edge_index, edge_type, edge_attr, idx_perturb = to_undirected(
            data_batch_temp.edge_index,
            data_batch_temp.edge_type,
            data_batch_temp.edge_attr,
            data_batch_temp.idx_perturb[data_batch_temp.idx_perturb > -1],
        )

        data_batch = Data(
            x=data_batch_temp.x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            g_idx=data_batch_temp.g_idx,
            y=data_batch_temp.y,
            idx_perturb=idx_perturb,
            head_idx=data_batch_temp.ptr[:-1],
        )

        return data_batch

    def _make_graphlet(
        self,
        node_idx: Union[int, List[int], Tensor],
        exclude_center: bool = True,
    ):

        if isinstance(node_idx, Tensor):
            node_idx = int(node_idx)
        elif isinstance(node_idx, List):
            node_idx = node_idx[0]

        edge_index, edge_type, mapping = k_hop_subgraph(
            edge_index=self.edge_index_reduced,
            node_idx=node_idx,
            num_hops=self.num_hops,
            edge_type=self.edge_type_reduced,
        )

        if (self.max_nodes is not None) & (mapping.size(1) > self.max_nodes):
            idx_keep = (mapping[1, :] > 0) & (mapping[1, :] < edge_index.max().item())
            idx_keep[
                idx_keep.nonzero().squeeze()[
                    torch.randperm(idx_keep.nonzero().squeeze().size(0))[
                        0 : self.max_nodes - 2
                    ]
                ]
            ] = False
            idx_keep = ~idx_keep
            idx_keep = idx_keep.nonzero().squeeze()
            edge_index, edge_mask, mask_ = subgraph(idx_keep, edge_index)
            edge_type = edge_type[edge_mask]
            mapping = torch.vstack((mapping[0, mask_[0, :]], mask_[1, :]))

        edge_index, edge_type = add_self_loops(
            edge_index=edge_index,
            edge_type=edge_type,
            exclude_center=exclude_center,
        )

        x, edge_feat = feature_extract_lm(
            main_data=self.main_data,
            node_idx=mapping[0, :],
            edge_type=edge_type,
            exclude_center=exclude_center,
        )

        data_out = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_feat,
            g_idx=node_idx,
            y=torch.tensor([1]),
            mapping=torch.transpose(mapping, 0, 1),
        )

        return data_out


def _perturb_mask_edge(data, idx_perturb, n_perturb, **kwargs):
    idx_perturb_ = idx_perturb[torch.randperm(idx_perturb.size(0))[0:n_perturb]]
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb_
    data_perturb.edge_attr[idx_perturb_, :] = torch.ones(
        idx_perturb_.size(0), data_perturb.edge_attr.size(1)
    )
    data_perturb.y = torch.tensor([1])
    return data_perturb


def _perturb_mask_node(data, idx_perturb, n_perturb, **kwargs):
    idx_perturb_ = idx_perturb[torch.randperm(idx_perturb.size(0))[0:n_perturb]]
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb_
    tail_idx_ = data_perturb.edge_index[1, idx_perturb_].unique()
    data_perturb.x[tail_idx_, :] = -9e15 * torch.ones(tail_idx_.size(0), data.x.size(1))
    data_perturb.y = torch.tensor([1])
    return data_perturb


def _perturb_replace_edge(main_data, data, idx_perturb, n_perturb, **kwargs):
    idx_perturb_ = idx_perturb[
        torch.randperm(idx_perturb.size(0))[0 : (idx_perturb.size(0) - n_perturb + 1)]
    ]
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb_
    neg_edge_ = np.setdiff1d(
        torch.arange(len(main_data.rel2idx)), data.edge_type[idx_perturb_]
    )
    neg_edge_type = np.random.choice(neg_edge_, idx_perturb_.size(0), replace=True)
    data_perturb.edge_attr[idx_perturb_, :] = feature_extract_lm(
        main_data, edge_type=neg_edge_type
    )
    data_perturb.y = torch.tensor([0])
    return data_perturb


def _perturb_replcae_node(main_data, data, neg_node_set, idx_perturb, n_perturb):
    idx_perturb_ = idx_perturb[
        torch.randperm(idx_perturb.size(0))[0 : (idx_perturb.size(0) - n_perturb + 1)]
    ]
    data_perturb = data.clone()
    data_perturb.idx_perturb = idx_perturb_
    neg_node_ = np.setdiff1d(neg_node_set, data.mapping[:, 0])
    neg_node_idx = np.random.choice(neg_node_, idx_perturb_.size(0), replace=True)
    data_perturb.x[data_perturb.edge_index[1, idx_perturb_], :] = feature_extract_lm(
        main_data, node_idx=neg_node_idx, exclude_center=False
    )
    data_perturb.y = torch.tensor([0])
    return data_perturb


# def _idx_perturb_mask(idx_perturb_, n_candidate, n_perturb):
#     return idx_perturb_[torch.randperm(n_candidate)[0:n_perturb]]


# def _idx_perturb_neg(idx_perturb_, n_candidate, n_perturb):
#     return idx_perturb_[torch.randperm(n_candidate)[0 : (n_candidate - n_perturb)]]


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
