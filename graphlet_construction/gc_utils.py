"""
Functions that can be utilized in graphlet construction

"""

# Python
import math
import numpy as np
from typing import List, Tuple, Union, Optional

# Pytorch
import torch
from torch import Tensor


## K-hop Subgraph Extraction
def k_hop_subgraph(
    node_idx: int,
    num_hops: int,
    max_nodes: int,
    edge_index: Tensor,
    edge_type: Union[int, List[int], Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.
    """

    num_nodes = edge_index.max().item() + 1
    head, tail = edge_index

    node_mask = head.new_empty(num_nodes, dtype=torch.bool)
    reduce_mask_ = head.new_empty(edge_index.size(1), dtype=torch.bool)
    reduce_mask_.fill_(False)
    subset = int(node_idx)
    limit = [int(math.ceil(max_nodes * 10 ** (-i))) for i in range(num_hops)]

    for i in range(num_hops):
        node_mask.fill_(False)
        node_mask[subset] = True
        idx_rm = node_mask[head].nonzero().view(-1)
        idx_rm = idx_rm[torch.randperm(idx_rm.size(0))[: limit[i]]]
        reduce_mask_[idx_rm] = True
        subset = tail[reduce_mask_].unique()

    edge_index = edge_index[:, reduce_mask_]
    edge_type = edge_type[reduce_mask_]

    subset = edge_index.unique()

    mapping = torch.reshape(torch.tensor((node_idx, 0)), (2, 1))
    mapping_temp = torch.vstack(
        (subset[subset != node_idx], torch.arange(1, subset.size()[0]))
    )
    mapping = torch.hstack((mapping, mapping_temp))

    head_ = edge_index[0, :]
    tail_ = edge_index[1, :]

    sort_idx = torch.argsort(mapping[0, :])
    idx_h = torch.searchsorted(mapping[0, :], head_, sorter=sort_idx)
    idx_t = torch.searchsorted(mapping[0, :], tail_, sorter=sort_idx)

    out_h = mapping[1, :][sort_idx][idx_h]
    out_t = mapping[1, :][sort_idx][idx_t]

    edge_index_new = torch.vstack((out_h, out_t))

    edge_index_new = torch.hstack(
        (
            edge_index_new,
            torch.tensor(
                (
                    [0, edge_index_new.max().item() + 1],
                    [edge_index_new.max().item() + 1, 0],
                )
            ),
        )
    )
    edge_type = torch.hstack((edge_type, torch.zeros(2, dtype=torch.long)))

    mapping = torch.hstack(
        (mapping, torch.tensor([[node_idx], [edge_index_new.max().item()]]))
    )

    return edge_index_new, edge_type, mapping


## Subgraph with assigned nodes
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Union[Tensor, List[int]],
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.
    """

    device = edge_index.device
    num_nodes = edge_index.max().item() + 1

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    subset_ = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    subset_[subset] = True
    subset = subset_

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    mapping = torch.vstack((edge_index.unique(), torch.argsort(edge_index.unique())))

    head_ = edge_index[0, :]
    tail_ = edge_index[1, :]

    sort_idx = torch.argsort(mapping[0, :])
    idx_h = torch.searchsorted(mapping[0, :], head_, sorter=sort_idx)
    idx_t = torch.searchsorted(mapping[0, :], tail_, sorter=sort_idx)

    out_h = mapping[1, :][sort_idx][idx_h]
    out_t = mapping[1, :][sort_idx][idx_t]

    edge_list_new = torch.vstack((out_h, out_t))

    return edge_list_new, edge_mask, mapping


## Add self-loop function
def add_self_loops(
    edge_index: Tensor,
    edge_feat: Tensor,
    edge_type=None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index` or
    to the central node. Edgetype of self-loops will be added with '0'
    """

    N = edge_index.max().item() + 1

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_feat = torch.vstack((edge_feat, torch.ones((N, edge_feat.size(1)))))

    if edge_type is not None:
        edge_type = torch.cat(
            [edge_type, torch.zeros(N, dtype=torch.long, device=edge_index.device)],
            dim=0,
        )
        return edge_index, edge_feat, edge_type
    else:
        return edge_index, edge_feat


## Remove duplicate function
def remove_duplicates(
    edge_index: Tensor,
    edge_type: Tensor = None,
    edge_attr: Tensor = None,
    perturb_tensor: Tensor = None,
):
    nnz = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[0]
    idx[1:].mul_(num_nodes).add_(edge_index[1])

    if edge_type is not None:
        idx[1:].add_((edge_type + 1) * (10 ** (len(str(num_nodes)) + 3)))

    idx[1:], perm = torch.sort(
        idx[1:],
    )

    mask = idx[1:] > idx[:-1]

    edge_index = edge_index[:, perm]
    edge_index = edge_index[:, mask]

    if edge_type is not None:
        edge_type, edge_attr = edge_type[perm], edge_attr[perm, :]
        edge_type, edge_attr = edge_type[mask], edge_attr[mask, :]
        if perturb_tensor is not None:
            perturb_tensor = perturb_tensor[perm]
            perturb_tensor = perturb_tensor[mask]
            return edge_index, edge_type, edge_attr, perturb_tensor
        else:
            return edge_index, edge_type, edge_attr
    else:
        return edge_index


## To undirected function
def to_undirected(
    edge_index: Tensor,
    edge_type: Tensor = None,
    edge_attr: Tensor = None,
    idx_perturb=None,
):
    row = torch.cat([edge_index[0, :], edge_index[1, :]])
    col = torch.cat([edge_index[1, :], edge_index[0, :]])

    edge_index = torch.stack([row, col], dim=0)

    if edge_type is not None:
        edge_type = torch.cat([edge_type, edge_type])
        edge_attr = torch.vstack((edge_attr, edge_attr))
        if idx_perturb is not None:
            perturb_tensor = torch.zeros(edge_type.size(0))
            perturb_tensor[idx_perturb] = -1
            perturb_tensor = torch.cat([perturb_tensor, perturb_tensor])
            edge_index, edge_type, edge_attr, perturb_tensor = remove_duplicates(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=edge_attr,
                perturb_tensor=perturb_tensor,
            )
            idx_perturb = (perturb_tensor < 0).nonzero().squeeze()
            return edge_index, edge_type, edge_attr, idx_perturb
        else:
            edge_index, edge_type, edge_attr = remove_duplicates(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=edge_attr,
            )
        idx_perturb = []
        return edge_index, edge_type, edge_attr, idx_perturb
    else:
        edge_index = remove_duplicates(edge_index=edge_index)
        return edge_index


## To the original(directed) with
def to_directed(
    edge_index: Tensor,
    edge_type: Tensor,
    edge_index_mod: Tensor,
    edge_type_mod: Tensor,
    edge_attr_mod: Tensor,
):
    num_nodes = edge_index.max().item() + 1
    idx = edge_index[0].clone()
    idx.mul_(num_nodes).add_(edge_index[1]).add_(
        edge_type * (10 ** (len(str(num_nodes)) + 1))
    )

    idx_mod = edge_index_mod[0].clone()
    idx_mod.mul_(num_nodes).add_(edge_index_mod[1]).add_(
        edge_type_mod * (10 ** (len(str(num_nodes)) + 1))
    )

    map = ~(idx_mod[:, None] != idx).all(dim=1)
    edge_index = edge_index_mod[:, map]
    edge_type = edge_type_mod[map]
    edge_attr = edge_attr_mod[map, :]

    return edge_index, edge_type, edge_attr


##############


## Word-Feature Extraction (Subject to change depending on the naming of entities/relations)
def feature_extract_lm(
    main_data,
    node_idx: Optional[Union[int, List[int], Tensor]] = None,
    edge_type: Optional[Union[int, List[int], Tensor]] = None,
    control_center: bool = False,
):
    r"""Extracts node/edge features from language model."""

    if node_idx is not None:
        if isinstance(node_idx, int):
            node_idx = [node_idx]
        elif isinstance(node_idx, Tensor):
            node_idx = node_idx.squeeze()
        if isinstance(node_idx, Tensor) and node_idx.dim() == 0:
            node_idx = [node_idx]

        gent_names = main_data.ent2idx[node_idx]

        x = [main_data.x_model.get_sentence_vector(x) for x in gent_names]

        x = np.array(x)
        x = torch.tensor(x)
        if control_center:
            x = torch.vstack((torch.ones(x.size(1)), x))

    if edge_type is not None:
        if isinstance(edge_type, int):
            edge_type = [edge_type]
        elif isinstance(edge_type, Tensor):
            edge_type = edge_type.squeeze()
        if isinstance(edge_type, Tensor) and edge_type.dim() == 0:
            edge_type = [edge_type]

        grel_names = main_data.rel2idx[edge_type]

        edge_feat = [main_data.x_model.get_sentence_vector(x) for x in grel_names]

        edge_feat = np.array(edge_feat)
        edge_feat = torch.tensor(
            edge_feat,
        )

    if (node_idx is not None) and (edge_type is None):
        return x
    elif (node_idx is None) and (edge_type is not None):
        return edge_feat
    elif (node_idx is not None) and (edge_type is not None):
        return x, edge_feat

        # re.sub(r"[a-z]{2,}/", "", x)
        # .replace("_", " ")
        # .replace("<", "")
        # .replace("\n", "")
        # .replace(">", "")
        # .lower()

        # re.sub(r"\B([A-Z])", r" \1", x)
        # .replace("_", " ")
        # .replace("\n", "")
        # .replace("<", "")
        # .replace(">", "")
        # .lower()
