"""
Functions that can be utilized in graphlet construction

"""

# Python
import re
import numpy as np
from typing import List, Tuple, Union, Optional

# Pytorch
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import scatter
from torch_geometric.utils import index_to_mask
from torch_geometric.loader import DataLoader

## K-hop Subgraph Extraction
def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Adj,
    edge_type: Union[int, List[int], Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.
    """

    num_nodes = edge_index.max().item() + 1
    cen_node = node_idx

    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[: node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]

    edge_type_new = edge_type[edge_mask]

    # mapping = torch.vstack((edge_index.unique(), torch.argsort(edge_index.unique())))

    mapping = torch.reshape(torch.tensor((cen_node, 0)), (2, 1))
    mapping_temp = torch.vstack(
        (subset[subset != cen_node], torch.arange(1, subset.size()[0]))
    )
    mapping = torch.hstack((mapping, mapping_temp))

    edge_list_new = edge_index.clone()
    for i in range(mapping.size()[1]):
        edge_list_new[edge_index == mapping[0, i]] = mapping[1, i]

    # edge_list_new = edge_index.clone()
    # for i in range(subset.size()[0]):
    #     edge_list_new[edge_list_new == subset[i]] = mapping[
    #         1, mapping[0, :] == subset[i]
    #     ]

    return edge_index, edge_list_new, edge_type_new, mapping


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

    subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    mapping = torch.vstack((edge_index.unique(), torch.argsort(edge_index.unique())))

    edge_list_new = edge_index.clone()
    for i in range(mapping.size()[1]):
        edge_list_new[edge_index == mapping[0, i]] = mapping[1, i]

    return edge_list_new, edge_mask, mapping


## Word-Feature Extraction (Subject to change depending on the naming of entities/relations)
def feature_extract_lm(
    main_data,
    node_idx: Optional[Union[int, List[int], Tensor]] = None,
    edge_type: Optional[Union[int, List[int], Tensor]] = None,
):
    r"""Extracts node/edge features from language model."""

    if node_idx is not None:
        if isinstance(node_idx, int):
            node_idx = [node_idx]
        elif isinstance(node_idx, Tensor) and node_idx.size()[0] == 1:
            node_idx = [node_idx.tolist()]

        gent_names = main_data.ent2idx[node_idx]

        x = [
            main_data.x_model.get_sentence_vector(
                x.replace("_", " ").replace("<", "").replace(">", "").lower()
            )
            for x in gent_names
        ]

        x = np.array(x)
        x = torch.tensor(
            x,
        )

    if edge_type is not None:
        if isinstance(edge_type, int):
            edge_type = [edge_type]
        elif isinstance(edge_type, Tensor) and edge_type.size()[0] == 1:
            edge_type = edge_type.tolist()

        grel_names = main_data.rel2idx[edge_type]

        edge_feat = [
            main_data.x_model.get_sentence_vector(
                re.sub(r"\B([A-Z])", r" \1", x)
                .replace("_", " ")
                .replace("<", "")
                .replace(">", "")
                .lower()
            )
            for x in grel_names
        ]

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


## Add self-loop function
def add_self_loops(
    edge_index: Adj,
    edge_type: Adj,
    only_center: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index` or
    to the central node. Edgetype of self-loops will be added with '0'
    """

    if only_center == True:
        N = 1
    elif only_center == False:
        N = edge_index.max().item() + 1
    else:
        raise AttributeError("Valid input required")

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    edge_type = torch.cat(
        [edge_type, torch.zeros(N, dtype=torch.long, device=edge_index.device)], dim=0
    )

    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_type
