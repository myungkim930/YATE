"""
Function to make a batch of graph data objects that can be fed into YATE

"""

# Python
from typing import List, Union

# Pytorch
from torch import Tensor
from torch_geometric.loader import DataLoader

# Pytorch
from .gc_makeg import Graphlet, Augment


## Function to make batches
def make_batch(
    idx_cen: Union[int, List[int], Tensor],
    num_hops: int,
    main_data,
    flow: str = "target_to_source",
    max_nodes: int = 100,
    n_pos: int = 10,
    per_pos: float = 0.8,
    n_neg: int = 10,
    per_neg: float = 0.05,
):

    if isinstance(idx_cen, Tensor):
        idx_cen = idx_cen.tolist()
    elif isinstance(idx_cen, Tensor):
        idx_cen = [idx_cen]

    data = []
    start_idx = 0
    g = Graphlet(main_data)
    aug = Augment(
        max_nodes=max_nodes, n_pos=n_pos, per_pos=per_pos, n_neg=n_neg, per_neg=per_neg
    )

    for g_idx in range(len(idx_cen)):

        data_temp = g.make_graphlet(
            cen_ent=idx_cen[g_idx], num_hops=num_hops, flow=flow
        )
        data_total_temp = aug.generate(data_temp, main_data=main_data)

        data_total_temp.head_idx = data_total_temp.head_idx + start_idx
        start_idx += data_total_temp.num_nodes

        data.append(data_total_temp)

    d_batch = next(iter(DataLoader(data, batch_size=len(idx_cen))))

    return d_batch
