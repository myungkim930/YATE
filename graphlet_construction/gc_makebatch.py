# Python
from typing import List, Union

# Pytorch
from torch import Tensor
from torch_geometric.loader import DataLoader

# Pytorch
from .gc_makeg import Graphlet
from .gc_augmentation import Augment


## Function to make batches
def make_batch(idx_cen: Union[int, List[int], Tensor], num_hops: int, main_data):

    if isinstance(idx_cen, Tensor):
        idx_cen = idx_cen.tolist()
    elif isinstance(idx_cen, Tensor):
        idx_cen = [idx_cen]

    data = []
    g = Graphlet(main_data)
    aug = Augment(max_nodes=100, n_pos=1, per_pos=0.8, n_neg=1, per_neg=0.05)

    for g_idx in range(len(idx_cen)):

        data_temp = g.make_graphlet(cen_ent=idx_cen[g_idx], num_hops=num_hops)
        data_total_temp = aug.generate(data_temp, main_data=main_data)

        data.append(data_total_temp)

    d_batch = next(iter(DataLoader(data, batch_size=len(idx_cen))))

    return d_batch
