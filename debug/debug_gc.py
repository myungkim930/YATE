#%% gc_utils

import os

os.chdir("/storage/store3/work/mkim/gitlab/YATE")
import torch
import graphlet_construction as gc

## 1. k_hop_sugraph
# The result should contain four things: edgelist, edgelist(remapped), edgetype, mapping
edge_index = torch.tensor(
    [[0, 0, 0, 0, 1233, 454, 5, 6123], [1, 2, 1233, 454, 5, 6123, 8, 7]]
)
edge_type = torch.tensor([2, 3, 3, 3, 3, 1, 4, 5])
node_idx, num_hops = 0, 2

result = gc.k_hop_subgraph(
    node_idx=node_idx, num_hops=num_hops, edge_index=edge_index, edge_type=edge_type
)

## 2. subgraph
edge_index = torch.tensor([[2, 2, 4, 4, 6, 6], [0, 1, 2, 3, 4, 5]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 1])
subset = [6, 4, 5]

result = gc.subgraph(subset=subset, edge_index=edge_index)

## 3. add_self_loops
edge_index = torch.tensor([[2, 2, 4, 4, 6, 6], [0, 1, 2, 3, 4, 5]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 1])

result1 = gc.add_self_loops(edge_index=edge_index, edge_type=edge_type)
result2 = gc.add_self_loops(
    edge_index=edge_index, edge_type=edge_type, exclude_center=True
)

## 4. remove_duplicates
edge_index = torch.tensor([[0, 1, 2, 2, 2, 2, 1], [1, 0, 3, 3, 3, 3, 2]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 2, 1])
edge_attr = torch.rand((edge_type.size(0), 3))

result = gc.remove_duplicates(
    edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
)

## 5. to_undirected
edge_index = torch.tensor([[0, 1, 2, 2, 2, 2, 1], [1, 0, 3, 3, 3, 3, 2]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 2, 1])
edge_attr = torch.rand((edge_type.size(0), 3))

edge_index, edge_type, edge_attr = gc.remove_duplicates(
    edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
)
edge_index, edge_type = gc.add_self_loops(edge_index=edge_index, edge_type=edge_type)
edge_attr = torch.rand((edge_type.size(0), 3))

edge_index1, edge_type1, edge_attr1 = gc.to_undirected(
    edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
)

## 6. to_directed
edge_index = torch.tensor([[0, 1, 2, 2, 2, 2, 1], [1, 0, 3, 3, 3, 3, 2]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 2, 1])
edge_attr = torch.rand((edge_type.size(0), 3))

edge_index, edge_type, edge_attr = gc.remove_duplicates(
    edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
)

edge_index, edge_type = gc.add_self_loops(edge_index=edge_index, edge_type=edge_type)
edge_attr = torch.rand((edge_type.size(0), 3))

edge_index1, edge_type1, edge_attr1 = gc.to_undirected(
    edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
)

result = gc.to_directed(edge_index, edge_type, edge_index1, edge_type1, edge_attr1)

#%% gc_makeg

import os

os.chdir("/storage/store3/work/mkim/gitlab/YATE")

# load data
from data_utils import Load_data

data_name = "yago3_2022"  # others - yago3 'yago3_2022'
main_data = Load_data(data_name)

# making graphlets
import torch
import graphlet_construction as gc

g = gc.Graphlet(main_data, num_hops=2, max_nodes=100)
idx_head = main_data.edge_index[0, :].unique()

# single graphlet
# idx = idx_head[torch.randperm(idx_head.size()[0])][0]
# idx = 2859 # United States
# data = g.make_batch(cen_idx=idx, n_perturb_mask=0, n_perturb_replace=0)

# multiple graphlets in a batch with perturbations
idx_cen = idx_head[torch.randperm(idx_head.size(0))][0:64]
data_batch = g.make_batch(
    cen_idx=idx_cen, per_perturb=0.8, n_perturb_mask=1, n_perturb_replace=2
)

from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
    with record_function("graphlet_construction"):
        data_batch = g.make_batch(
            cen_idx=idx_cen, n_perturb_mask=1, n_perturb_replace=2
        )
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

import cProfile
from pstats import SortKey

with cProfile.Profile() as pf:
    data_batch = g.make_batch(
        cen_idx=idx_cen, per_perturb=0.8, n_perturb_mask=1, n_perturb_replace=2
    )

pf.print_stats(sort=SortKey.CUMULATIVE)


# %%
