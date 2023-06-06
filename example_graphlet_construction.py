### This example runs through various cases of graphlet construction that YATE encounters

## Load packages
import torch
import pandas as pd

from sklearn.model_selection import train_test_split

import graphlet_construction as gc

# %%
## 1. Graphlet construction from the knowledge graph YAGO
# Load YAGO data
data_name = "yago3_2022"  # others - yago3 'yago3_2022'
yago_data = gc.Load_Yago(data_name, numerical=True)

# The KG contains: 'edge_index', 'edge_type', 'ent2idx', 'ent2idx_original', 'rel2idx', 'x_model', 'data_name'
print(yago_data.__dict__.keys())

# Construct a graphlet for a given entity
# In the pretraining, the graphlets of entities which are the heads are only used
g = gc.Graphlet(main_data=yago_data, num_hops=2, max_nodes=100)
idx_head = yago_data.edge_index[0].unique()

n_batch = 1  # Here I use the word n_batch for simplification
idx_cen = idx_head[torch.randperm(idx_head.size(0))][0:n_batch]

# cen_idx: index of entity of interest
# aggregate: to make into Data object. If set to false, it spits out a list
# per_keep: for pretraining, we make positives which truncates the graphlet.
# This controls the maximum number of leaf nodes to keep for truncation.
# n_perturb: number of positives. If set to 0, does not make any positives only the graphlet itself
# If we increase n_batch, it creates the graphlets for all the entities in the batch with their positives
data_batch = g.make_batch(
    cen_idx=idx_cen,
    aggregate=True,
    per_keep=0.9,
    n_perturb=0,
)

# %%
## 2. Table2Graph - extracting graphlets for each row in the table
# import data - movies
data_pd_dir = (
    "/storage/store3/work/mkim/gitlab/YATE/data/eval_kg_data/raw/movies.parquet"
)
data_pd = pd.read_parquet(data_pd_dir)
target_name = "target"

# Split into train/test -
num_data = len(data_pd)
data_train, data_test = train_test_split(
    data_pd,
    test_size=0.2,
    shuffle=True,
    random_state=1,
)

# Apply table2graph
# numerical_cardinality_threshold: change numerical columns to categorical with cardinality less than threshold
# numerical_transformer: transformer for numerical values. If set to None, it keeps the original numericals
# num_transformer_params: paramaeters for the numerical transformer. For now only for quantile_transformer
num_transformer_params = {"n_quantiles": 125}
table2graph_transformer = gc.Table2Graph(
    numerical_cardinality_threshold=1,
    numerical_transformer="quantile",
    num_transformer_params=num_transformer_params,
)

# Having the fit_transform and transform is for the numerical columns.
# Without numericals, the rows are independent of one another.
train_dataset = table2graph_transformer.fit_transform(
    data_train, target_name=target_name
)
test_dataset = table2graph_transformer.transform(data_test)

# The result should contain a list of Data objects that can be fed into the proposed GNN architecture
# For now, the graphlet does not contain names. This may be included later for other purposes
print(train_dataset[0])

# Given a list of graphlets, we may augment it by using the YAGO KG
# This requires a good matching of entity name in the dataset with entities in YAGO, so it is rather difficult
# For the movies dataset, it already has a good matching (hard work from Riccardo),
# but we can only extract the ones in the head index (by the design of the proposed method)

# GraphAugmentor takes a bit of time and may need to optimize for later usages
# graphlet_settings = dict()
# graphlet_settings["num_hops"] = 2
# graphlet_settings["max_nodes"] = 100
# GraphAugmentor = gc.GraphAugmentKG(graphlet_settings=graphlet_settings)
# data_augment_train = GraphAugmentor.augment_data(data_train, augment_col_name="name", data_graph=train_dataset)
# data_augment_test = GraphAugmentor.augment_data(data_test, augment_col_name="name", data_graph=test_dataset)

# %%
## 3. Other various utils (to be updated)
# (a) k_hop_sugraph: The result should contain four things: edgelist, edgelist(remapped), edgetype, mapping
edge_index = torch.tensor(
    [[0, 0, 0, 0, 1233, 454, 5, 6123], [1, 2, 1233, 454, 5, 6123, 8, 7]]
)
edge_type = torch.tensor([2, 3, 3, 3, 3, 1, 4, 5])
node_idx, num_hops = 0, 2

result = gc.k_hop_subgraph(
    node_idx=node_idx,
    num_hops=num_hops,
    edge_index=edge_index,
    edge_type=edge_type,
    max_nodes=100,
)

# (b) subgraph
edge_index = torch.tensor([[2, 2, 4, 4, 6, 6], [0, 1, 2, 3, 4, 5]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 1])
subset = [6, 4, 5]

result = gc.subgraph(subset=subset, edge_index=edge_index)

# (c) add_self_loops
edge_index = torch.tensor([[2, 2, 4, 4, 6, 6], [0, 1, 2, 3, 4, 5]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 1])
edge_feat = torch.rand(6, 2)

result1 = gc.add_self_loops(
    edge_index=edge_index, edge_feat=edge_feat, edge_type=edge_type
)
result2 = gc.add_self_loops(
    edge_index=edge_index, edge_type=edge_type, exclude_center=True
)

# (d) remove_duplicates
edge_index = torch.tensor([[0, 1, 2, 2, 2, 2, 1], [1, 0, 3, 3, 3, 3, 2]])
edge_type = torch.tensor([2, 3, 3, 3, 3, 2, 1])
edge_attr = torch.rand((edge_type.size(0), 3))

result = gc.remove_duplicates(
    edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
)

## (e) to_undirected
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

## (f) to_directed - not really used
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
