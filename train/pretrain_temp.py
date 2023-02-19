import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import graphlet_construction as gc

from models import YATE_Encode
from data_utils import Load_data

#############

# set training objectives
def load_train_objs(data_name: str, num_rel=None):
    # load data
    main_data = Load_data(data_name=data_name)
    if num_rel is not None:
        main_data.reduce(num_rel=num_rel)

    # set graph_construction framework
    g = gc.Graphlet(main_data, num_hops=1)

    # set index for batch
    train_set = main_data.edge_index[0, :]

    # load your model
    model = YATE_Encode(
        input_dim=300,
        hidden_dim=100,
        num_layers=3,
        ff_dim=300,
        num_heads=1,
    )

    # experiment settings
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    return train_set, model, optimizer