#%% Setup

# Set working directory
import os

os.chdir("/storage/store3/work/mkim/gitlab/YATE")

# load data
from data_utils import Load_data

data_name = "yago3"
main_data = Load_data(data_name)
# main_data.reduce(num_rel=5)

#%%

import torch
import matplotlib.pyplot as plt

from model.yate_gnn import YATE_Encode as YE

# load model
model = YE(
    input_dim_x=300,
    input_dim_e=300,
    hidden_dim=300,
    edge_class_dim=len(main_data.rel2idx),
    num_layers=12,
    ff_dim=300,
    num_heads=1,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
from train import CosineAnnealingWarmUpRestarts

scheduler = CosineAnnealingWarmUpRestarts(
    optimizer, T_0=2500, T_mult=1, eta_max=1e-3, T_up=100, gamma=0.9
)

lrs = []

for i in range(10000):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

# plot rate
plt.plot(range(10000), lrs)

# %%
