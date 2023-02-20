import os
import torch
import graphlet_construction as gc
from models import YATE_Encode
from data_utils import Load_data

############
# Set working directory
os.chdir("/storage/store3/work/mkim/gitlab/YATE")

# load data
data_name = "yago3"  # others - 'yago3_10rel', 'yago3_cat', 'CLEAR_Corpus'
main_data = Load_data(data_name)
main_data.reduce(num_rel=10)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# load the base model
model = YATE_Encode(
    input_dim_x=300,
    input_dim_e=300,
    hidden_dim=300,
    num_layers=3,
    ff_dim=300,
    num_heads=1,
)

# replace with trained weights
dir = "/storage/store3/work/mkim/gitlab/YATE/data/saved_model/2023-02-19 23:27:25"
dir = dir + "/checkpoint_ep0.pt"
model.load_state_dict(torch.load(dir), strict=False)
model.eval()

# check if it is trained on Pos/Neg classification
g = gc.Graphlet(main_data, num_hops=1)
idx_head = main_data.edge_index[0, :].unique()
idx = idx_head[torch.randperm(idx_head.size()[0])][0:5]

n_pos = 1
n_neg = 1
data_batch = g.make_batch(idx_cen=idx, n_pos=n_pos, n_neg=n_neg, per_neg=0.8)

data_batch.to(device)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
output = model(data_batch)
target = data_batch.y.type(torch.int64)
loss = criterion(output, target)

# obataining representation - replace last layer with identity
# might need to tune this by concat several of the last layers
model.classifier = torch.nn.Identity()
output1 = model(data_batch)

idx_head = main_data.edge_index[0, :].unique()
g = gc.Graphlet(main_data, num_hops=1)

x = torch.zeros((idx_head.size(0), 300), device=device)

for i in range(idx_head.size(0)):
    data = g.make_batch(int(idx_head[i]), aug=False)
    data.to(device)
    with torch.no_grad():
        x[i,:] = model(data)


Dataloader()
torch.cuda.empty_cache()

y = main_data.headidx2type[1, :]


############


import torch

import graphlet_construction as gc
from data_utils import Load_data, load_pretrained_model

# Load data/model
model = load_pretrained_model(model_name="toy")
main_data = Load_data(data_name="clear_corpus")

g = gc.Graphlet(main_data=main_data, num_hops=1)
idx_cen = main_data.edge_index[0, :].unique()
data_batch = g.make_batch(idx_cen=idx_cen, aug=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_batch = data_batch.to(device)

output = model(data_batch)
output = output.to("cpu")


import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

from lazypredict.Supervised import LazyRegressor

X = output.detach().numpy()
X = X.astype(np.float32)
y = main_data.target

X, y = shuffle(X, y)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = RandomForestRegressor(max_depth=2, random_state=0)
reg.fit(X_train, y_train)


reg.score(X_test, y_test)


reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, _ = reg.fit(X_train, X_test, y_train, y_test)


# def main():


# if __name__ == "__main__":

# Evaluator class

# class Evaluator:
#     def __init__(self, data, model: torch.nn.Module,) -> None:
#         self.data = data
#         self.model = model


#     def evaluate():
