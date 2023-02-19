### Trainer for toy examples

import os
import torch
import datetime

import numpy as np
import torch.nn.functional as F
import graphlet_construction as gc

from torch.utils.data import DataLoader
from models import YATE_Encode
from data_utils import Load_data


## Trainer class
class Trainer:
    def __init__(
        self,
        exp_setting: dict,
    ) -> None:
        self.__dict__ = exp_setting
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def _run_batch(self, data):
        self.optimizer.zero_grad()
        output = self.model(data)
        target = data.y.type(torch.int64)
        weight = torch.tensor(
            [1 - self.label_ratio, self.label_ratio], device=self.device
        )
        loss = F.cross_entropy(output, target, weight=weight)
        self.loss = loss.clone()
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f"[GPU{self.device.index}] Epoch {epoch} | Batchsize: {self.n_batch}")
        batch_number = 0
        for idx in self.train_data:
            data = self.graphlet.make_batch(idx, *self.graphlet_setting)
            data = data.to(self.device)
            self._run_batch(data)
            print(
                f"[GPU{self.device.index}] Epoch {epoch} | Iteration: {batch_number}/{len(self.train_data)} | Loss: {round(self.loss.item(), 4)}"
            )
            batch_number += 1

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = self.save_dir + f"/checkpoint_ep{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self):
        for epoch in range(self.n_epoch):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(
    data_name: str,
    num_rel,
    num_hops: int,
    num_pos: int,
    per_pos: float,
    num_neg: int,
    per_neg: float,
    max_nodes: int,
    n_batch: int = 32,
    n_epoch: int = 50,
    save_every: int = 1,
):

    # create dictionary that set experiment settings
    exp_setting = dict()
    exp_setting["n_batch"] = n_batch
    exp_setting["n_epoch"] = n_epoch
    exp_setting["save_every"] = save_every

    exp_setting["graphlet_setting"] = [num_pos, per_pos, num_neg, per_neg, max_nodes]

    # load data
    main_data = Load_data(data_name=data_name)
    if num_rel is not None:
        main_data.reduce(num_rel=num_rel)

    # set graph_construction framework
    graphlet = gc.Graphlet(main_data, num_hops=num_hops)
    exp_setting["graphlet"] = graphlet

    # set train for batch
    idx_epoch = idx_extractor(main_data, max_nodes=max_nodes)
    train_data = DataLoader(
        idx_epoch, batch_size=n_batch, pin_memory=True, shuffle=True
    )
    exp_setting["train_data"] = train_data

    # load your model
    model = YATE_Encode(
        input_dim_x=300,
        input_dim_e=300,
        hidden_dim=300,
        num_layers=3,
        ff_dim=300,
        num_heads=1,
    )
    exp_setting["model"] = model

    # experiment settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    label_ratio = (num_pos + 1) / ((num_pos + 1) + (num_pos + 1) * num_neg)

    exp_setting["optimizer"] = optimizer
    exp_setting["label_ratio"] = label_ratio

    now = datetime.datetime.now()
    now.strftime("%Y-%m-%d %H:%M:%S")
    save_dir = os.getcwd() + "/data/saved_model/" + now.strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp_setting["save_dir"] = save_dir

    return exp_setting


## Index sampler according to the coverage of edge_index
def idx_extractor(main_data, max_nodes: int):
    ent_list, _ = torch.sort(main_data.edge_index[0, :].unique())
    count_head = np.ceil(np.bincount(ent_list) / max_nodes)
    count_head = np.array(count_head, dtype=np.dtype("int"))
    idx_epoch = ent_list[0].repeat(count_head[ent_list[0]])
    for i in range(1, ent_list.size(0)):
        idx_epoch = torch.hstack(
            (idx_epoch, ent_list[i].repeat(count_head[ent_list[i]]))
        )
    return idx_epoch


##############
def main():
    os.chdir("/storage/store3/work/mkim/gitlab/YATE")
    exp_setting = load_train_objs(
        data_name="yago3",
        num_rel=10,
        num_hops=1,
        num_pos=5,
        per_pos=0.8,
        num_neg=1,
        per_neg=0.8,
        max_nodes=100,
        n_batch=32,
        n_epoch=50,
        save_every=1,
    )
    trainer = Trainer(exp_setting)
    trainer.train()


if __name__ == "__main__":
    main()

##################

# def train():
#     exp_setting = load_train_objs(data_name = 'TUDataset/MUTAG')

#     model.train()


# def load_train_objs(data_name:str):

#     exp_setting = dict()

#     os.chdir("/storage/store3/work/mkim/gitlab/YATE")
#     data_dir = os.getcwd() + '/data_toy/' + data_name + '/processed/data.pt'
#     data = torch.load(data_dir)

#     model = YATE_Encode(
#         input_dim=300,
#         hidden_dim=100,
#         num_layers=3,
#         ff_dim=300,
#         num_heads=5,
#     )

#     return exp_setting


# def run_batch(self, data):
#     self.optimizer.zero_grad()
#     output = self.model(data)
#     target = data.y.type(torch.int64)
#     loss = F.cross_entropy(output, target)
#     loss.backward()
#     self.optimizer.step()


# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# # from datautils import MyTrainDataset

# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
# import os

# # From YATE
# from graphlet_construction import make_batch

# ## Load data
# from utils import Load_data

# main_data = Load_data(data_name="yago310")
# ent_list = main_data.edge_index[0, :]

# ## Define model and parameters
# from models import YATE_Encode

# ##################
# def ddp_setup(rank, world_size):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)


# class Trainer:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         train_data: DataLoader,
#         optimizer: torch.optim.Optimizer,
#         gpu_id: int,
#         save_every: int,
#     ) -> None:
#         self.gpu_id = gpu_id
#         self.model = model.to(gpu_id)
#         self.train_data = train_data
#         self.optimizer = optimizer
#         self.save_every = save_every
#         self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

#     def _run_batch(self, data):
#         self.optimizer.zero_grad()
#         output = self.model(data)
#         target = data.y.type(torch.int64)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
#         self.optimizer.step()

#     def _run_epoch(self, epoch):
#         b_sz = 32
#         print(
#             f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
#         )
#         self.train_data.sampler.set_epoch(epoch)
#         for idx in self.train_data:

#             data = make_batch(
#                 idx_cen=idx,
#                 num_hops=1,
#                 main_data=main_data,
#                 n_pos=5,
#                 per_pos=0.8,
#                 n_neg=1,
#                 per_neg=0.9,
#             )

#             data = data.to(self.gpu_id)

#             self._run_batch(data)

#     def _save_checkpoint(self, epoch):
#         ckp = self.model.module.state_dict()
#         PATH = "/storage/store3/work/mkim/gitlab/YATE/models/saved_model/checkpoint.pt"
#         torch.save(ckp, PATH)
#         print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

#     def train(self, max_epochs: int):
#         for epoch in range(max_epochs):
#             self._run_epoch(epoch)
#             if self.gpu_id == 0 and epoch % self.save_every == 0:
#                 self._save_checkpoint(epoch)


# def load_train_objs():
#     train_set = ent_list  # load your dataset
#     model = YATE_Encode(
#         input_dim=300, emb_dim=300, output_dim=100, num_heads=1, num_layers=2
#     )  # load your model
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     return train_set, model, optimizer


# def prepare_dataloader(dataset: Dataset, batch_size: int):
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=False,
#         sampler=DistributedSampler(dataset),
#     )


# def main(
#     rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int
# ):
#     ddp_setup(rank, world_size)
#     dataset, model, optimizer = load_train_objs()
#     train_data = prepare_dataloader(dataset, batch_size)
#     trainer = Trainer(model, train_data, optimizer, rank, save_every)
#     trainer.train(total_epochs)
#     destroy_process_group()


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="simple distributed training job")
#     parser.add_argument(
#         "total_epochs", type=int, help="Total epochs to train the model"
#     )
#     parser.add_argument("save_every", type=int, help="How often to save a snapshot")
#     parser.add_argument(
#         "--batch_size",
#         default=32,
#         type=int,
#         help="Input batch size on each device (default: 32)",
#     )
#     args = parser.parse_args()

#     world_size = torch.cuda.device_count()
#     mp.spawn(
#         main,
#         args=(world_size, args.save_every, args.total_epochs, args.batch_size),
#         nprocs=world_size,
#     )
