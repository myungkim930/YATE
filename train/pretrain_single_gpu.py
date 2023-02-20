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
            data = self.graphlet.make_batch(idx, **self.graphlet_setting)
            data = data.to(self.device)
            self._run_batch(data)
            print(
                f"[GPU{self.device.index}] Epoch {epoch} | Iteration: {batch_number}/{len(self.train_data)} | Loss: {round(self.loss.item(), 4)}"
            )
            batch_number += 1

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
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

    # exp_setting["graphlet_setting"] = [num_pos, per_pos, num_neg, per_neg, max_nodes]
    exp_setting["graphlet_setting"] = dict(
        {
            "n_pos": num_pos,
            "per_pos": per_pos,
            "n_neg": num_neg,
            "per_neg": per_neg,
            "max_nodes": max_nodes,
        }
    )

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
