### Trainer for toy examples

import os
import torch
import datetime
import math

from time import time

from model import YATE_Encode
from data_utils import Load_data
from graphlet_construction import Graphlet
from torch.utils.data import DataLoader

# from torch.optim.lr_scheduler import _LRScheduler

from train.utils import CosineAnnealingWarmUpRestarts, Index_extractor

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


##############
## DDP setup
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


## Trainer class
class Trainer:
    def __init__(
        self,
        exp_setting: dict,
        gpu_id: int,
    ) -> None:
        self.__dict__ = exp_setting
        self.gpu_id = gpu_id
        self.model = self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.log = []

    def _run_step(self, step, idx):
        start_time = time()
        self.optimizer.zero_grad()
        data = self.graphlet.make_batch(idx, **self.graphlet_setting)
        data = data.to(self.gpu_id)
        output_x, output_edge_attr = self.model(data)

        # loss on nodes
        target_node = data.y
        loss_node = self.criterion_node(output_x, target_node)

        # loss on edges
        target_edge = data.edge_type[data.idx_perturb]
        loss_edge = self.criterion_edge(output_edge_attr, target_edge)

        loss = loss_node + loss_edge
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        end_time = time()
        duration = round(end_time - start_time, 4)

        loss_node = round(loss_node.detach().item(), 4)
        loss_edge = round(loss_edge.detach().item(), 4)

        self.log.append(
            f"Step {step} | Loss(n/e): {loss_node}/{loss_edge} | Duration: {duration}"
        )

        print(
            f"[GPU{self.gpu_id}] Step {step} | Loss(n/e): {loss_node}/{loss_edge} | Duration: {duration}"
        )

        del (
            loss,
            loss_node,
            loss_edge,
            output_x,
            output_edge_attr,
            target_node,
            target_edge,
            data,
        )

    def _set_perturb(self):
        self.graphlet_setting["per_perturb"] += self.perturb_window[
            "per_perturb_increase"
        ]
        self.perturb_window["step_perturb_change"] += self.perturb_window["step_diff"]
        # self.perturb_window["step_diff"] -= self.perturb_window["change_step_diff"]

    def _save_checkpoint(self, step):
        ckp = self.model.module.state_dict()
        PATH = self.save_dir + f"/ckpt_step{step}.pt"
        torch.save(ckp, PATH)
        PATH_LOG = self.save_dir + f"/log_train.txt"
        with open(PATH_LOG, "w") as output:
            for row in self.log:
                output.write(str(row) + "\n")
        print(f"Step {step-1} | Training checkpoint saved at {self.save_dir}")

    def train(self):
        self.idx_extract.reset()
        self.model.train()
        step = 0
        idx = self.idx_extract.sample(n_batch=self.n_batch * torch.cuda.device_count())
        train_idx = DataLoader(
            idx,
            batch_size=self.n_batch,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(idx),
        )
        # self.train_data.sampler.set_epoch(epoch)
        while step < self.n_steps:
            for idx in train_idx:
                self._run_step(step, idx)
            step += 1
            if step > self.perturb_window["step_perturb_change"] - 1:
                self._set_perturb()
            if self.gpu_id == 0 and step % self.save_every == 0:
                self._save_checkpoint(step)


def load_train_objs(
    data_name: str,
    num_hops: int,
    per_perturb: float,
    n_perturb_mask: int,
    n_perturb_replace: int,
    max_nodes: int,
    n_batch: int,
    n_steps: int,
    save_every: int,
    num_rel=None,
):

    # create dictionary that set experiment settings
    exp_setting = dict()
    exp_setting["n_batch"] = n_batch
    exp_setting["n_steps"] = n_steps
    exp_setting["save_every"] = save_every

    exp_setting["graphlet_setting"] = dict(
        {
            "per_perturb": per_perturb,
            "n_perturb_mask": n_perturb_mask,
            "n_perturb_replace": n_perturb_replace,
        }
    )
    world_size = torch.cuda.device_count()
    exp_setting["perturb_window"] = dict(
        {
            "step_perturb_change": int(n_steps / world_size),
            "step_diff": int(n_steps / world_size),
            # "change_step_diff": int(100000 / world_size),
            "per_perturb_increase": 0.2,
        }
    )

    # load data
    main_data = Load_data(data_name=data_name)
    if num_rel is not None:
        main_data.reduce(num_rel=num_rel)

    # set graph_construction framework
    graphlet = Graphlet(main_data, num_hops=num_hops, max_nodes=max_nodes)
    exp_setting["graphlet"] = graphlet

    # set train for batch
    idx_extract = Index_extractor(main_data)
    exp_setting["idx_extract"] = idx_extract

    # load your model
    model = YATE_Encode(
        input_dim_x=300,
        input_dim_e=300,
        hidden_dim=300,
        edge_class_dim=len(main_data.rel2idx),
        num_layers=12,
        ff_dim=300,
        num_heads=1,
    )
    exp_setting["model"] = model

    # training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    exp_setting["optimizer"] = optimizer

    criterion_node = torch.nn.CrossEntropyLoss()
    # Other losses: torch.nn. BCEWithLogitsLoss BCELoss L1Loss / Infonce_loss
    criterion_edge = torch.nn.CrossEntropyLoss()
    exp_setting["criterion_node"] = criterion_node
    exp_setting["criterion_edge"] = criterion_edge

    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=int(n_steps / world_size),
        T_mult=1,
        eta_max=5e-5,
        T_up=int(10000 / world_size),
        gamma=0.9,
    )
    exp_setting["scheduler"] = scheduler

    # saving directory
    now = datetime.datetime.now()
    save_dir = (
        os.getcwd()
        + "/data/saved_model/"
        + data_name
        + "_"
        + now.strftime(
            f"%d%m_NB{n_batch}_NS{n_steps/10000}_NH{num_hops}_NP{n_perturb_mask+n_perturb_replace+1}"
        )
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    exp_setting["save_dir"] = save_dir

    return exp_setting


##############
def main(rank: int, world_size: int):
    os.chdir("/storage/store3/work/mkim/gitlab/YATE")
    exp_setting = load_train_objs(
        data_name="yago3",
        num_hops=2,
        per_perturb=0.2,
        n_perturb_mask=1,
        n_perturb_replace=2,
        max_nodes=100,
        n_batch=64,
        n_steps=int(1000000 / world_size),
        save_every=10000,
    )
    ddp_setup(rank, world_size)
    trainer = Trainer(exp_setting, rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

##################
