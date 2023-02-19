import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
import graphlet_construction as gc

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import YATE_Encode
from data_utils import Load_data

############
## Multi-gpu setup
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
        train_data: DataLoader,
        exp_setting: dict,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.train_data = train_data
        self.exp_setting = exp_setting
        self.model = exp_setting["model"].to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.graphlet = exp_setting["graphlet"]
        self.optimizer = exp_setting["optimizer"]

    def _run_batch(self, data):
        self.optimizer.zero_grad()
        output = self.model(data)
        target = data.y.type(torch.int64)
        loss = F.cross_entropy(output, target, weight=self.exp_setting["weight_loss"])
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = self.exp_setting["n_batch"]
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for idx in self.train_data:
            data = self.graphlet.make_batch(idx, **self.exp_setting["graphlet_setting"])
            data = data.to(self.gpu_id)
            self._run_batch(data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = os.getcwd() + f"/data/saved_model/checkpoint_ep{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


# set training objectives
def load_train_objs(
    data_name: str,
    num_rel,
    num_pos: int,
    per_pos: float,
    num_neg: int,
    per_neg: float,
    max_nodes: int,
    n_batch: int = 100,
):

    # create dictionary that set experiment settings
    exp_setting = dict()
    exp_setting["n_batch"] = n_batch
    exp_setting["graphlet_setting"] = [num_pos, per_pos, num_neg, per_neg, max_nodes]

    # load data
    main_data = Load_data(data_name=data_name)
    if num_rel is not None:
        main_data.reduce(num_rel=num_rel)

    # set graph_construction framework
    graphlet = gc.Graphlet(main_data, num_hops=1)
    exp_setting["graphlet"] = graphlet

    # set index for batch
    idx_epoch = idx_extractor(main_data, max_nodes=max_nodes)
    exp_setting["idx_epoch"] = idx_epoch

    # load your model
    model = YATE_Encode(
        input_dim=300,
        hidden_dim=100,
        num_layers=3,
        ff_dim=300,
        num_heads=5,
    )
    exp_setting["model"] = model

    # experiment settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ratio = (num_pos + 1) / ((num_pos + 1) + (num_pos + 1) * num_neg)
    weight_loss = torch.tensor([1 - ratio, ratio])

    exp_setting["optimizer"] = optimizer
    exp_setting["weight_loss"] = weight_loss

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


# prepare dataloader
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        sampler=DistributedSampler(dataset),
    )


##############
def main(rank: int, world_size: int, save_every: int, total_epochs: int):
    os.chdir("/storage/store3/work/mkim/gitlab/YATE")
    ddp_setup(rank, world_size)
    exp_setting = load_train_objs(
        data_name="yago3",
        num_rel=10,
        num_pos=5,
        per_pos=0.8,
        num_neg=1,
        per_neg=0.8,
        max_nodes=100,
    )
    train_data = prepare_dataloader(
        exp_setting["idx_epoch"], batch_size=exp_setting["n_batch"]
    )
    trainer = Trainer(train_data, exp_setting, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


##############
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    # parser.add_argument(
    #     "--batch_size",
    #     default=32,
    #     type=int,
    #     help="Input batch size on each device (default: 32)",
    # )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size, args.save_every, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
