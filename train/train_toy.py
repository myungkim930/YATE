import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# From YATE
from graphlet_construction import make_batch

## Load data
from utils import Load_data

main_data = Load_data(data_name="yago310")
ent_list = main_data.edge_index[0, :]

## Define model and parameters
from models import YATE_Encode

##################
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def _run_batch(self, data):
        self.optimizer.zero_grad()
        output = self.model(data)
        target = data.y.type(torch.int64)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = 32
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for idx in self.train_data:

            data = make_batch(
                idx_cen=idx,
                num_hops=1,
                main_data=main_data,
                n_pos=5,
                per_pos=0.8,
                n_neg=1,
                per_neg=0.9,
            )

            data = data.to(self.gpu_id)

            self._run_batch(data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "/storage/store3/work/mkim/gitlab/YATE/models/saved_model/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = ent_list  # load your dataset
    model = YATE_Encode(
        input_dim=300, emb_dim=300, output_dim=100, num_heads=1, num_layers=2
    )  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(
    rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int
):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size, args.save_every, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
