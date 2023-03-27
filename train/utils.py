## Functions used for training
import math
import torch

from torch.optim.lr_scheduler import _LRScheduler


## Index sampler according to the coverage of edge_index
class Index_extractor:
    def __init__(self, main_data):
        self.main_data = main_data

    def reset(self):
        self.count_head = torch.bincount(self.main_data.edge_index[0, :])
        self.count_head = self.count_head.to(torch.float)

    def sample(self, n_batch: int):
        if n_batch > self.count_head[self.count_head > 0].size(0):
            self.reset()
        idx_sample = torch.multinomial(self.count_head, n_batch)
        self.count_head[idx_sample] -= 1
        return idx_sample


## Index sampler according to the coverage of edge_index (by type)
class Index_extractor_type:
    def __init__(self, main_data):
        self.main_data = main_data
        self.num_types = main_data.headidx2type[1].unique().size(0)
        self.count_head = torch.bincount(main_data.edge_index[0])
        self.count_head_type = dict()
        for i in range(self.num_types):
            temp = torch.zeros(self.count_head.size(0), dtype=torch.long)
            temp[
                main_data.headidx2type[0, main_data.headidx2type[1] == i]
            ] = self.count_head[
                main_data.headidx2type[0, main_data.headidx2type[1] == i]
            ]
            self.count_head_type[f"{i}"] = temp.to(torch.float)
        self.count_head_type_original = self.count_head_type.copy()
        self.head_type = 0

    def reset(self):
        self.count_head_type[f"{self.head_type}"] = self.count_head_type_original[
            f"{self.head_type}"
        ].clone()
        # self.count_head = torch.bincount(self.main_data.edge_index[0, :])
        # self.count_head = self.count_head.to(torch.float)

    def sample(self, n_batch: int):
        # self.count_head_type[f'{self.head_type}'].nonzero().size(0)
        if n_batch > self.count_head_type[f"{self.head_type}"].nonzero().size(0):
            self.reset()
        idx_sample = torch.multinomial(
            self.count_head_type[f"{self.head_type}"], n_batch
        )
        self.count_head_type[f"{self.head_type}"][idx_sample] -= 1

        # idx_subtract_ = (
        #     (torch.bincount(idx_sample) > 0).nonzero().view(-1).to(torch.long)
        # )
        # self.count_head[idx_subtract_] = (
        #     self.count_head[idx_subtract_] - torch.bincount(idx_sample)[idx_subtract_]
        # )

        if self.head_type == self.num_types - 1:
            self.head_type = 0
        else:
            self.head_type += 1
        return idx_sample


## scheduler for the learning rate
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
