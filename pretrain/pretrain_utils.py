## Functions used for training
import math
import torch

from torch.optim.lr_scheduler import _LRScheduler


## Index sampler class
class Index_extractor:
    def __init__(
        self, main_data, num_rel: int = 6, per: float = 0.9, max_nodes: int = 100
    ):
        self.count_h = main_data.edge_index[0].bincount()
        self.list_u = (self.count_h > num_rel - 1).nonzero().view(-1)
        self.list_d = (self.count_h < num_rel).nonzero().view(-1)

        self.prob = dict()
        self.prob["u"] = torch.ceil(
            self.count_h[self.list_u].to(torch.float) / max_nodes
        )
        self.prob["d"] = torch.ceil(
            self.count_h[self.list_d].to(torch.float) / max_nodes
        )
        self.prob_original = self.prob.copy()

        self.per = per

    def reset(self, up: bool):
        if up:
            self.prob["u"] = self.prob_original["u"].clone()
        else:
            self.prob["d"] = self.prob_original["d"].clone()

    def sample(self, n_batch: int):
        num_u = math.ceil(n_batch * self.per)
        num_d = n_batch - num_u

        if num_u > self.prob["u"].nonzero().size(0):
            self.reset(up=True)
        idx_sample_u = torch.multinomial(self.prob["u"], num_u)
        # self.prob["u"][idx_sample_u] -= 1
        idx_sample_u = self.list_u[idx_sample_u]

        if num_d > self.prob["d"].nonzero().size(0):
            self.reset(up=False)
        idx_sample_d = torch.multinomial(self.prob["d"], num_d)
        # self.prob["d"][idx_sample_d] -= 1
        idx_sample_d = self.list_d[idx_sample_d]

        idx_sample = torch.hstack((idx_sample_u, idx_sample_d))
        return idx_sample


## cosine scheduler for the learning rate
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


## Target construction for the pretraining
def create_target_node(data):
    graph_idx = data.g_idx
    pos_mask = (
        graph_idx.repeat(graph_idx.size(0), 1)
        - graph_idx.repeat(graph_idx.size(0), 1).t()
    )

    target = pos_mask.clone()
    target[pos_mask == 0] = 1
    target[pos_mask != 0] = 0
    target = target.type("torch.cuda.FloatTensor")

    return target
