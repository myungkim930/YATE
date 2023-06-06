""" Pretrain loss - INFONCE / MAXSIM
"""
import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor


## INFONCE Loss
class Infonce_loss(_Loss):
    def __init__(self, tau: float) -> None:
        super(Infonce_loss, self).__init__()

        self.tau = tau

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _infonce(input, target, self.tau)


def _infonce(input: torch.tensor, target: Tensor, tau: float = 1.0):
    input_ = input.clone()
    input_ = input_ / tau
    pos_mask = (target - torch.eye(target.size(0), device=input.device)) > 0
    self_mask = torch.eye(input.size(0), dtype=torch.bool, device=input.device)
    input_.masked_fill_(self_mask, -9e15)
    num_pos_ = int(sum(pos_mask)[0])
    loss = torch.sum(-input_[pos_mask].reshape((target.size(0), num_pos_)), dim=1)
    loss += torch.logsumexp(input_, dim=0)
    loss = loss.mean()
    return loss


## Max. Similarity Loss
class Max_sim_loss(_Loss):
    def __init__(self, tau: float = 1.0) -> None:
        super(Max_sim_loss, self).__init__()

        self.tau = tau

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _max_sim(input, target, self.tau)


def _max_sim(input: torch.tensor, target: Tensor, tau: float = 1.0):
    input_ = input.clone()
    input_ = input_ / tau
    pos_mask = (target - torch.eye(target.size(0), device=input.device)) > 0
    self_mask = torch.eye(input.size(0), dtype=torch.bool, device=input.device)
    input_.masked_fill_(self_mask, -9e15)

    loss = target - input_
    loss = loss[pos_mask]
    loss = loss.mean()

    return loss
