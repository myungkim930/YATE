import torch


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    return h1 @ h2.t()
    return h1 @ h2.t()
