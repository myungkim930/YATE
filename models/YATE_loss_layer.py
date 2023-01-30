"""
YATE loss layer for unsupervised learning.

"""

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

#####

## Fact(relation + tail) classification loss (T/F) - Cross entropy
class YATE_Contrast_TF(nn.Module):
    def __init__(self):

        super(YATE_Contrast_TF, self).__init__()

    def forward(self, aug_data):

        x = aug_data.x
        loss_crit = nn.CrossEntropyLoss()
        loss = loss_crit(x, aug_data.y)

        return loss


## Contrastive loss
class YATE_Contrast_CL(nn.Module):
    def __init__(self, loss):

        super(YATE_Contrast_CL, self).__init__()

        self.loss = loss

    def forward(self, h1, h2):

        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)

        return (l1 + l2) * 0.5


## Reconstruction loss for denoising
class YATE_Contrast_Denoise(nn.Module):
    def __init(self, loss):

        super(YATE_Contrast_Denoise, self).__init__()

        self.loss = loss

    def forward(self, x, emb_x):

        l = self.loss(x, emb_x)

        return l
