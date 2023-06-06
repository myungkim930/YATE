"""
YATE gnn finetune model with defined layers.

"""

# PyTorch
import torch
import torch.nn as nn

from .yate_block import YATE_Base


##################################################
# YATE - Finetune block for regression
class YATE_FinetuneReg(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        input_numeric_dim: int,
        output_dim: int,
        num_layers: int,
        include_numeric: bool = True,
        **block_args
    ):
        super(YATE_FinetuneReg, self).__init__()

        self.ft_base = YATE_Base(
            input_dim_x=input_dim_x,
            input_dim_e=input_dim_e,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **block_args
        )

        self.include_numeric = include_numeric
        if include_numeric:
            self.ft_numerical = nn.Sequential(
                nn.BatchNorm1d(input_numeric_dim),
            )
            cls_input_dim = hidden_dim + input_numeric_dim
        else:
            cls_input_dim = hidden_dim

        # self.ft_resnet1 = nn.Sequential(
        #     nn.Linear(cls_input_dim, cls_input_dim),
        # )

        # self.ft_resnet2 = nn.Sequential(
        #     nn.Linear(cls_input_dim, cls_input_dim),
        # )

        # self.ft_classifier = nn.Sequential(
        #     nn.Linear(cls_input_dim, output_dim),
        # )

        self.ft_classifier = nn.Sequential(
            nn.Linear(cls_input_dim, cls_input_dim),
            nn.Linear(cls_input_dim, cls_input_dim),
            nn.Linear(cls_input_dim, output_dim),
        )

    def forward(self, input):
        x, edge_index, edge_attr, head_idx = (
            input.x,
            input.edge_index,
            input.edge_attr,
            input.ptr[:-1],
        )

        x = self.ft_base(x, edge_index, edge_attr)
        x = x[head_idx, :]

        if self.include_numeric:
            x_num = input.x_num.to(torch.float)
            x_num = self.ft_numerical(x_num)
            x = torch.hstack((x, x_num))

        # x = x + self.ft_resnet1(x)
        # x = x + self.ft_resnet2(x)

        x = self.ft_classifier(x)

        return x


# YATE - Finetune block for classification
class YATE_FinetuneCls(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        input_numeric_dim: int,
        output_dim: int,
        num_layers: int,
        include_numeric: bool = True,
        **block_args
    ):
        super(YATE_FinetuneCls, self).__init__()

        self.ft_base = YATE_Base(
            input_dim_x=input_dim_x,
            input_dim_e=input_dim_e,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **block_args
        )

        self.include_numeric = include_numeric
        if include_numeric:
            self.ft_numerical = nn.Sequential(
                nn.BatchNorm1d(
                    input_numeric_dim, affine=True, track_running_stats=True
                ),
            )
            cls_input_dim = hidden_dim + input_numeric_dim
        else:
            cls_input_dim = hidden_dim

        self.ft_classifier = nn.Sequential(
            nn.Linear(cls_input_dim, cls_input_dim),
            nn.ReLU(),
            nn.Linear(cls_input_dim, cls_input_dim),
            nn.ReLU(),
            nn.Linear(cls_input_dim, output_dim),
        )

    def forward(self, input):
        x, edge_index, edge_attr, head_idx = (
            input.x,
            input.edge_index,
            input.edge_attr,
            input.ptr[:-1],
        )

        x = self.ft_base(x, edge_index, edge_attr)
        x = x[head_idx, :]

        if self.include_numeric:
            x_num = input.x_num.to(torch.float)
            x_num = self.ft_numerical(x_num)
            x = torch.hstack((x, x_num))

        x = self.ft_classifier(x)

        return x
