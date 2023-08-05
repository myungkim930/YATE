import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import sys

sys.path.append("")
import skorch
from baselines.utils import get_activation_fn, get_nonglu_activation_fn
import numpy as np
import pandas as pd


class ResNet(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
        regression: bool,
        categorical_indicator
    ) -> None:
        super().__init__()

        # categories = None #TODO
        def make_normalization():
            return {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[
                normalization
            ](d)

        self.categorical_indicator = categorical_indicator  # Added
        self.regression = regression
        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)
        self.categories = categories
        if categories is not None and len(categories) > 0:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(int(sum(categories)), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # set the embedding of the last category of each feature to zero
            # it represents the "missing" category, i.e. the categories that is not present
            # in the training set
            for i, c in enumerate(categories):
                self.category_embeddings.weight.data[
                    category_offsets[i] + c - 1
                ].zero_()

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": make_normalization(),
                        "linear0": nn.Linear(
                            d, d_hidden * (2 if activation.endswith("glu") else 1)
                        ),
                        "linear1": nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x) -> Tensor:
        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long()  # TODO
        else:
            x_num = x
            x_cat = None
        x = []
        if x_num is not None and x_num.numel() > 0:
            x.append(x_num)
        if x_cat is not None and x_cat.numel() > 0:
            # replace -1 by the last category
            for i in range(x_cat.shape[1]):
                x_cat[:, i][x_cat[:, i] == -1] = self.categories[i] - 1
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer["norm"](z)
            z = layer["linear0"](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer["linear1"](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x


class InputShapeSetterResnet(skorch.callbacks.Callback):
    def __init__(
        self, regression=False, batch_size=None, cat_features=None, categories=None
    ):
        self.cat_features = cat_features
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories

    def on_train_begin(self, net, X, y):
        if not (self.cat_features is None):
            self.categorical_indicator = [
                i in self.cat_features for i in range(X.shape[1])
            ]
        else:
            self.categorical_indicator = None
        if self.categorical_indicator is None:
            d_numerical = X.shape[1]
            categories = None
        else:
            d_numerical = X.shape[1] - sum(self.categorical_indicator)
            if self.categories is None:
                # if numpy array
                if isinstance(X, np.ndarray):
                    categories = list(
                        (X[:, self.categorical_indicator].max(0) + 2).astype(int)
                    )  # +2 for unknown category
                # if pandas dataframe
                elif isinstance(X, pd.DataFrame):
                    categories = list(
                        (X.iloc[:, self.categorical_indicator].max(0) + +2).astype(int)
                    )  # +2 for unknown category
            else:
                categories = self.categories
        net.set_params(
            module__d_numerical=d_numerical,
            module__categories=categories,  # FIXME #lib.get_categories(X_cat),
            module__categorical_indicator=torch.BoolTensor(self.categorical_indicator)
            if self.categorical_indicator is not None
            else None,
            module__d_out=2 if self.regression == False else 1,
        )  # FIXME#D.info['n_classes'] if D.is_multiclass else 1,
        # print("Numerical features: {}".format(d_numerical))
        # print("Categories {}".format(categories))
