""" Feature Extractor for the pretrained model
"""

import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Batch
from model import YATE_Pretrain

from graphlet_construction import transform_table_to_graph
from utils import load_config

################################################


class YATE_feat_extractor:
    def __init__(self, device: str = "cuda", n_layers: int = 4):
        config = load_config()

        self.makebatch = Batch()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = YATE_Pretrain(
            input_dim_x=300,
            input_dim_e=300,
            hidden_dim=300,
            num_layers=n_layers,
            ff_dim=300,
            num_heads=12,
        )

        # replace with trained weights and replace final layers
        dir_model = config["pretrained_model_dir"]
        self.model.load_state_dict(
            torch.load(dir_model, map_location=self.device), strict=False
        )
        self.model = self.model.to(self.device)

        # Replace the last layer to for representation extration
        self.model.pretrain_classifier = torch.nn.Identity()
        self.model.eval()

    def extract(
        self,
        X: pd.DataFrame,
        state: str,
        include_numeric: bool = True,
        format: str = "numpy",
    ):
        data = transform_table_to_graph(X)

        data_batch = self.makebatch.from_data_list(data, follow_batch=["edge_index"])
        data_batch.head_idx = data_batch.ptr[:-1]
        data_batch.to(self.device)

        X_numerical = data_batch.x_num

        if state == "initial":
            # Initial Features
            X_preprocessed = data_batch.x[data_batch.head_idx]
        elif state == "pretrained":
            # Pretrained Features
            with torch.no_grad():
                X_preprocessed = self.model(data_batch)
        if include_numeric:
            X_preprocessed = torch.hstack((X_preprocessed, X_numerical))

        X_preprocessed = X_preprocessed.cpu().detach().numpy()
        X_preprocessed = X_preprocessed.astype(np.float32)
        X_preprocessed = pd.DataFrame(X_preprocessed)
        col_names = [f"X{i}" for i in range(X_preprocessed.shape[1])]
        X_preprocessed = X_preprocessed.set_axis(col_names, axis="columns")
        return X_preprocessed

        #     return X_yate_initial
        #     X_yate_pretrained = X_yate_pretrained.cpu().detach().numpy()
        #     X_yate_pretrained = X_yate_pretrained.astype(np.float32)
        #     X_yate_pretrained = pd.DataFrame(X_yate_pretrained)
        #     return X_yate_pretrained
        # # elif state == "numerical":
        # #     if format == "numpy":
        # #         X_numerical = X_numerical.cpu().detach().numpy()
        # #         X_numerical = X_numerical.astype(np.float32)
        # #     return X_numerical

        # data_x_cat = data_x_cat.set_axis(col_names, axis="columns")
