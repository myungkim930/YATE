import torch
import numpy as np

from torch_geometric.data import Batch
from model import YATE_Pretrain


class YATE_feat_extractor:
    def __init__(self, gpu_id: int = 0, n_layers: int = 4):
        self.makebatch = Batch()
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.model = YATE_Pretrain(
            input_dim_x=300,
            input_dim_e=300,
            hidden_dim=300,
            num_layers=n_layers,
            ff_dim=300,
            num_heads=12,
        )

        # replace with trained weights and replace final layers
        model_name = "yago3_2022_num_2304_NB128_NS100_NH2_NP1_MN100_MS"
        n_step = 250000
        dir_model = (
            "/storage/store3/work/mkim/gitlab/YATE/data/saved_model/" + model_name
        )
        dir_model = dir_model + f"/ckpt_step{n_step}.pt"
        self.model.load_state_dict(
            torch.load(dir_model, map_location="cuda:0"), strict=False
        )
        self.model = self.model.to(self.device)

        # Replace the last layer to for representation extration
        self.model.classifier_node = torch.nn.Identity()
        self.model.eval()

    def extract(self, data, state: str, format: str = "torch"):
        data_batch = self.makebatch.from_data_list(data, follow_batch=["edge_index"])
        data_batch.head_idx = data_batch.ptr[:-1]
        data_batch.to(self.device)

        if state == "initial":
            # Initial Features
            X_yate_initial = data_batch.x[data_batch.head_idx]
            if format == "numpy":
                X_yate_initial = X_yate_initial.cpu().detach().numpy()
                X_yate_initial = X_yate_initial.astype(np.float32)
            return X_yate_initial
        elif state == "pretrained":
            # Pretrained Features
            with torch.no_grad():
                X_yate_pretrained = self.model(data_batch)
            if format == "numpy":
                X_yate_pretrained = X_yate_pretrained.cpu().detach().numpy()
                X_yate_pretrained = X_yate_pretrained.astype(np.float32)
            return X_yate_pretrained
        elif state == "numerical":
            X_numerical = data_batch.x_num
            if format == "numpy":
                X_numerical = X_numerical.cpu().detach().numpy()
                X_numerical = X_numerical.astype(np.float32)
            return X_numerical
