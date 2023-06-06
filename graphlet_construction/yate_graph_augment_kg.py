""" Class for augmenting graph with knowledge graph
"""

import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Data, Batch
from tqdm import tqdm
from typing import Union

from graphlet_construction import Load_Yago, Graphlet

from utils import load_config


class GraphAugmentor_KG:
    def __init__(
        self,
        graphlet_settings: Union[dict, None] = None,
    ):
        self.config = load_config()

        yago_data = Load_Yago(data_name="yago3_2022", numerical=True)
        head_idx = yago_data.edge_index[0].unique()
        self.ent2idx_yago = dict(yago_data.ent2idx_original[head_idx])

        if graphlet_settings is None:
            graphlet_settings = dict()

        self.graphlet = Graphlet(yago_data, **graphlet_settings)
        self.make_batch = Batch()

    def augment_data(self, data: pd.DataFrame, augment_col_name: str, data_graph: list):
        mapping = data[augment_col_name].map(self.ent2idx_yago)
        mapping = np.array(mapping)

        data = [
            self._aggregate(data_graph, mapping, i)
            for i in tqdm(range(len(data_graph)), desc="Augment")
        ]

        return data

    def _aggregate(self, data_graph: list, mapping: np.array, idx: int):
        data_original = data_graph[idx]

        if str(mapping[idx]) == "nan":
            data = data_original
        else:
            data_yago = self.graphlet.make_batch(
                cen_idx=int(mapping[idx]), aggregate=False
            )
            data_yago = Data(
                x=data_yago[0].x,
                x_num=torch.tensor([]),
                edge_index=data_yago[0].edge_index[:, data_yago[0].edge_type != 0],
                edge_type=data_yago[0].edge_type[data_yago[0].edge_type != 0]
                + data_original.edge_type.max().item(),
                edge_attr=data_yago[0].edge_attr[data_yago[0].edge_type != 0, :],
                g_idx=data_original.g_idx,
                y=data_original.y,
            )

            data_aggregate = [data_original] + [data_yago]
            data_aggregate = self.make_batch.from_data_list(data_aggregate)

            edge_index = data_aggregate.edge_index
            edge_index[edge_index == data_aggregate.ptr[1].item()] = 0
            edge_index[edge_index > data_aggregate.ptr[1].item()] -= 1

            edge_attr = data_aggregate.edge_attr

            x = data_aggregate.x
            x = torch.vstack(
                (
                    x[: data_aggregate.ptr[1].item()],
                    x[data_aggregate.ptr[1].item() + 1 : -1],
                )
            )

            Z = torch.mul(edge_attr, x[edge_index[1]])
            x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

            edge_type = data_aggregate.edge_type

            data = Data(
                x=x,
                x_num=data_original.x_num,
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=edge_attr,
                g_idx=data_original.g_idx,
                y=data_original.y,
            )

        return data
