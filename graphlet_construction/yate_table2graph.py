import torch
import numpy as np
import pandas as pd
import fasttext

from typing import Union
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

from utils import load_config


class Table2Graph:
    def __init__(
        self,
        numerical_cardinality_threshold: Union[int, None] = 100,
        numerical_transformer: Union[str, None] = "quantile",
        num_transformer_params: Union[str, None] = None,
    ):
        self.config = load_config()
        self.lm_model = fasttext.load_model(self.config["fasttext_dir"])

        self.numerical_cardinality_threshold = numerical_cardinality_threshold

        if num_transformer_params is None:
            num_transformer_params = dict()

        self.numerical_transformer = numerical_transformer
        if numerical_transformer == "quantile":
            self.numerical_transformer = QuantileTransformer(**num_transformer_params)
        elif numerical_transformer == "standard":
            self.numerical_transformer = StandardScaler(**num_transformer_params)
        elif numerical_transformer == "minmax":
            self.numerical_transformer = MinMaxScaler(**num_transformer_params)

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_name: Union[str, None],
    ):
        self.target_name = target_name

        # Set low cardinality columns
        if self.numerical_cardinality_threshold is None:
            self.numerical_cardinality_threshold = 0

        data_x = data.drop(columns=[target_name])
        data_x_num = data_x.select_dtypes(exclude="object")

        if data_x_num.shape[1] == 0:
            self.numerical_transformer = None

        self.col_numeric_low_card = [
            col
            for col in data_x_num.columns
            if data_x_num[col].nunique() < self.numerical_cardinality_threshold
        ]

        data_fit_transform = self._generate_data(data=data, option="fit_transform")

        return data_fit_transform

    def transform(
        self,
        data: pd.DataFrame,
    ):
        data_transform = self._generate_data(data=data, option="transform")

        return data_transform

    def _generate_data(
        self,
        data: pd.DataFrame,
        option: str,
    ):
        # Preprocessing for the strings (subject to specifics of the data)
        data = data.replace("\n", " ", regex=True)
        num_data = len(data)

        data_total = dict()
        if self.target_name is not None:
            data_y = data[self.target_name].copy()
            data_y = np.array(data_y)
            data_y = torch.tensor(data_y).reshape((num_data, 1))
            data_total["data_y"] = data_y
            data_x = data.drop(labels=self.target_name, axis=1)
        else:
            data_x = data.copy()

        data_x[self.col_numeric_low_card] = data_x[self.col_numeric_low_card].astype(
            "object"
        )

        data_x_cat = data_x.select_dtypes(include="object")

        # Extract relations
        rel_names = "has " + data_x_cat.columns  # 'has'

        rel_names = (
            rel_names.str.replace("<", "")
            .str.replace(">", "")
            .str.replace("\n", "")
            .str.replace("_", " ")
            .str.lower()
        )

        rel_names = rel_names.str.replace("\n", " ", regex=True).str.replace("_", " ")
        rel_names = list(rel_names)
        # rel2idx = dict({i: rel_names[i] for i in range(len(rel_names))})

        edge_attr_total = [self.lm_model.get_sentence_vector(i) for i in rel_names]
        edge_attr_total = np.array(edge_attr_total).astype(np.float32)
        edge_attr_total = torch.tensor(edge_attr_total)

        data_x_cat.columns = np.arange(len(data_x_cat.columns))

        data_x_num = data_x.select_dtypes(exclude="object")
        data_x_num = np.array(data_x_num)
        data_x_num = data_x_num.astype("float32")

        if self.numerical_transformer is not None:
            if option == "fit_transform":
                data_x_num = self.numerical_transformer.fit_transform(data_x_num)
            else:
                data_x_num = self.numerical_transformer.transform(data_x_num)

        data_total["data_x_cat"] = data_x_cat
        data_total["data_x_num"] = data_x_num
        data_total["edge_attr_total"] = edge_attr_total

        data_graph = [
            self._graph_construct(
                data_total=data_total,
                idx=i,
            )
            for i in tqdm(range(num_data), desc=option)
        ]

        return data_graph

    def _graph_construct(
        self,
        data_total: dict,
        idx: int,
    ):
        # Obtain the data for a 'idx'-th row
        data_cat = data_total["data_x_cat"].iloc[idx]
        data_cat = data_cat.dropna()  # Exclude cells with NaN values

        # Extract edge_type
        edge_type = np.array(data_cat.index)
        edge_type = torch.tensor(edge_type)

        # Extract edge_attr
        edge_attr = data_total["edge_attr_total"][edge_type]

        # Extract edge_index
        head = torch.zeros((len(data_cat),), dtype=torch.long)
        tail = torch.arange(1, len(data_cat) + 1, dtype=torch.long)
        edge_index = torch.vstack((head, tail))
        edge_index_r = torch.vstack((tail, head))

        # Extract x (node features)
        # ent_names = data_cat
        # ent_names = (
        #     ent_names.str.replace("<", "")
        #     .str.replace(">", "")
        #     .str.replace("_", " ")
        #     .str.lower()
        # )
        ent_names = np.array(data_cat)

        x = [self.lm_model.get_sentence_vector(str(x)) for x in ent_names]
        x = np.array(x)
        x = torch.tensor(x)
        x = torch.vstack((torch.zeros((1, edge_attr.size(1))), x))

        Z = torch.mul(edge_attr, x[edge_index[1]])
        x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)
        # Z_head = Z[1:]
        # Z_head = torch.nn.functional.layer_norm(Z_head, Z_head.size())

        # Set numericals
        x_num = data_total["data_x_num"][idx]
        x_num = torch.tensor(x_num)
        x_num = x_num.reshape((1, x_num.size(0)))

        # To undirected
        edge_attr = torch.vstack((edge_attr, edge_attr))
        edge_type = torch.hstack((edge_type, edge_type))
        edge_index = torch.hstack((edge_index, edge_index_r))

        # Target
        if data_total["data_y"] is not None:
            y = data_total["data_y"][idx].clone()

        # graph index (g_idx)
        g_idx = idx

        data = Data(
            x=x,
            x_num=x_num,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            g_idx=g_idx,
            y=y,
        )

        return data
