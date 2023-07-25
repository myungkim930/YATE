import torch
import numpy as np
import pandas as pd
import fasttext

from torch_geometric.data import Data

from utils import load_config

from sklearn.base import BaseEstimator, TransformerMixin


def transform_table_to_graph(X: pd.DataFrame, y=None):
    """Function to transform tables to list of graphs.

    The list of graphs are generated in a row-wise fashion.

    Parameters
    ----------
    X : The input pandas DataFrame.
    y : The target variable

    Returns
    -------
    data_graph : list of data objects in which each object is a graph with node features, edge features, edge types, numerical features.

    """

    # Loading fasttext
    config = load_config()
    lm_model = fasttext.load_model(config["fasttext_dir"])

    # Preprocessing for the strings (subject to specifics of the data)
    X = X.replace("\n", " ", regex=True)
    num_data = X.shape[0]

    # Preprocess the target
    if y is not None:
        y = np.array(y)
        y = torch.tensor(y).reshape((num_data, 1))

    # Extract relations
    X_categorical = X.select_dtypes(include="object")
    rel_names = "has " + X_categorical.columns  # 'has'
    rel_names = (
        rel_names.str.replace("\n", " ", regex=True).str.replace("_", " ").str.lower()
    )
    rel_names = list(rel_names)

    edge_attr_total = [lm_model.get_sentence_vector(i) for i in rel_names]
    edge_attr_total = np.array(edge_attr_total).astype(np.float32)
    edge_attr_total = torch.tensor(edge_attr_total)

    X_categorical.columns = np.arange(len(X_categorical.columns))

    X_numerical = X.select_dtypes(exclude="object")
    X_numerical = np.array(X_numerical)
    X_numerical = X_numerical.astype("float32")

    data_graph = [
        _graph_construct(
            X_categorical,
            X_numerical,
            edge_attr_total,
            y,
            lm_model,
            idx=i,
        )
        for i in range(num_data)
    ]

    return data_graph


def _graph_construct(
    X_categorical,
    X_numerical,
    edge_attr_total,
    y,
    lm_model,
    idx,
):
    # Obtain the data for a 'idx'-th row
    data_cat = X_categorical.iloc[idx]
    data_cat = data_cat.dropna()  # Exclude cells with NaN values

    # Extract edge_type
    edge_type = np.array(data_cat.index)
    edge_type = torch.tensor(edge_type)

    # Extract edge_attr
    edge_attr = edge_attr_total[edge_type]

    # Extract edge_index
    head = torch.zeros((len(data_cat),), dtype=torch.long)
    tail = torch.arange(1, len(data_cat) + 1, dtype=torch.long)
    edge_index = torch.vstack((head, tail))
    edge_index_r = torch.vstack((tail, head))

    # Extract x (node features)
    ent_names = data_cat
    ent_names = (
        ent_names.str.replace("<", "")
        .str.replace(">", "")
        .str.replace("_", " ")
        .str.lower()
    )
    ent_names = np.array(data_cat)

    x = [lm_model.get_sentence_vector(str(x).lower()) for x in ent_names]
    x = np.array(x)
    x = torch.tensor(x)
    x = torch.vstack((torch.zeros((1, edge_attr.size(1))), x))

    Z = torch.mul(edge_attr, x[edge_index[1]])
    x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

    # Set numericals
    x_num = X_numerical[idx]
    x_num = torch.tensor(x_num)
    # x_num = torch.nan_to_num(x_num, nan=0.5)
    x_num = x_num.reshape((1, x_num.size(0)))

    # To undirected
    edge_attr = torch.vstack((edge_attr, edge_attr))
    edge_type = torch.hstack((edge_type, edge_type))
    edge_index = torch.hstack((edge_index, edge_index_r))

    # Target
    if y is not None:
        y = y[idx].clone()
    else:
        y = torch.tensor([])

    # graph index (g_idx)
    g_idx = idx

    data = Data(
        x=x,
        x_num=x_num,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        y=y,
        g_idx=g_idx,
    )

    return data


####


class Table2GraphTransformer(TransformerMixin, BaseEstimator):
    """Class to transform tables to list of graphs.

    The list of graphs are generated in a row-wise fashion.

    Parameters
    ----------
    X : The input pandas DataFrame.
    y : The target variable

    Returns
    -------
    data_graph : list of data objects in which each object is a graph with node features, edge features, edge types, numerical features.

    """

    def fit(self, X, y=None):
        self.y_ = y
        return self

    def transform(self, X, y=None):
        if hasattr(self, "lm_model_") == False:
            self._load_fasttext_model()

        X = X.replace("\n", " ", regex=True)
        num_data = X.shape[0]

        # Preprocess the target
        if self.y_ is not None:
            y = np.array(self.y_)
            y = torch.tensor(y).reshape((num_data, 1))

        # Extract relations
        X_categorical = X.select_dtypes(include="object")
        rel_names = "has " + X_categorical.columns  # 'has'
        rel_names = (
            rel_names.str.replace("\n", " ", regex=True)
            .str.replace("_", " ")
            .str.lower()
        )
        rel_names = list(rel_names)

        edge_attr_total = [self.lm_model_.get_sentence_vector(i) for i in rel_names]
        edge_attr_total = np.array(edge_attr_total).astype(np.float32)
        edge_attr_total = torch.tensor(edge_attr_total)

        X_categorical.columns = np.arange(len(X_categorical.columns))

        X_numerical = X.select_dtypes(exclude="object")
        X_numerical = np.array(X_numerical)
        X_numerical = X_numerical.astype("float32")

        data_graph = [
            self._graph_construct(
                X_categorical,
                X_numerical,
                edge_attr_total,
                y,
                idx=i,
            )
            for i in range(num_data)
        ]

        if self.y_ is not None:
            self.y_ = None

        return data_graph

    def _load_fasttext_model(
        self,
    ):
        # Loading fasttext
        config = load_config()
        self.lm_model_ = fasttext.load_model(config["fasttext_dir"])

    def _graph_construct(
        self,
        X_categorical,
        X_numerical,
        edge_attr_total,
        y,
        idx,
    ):
        # Obtain the data for a 'idx'-th row
        data_cat = X_categorical.iloc[idx]
        data_cat = data_cat.dropna()  # Exclude cells with NaN values

        # Extract edge_type
        edge_type = np.array(data_cat.index)
        edge_type = torch.tensor(edge_type)

        # Extract edge_attr
        edge_attr = edge_attr_total[edge_type]

        # Extract edge_index
        head = torch.zeros((len(data_cat),), dtype=torch.long)
        tail = torch.arange(1, len(data_cat) + 1, dtype=torch.long)
        edge_index = torch.vstack((head, tail))
        edge_index_r = torch.vstack((tail, head))

        # Extract x (node features)
        ent_names = data_cat
        ent_names = (
            ent_names.str.replace("<", "")
            .str.replace(">", "")
            .str.replace("_", " ")
            .str.lower()
        )
        ent_names = np.array(data_cat)

        x = [self.lm_model_.get_sentence_vector(str(x).lower()) for x in ent_names]
        x = np.array(x)
        x = torch.tensor(x)
        x = torch.vstack((torch.zeros((1, edge_attr.size(1))), x))

        Z = torch.mul(edge_attr, x[edge_index[1]])
        x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

        # Set numericals
        x_num = X_numerical[idx]
        x_num = torch.tensor(x_num)
        # x_mean = torch.nanmean(x_num).item()
        # x_num = torch.nan_to_num(x_num, nan=x_mean)
        x_num = x_num.reshape((1, x_num.size(0)))

        # To undirected
        edge_attr = torch.vstack((edge_attr, edge_attr))
        edge_type = torch.hstack((edge_type, edge_type))
        edge_index = torch.hstack((edge_index, edge_index_r))

        # Target
        if self.y_ is not None:
            y = y[idx].clone()
        else:
            y = torch.tensor([])

        # graph index (g_idx)
        g_idx = idx

        data = Data(
            x=x,
            x_num=x_num,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            y=y,
            g_idx=g_idx,
        )

        return data
