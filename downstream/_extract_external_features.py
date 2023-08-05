""" Transform tables / Extract features from external sources
"""

import torch
import numpy as np
import pandas as pd
from typing import Union
from utils import load_config


# Transformation of the original table with external information


def table_to_yate_features(
    X: pd.DataFrame,
    state: str,
    include_numeric: bool = True,
    n_layers: int = 4,
    device: str = "cuda:0",
):
    from torch_geometric.data import Batch
    from model import YATE_Pretrain
    from graphlet_construction import transform_table_to_graph

    # Preliminary Settings
    config = load_config()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    makebatch = Batch()

    # Transform data into graph objects
    data = transform_table_to_graph(X)

    # Make batch for feature extraction
    data_batch = makebatch.from_data_list(data, follow_batch=["edge_index"])
    data_batch.head_idx = data_batch.ptr[:-1]
    data_batch.to(device)
    X_numerical = data_batch.x_num

    if state == "initial":
        # Initial Features
        X_preprocessed = data_batch.x[data_batch.head_idx]
    elif state == "pretrained":
        # Set model
        model = YATE_Pretrain(
            input_dim_x=300,
            input_dim_e=300,
            hidden_dim=300,
            num_layers=n_layers,
            ff_dim=300,
            num_heads=12,
        )
        # Replace with trained weights and replace final layers
        dir_model = config["pretrained_model_dir"]
        model.load_state_dict(torch.load(dir_model, map_location=device), strict=False)
        model.to(device)
        # Replace the last layer to for representation extration
        model.pretrain_classifier = torch.nn.Identity()
        model.eval()
        # Pretrained Features
        with torch.no_grad():
            X_preprocessed = model(data_batch)
    if include_numeric:
        X_preprocessed = torch.hstack((X_preprocessed, X_numerical))
    X_preprocessed = X_preprocessed.cpu().detach().numpy()
    X_preprocessed = X_preprocessed.astype(np.float32)
    X_preprocessed = pd.DataFrame(X_preprocessed)
    col_names = [f"X{i}" for i in range(X_preprocessed.shape[1])]
    X_preprocessed = X_preprocessed.set_axis(col_names, axis="columns")
    return X_preprocessed


class Table2LMFeatures:
    def __init__(
        self,
        device: str = "cuda:0",
    ):
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        import fasttext

        self.config = load_config()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        config_bert = RobertaConfig.from_pretrained(
            "roberta-base", output_hidden_states=True
        )
        self.lm_model_roberta = RobertaModel.from_pretrained(
            "roberta-base", config=config_bert
        )
        self.lm_model_roberta.to(self.device)
        self.lm_model_ft = fasttext.load_model(self.config["fasttext_dir"])

    def generate_data(
        self,
        data: pd.DataFrame,
        lm_method: str,
        include_numeric: bool,
    ):
        # Preprocessing for the strings (subject to specifics of the data)
        data = data.replace("\n", " ", regex=True)
        num_data = len(data)
        data_x = data.copy()

        data_x_cat = data_x.select_dtypes(include="object")
        data_x_num = data_x.select_dtypes(exclude="object")

        X_categorical = [
            self._vector_construct(lm_method=lm_method, data_x_cat=data_x_cat, idx=i)
            for i in range(num_data)
        ]
        X_categorical = np.array(X_categorical)
        data_x_cat = pd.DataFrame(X_categorical)

        col_names = [f"X{i}" for i in range(data_x_cat.shape[1])]
        data_x_cat = data_x_cat.set_axis(col_names, axis="columns")

        if include_numeric:
            data_total = pd.concat([data_x_cat, data_x_num], axis=1)
        else:
            data_total = data_x_cat

        return data_total

    def _vector_construct(self, lm_method: str, data_x_cat: pd.DataFrame, idx: int):
        data_temp = data_x_cat.iloc[idx]  # Obtain the data for a 'idx'-th row
        data_temp = data_temp.dropna()  # Exclude cells with Null values
        data_temp = (
            data_temp.str.replace("<", "")
            .str.replace(">", "")
            .str.replace("\n", "")
            .str.replace("_", " ")
            .str.lower()
        )

        serialization = np.array(data_temp.index) + " " + np.array(data_temp) + " . "
        np.random.shuffle(serialization)
        sentence = ""
        for i in range(len(data_temp)):
            sentence += serialization[i]
        sentence = sentence[:-1]

        if lm_method == "roberta":
            encoded_input = self.tokenizer(sentence, return_tensors="pt")
            encoded_input.to(self.device)
            with torch.no_grad():
                output = self.lm_model_roberta(**encoded_input)

            # last_four_layers = [
            #     (output["hidden_states"][i][0][1:]).mean(dim=0)
            #     for i in (-1, -2, -3, -4)
            # ]
            # embedding = torch.cat(tuple(last_four_layers), dim=-1)

            embedding = (output["hidden_states"][-1][0][1:]).mean(dim=0)
            embedding = embedding.cpu().detach().numpy()
            embedding = embedding.astype(np.float32)
        elif lm_method == "fasttext":
            embedding = self.lm_model_ft.get_sentence_vector(sentence)
            embedding = np.array(embedding)

        return embedding


# Additional information from external sources


def extract_ken_features(
    data: pd.DataFrame,
    augment_col_name: str,
):
    # Preliminary Settings
    config = load_config()

    # KEN embeddings
    ken_emb = pd.read_parquet(config["ken_embed_dir"])
    ken_ent = ken_emb["Entity"].str.lower()
    ken_embed_ent2idx = {ken_ent[i]: i for i in range(len(ken_emb))}

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)
    data_ = data.copy()
    data_[augment_col_name] = data_[augment_col_name].str.lower()

    # Mapping
    mapping = data_[augment_col_name].map(ken_embed_ent2idx)
    mapping = mapping.dropna()
    mapping = mapping.astype(np.int64)
    mapping = np.array(mapping)

    # KEN data
    data_ken = ken_emb.iloc[mapping]
    data_ken.rename(columns={"Entity": "name"}, inplace=True)
    data_ken.drop_duplicates(inplace=True)
    data_ken = data_ken.reset_index(drop=True)

    return data_ken


def extract_fasttext_features(
    data: pd.DataFrame,
    augment_col_name: str,
):
    import fasttext

    # Preliminary Settings
    config = load_config()
    lm_model = fasttext.load_model(config["fasttext_dir"])

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)
    data_ = data.copy()

    # Entity Names
    ent_names = (
        data[augment_col_name]
        .str.replace("<", "")
        .str.replace(">", "")
        .str.replace("_", " ")
        .str.lower()
    )
    ent_names = np.array(ent_names)

    # Data Fasttext for entity names
    data_fasttext = [lm_model.get_sentence_vector(str(x)) for x in ent_names]
    data_fasttext = np.array(data_fasttext)
    data_fasttext = pd.DataFrame(data_fasttext)
    col_names = [f"X{i}" for i in range(data_fasttext.shape[1])]
    data_fasttext = data_fasttext.set_axis(col_names, axis="columns")
    data_fasttext = pd.concat([data_fasttext, data[augment_col_name]], axis=1)
    # data_fasttext.drop_duplicates(inplace=True)
    data_fasttext = data_fasttext.reset_index(drop=True)

    return data_fasttext
