# Table_to_vec_lm: table to vectors with language model roberta

import numpy as np
import pandas as pd
import pickle
import torch
import fasttext

from typing import Union
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from utils import load_config


class Table2Vector_LM:
    def __init__(
        self,
        gpu_id: int = 0,
    ):
        self.config = load_config()
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

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
        target_name: Union[str, None],
        lm_method: str,
        numerical_cardinality_threshold: Union[int, None] = 10,
        save_dir: Union[str, None] = None,
    ):
        if numerical_cardinality_threshold is None:
            numerical_cardinality_threshold = 0

        # Preprocessing for the strings (subject to specifics of the data)
        data = data.replace("\n", " ", regex=True)
        num_data = len(data)

        # Designate the target
        data_total = dict()
        if target_name is not None:
            data_y = data[target_name]
            data_total["y"] = np.array(data_y)
            data_x = data.drop(labels=target_name, axis=1)
        else:
            data_x = data.copy()

        data_x_num = data_x.select_dtypes(exclude="object")
        col_numeric_low_card = [
            col
            for col in data_x_num.columns
            if data_x_num[col].nunique() < numerical_cardinality_threshold
        ]
        data_x[col_numeric_low_card] = data_x[col_numeric_low_card].astype("str")
        data_x_cat = data_x.select_dtypes(include="object")
        data_x_num = data_x.select_dtypes(exclude="object")

        X_categorical = [
            self._vector_construct(lm_method=lm_method, data_x_cat=data_x_cat, idx=i)
            for i in tqdm(range(num_data))
        ]
        X_categorical = np.array(X_categorical)
        data_x_cat = pd.DataFrame(X_categorical)

        col_names = [f"X{i}" for i in range(data_x_cat.shape[1])]
        data_x_cat = data_x_cat.set_axis(col_names, axis="columns")

        data_total = pd.concat([data_x_cat, data_x_num, data_y], axis=1)
        # data_total = pd.concat([data_x_cat, data_y], axis=1)

        if save_dir is not None:
            data_total.to_parquet(save_dir, index=False)

            # with open(save_dir, "wb") as pickle_file:
            #     pickle.dump(data_total, pickle_file)

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
