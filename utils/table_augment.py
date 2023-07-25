""" Class for obtaining external information for table with KEN/Fasttext
"""

import pandas as pd
import numpy as np
import fasttext

from typing import Union

from utils import load_config


class TableExternalInfoExtractor:
    def __init__(self):
        self.config = load_config()

        self.ken_emb = pd.read_parquet(self.config["ken_embed_dir"])
        ken_ent = self.ken_emb["Entity"]
        self.ken_embed_ent2idx = {ken_ent[i]: i for i in range(len(self.ken_emb))}

        self.lm_model = fasttext.load_model(self.config["fasttext_dir"])

    def augment(
        self,
        data: pd.DataFrame,
        method: str,
        augment_col_name: str,
        target_name: Union[str, None] = None,
    ):
        # Original data
        data_ = data.copy()
        data_.replace("\n", " ", regex=True, inplace=True)
        data_ = data.copy()
        if target_name is not None:
            target = data_[target_name]
            data_.drop(labels=target_name, axis=1, inplace=True)

        if method == "ken":
            data_aug = self._augment_with_ken(data_, augment_col_name)
        elif method == "fasttext":
            data_aug = self._augment_with_fasttext(data_, augment_col_name)

        return data_aug

    def _augment_with_ken(
        self,
        data: pd.DataFrame,
        augment_col_name: str,
    ):
        mapping = data[augment_col_name].map(self.ken_embed_ent2idx)
        mapping = mapping.dropna()
        mapping = mapping.astype(np.int64)
        mapping = np.array(mapping)

        data_aug = self.ken_emb.iloc[mapping]
        data_aug = data_aug.reset_index(drop=True)
        data_aug.rename(columns={"Entity": "name"}, inplace=True)
        # data_aug = data_aug.drop(labels="Entity", axis=1)

        data_aug.drop_duplicates(inplace=True)

        return data_aug

    def _augment_with_fasttext(
        self,
        data: pd.DataFrame,
        augment_col_name: str,
    ):
        ent_names = (
            data[augment_col_name]
            .str.replace("<", "")
            .str.replace(">", "")
            .str.replace("_", " ")
            .str.lower()
        )
        ent_names = np.array(ent_names)
        data_aug = [self.lm_model.get_sentence_vector(str(x)) for x in ent_names]
        data_aug = np.array(data_aug)

        data_aug = pd.DataFrame(data_aug)

        col_names = [f"X{i}" for i in range(data_aug.shape[1])]
        data_aug = data_aug.set_axis(col_names, axis="columns")

        data_aug = pd.concat([data_aug, data[augment_col_name]], axis=1)

        data_aug.drop_duplicates(inplace=True)

        return data_aug
