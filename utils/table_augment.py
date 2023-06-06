""" Class for augmenting table with KEN/Fasttext
"""

import pandas as pd
import numpy as np
import fasttext

from typing import Union

from utils import load_config


class TableAugmentor:
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
        drop_augment_col: bool = False,
        target_name: Union[str, None] = None,
        concat: bool = True,
    ):
        # Oritinal data
        data_x = data.copy()
        if target_name is not None:
            data_x = data_x.drop(labels=target_name, axis=1)
            target = data[target_name]

        if method == "ken":
            data_aug = self._augment_with_ken(data, augment_col_name)
        elif method == "fasttext":
            data_aug = self._augment_with_fasttext(data, augment_col_name)

        if concat:
            data_aug = pd.concat([data_x, data_aug], axis=1)
            if drop_augment_col:
                data_aug.drop(columns=["augment_col_name"], inplace=True)

        data_aug = pd.concat([data_aug, target], axis=1)

        return data_aug

    def _augment_with_ken(
        self,
        data: pd.DataFrame,
        augment_col_name: str,
    ):
        mapping = data[augment_col_name].map(self.ken_embed_ent2idx)
        mapping = np.array(mapping)

        data_aug = self.ken_emb.iloc[mapping]
        data_aug = data_aug.reset_index(drop=True)
        data_aug = data_aug.drop(labels="Entity", axis=1)

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

        return data_aug
