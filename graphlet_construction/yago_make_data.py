""" Function for making yago data suitable for pretraining step
"""

import pandas as pd
import numpy as np
import torch
import pickle

from utils import load_config


def make_data_yago(
    data_name: str,
    numerical: bool = False,
    save: bool = False,
):
    config = load_config()
    data_folder_dir = config["data_kg_dir"] + "raw/" + data_name + "/"

    triplet_cat_dir = data_folder_dir + "yagoFacts.tsv"
    triplet_cat = pd.read_csv(
        triplet_cat_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    )
    triplet_cat = triplet_cat.dropna()

    head_cat = triplet_cat["h"]
    tail_cat = triplet_cat["t"]
    rel_cat = triplet_cat["r"]

    triplet_num_dir = data_folder_dir + "yagoLiteralFacts.tsv"
    triplet_num = pd.read_csv(
        triplet_num_dir,
        sep="\t",
        header=0,
        usecols=[1, 2, 3],
        names=["h", "r", "t"],
    )
    triplet_num = triplet_num.dropna()
    triplet_num = triplet_num[triplet_num["r"] != "rdfs:comment"]

    triplet_num_cat = triplet_num[triplet_num["t"].str.contains("\^") == False]
    head_cat = pd.concat([head_cat, triplet_num_cat["h"]])
    tail_cat = pd.concat([tail_cat, triplet_num_cat["t"]])
    rel_cat = pd.concat([rel_cat, triplet_num_cat["r"]])

    triplet_num = triplet_num[triplet_num["t"].str.contains("\^") == True]

    triplet_num = triplet_num.copy()
    tail_num = triplet_num["t"].str.partition("^^")
    tail_num = tail_num[0]
    tail_num = tail_num.astype("float")
    rel_num_temp = pd.unique(triplet_num["r"])

    for rel in rel_num_temp:
        tail_temp = tail_num[triplet_num["r"] == rel]
        tail_temp = pd.qcut(
            tail_temp.rank(method="first"), q=125, labels=False, duplicates="drop"
        )
        triplet_num["t"][triplet_num["r"] == rel] = tail_temp

    triplet_num["t"] = triplet_num["t"].astype("int")
    triplet_num["t"] = "q " + triplet_num["t"].astype("str")
    triplet_num["t"] = triplet_num["t"].astype("object")

    head_num = triplet_num["h"]
    tail_num = triplet_num["t"]
    rel_num = triplet_num["r"]

    if numerical:
        head = pd.concat([head_cat, head_num])
        tail = pd.concat([tail_cat, tail_num])
        rel = pd.concat([rel_cat, rel_num])
    else:
        head, tail, rel = head_cat, tail_cat, rel_cat

    # Extracting edge_index
    head_name = pd.DataFrame(head.unique())
    tail_name = pd.DataFrame(tail.unique())
    ent_name = pd.concat([head_name, tail_name])[0]
    ent_name = pd.DataFrame(ent_name.unique())
    ent2idx = pd.concat(
        [ent_name, pd.DataFrame({"idx": np.arange(len(ent_name))})], axis=1
    )
    ent2idx = np.array(ent2idx)
    ent2idx_original = ent2idx.copy()
    head = head.map(dict(ent2idx))
    tail = tail.map(dict(ent2idx))
    edge_index = np.stack([head, tail])
    edge_index = torch.tensor(edge_index)

    # Extracting edge_type
    rel_name = pd.DataFrame(rel.unique())
    rel_name.index = rel_name.index + 1
    rel_name.loc[0] = "<hasName>"
    rel_name = rel_name.sort_index()
    rel2idx = pd.concat(
        [rel_name, pd.DataFrame({"idx": np.arange(len(rel_name))})], axis=1
    )
    rel2idx = np.array(rel2idx)
    rel = np.array(rel.map(dict(rel2idx)))
    edge_type = torch.tensor(rel)

    # Preprocess the names of entities and relations
    ent2idx = pd.DataFrame(ent2idx)
    ent2idx = ent2idx[0]
    ent2idx = (
        ent2idx.str.replace("[a-z]{2,}/", "", regex=True)
        .str.replace("_", " ")
        .str.replace("<", "")
        .str.replace(">", "")
        .str.replace("\n", "")
        .str.lower()
    )
    ent2idx = np.array(ent2idx)
    rel2idx = pd.DataFrame(rel2idx)
    rel2idx = rel2idx[0]
    rel2idx = (
        rel2idx.str.replace(r"\B([A-Z][a-z])", r" \1", regex=True)
        .str.replace(r"\B([A-Z][A-Z])", r" \1", regex=True)
        .str.replace("<", "")
        .str.replace(">", "")
        .str.replace("\n", "")
        .str.lower()
    )
    rel2idx = np.array(rel2idx)

    data = {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "ent2idx": ent2idx,
        "ent2idx_original": ent2idx_original,
        "rel2idx": rel2idx,
    }

    if numerical:
        data_name = data_name + "_num"

    if save:
        save_dir = config["data_kg_dir"] + "/processed/" + data_name + ".pickle"
        with open(save_dir, "wb") as pickle_file:
            pickle.dump(data, pickle_file)
