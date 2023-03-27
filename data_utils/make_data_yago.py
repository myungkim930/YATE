import os
import pandas as pd
import numpy as np
import torch
import pickle

# Need to make changes with numberical values. For now, keep numerical=False.
# Need to fix for 2022 version


def make_data_yago(
    data_name: str, numerical: bool = False, save: bool = False, file_name=None
):

    data_folder_dir = os.getcwd() + "/data/raw/" + data_name + "/"

    triplet_cat_dir = data_folder_dir + "yagoFacts.tsv"
    triplet_cat = pd.read_csv(
        triplet_cat_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    )
    triplet_cat = triplet_cat.dropna()

    head_cat = triplet_cat["h"]
    tail_cat = triplet_cat["t"]
    rel_cat = triplet_cat["r"]

    head, tail, rel = head_cat, tail_cat, rel_cat

    # if numerical:
    #     triplet_num_dir = data_folder_dir + "yagoLiteralFacts.tsv"
    # triplet_num = pd.read_csv(
    #     triplet_num_dir,
    #     sep="\t",
    #     header=0,
    #     usecols=[1, 2, 3],
    #     names=["h", "r", "t"],
    # )
    # head_num = triplet_num["h"]
    #     tail_num = triplet_num["t"]
    #     rel_num = triplet_num["r"]

    #     head, tail, rel = head_cat, tail_cat, rel_cat

    # else:
    #     head, tail, rel = head_cat, tail_cat, rel_cat

    # Extracting edge_index
    head_name = pd.DataFrame(head.unique())
    tail_name = pd.DataFrame(tail.unique())
    ent_name = pd.concat([head_name, tail_name])[0]
    ent_name = pd.DataFrame(ent_name.unique())
    ent2idx = pd.concat(
        [ent_name, pd.DataFrame({"idx": np.arange(len(ent_name))})], axis=1
    )
    ent2idx = np.array(ent2idx)

    head = head.map(dict(ent2idx))
    tail = tail.map(dict(ent2idx))

    edge_index = np.stack([head, tail])
    edge_index = torch.tensor(edge_index)

    # Extracting edge_type
    rel_name = pd.DataFrame(rel.unique())
    rel_name.index = rel_name.index + 2
    rel_name.loc[0] = "<selfLoop>"
    rel_name.loc[1] = "<hasName>"
    rel_name = rel_name.sort_index()
    # rel_name.loc[38] = "<hasName>"

    rel2idx = pd.concat(
        [rel_name, pd.DataFrame({"idx": np.arange(len(rel_name))})], axis=1
    )
    rel2idx = np.array(rel2idx)
    rel = np.array(rel.map(dict(rel2idx)))

    edge_type = torch.tensor(rel)

    # Extracting types for head
    taxo_dir = (
        data_folder_dir + "yagoSimpleTaxonomy.tsv"
    )  # yagoSimpleTaxonomy yagoTaxonomy
    taxo = pd.read_csv(
        taxo_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    )

    taxo = taxo.dropna()
    taxo = taxo[taxo["r"] == "rdfs:subClassOf"]
    taxo = taxo[taxo["t"] != "owl:Thing"]

    taxomap = pd.concat([taxo["h"], taxo["t"]], axis=1)
    taxomap_wiki = taxomap[taxomap["h"].str.contains("wiki") == True]
    taxomap_wiki = np.array(taxomap_wiki)
    taxomap_wordnet = taxomap[taxomap["h"].str.contains("wordnet") == True]
    taxomap_wordnet = np.array(taxomap_wordnet)

    types_dir = data_folder_dir + "yagoSimpleTypes.tsv"  # yagoSimpleTypes yagoTypes
    types = pd.read_csv(
        types_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    )

    types = types.dropna()
    types_temp1 = types[types["t"].str.contains("wordnet") == True]
    types_temp2 = types[types["t"].str.contains("wiki") == True]
    types_temp = pd.concat([types_temp1, types_temp2], axis=0)
    types_temp = types_temp.drop_duplicates(subset=["h"])

    temp = pd.concat([types_temp["t"], types_temp["t"].map(dict(taxomap_wiki))], axis=1)
    temp.columns = ["0", "1"]
    temp["1"][temp["1"].isnull()] = temp["0"][temp["1"].isnull()]

    typemap_temp = pd.concat([types_temp["h"], temp["1"]], axis=1)
    typemap = typemap_temp
    # typemap = pd.concat([types["h"], types["t"]], axis=1)
    typemap = np.array(typemap)

    head2type = head_name[0].map(dict(typemap))
    # head2type = pd.concat([head2type, head2type.map(dict(taxomap_wiki))], axis=1)
    # head2type.columns = ["0", "1"]

    # head2type["1"][head2type["1"].isnull()] = head2type["0"][head2type["1"].isnull()]

    type_name = pd.unique(head2type)
    # type_name = pd.unique(head2type["1"])
    type2idx = pd.concat(
        [pd.DataFrame(type_name), pd.DataFrame({"idx": np.arange(len(type_name))})],
        axis=1,
    )
    type2idx = np.array(type2idx)

    headidx2type = np.stack(
        [head_name[0].map(dict(ent2idx)), head2type.map(dict(type2idx))]
    )
    headidx2type = torch.tensor(headidx2type)

    data = {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "ent2idx": ent2idx,
        "rel2idx": rel2idx,
        "headidx2type": headidx2type,
        "type2idx": type2idx,
    }

    if save:
        if file_name is not None:
            save_dir = (
                os.getcwd()
                + "/data/processed/"
                + data_name
                + str(file_name)
                + ".pickle"
            )
        else:
            save_dir = os.getcwd() + "/data/processed/" + data_name + ".pickle"

        with open(save_dir, "wb") as pickle_file:
            pickle.dump(data, pickle_file)

    # headidx2type[1].bincount()
    # (headidx2type[1].bincount()>64).nonzero().size()
    # type2idx[(headidx2type[1].bincount()>64).nonzero().view(-1)]
    ###

    # headidx2type = np.stack(
    #     [head_name[0].map(dict(ent2idx)), temp[0].map(dict(type2idx))]
    # )
    # headidx2type = torch.tensor(headidx2type)

    # len(pd.unique(head2type["1"]))

    # temp = pd.DataFrame(head_name[0].map(dict(temp_map)))

    # ##

    # # taxo_dir = data_folder_dir + "yagoTaxonomy.tsv"  # yagoSimpleTaxonomy yagoTaxonomy
    # # taxo = pd.read_csv(
    # #     taxo_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    # # )

    # # taxo = taxo.drop_duplicates(subset=["h"])
    # # taxo = taxo.dropna()

    # # taxo = taxo[taxo["r"] == "rdfs:subClassOf"]
    # # taxo = taxo[taxo["t"] != "owl:Thing"]

    # #

    # taxo_temp = taxo["t"].replace("wordnet_", "", regex=True)
    # # taxo_temp = taxo['t'].replace("wordnet_", "", regex=True)
    # taxomap = pd.concat([taxo_s["h"], taxo_s["t"]], axis=1)
    # taxomap = np.array(taxomap)

    # taxo_temp1 = pd.DataFrame(taxo_temp.map(dict(taxomap)))
    # taxo_temp = pd.DataFrame(taxo_temp)
    # taxo_temp[taxo_temp1["t"].notnull()] = taxo_temp1[taxo_temp1["t"].notnull()]
    # taxo_temp = taxo_temp[taxo_temp["t"].str.contains("wiki") == False]

    # taxo_ = (
    #     taxo_temp["t"]
    #     .replace("wordnet_", "", regex=True)
    #     .replace("_\d+", "", regex=True)
    #     .replace("yagoGeoEntity", "geological_entity", regex=True)
    # )

    # type2taxo1 = pd.concat([taxo["h"], taxo_], axis=1)
    # type2taxo2 = pd.concat([taxo_s["h"], taxo_s["t"]], axis=1)
    # type2taxo = pd.concat([type2taxo1, type2taxo2], axis=0)
    # type2taxo = type2taxo.drop_duplicates(subset=["h"])
    # type2taxo = np.array(type2taxo)

    # types_dir = data_folder_dir + "yagoSimpleTypes.tsv"  # yagoSimpleTypes yagoTypes
    # types = pd.read_csv(
    #     types_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    # )

    # types = types.drop_duplicates(subset=["h"])
    # types = types.dropna()

    # ent2taxo = pd.concat([types["h"], types["t"].map(dict(type2taxo))], axis=1)
    # ent2taxo = np.array(ent2taxo)

    # ###
    # temp_map = pd.concat([types["h"], types["t"]], axis=1)
    # temp_map = np.array(temp_map)

    # temp = pd.DataFrame(head_name[0].map(dict(temp_map)))

    # taxomap = pd.concat([taxo_s["h"], taxo_s["t"]], axis=1)
    # taxomap_wiki = taxomap[taxomap["h"].str.contains("wiki") == True]
    # taxomap_wiki = np.array(taxomap_wiki)

    # temp1 = pd.concat([temp[0], temp[0].map(dict(taxomap_wiki))], axis=1)
    # temp1.columns = ["0", "1"]

    # temp1["1"][temp1["1"].isnull()] = temp1["0"][temp1["1"].isnull()]

    # len(pd.unique(temp1["1"]))
    # any(temp1["1"].isna())

    # len(pd.unique(temp1[temp1["0"].str.contains("wiki") == False]["0"]))
    # len(pd.unique(temp1["1"]))

    # taxo_temp[taxo_temp1["t"].notnull()] = taxo_temp1[taxo_temp1["t"].notnull()]
    # taxo_temp = taxo_temp[taxo_temp["t"].str.contains("wiki") == False]

    # taxo_temp[taxo_temp1["t"].isnull()] = taxo_temp1[taxo_temp1["t"].notnull()]

    # temp = pd.DataFrame(head_name[0].map(dict(ent2taxo)))
    # temp[0][2] = "<person>"
    # temp1 = pd.DataFrame(temp[0].unique())
    # type2idx = pd.concat([temp1, pd.DataFrame({"idx": np.arange(len(temp1))})], axis=1)
    # type2idx = np.array(type2idx)

    # headidx2type = np.stack(
    #     [head_name[0].map(dict(ent2idx)), temp[0].map(dict(type2idx))]
    # )
    # headidx2type = torch.tensor(headidx2type)

    # temp1 = pd.DataFrame(temp[0].unique())

    # any(temp[0].isna())

    # a = taxo["t"]
    # a1 = pd.concat([taxo_s["h"], taxo_s["t"]], axis=1)
    # a1 = np.array(a1)

    # a2 = pd.DataFrame(a.map(dict(a1)))
    # a2.dropna()
    # pd.DataFrame(a2["t"].unique())

    # ent2type = np.array(ent2type)

    # types_dir = data_folder_dir + "yagoSimpleTypes.tsv"  # yagoSimpleTypes yagoTypes
    # types = pd.read_csv(
    #     types_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    # )

    # types = types.drop_duplicates(subset=["h"])
    # types = types.dropna()

    # type_name = pd.DataFrame(taxo["t"].unique())
    # type_name[0] = (
    #     type_name[0]
    #     .replace("wordnet_", "", regex=True)
    #     .replace("_\d+", "", regex=True)
    #     .replace("yagoGeoEntity", "geological_entity", regex=True)
    # )

    # type2idx = pd.concat(
    #     [type_name, pd.DataFrame({"idx": np.arange(len(type_name))})], axis=1
    # )
    # type2idx = np.array(type2idx)

    # type_map = pd.concat([taxo["h"], taxo["t"].map(dict(type2idx))], axis=1)
    # type_map = np.array(type_map)

    # types_dir = data_folder_dir + "yagoSimpleTypes.tsv"  # yagoSimpleTypes yagoTypes
    # types = pd.read_csv(
    #     types_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    # )

    # types = types.drop_duplicates(subset=["h"])
    # types = types.dropna()
    # ent2type = pd.concat([types["h"], types["t"].map(dict(type_map))], axis=1)
    # ent2type = np.array(ent2type)

    # headidx2type = np.stack(
    #     [head_name[0].map(dict(ent2idx)), head_name[0].map(dict(ent2type))]
    # )
    # headidx2type = torch.tensor(headidx2type)

    # idx_head = edge_index[0, :].unique()
    # head_, tail_ = edge_index[0], edge_index[1]
    # num_hops = 3
    # neighbor = dict({f"{i}_hop": dict() for i in range(1, num_hops + 1)})

    # for idx in idx_head:
    #     subset = idx.clone()
    #     node_mask = head_.new_empty(edge_index.max().item() + 1, dtype=torch.bool)
    #     node_mask.fill_(False)
    #     for i in range(num_hops):
    #         node_mask[subset] = True
    #         reduce_mask = node_mask[head_]
    #         neighbor[f"{i+1}_hop"][str(int(idx))] = head_[reduce_mask].unique()
    #         subset = tail_[reduce_mask].unique()
    #     del reduce_mask

    # return data
