"""
Load YAGO as a class object
"""

# Python
import pickle
import fasttext
import numpy as np

from utils import load_config


class Load_Yago:
    def __init__(
        self,
        data_name: str,
        numerical: bool = True,
    ):
        super(Load_Yago, self).__init__()

        config = load_config()

        if numerical:
            data_name += "_num"
        data_dir = config["data_kg_dir"] + "processed/" + data_name + ".pickle"

        with open(data_dir, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        self.__dict__ = data

        self.x_model = fasttext.load_model(config["fasttext_dir"])

        if isinstance(data["ent2idx"], list):
            self.ent2idx = np.array(data["ent2idx"])
        else:
            self.ent2idx = data["ent2idx"]
        if isinstance(data["rel2idx"], list):
            self.rel2idx = np.array(data["rel2idx"])
        else:
            self.rel2idx = data["rel2idx"]

        if len(np.shape(self.ent2idx)) > 1:
            self.ent2idx = self.ent2idx[:, 0]
            self.rel2idx = self.rel2idx[:, 0]

        self.data_name = data_name

    def __repr__(self):
        return f"KG: {self.data_name}"
