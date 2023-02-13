"""
Loading data or saved models
"""

# Python
import pickle
import fasttext
import os
import numpy as np

# Pytorch
import torch

## Load data and place it as an object
class Load_data:
    def __init__(
        self,
        data_name: str,
        x_model: str = "fasttext",
    ):

        super(Load_data, self).__init__()

        data_dir = os.getcwd() + "/data/preprocessed/" + data_name + ".pickle"

        with open(data_dir, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        self.__dict__ = data

        if x_model == "fasttext":
            self.x_model = fasttext.load_model(
                "/storage/store3/work/mkim/gitlab/cc.en.300.bin"
            )
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


## Loading Pretrained models
def load_pretrained_model(model_name: str, fine_tune: bool = False):

    model_dir = os.getcwd() + "/saved_model/" + model_name + ".pt"
    model = torch.load(model_dir)
    model.eval()

    if fine_tune == False:
        model.classifier = Identity()

    return model


# Identity class for replacing the classifier layer
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
