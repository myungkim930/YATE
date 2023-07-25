"""
YATE-GNN estimator
"""

import torch
import numpy as np
import copy
from typing import Union
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.preprocessing import power_transform
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from model import YATE_GNNModel_Reg, YATE_GNNModel_Cls
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import load_config



class YateFinetune(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        self.y_ = y
        return self

    def transform(self, X, y=None):


