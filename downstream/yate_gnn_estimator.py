""" YATE GNN regressor and classifier class
"""

import torch
import copy
import numpy as np

from typing import Union

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from model import YATE_FinetuneCls, YATE_FinetuneReg
from joblib import Parallel, delayed

from utils import load_config


######################################################
class Yate_FineTune_Regressor:
    def __init__(
        self,
        num_layers: int = 0,
        include_numeric: bool = True,
        load_pretrain: bool = True,
        freeze: bool = True,
        loss: str = "squared_error",
        learning_rate: float = 1e-4,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 10,
        tol: Union[None, float] = None,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        gpu_id: int = 0,
        n_jobs: int = 1,
        disable_pbar: bool = True,
    ):
        # Model Settings
        self.num_layers = num_layers
        self.include_numeric = include_numeric
        self.load_pretrain = load_pretrain
        self.freeze = freeze

        # Experiment Settings
        if loss == "squared_error":
            self.criterion = torch.nn.MSELoss()
        elif loss == "absolute_error":
            self.criterion = torch.nn.L1Loss()
        self.loss = loss
        self.lr = learning_rate
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_model = num_model
        self.tol = tol
        self.max_epoch = max_epoch
        self.patience = early_stopping_patience
        if early_stopping_patience is None:
            self.patience = max_epoch + 1

        # Device Settings
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.n_jobs = n_jobs
        self.disable_pbar = disable_pbar

        # Other
        self.make_batch = Batch()
        self.config = load_config()

    def fit(self, dataset_train: list, dataset_valid: Union[list, None] = None):
        input_numeric_dim = dataset_train[0].x_num.size(1)
        output_dim = 1
        model_base = self._load_model(
            input_numeric_dim=input_numeric_dim, output_dim=output_dim
        )
        model_base.to(self.device)

        if dataset_valid is None:
            result = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_train)(dataset_train, dataset_valid, model_base, i)
                for i in range(self.num_model)
            )
            # result_valid_loss = [result[i]["valid_loss"] for i in range(self.num_model)]
            result_train_loss = [result[i]["train_loss"] for i in range(self.num_model)]
            # self.valid_loss_best = min(result_valid_loss)
            # self.valid_loss_mean = np.mean(result_valid_loss)
            self.train_loss_best = min(result_train_loss)
            self.train_loss_mean = np.mean(result_train_loss)
            num_train = int(len(dataset_train) * (1 - self.val_size))
            self.train_loss_std = np.std(result_train_loss) / np.sqrt(num_train)
            self._run_refit(dataset_train, model_base)
        else:
            self._run_train(
                dataset_train=dataset_train,
                dataset_valid=dataset_valid,
                model=model_base,
                idx_model=0,
            )

    def predict(self, dataset: list, format: str = "numpy", return_y: bool = False):
        with torch.no_grad():
            data_eval = self.make_batch.from_data_list(
                dataset, follow_batch=["edge_index"]
            )
            data_eval.head_idx = data_eval.ptr[:-1]
            data_eval.to(self.device)
            self.model_best.eval()
            out = self.model_best(data_eval)
            target = data_eval.y
            if format == "numpy":
                out = out.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
        if return_y:
            return out, target
        else:
            return out

    def _run_refit(self, dataset_train: list, model_base: torch.nn.Module):
        train_loader = DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True
        )
        model_run_refit = copy.deepcopy(model_base)
        self.optimizer = torch.optim.AdamW(model_run_refit.parameters(), lr=self.lr)
        max_epoch = 500
        if self.tol is None:
            self.tol = 1.96 * self.train_loss_std
        train_loss_best = 9e15
        for ep in tqdm(
            range(1, max_epoch + 1), desc="Refit", disable=self.disable_pbar
        ):
            self._run_epoch(model_run_refit, train_loader)
            train_loss = self._eval(model_run_refit, dataset_train)
            if train_loss < train_loss_best:
                train_loss_best = train_loss
                self.model_best = model_run_refit
                es_counter = 0
                if (train_loss - self.train_loss_mean) < self.tol:
                    break
            else:
                es_counter += 1
                if es_counter > self.patience:
                    break
            if ep == max_epoch:
                self.model_best = model_run_refit

    def _run_train(
        self,
        dataset_train: list,
        dataset_valid: Union[list, None],
        model: torch.nn.Module,
        idx_model: int,
    ):
        # In the case of no validation set, the train is divided by val_size
        if dataset_valid is None:
            idx_train, idx_valid = train_test_split(
                np.arange(0, len(dataset_train)),
                test_size=self.val_size,
                shuffle=True,
            )
            ds_train = [dataset_train[i] for i in idx_train]
            ds_valid = [dataset_train[i] for i in idx_valid]
        else:
            ds_train = dataset_train
            ds_valid = dataset_valid

        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        model_run_train = copy.deepcopy(model)
        self.optimizer = torch.optim.AdamW(model_run_train.parameters(), lr=self.lr)
        valid_loss_best = 9e15
        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. {idx_model}",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_train, train_loader)
            valid_loss = self._eval(model_run_train, ds_valid)
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                train_loss_best = self._eval(model_run_train, ds_train)
                if dataset_valid is not None:
                    self.model_best = copy.deepcopy(model_run_train)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > self.patience:
                    break

        result_run = dict()
        result_run["valid_loss"] = valid_loss_best
        result_run["train_loss"] = train_loss_best
        return result_run

    def _run_epoch(self, model: torch.nn.Module, train_loader):
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(self.device)
            data.head_idx = data.ptr[:-1]
            out = model(data)  # Perform a single forward pass.
            out = out
            target = data.y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss = self.criterion(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def _eval(self, model: torch.nn.Module, dataset: list):
        with torch.no_grad():
            data_eval = self.make_batch.from_data_list(
                dataset, follow_batch=["edge_index"]
            )
            data_eval.head_idx = data_eval.ptr[:-1]
            data_eval.to(self.device)
            model.eval()
            out = model(data_eval)
            target = data_eval.y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss_eval = self.criterion(out, target)
            loss_eval = round(loss_eval.detach().item(), 4)
        return loss_eval

    def _load_model(self, input_numeric_dim: int, output_dim: int):
        model_config = dict()
        model_config["input_dim_x"] = 300
        model_config["input_dim_e"] = 300
        model_config["hidden_dim"] = 300
        model_config["ff_dim"] = 300
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers
        model_config["input_numeric_dim"] = input_numeric_dim
        model_config["output_dim"] = output_dim
        model_config["include_numeric"] = self.include_numeric

        model = YATE_FinetuneReg(**model_config)

        if self.load_pretrain:
            dir_model = self.config["pretrained_model_dir"]
            model.load_state_dict(
                torch.load(dir_model, map_location=self.device), strict=False
            )
        if self.freeze:
            for param in model.ft_base.read_out_block.parameters():
                param.requires_grad = False
            for param in model.ft_base.layers.parameters():
                param.requires_grad = False
        return model


######################################################
class Yate_FineTune_Classifier:
    def __init__(
        self,
        num_layers: int = 0,
        include_numeric: bool = True,
        load_pretrain: bool = True,
        freeze: bool = True,
        loss: str = "binary_crossentropy",
        learning_rate: float = 1e-4,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 10,
        tol: Union[None, float] = None,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        gpu_id: int = 0,
        n_jobs: int = 5,
        disable_pbar: bool = True,
    ):
        # Model Settings
        self.num_layers = num_layers
        self.include_numeric = include_numeric
        self.load_pretrain = load_pretrain
        self.freeze = freeze

        # Experiment Settings
        if loss == "binary_crossentropy":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif loss == "categorical_crossentropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        self.loss = loss
        self.lr = learning_rate
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_model = num_model
        self.tol = tol
        self.max_epoch = max_epoch
        self.patience = early_stopping_patience
        if early_stopping_patience is None:
            self.patience = max_epoch + 1

        # Device Settings
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.n_jobs = n_jobs
        self.disable_pbar = disable_pbar

        # Other
        self.make_batch = Batch()
        self.config = load_config()

    def fit(self, dataset_train: list, dataset_valid: Union[list, None] = None):
        input_numeric_dim = dataset_train[0].x_num.size(1)
        data_temp = self.make_batch.from_data_list(dataset_train)
        self.y_temp = data_temp.y
        output_dim = data_temp.y.unique().size(0)
        if output_dim == 2:
            output_dim -= 1
            self.loss = "binary_crossentropy"
            self.criterion = torch.nn.BCEWithLogitsLoss()

        model_base = self._load_model(
            input_numeric_dim=input_numeric_dim, output_dim=output_dim
        )
        model_base.to(self.device)

        if dataset_valid is None:
            result = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_train)(dataset_train, model_base, i)
                for i in range(self.num_model)
            )
            # result_valid_loss = [result[i]["valid_loss"] for i in range(self.num_model)]
            result_train_loss = [result[i]["train_loss"] for i in range(self.num_model)]
            # self.valid_loss_best = min(result_valid_loss)
            # self.valid_loss_mean = np.mean(result_valid_loss)
            self.train_loss_best = min(result_train_loss)
            self.train_loss_mean = np.mean(result_train_loss)
            num_train = int(len(dataset_train) * (1 - self.val_size))
            self.train_loss_std = np.std(result_train_loss) / np.sqrt(num_train)
            self._run_refit(dataset_train, model_base)
        else:
            self._run_train(
                dataset_train=dataset_train,
                dataset_valid=dataset_valid,
                model=model_base,
                idx_model=0,
            )

    def predict(self, dataset: list, format: str = "torch", return_y: bool = False):
        with torch.no_grad():
            data_eval = self.make_batch.from_data_list(
                dataset, follow_batch=["edge_index"]
            )
            data_eval.head_idx = data_eval.ptr[:-1]
            data_eval.to(self.device)
            self.model_best.eval()
            out = self.model_best(data_eval)
            target = data_eval.y
        if self.loss == "binary_crossentropy":
            out = torch.sigmoid(out)
        elif self.loss == "categorical_crossentropy":
            out = torch.softmax(out, dim=1)
        if format == "numpy":
            out = out.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
        if return_y:
            return out, target
        else:
            return out

    def _run_refit(self, dataset_train: list, model_base: torch.nn.Module):
        train_loader = DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True
        )
        model_run_refit = copy.deepcopy(model_base)
        self.optimizer = torch.optim.AdamW(model_run_refit.parameters(), lr=self.lr)
        max_epoch = 500
        if self.tol is None:
            self.tol = 1.96 * self.train_loss_std
        train_loss_best = 9e15
        for ep in tqdm(
            range(1, max_epoch + 1), desc="Refit", disable=self.disable_pbar
        ):
            self._run_epoch(model_run_refit, train_loader)
            train_loss = self._eval(model_run_refit, dataset_train)
            if train_loss < train_loss_best:
                train_loss_best = train_loss
                self.model_best = model_run_refit
                es_counter = 0
                if (train_loss - self.train_loss_mean) < self.tol:
                    break
            else:
                es_counter += 1
                if es_counter > self.patience:
                    break
            if ep == max_epoch:
                self.model_best = model_run_refit

    def _run_train(
        self,
        dataset_train: list,
        dataset_valid: Union[list, None],
        model: torch.nn.Module,
        idx_model: int,
    ):
        # In the case of no validation set, the train is divided by val_size
        if dataset_valid is None:
            idx_train, idx_valid = train_test_split(
                np.arange(0, len(dataset_train)),
                test_size=self.val_size,
                shuffle=True,
                stratify=self.y_temp,
            )
            ds_train = [dataset_train[i] for i in idx_train]
            ds_valid = [dataset_train[i] for i in idx_valid]
        else:
            ds_train = dataset_train
            ds_valid = dataset_valid

        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        model_run_train = copy.deepcopy(model)
        self.optimizer = torch.optim.AdamW(model_run_train.parameters(), lr=self.lr)
        valid_loss_best = 9e15
        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. {idx_model}",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_train, train_loader)
            valid_loss = self._eval(model_run_train, ds_valid)
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                train_loss_best = self._eval(model_run_train, ds_train)
                if dataset_valid is not None:
                    self.model_best = copy.deepcopy(model_run_train)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > self.patience:
                    break
        result_run = dict()
        result_run["valid_loss"] = valid_loss_best
        result_run["train_loss"] = train_loss_best
        return result_run

    def _run_epoch(self, model: torch.nn.Module, train_loader):
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(self.device)
            data.head_idx = data.ptr[:-1]
            out = model(data)  # Perform a single forward pass.
            out = out
            target = data.y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss = self.criterion(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def _eval(self, model: torch.nn.Module, dataset: list):
        with torch.no_grad():
            data_eval = self.make_batch.from_data_list(
                dataset, follow_batch=["edge_index"]
            )
            data_eval.head_idx = data_eval.ptr[:-1]
            data_eval.to(self.device)
            model.eval()
            out = model(data_eval)
            target = data_eval.y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss_eval = self.criterion(out, target)
            loss_eval = round(loss_eval.detach().item(), 4)
        return loss_eval

    def _load_model(self, input_numeric_dim: int, output_dim: int):
        model_config = dict()
        model_config["input_dim_x"] = 300
        model_config["input_dim_e"] = 300
        model_config["hidden_dim"] = 300
        model_config["ff_dim"] = 300
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers
        model_config["input_numeric_dim"] = input_numeric_dim
        model_config["output_dim"] = output_dim
        model_config["include_numeric"] = self.include_numeric

        model = YATE_FinetuneCls(**model_config)

        if self.load_pretrain:
            dir_model = self.config["pretrained_model_dir"]
            model.load_state_dict(
                torch.load(dir_model, map_location=self.device), strict=False
            )
        if self.freeze:
            for param in model.ft_base.read_out_block.parameters():
                param.requires_grad = False
            for param in model.ft_base.layers.parameters():
                param.requires_grad = False
        return model
