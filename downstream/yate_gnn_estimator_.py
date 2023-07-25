"""
YATE-GNN estimator
"""

import torch
import numpy as np
import copy
from typing import Union
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.preprocessing import power_transform, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from model import YATE_GNNModel_Reg, YATE_GNNModel_Cls
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import load_config


class BaseYateGNNEstimator(BaseEstimator):
    """Base class for Yate Graph Neural Network."""

    def __init__(
        self,
        *,
        num_layers,
        include_numeric,
        load_pretrain,
        freeze_pretrain,
        learning_rate,
        batch_size,
        val_size,
        num_model,
        max_epoch,
        early_stopping_patience,
        n_jobs,
        device,
        random_state,
        disable_pbar,
    ):
        self.num_layers = num_layers
        self.include_numeric = include_numeric
        self.load_pretrain = load_pretrain
        self.freeze_pretrain = freeze_pretrain
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_model = num_model
        self.max_epoch = max_epoch
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state
        self.disable_pbar = disable_pbar

    def fit(self, X, y):
        # Preliminary settings
        self.is_fitted_ = False
        self.device_ = torch.device(self.device)
        self.X_ = X
        self.y_ = y
        self._set_task_specific_settings()

        if self.include_numeric:
            if X[0].x_num.size(1) == 0:
                raise ValueError(
                    "No numerical data found in the dataset. Please check the 'include_numeric' argument."
                )
            else:
                pass
            # Transform numericals
            # self.num_transformer_ = PowerTransformer()
            X = self._transform_numerical(X)

        if self.num_model == 1:
            _ = self._run_train(X, self.random_state)
        else:
            random_state = check_random_state(self.random_state)
            random_state_list = [
                random_state.randint(1000) for _ in range(self.num_model)
            ]
            result_valid_loss = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_train)(X, rs) for rs in random_state_list
            )
            self._run_refit(X, result_valid_loss)

        self.is_fitted_ = True

        return self

    def _run_train(self, X, random_state):
        # Input dimension for numerical values
        input_numeric_dim = X[0].x_num.size(1)

        # Set validation by val_size
        idx_train, idx_valid = train_test_split(
            np.arange(0, len(X)),
            test_size=self.val_size,
            shuffle=True,
            random_state=random_state,
        )
        ds_train = [X[i] for i in idx_train]
        ds_valid = [X[i] for i in idx_valid]

        # Set validation batch for evaluation
        ds_valid_eval = self._set_data_eval(data=ds_valid)

        # Load model and optimizer
        model_run_train = self._load_model(input_numeric_dim)
        model_run_train.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_train.parameters(), lr=self.learning_rate
        )

        # Train model
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        valid_loss_best = 9e15
        if self.early_stopping_patience is None:
            patience = 200
        else:
            patience = self.early_stopping_patience

        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. {random_state}",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_train, optimizer, train_loader)
            valid_loss = self._eval(model_run_train, ds_valid_eval)
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                if self.num_model == 1:
                    self.model_best_ = copy.deepcopy(model_run_train)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > patience:
                    break
        return valid_loss_best

    def _run_refit(self, X: list, result_valid_loss):
        # Load model and optimizer
        input_numeric_dim = X[0].x_num.size(1)
        model_run_refit = self._load_model(input_numeric_dim=input_numeric_dim)
        model_run_refit.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_refit.parameters(), lr=self.learning_rate
        )

        # Statistics for stopping criterion in the refit
        # valid_loss_best = min(result_valid_loss)
        valid_loss_mean = np.mean(result_valid_loss)
        #     np.sort(result_valid_loss)[: int(self.num_model * 0.6)]
        # )
        valid_loss_std = np.std(result_valid_loss) / np.sqrt(self.num_model)

        # Set train settings
        max_epoch = 1000
        tol = 1.85 * valid_loss_std

        # Set train batch for loss evaluation
        ds_train_eval = self._set_data_eval(data=X)

        # Train
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        train_loss_best = 9e15
        for ep in tqdm(
            range(1, max_epoch + 1),
            desc="Refit",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_refit, optimizer, train_loader)
            train_loss = self._eval(model_run_refit, ds_train_eval)
            if train_loss < train_loss_best:
                train_loss_best = train_loss
                self.model_best_ = model_run_refit
                es_counter = 0
                if (train_loss - valid_loss_mean) < -tol:
                    break
            else:
                es_counter += 1
                if es_counter > max_epoch:  # self.patience:
                    break
            if ep == max_epoch and es_counter == 0:
                self.model_best_ = model_run_refit

    def _run_epoch(self, model: torch.nn.Module, optimizer, train_loader):
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(self.device_)
            data.head_idx = data.ptr[:-1]
            out = model(data)  # Perform a single forward pass.
            target = data.y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss = self.criterion_(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def _eval(self, model: torch.nn.Module, ds_eval):
        with torch.no_grad():
            model.eval()
            out = model(ds_eval)
            target = ds_eval.y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss_eval = self.criterion_(out, target)
            loss_eval = round(loss_eval.detach().item(), 4)
        return loss_eval

    def _set_data_eval(self, data: list):
        make_batch = Batch()
        with torch.no_grad():
            ds_eval = make_batch.from_data_list(data, follow_batch=["edge_index"])
            ds_eval.head_idx = ds_eval.ptr[:-1]
            ds_eval.to(self.device_)
        return ds_eval

    def _set_task_specific_settings(self):
        self.criterion_ = None
        self.output_dim_ = None
        self.model_task_ = None

    def _transform_numerical(self, X):
        make_batch = Batch()
        data_batch = make_batch.from_data_list(X, follow_batch=["edge_index"])
        X_num = data_batch.x_num.cpu().detach().numpy()
        X_num = X_num.astype(np.float64)
        # if self.is_fitted_ == False:
        #     X_num = self.num_transformer_.fit_transform(X_num)
        # else:
        #     X_num = self.num_transformer_.transform(X_num)
        X_num = power_transform(X_num)
        X_num = torch.tensor(X_num)
        X_num = torch.nan_to_num(X_num, nan=0)
        data_batch.x_num = X_num
        return data_batch.to_data_list()

    def _load_model(self, input_numeric_dim: int):
        model_config = dict()
        model_config["input_dim_x"] = 300
        model_config["input_dim_e"] = 300
        model_config["hidden_dim"] = 300
        model_config["ff_dim"] = 300
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers
        model_config["input_numeric_dim"] = input_numeric_dim
        model_config["output_dim"] = self.output_dim_
        model_config["include_numeric"] = self.include_numeric
        if input_numeric_dim == 0:
            model_config["include_numeric"] = False

        if self.model_task_ == "regression":
            model = YATE_GNNModel_Reg(**model_config)
        else:
            model = YATE_GNNModel_Cls(**model_config)

        if self.load_pretrain:
            config = load_config()
            dir_model = config["pretrained_model_dir"]
            model.load_state_dict(
                torch.load(dir_model, map_location=self.device_), strict=False
            )
        if self.freeze_pretrain:
            for param in model.ft_base.read_out_block.parameters():
                param.requires_grad = False
            for param in model.ft_base.layers.parameters():
                param.requires_grad = False
        return model


class YateGNNRegressor(RegressorMixin, BaseYateGNNEstimator):
    """Yate Graph Neural Network for Regression.

    This estimator is GNN model compatible with the YATE pretrained model

    """

    def __init__(
        self,
        *,
        loss: str = "absolute_error",
        num_layers: int = 0,
        include_numeric: bool = True,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 10,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 40,
        n_jobs: int = 1,
        device="cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(YateGNNRegressor, self).__init__(
            num_layers=num_layers,
            include_numeric=include_numeric,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")

        if self.include_numeric:
            # Transform numericals
            X_total = self.X_ + X
            X = self._transform_numerical(X_total)
            X = X[len(self.X_) :]

        # Obtain the batch to feed into the network
        ds_predict_eval = self._set_data_eval(data=X)

        # Obtain the predicitve output
        with torch.no_grad():
            self.model_best_.eval()
            out = self.model_best_(ds_predict_eval)
        out = out.cpu().detach().numpy()
        return out

    def _set_task_specific_settings(self):
        if self.loss == "squared_error":
            self.criterion_ = torch.nn.MSELoss()
        elif self.loss == "absolute_error":
            self.criterion_ = torch.nn.L1Loss()

        self.output_dim_ = 1
        self.model_task_ = "regression"


class YateGNNClassifier(ClassifierMixin, BaseYateGNNEstimator):
    """Yate Graph Neural Network for Classification.

    This estimator is GNN model compatible with the YATE pretrained model

    """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        num_layers: int = 0,
        include_numeric: bool = True,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 10,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 40,
        n_jobs: int = 1,
        device="cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(YateGNNClassifier, self).__init__(
            num_layers=num_layers,
            include_numeric=include_numeric,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        return self._get_predict_prob(X)

    def _get_predict_prob(self, X):
        if self.include_numeric:
            # Transform numericals
            # X_total = self.X_ + X
            X = self._transform_numerical(X)
            # X = X[len(self.X_) :]

        # Obtain the batch to feed into the network
        ds_predict_eval = self._set_data_eval(data=X)

        # Obtain the predicitve output
        with torch.no_grad():
            self.model_best_.eval()
            out = self.model_best_(ds_predict_eval)
        if self.loss == "binary_crossentropy":
            out = torch.sigmoid(out)
        elif self.loss == "categorical_crossentropy":
            out = torch.softmax(out, dim=1)
        out = out.cpu().detach().numpy()
        return out

    def _set_task_specific_settings(self):
        if self.loss == "binary_crossentropy":
            self.criterion_ = torch.nn.BCEWithLogitsLoss()
        elif self.loss == "categorical_crossentropy":
            self.criterion_ = torch.nn.CrossEntropyLoss()

        self.output_dim_ = len(np.unique(self.y_))
        if self.output_dim_ == 2:
            self.output_dim_ -= 1
            self.criterion_ = torch.nn.BCEWithLogitsLoss()

        self.model_task_ = "classification"
