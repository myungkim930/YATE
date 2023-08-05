import torch
import numpy as np
import copy
from typing import Union
from torch import Tensor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_random_state
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed
from model import SimpleMLP


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BaseSimpleMLPEstimator(BaseEstimator):
    """Base class for Simple MLP."""

    def __init__(
        self,
        *,
        num_layers,
        hidden_dim,
        dropout_prob,
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
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
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

        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, Tensor) == False:
            y = torch.tensor(y, dtype=torch.float32)

        self.X_mean_ = torch.nanmean(X, dim=0)
        for i in range(X.size(1)):
            X[:, i] = torch.nan_to_num(X[:, i], nan=self.X_mean_[i])

        if self.num_model == 1:
            _ = self._run_train(X, y, self.random_state)
        else:
            random_state = check_random_state(self.random_state)
            random_state_list = [
                random_state.randint(1000) for _ in range(self.num_model)
            ]
            result_valid_loss = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_train)(X, y, rs) for rs in random_state_list
            )
            self._run_refit(X, y, result_valid_loss)

        self.is_fitted_ = True
        return self

    def _run_train(self, X, y, random_state):
        input_dim = X.size(1)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.val_size,
            shuffle=True,
            random_state=random_state,
        )

        ds_train = TabularDataset(X_train, y_train)

        # Load model and optimizer
        model_run_train = self._load_model(input_dim)
        model_run_train.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_train.parameters(), lr=self.learning_rate
        )

        # Train model
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        valid_loss_best = 9e15
        es_counter = 0
        if self.early_stopping_patience is None:
            patience = self.max_epoch
        else:
            patience = self.early_stopping_patience

        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. {random_state}",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_train, optimizer, train_loader)
            valid_loss = self._eval(model_run_train, X_valid, y_valid)
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

    def _run_refit(self, X, y, result_valid_loss):
        # Load model and optimizer
        input_dim = X.size(1)
        model_run_refit = self._load_model(input_dim=input_dim)
        model_run_refit.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_refit.parameters(), lr=self.learning_rate
        )

        # Statistics for stopping criterion in the refit
        valid_loss_mean = np.mean(result_valid_loss)
        valid_loss_std = np.std(result_valid_loss) / np.sqrt(self.num_model)
        # valid_loss_best = min(result_valid_loss)

        # Set train settings
        max_epoch = 5000
        tol = 1.85 * valid_loss_std

        ds_train = TabularDataset(X, y)

        # Train
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        train_loss_best = 9e15
        for ep in tqdm(
            range(1, max_epoch + 1),
            desc="Refit",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_refit, optimizer, train_loader)
            train_loss = self._eval(model_run_refit, X, y)
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
        for data_X, data_y in train_loader:
            data_X = data_X.to(self.device_)
            data_y = data_y.to(self.device_)
            out = model(data_X)  # Perform a single forward pass.
            target = data_y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss = self.criterion_(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def _eval(self, model: torch.nn.Module, X, y):
        X = X.to(self.device_)
        y = y.to(self.device_)
        with torch.no_grad():
            model.eval()
            out = model(X)
            target = y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss_eval = self.criterion_(out, target)
            loss_eval = round(loss_eval.detach().item(), 4)
        return loss_eval

    def _set_task_specific_settings(self):
        self.criterion_ = None
        self.output_dim_ = None
        self.model_task_ = None

    def _load_model(self, input_dim: int):
        model_config = dict()
        model_config["input_dim"] = input_dim
        model_config["hidden_dim"] = self.hidden_dim
        model_config["output_dim"] = self.output_dim_
        model_config["dropout_prob"] = self.dropout_prob
        model_config["num_layers"] = self.num_layers
        model = SimpleMLP(**model_config)
        return model


class SimpleMLPRegressor(RegressorMixin, BaseSimpleMLPEstimator):
    """ """

    def __init__(
        self,
        *,
        loss: str = "absolute_error",
        num_layers: int = 4,
        hidden_dim: int = 256,
        dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 40,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(SimpleMLPRegressor, self).__init__(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
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
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        for i in range(X.size(1)):
            X[:, i] = torch.nan_to_num(X[:, i], nan=self.X_mean_[i])
        X = X.to(self.device_)

        # Obtain the predicitve output
        with torch.no_grad():
            self.model_best_.eval()
            out = self.model_best_(X)
        out = out.cpu().detach().numpy()
        return out

    def _set_task_specific_settings(self):
        if self.loss == "squared_error":
            self.criterion_ = torch.nn.MSELoss()
        elif self.loss == "absolute_error":
            self.criterion_ = torch.nn.L1Loss()

        self.output_dim_ = 1
        self.model_task_ = "regression"


class SimpleMLPClassifier(ClassifierMixin, BaseSimpleMLPEstimator):
    """ """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        num_layers: int = 4,
        hidden_dim: int = 256,
        dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 40,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(SimpleMLPClassifier, self).__init__(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
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
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        for i in range(X.size(1)):
            X[:, i] = torch.nan_to_num(X[:, i], nan=self.X_mean_[i])
        X = X.to(self.device_)

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        for i in range(X.size(1)):
            X[:, i] = torch.nan_to_num(X[:, i], nan=self.X_mean_[i])
        X = X.to(self.device_)
        return self._get_predict_prob(X)

    def decision_function(self, X):
        decision = self.predict_proba(X)
        if decision.shape[1] == 1:
            decision = decision.ravel()
        return decision

    def _get_predict_prob(self, X):
        # Obtain the predicitve output
        with torch.no_grad():
            self.model_best_.eval()
            out = self.model_best_(X)
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
