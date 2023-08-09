import torch.nn as nn
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam, SGD
from skorch.callbacks import WandbLogger
import sys
sys.path.append("")
from baselines.resnet_model import ResNet, InputShapeSetterResnet
from skorch.callbacks import Callback, Checkpoint
import numpy as np
import os


class LearningRateLogger(Callback):
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        callbacks = net.callbacks
        for callback in callbacks:
            if isinstance(callback, WandbLogger):
                callback.wandb_run.log(
                    {"log_lr": np.log10(net.optimizer_.param_groups[0]["lr"])}
                )


class UniquePrefixCheckpoint(Checkpoint):
    def initialize(self):
        print("Initializing UniquePrefixCheckpoint")
        self.fn_prefix = str(id(self))
        print("fn_prefix is {}".format(self.fn_prefix))
        return super(UniquePrefixCheckpoint, self).initialize()


class NeuralNetRegressorBis(NeuralNetRegressor):
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y)
    def on_train_begin(self, net, X, y):
        self.training = True
        for callback in self.callbacks_:
            if isinstance(callback[1], UniquePrefixCheckpoint):
                callback[1].fn_prefix += str(np.random.randint(0, 1000000))

    def on_train_end(self, net, X, y):
        self.training = False

    def predict(self, X):
        y_pred = super().predict(X)
        # remove the checkpoint file if it exist
        # after the prediction
        # this could be done at the end of the training
        # but we want to do it after load_best
        if not self.training:
            for callback in self.callbacks_:
                if isinstance(callback[1], UniquePrefixCheckpoint):
                    fn_prefix = callback[1].fn_prefix
                    print(f"removing skorch_cp/{fn_prefix}params.pt")
                    os.remove(f"skorch_cp/{fn_prefix}params.pt")
        return y_pred


class NeuralNetClassifierBis(NeuralNetClassifier):
    def fit(self, X, y):
        y = y.astype(np.int64)
        return super().fit(X, y)
    def on_train_begin(self, net, X, y):
        self.training = True
        for callback in self.callbacks_:
            if isinstance(callback[1], UniquePrefixCheckpoint):
                callback[1].fn_prefix += str(np.random.randint(0, 1000000))

    def on_train_end(self, net, X, y):
        self.training = False

    def predict(self, X):
        y_pred = super().predict(X)
        # remove the checkpoint file if it exist
        # after the prediction
        # this could be done at the end of the training
        # but we want to do it after load_best
        if not self.training:
            for callback in self.callbacks_:
                if isinstance(callback[1], UniquePrefixCheckpoint):
                    fn_prefix = callback[1].fn_prefix
                    print(f"removing skorch_cp/{fn_prefix}params.pt")
                    os.remove(f"skorch_cp/{fn_prefix}params.pt")
        return y_pred


def create_resnet_regressor_skorch(
    id=None, wandb_run=None, use_checkpoints=True, cat_features=None, **kwargs
):
    print("resnet regressor")
    if "lr_scheduler" not in kwargs:
        print("no lr scheduler")
        lr_scheduler = False
    else:
        print("lr scheduler")
        lr_scheduler = kwargs.pop("lr_scheduler")
    if "es_patience" not in kwargs.keys():
        es_patience = 16
    else:
        es_patience = kwargs.pop("es_patience")
    if "max_epochs" not in kwargs.keys():
        max_epochs = 100
    else:
        max_epochs = kwargs.pop("max_epochs")
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop("lr_patience")
    if "optimizer" not in kwargs.keys():
        optimizer = "adamw"
    else:
        optimizer = kwargs.pop("optimizer")
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    if "batch_size" not in kwargs.keys():
        batch_size = 128
    else:
        batch_size = kwargs.pop("batch_size")
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop("categories")
    callbacks = [
        InputShapeSetterResnet(
            regression=True, cat_features=cat_features, categories=categories
        ),
        EarlyStopping(monitor="valid_loss", patience=es_patience),
    ]  # TODO try with train_loss, and in this case use checkpoint
    callbacks.append(
        EpochScoring(
            scoring="neg_root_mean_squared_error", name="train_accuracy", on_train=True
        )
    )

    if lr_scheduler:
        callbacks.append(
            LRScheduler(
                policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2
            )
        )  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(
            UniquePrefixCheckpoint(
                dirname="skorch_cp",
                f_params=r"params.pt",
                f_optimizer=None,
                f_criterion=None,
                f_history=None,
                load_best=True,
                monitor="valid_loss_best",
            )
        )
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    model = NeuralNetRegressorBis(
        ResNet,
        # Shuffle training data on each epoch
        max_epochs=max_epochs,
        optimizer=optimizer,
        batch_size=max(
            batch_size, 1
        ),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=True,
        module__categorical_indicator=None,  # will be change when fitted
        callbacks=callbacks,
        verbose=0,
        **kwargs,
    )

    return model


def create_resnet_classifier_skorch(
    id=None, wandb_run=None, use_checkpoints=True, cat_features=None, **kwargs
):
    print("resnet classifier")
    if "lr_scheduler" not in kwargs:
        print("no lr scheduler")
        lr_scheduler = False
    else:
        print("lr scheduler")
        lr_scheduler = kwargs.pop("lr_scheduler")
    if "es_patience" not in kwargs.keys():
        es_patience = 16
    else:
        es_patience = kwargs.pop("es_patience")
    if "max_epochs" not in kwargs.keys():
        max_epochs = 100
    else:
        max_epochs = kwargs.pop("max_epochs")
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop("lr_patience")
    if "optimizer" not in kwargs.keys():
        optimizer = "adamw"
    else:
        optimizer = kwargs.pop("optimizer")
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    if "batch_size" not in kwargs.keys():
        batch_size = 128
    else:
        batch_size = kwargs.pop("batch_size")
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop("categories")
    callbacks = [
        InputShapeSetterResnet(
            regression=False, cat_features=cat_features, categories=categories
        ),
        EarlyStopping(monitor="valid_loss", patience=es_patience),
    ]  # TODO try with train_loss, and in this case use checkpoint
    callbacks.append(
        EpochScoring(scoring="accuracy", name="train_accuracy", on_train=True)
    )

    if lr_scheduler:
        callbacks.append(
            LRScheduler(
                policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2
            )
        )  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(
            UniquePrefixCheckpoint(
                dirname="skorch_cp",
                f_params=r"params.pt",
                f_optimizer=None,
                f_criterion=None,
                f_history=None,
                load_best=True,
                monitor="valid_loss_best",
            )
        )
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    model = NeuralNetClassifierBis(
        ResNet,
        # Shuffle training data on each epoch
        max_epochs=max_epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=optimizer,
        batch_size=max(
            batch_size, 1
        ),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=False,
        module__categorical_indicator=None,  # will be change when fitted
        callbacks=callbacks,
        verbose=0,
        **kwargs,
    )

    return model
