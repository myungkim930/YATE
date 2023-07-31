import torch.nn
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam, SGD
from skorch.callbacks import WandbLogger
import sys
sys.path.append("")
from baselines.resnet_model import ResNet, InputShapeSetterResnet
from skorch.callbacks import Callback
import numpy as np

class LearningRateLogger(Callback):
    def on_epoch_begin(self, net,
                       dataset_train=None, dataset_valid=None, **kwargs):
        callbacks = net.callbacks
        for callback in callbacks:
            if isinstance(callback, WandbLogger):
                callback.wandb_run.log({'log_lr': np.log10(net.optimizer_.param_groups[0]['lr'])})




class NeuralNetRegressorBis(NeuralNetRegressor):
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y)

def create_resnet_regressor_skorch(id=None, wandb_run=None, use_checkpoints=True,
                                   cat_features=None, **kwargs):
    if id is None:
        # generate id at random
        id = np.random.randint(0, 1000000)
        print("id is None, generated id is {}".format(id))
    print("resnet regressor")
    if "lr_scheduler" not in kwargs:
        print("no lr scheduler")
        lr_scheduler = False
    else:
        print("lr scheduler")
        lr_scheduler = kwargs.pop("lr_scheduler")
    if "es_patience" not in kwargs.keys():
        es_patience = 40
    else:
        es_patience = kwargs.pop('es_patience')
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop('lr_patience')
    if "optimizer" not in kwargs.keys():
        optimizer = "adamw"
    else:
        optimizer = kwargs.pop('optimizer')
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    if "batch_size" not in kwargs.keys():
        batch_size = 128
    else:
        batch_size = kwargs.pop('batch_size')
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop('categories')
    callbacks = [InputShapeSetterResnet(regression=True,
                                        cat_features=cat_features,
                                        categories=categories),
                       EarlyStopping(monitor="valid_loss", patience=es_patience)] #TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True))

    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)) #FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None, load_best=True, monitor="valid_loss_best"))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    model = NeuralNetRegressorBis(
        ResNet,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        batch_size=max(batch_size, 1), # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None, # will be change when fitted
        module__d_out=1,  # idem
        module__regression=True,
        module__categorical_indicator=None, # will be change when fitted
        callbacks=callbacks,
        verbose=0,
        **kwargs
    )

    return model
