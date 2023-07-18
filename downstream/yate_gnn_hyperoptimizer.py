""" Class for optimizing the hyperparameters in yate_finetune
"""

import time
import numpy as np
import pandas as pd

from typing import Union
from joblib import Parallel, delayed

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RepeatedKFold

from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope

from downstream import Yate_FineTune_Regressor, Yate_FineTune_Classifier
from graphlet_construction import Table2Graph


####
@scope.define
def shifter(val, min_val, max_val):
    val += min_val
    if val > max_val:
        val = max_val
    return int(val)


################################################
class Yate_HyperParamOptimizer:
    def __init__(
        self,
        task: str,
        fixed_params: Union[None, dict] = None,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: Union[int, None] = None,
        n_jobs: int = 1,
        **kwargs,
    ):
        if fixed_params is None:
            self.fixed_params = dict()
        else:
            self.fixed_params = fixed_params

        self.rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        self.num_trial = int(n_splits * n_repeats)
        self.task = task
        self.n_jobs = n_jobs

    def optimize(
        self,
        data: pd.DataFrame,
        target_name: str,
        scoring: str,
        min_evals: Union[int, None] = None,
        max_evals: int = 10,
        early_stopping_patience: Union[int, None] = None,
        time_budget: Union[int, None] = None,
        **kwargs,
    ):
        self.trials = Trials()
        self.cum_time = 0
        self.es_count = 0
        if min_evals is None:
            self.min_evals = 0
        else:
            self.min_evals = min_evals

        if early_stopping_patience is None:
            self.early_stopping_patience = 9e15
        else:
            self.early_stopping_patience = early_stopping_patience

        if time_budget is None:
            self.time_budget = 9e15
        else:
            self.time_budget = time_budget

        self.iter = 0

        self.data = data
        self.target_name = target_name
        self.scoring = scoring
        self.split_index = [
            (train_index, test_index)
            for train_index, test_index in self.rkf.split(np.arange(0, len(data)))
        ]

        param_space = {
            "learning_rate": hp.uniform("learning_rate", 1e-5, 5e-3),
            "batch_size": hp.choice("batch_size", [128, 256, 512]),
            "early_stopping_patience": scope.shifter(
                hp.qlognormal("early_stopping_patience", np.log(33), 1, 1),
                min_val=10,
                max_val=200,
            ),
            # "early_stopping_patience": hp.choice(
            #     "early_stopping_patience", [10, 20, 30, 40, None]
            # ),
            "quantile_prop": hp.uniform("quantile_prop", 1e-3, 1),
        }
        if len(data) < 513:
            param_space["batch_size"] = hp.choice("batch_size", [len(data)])

        hyperparams = fmin(
            fn=self._score_fn,
            space=param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            early_stop_fn=self._early_stop_condition,
        )

        best_params = space_eval(param_space, hyperparams)
        self.best_params = best_params

        self.estimator = self._set_estimator(best_params)
        self.table2graph_transformer = self._set_table2graph_transformer(best_params)

    def _score_fn(self, params):
        self.iter += 1
        start = time.time()

        # cv_score = [
        #     self._run_fit_pred(params, idx_trial) for idx_trial in range(self.num_trial)
        # ]

        cv_score = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_fit_pred)(params, idx_trial)
            for idx_trial in range(self.num_trial)
        )

        end = time.time()

        self.cum_time += end - start

        if self.iter < self.min_evals or self.iter == 1:
            pass
        else:
            if np.mean(cv_score) < self.trials.best_trial["result"]["loss"]:
                self.es_count = 0
            else:
                self.es_count += 1

        return {
            "loss": np.mean(cv_score),
            "status": STATUS_OK,
            "eval_time": end - start,
            "cum_time": self.cum_time,
            "es_count": self.es_count,
        }

    def _early_stop_condition(self, trials, *kwargs):
        return trials.results[self.iter - 1][
            "cum_time"
        ] > self.time_budget or trials.results[self.iter - 1][
            "es_count"
        ] > self.early_stopping_patience, [
            1,
            1,
        ]

    def _run_fit_pred(
        self,
        params,
        idx_trial,
    ):
        idx_train = self.split_index[idx_trial][0]
        idx_valid = self.split_index[idx_trial][1]

        estimator = self._set_estimator(params)
        table2graph_transformer = Table2Graph(
            numerical_cardinality_threshold=0,
            numerical_transformer="quantile",
        )
        table2graph_transformer.reset_param_num_transformer(**params)
        dataset = table2graph_transformer.fit_transform(
            self.data,
            target_name=self.target_name,
        )

        X_train = [dataset[i] for i in idx_train]
        X_valid = [dataset[i] for i in idx_valid]

        # data_train = self.data.iloc[idx_train]
        # data_valid = self.data.iloc[idx_valid]

        # X_train = self.table2graph_transformer.fit_transform(
        #     data_train,
        #     target_name=self.target_name,
        # )
        # X_valid = self.table2graph_transformer.transform(data_valid)

        estimator.fit(dataset_train=X_train, dataset_valid=X_valid)
        y_pred, y_valid = estimator.predict(X_valid, format="numpy", return_y=True)

        if self.scoring == "r2":
            score = -r2_score(y_valid, y_pred)
        elif self.scoring == "mse":
            score = mean_squared_error(y_valid, y_pred, squared=False)
        elif self.scoring == "auc":
            score = -roc_auc_score(y_valid, y_pred)
        elif self.scoring == "average_precision":
            score = -average_precision_score(y_valid, y_pred)

        del (estimator, table2graph_transformer, y_pred, y_valid, idx_train, idx_valid)

        return score

    def _set_estimator(
        self,
        params,
    ):
        if self.task == "regression":
            estimator = Yate_FineTune_Regressor(**params, **self.fixed_params)
        elif self.task == "classification":
            estimator = Yate_FineTune_Classifier(**params, **self.fixed_params)

        return estimator

    def _set_table2graph_transformer(
        self,
        params,
    ):
        table2graph_transformer = Table2Graph(**params, **self.fixed_params)

        return table2graph_transformer
