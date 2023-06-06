import numpy as np

from typing import Union

from hyperopt import fmin, tpe, space_eval

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import (
    HistGradientBoostingRegressor as HGBR,
    HistGradientBoostingClassifier as HGBC,
    GradientBoostingRegressor as GBR,
    GradientBoostingClassifier as GBC,
)
from xgboost import XGBRegressor as XGBR, XGBClassifier as XGBC
from catboost import CatBoostRegressor as CBR, CatBoostClassifier as CBC

from .config_loader import load_config
from utils import Yate_FineTune_Regressor as YFR, Yate_FineTune_Classifier as YFC

from joblib import Parallel, delayed

################################################################


class HypOptimizer:
    def __init__(
        self,
        task: str,
        estim_method: str,
        fixed_params: Union[None, dict] = None,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: Union[None, int] = None,
        n_jobs: int = 5,
    ):
        self.config = load_config()
        self.task = task
        self.estim_method = estim_method
        if fixed_params is None:
            self.fixed_params = dict()
        else:
            self.fixed_params = fixed_params

        self.params = self.config[estim_method + "_hparam"]
        self.rkf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        self.num_trial = int(n_splits * n_repeats)

        self.n_jobs = n_jobs

    def optimize(
        self,
        X: Union[np.array, list],
        y: Union[np.array, None],
        scoring: str,
        max_evals: int = 10,
        refit: bool = True,
    ):
        self._reset(X, y, scoring)
        hyperparams = fmin(
            fn=self._score_fn, space=self.params, algo=tpe.suggest, max_evals=max_evals
        )
        best_params = space_eval(self.params, hyperparams)
        self.best_params = best_params

        estimator = self._set_estimator(best_params)
        if refit:
            if self.estim_method == "yate_finetune":
                estimator.fit(dataset_train=X)
            else:
                estimator.fit(X=X, y=y)

        return estimator

    def _reset(self, X, y, scoring):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.best_params = []
        if self.estim_method == "yate_finetune":
            num_data = len(X)
        else:
            num_data = X.shape[0]
        self.split_index = [
            (train_index, test_index)
            for train_index, test_index in self.rkf.split(np.arange(0, num_data))
        ]

    def _score_fn(self, params):
        # cv_score = [
        #     self._run_fit_pred(params, idx_trial) for idx_trial in range(self.num_trial)
        # ]

        cv_score = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_fit_pred)(params, idx_trial)
            for idx_trial in range(self.num_trial)
        )

        return np.mean(cv_score)

    def _run_fit_pred(
        self,
        params,
        idx_trial,
    ):
        estimator = self._set_estimator(params)

        idx_train = self.split_index[idx_trial][0]
        idx_valid = self.split_index[idx_trial][1]

        if self.estim_method == "yate_finetune":
            X_train = [self.X[i] for i in idx_train]
            X_valid = [self.X[i] for i in idx_valid]
            estimator.fit(dataset_train=X_train, dataset_valid=X_valid)
            y_pred, y_valid = estimator.predict(X_valid, format="numpy", return_y=True)
        else:
            X_train, y_train = self.X[idx_train], self.y[idx_train]
            X_valid, y_valid = self.X[idx_valid], self.y[idx_valid]
            estimator.fit(X=X_train, y=y_train)
            if self.task == "reg":
                y_pred = estimator.predict(X_valid)
            elif self.task == "cls":
                y_pred = estimator.predict_proba(X_valid)

        if self.scoring == "r2":
            score = -r2_score(y_valid, y_pred)
        elif self.scoring == "mse":
            score = mean_squared_error(y_valid, y_pred, squared=False)
        elif self.scoring == "auc":
            score = -roc_auc_score(y_valid, y_pred)
        elif self.scoring == "average_precision":
            score = -average_precision_score(y_valid, y_pred)

        del (estimator, y_pred, y_valid, idx_train, idx_valid)

        return score

    def _set_estimator(self, params):
        if self.estim_method == "histgb":
            estimator = HGBR(**params, **self.fixed_params)
            if self.task == "cls":
                estimator = HGBC(**params, **self.fixed_params)
        elif self.estim_method == "gb":
            estimator = GBR(**params, **self.fixed_params)
            if self.task == "cls":
                estimator = GBC(**params, **self.fixed_params)
        elif self.estim_method == "xgboost":
            estimator = XGBR(**params, **self.fixed_params)
            if self.task == "cls":
                estimator = XGBC(**params, **self.fixed_params)
        elif self.estim_method == "catboost":
            estimator = CBR(**params, **self.fixed_params)
            if self.task == "cls":
                estimator = CBC(**params, **self.fixed_params)
        elif self.estim_method == "yate_finetune":
            estimator = YFR(**params, **self.fixed_params)
            if self.task == "cls":
                estimator = YFC(**params, **self.fixed_params)
        return estimator


################################################################
