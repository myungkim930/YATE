"""
Simple check to see problems of running baselines
"""
import os
import pickle
from time import perf_counter
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    RepeatedKFold,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from baselines.resnet import create_resnet_regressor_skorch
from catboost import CatBoostRegressor, CatBoostClassifier
from downstream import YateGNNRegressor, YateGNNClassifier
from utils import load_config, TabpfnClassifier
from scipy.stats import loguniform, lognorm, randint, uniform, norm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Run evaluation
def _run_model(
    config,
    data_name,
    num_train,
    method,
    include_numeric,
    random_state,
    device,
):
    # Load data
    data_pd, data_additional = _load_data(data_name, config)

    # Basic settings
    target_name = data_additional["target_name"]
    task = data_additional["task"]
    scoring, result_criterion = _set_score_criterion(task)

    # Set methods
    method_parse = method.split("_")
    estim_method = method_parse[-1]
    preprocess_method = method_parse[0]

    # Exclude numeric if include_numeric = False
    num_col_names = data_pd.select_dtypes(exclude="object").columns.tolist()
    if target_name in num_col_names:
        num_col_names.remove("target")
    cat_col_names = data_pd.select_dtypes(include="object").columns.tolist()
    if target_name in cat_col_names:
        cat_col_names.remove("target")
    if len(num_col_names) == 0:
        include_numeric = False
    if include_numeric == False:
        data_pd.drop(columns=num_col_names, inplace=True)

    # Prepare data
    if "lm" in preprocess_method:
        data = data_additional[preprocess_method].copy()
    elif "fasttext" in preprocess_method:
        data_fasttext = data_additional["fasttext"].copy()
        #TODO: move this into a function
        name_col = data_fasttext["name"]
        name_col = (
            name_col.str.replace("<", "").str.replace(">", "").str.replace("_", " ")
        )
        data_fasttext["name"] = name_col
        data = pd.merge(
            data_pd[["name", target_name]], data_fasttext, how="left", on="name"
        )
        data.drop(columns="name", inplace=True)
    else:
        raise NotImplementedError
        #data = data_pd.copy()

    # Preprocess data with splits

    if "yate-gnn" in preprocess_method:
        X_train, X_test, y_train, y_test = _prepare_yate_gnn(
            data, target_name, num_train, random_state
        )
    elif "yate-feature" in preprocess_method:
        extract_state = preprocess_method.split("-")[-1]
        X_train, X_test, y_train, y_test = _prepare_yate_feature_based(
            data,
            target_name,
            num_train,
            extract_state,
            include_numeric,
            random_state,
            device,
        )
    elif "tablevectorizer" in preprocess_method:
        include_ken = False
        if "ken" in preprocess_method:
            include_ken = True
        X_train, X_test, y_train, y_test = _prepare_tablevectorizer(
            data,
            data_additional,
            include_ken,
            target_name,
            num_train,
            random_state,
        )
    elif "catboost" in preprocess_method:
        X_train, X_test, y_train, y_test = _prepare_catboost(
            data,
            target_name,
            cat_col_names,
            num_train,
            random_state,
        )
    elif "resnet" in preprocess_method:
        X_train, X_test, y_train, y_test = _prepare_resnet(
            data,
            target_name,
            cat_col_names,
            num_train,
            random_state,
        )
    else:
        X_train, X_test, y_train, y_test = _set_split(
            data,
            target_name,
            num_train,
            random_state=random_state,
        )


    # Set cross-validation settings
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1234)

    # Set Parameter distributions
    param_distributions = _set_param_distributions(
        estim_method,
        num_train,
    )

    # Set estimator
    if "catboost" in estim_method or "resnet" in estim_method:
        cat_features = [data.columns.get_loc(i) for i in cat_col_names]
    else:
        cat_features = None
    if "resnet" in estim_method:
        # +1 for unknown category
        categories = [len(pd.unique(data[col])) + 1 for col in cat_col_names]
    else:
        categories = None
    estimator = _assign_estimator(
        estim_method,
        task,
        include_numeric,
        device,
        cat_features,
        categories,
    )

    # Optimization
    if "yate-gnn" in method:
        refit, n_jobs = False, -1
        hyperparameter_search = GridSearchCV(
            estimator,
            param_grid=param_distributions,
            cv=cv,
            scoring=scoring,
            refit=refit,
            n_jobs=n_jobs,
        )
    elif "tabpfn" in method:
        refit, n_jobs = True, 25
        hyperparameter_search = GridSearchCV(
            estimator,
            param_grid=param_distributions,
            cv=cv,
            scoring=scoring,
            refit=refit,
            n_jobs=n_jobs,
        )
    elif "catboost" in method:
        hyperparameter_search = estimator
    else:
        n_iter, refit, n_jobs = 100, True, -1
        hyperparameter_search = RandomizedSearchCV(
            estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            refit=refit,
            n_jobs=n_jobs,
            error_score='raise',
            verbose=100,
        )

    marker = f"{data_name}_{method}_num_train-{num_train}_numeric-{include_numeric}_rs-{random_state}"
    print(marker + " start")

    start_time = perf_counter()
    hyperparameter_search.fit(X_train, y_train)

    # Prediction
    if "yate-gnn" in method:
        estimator_refit = _set_refit_yate_gnn(
            hyperparameter_search.best_params_,
            task,
            include_numeric,
            device,
        )
        estimator_refit.fit(X_train, y_train)
        if task == "regression":
            y_pred = estimator_refit.predict(X_test)
        else:
            y_pred = estimator_refit.predict_proba(X_test)
    else:
        if task == "regression":
            y_pred = hyperparameter_search.predict(X_test)
        else:
            y_pred = hyperparameter_search.predict_proba(X_test)

    score = _return_score(y_test, y_pred, task)
    end_time = perf_counter()
    duration = round(end_time - start_time, 4)

    # Saving results
    result_save_dir_base = config["results_dir"] + data_name
    if not os.path.exists(result_save_dir_base):
        os.makedirs(result_save_dir_base, exist_ok=True)
    if not os.path.exists(result_save_dir_base + "/score"):
        os.makedirs(result_save_dir_base + "/score", exist_ok=True)
    if not os.path.exists(result_save_dir_base + "/log"):
        os.makedirs(result_save_dir_base + "/log", exist_ok=True)

    results_ = dict()
    results_[result_criterion[0]] = score[0]
    results_[result_criterion[1]] = score[1]
    results_[result_criterion[2]] = duration
    results_model = pd.DataFrame([results_], columns=result_criterion)
    results_model.columns = f"{method}_" + results_model.columns
    results_model["random_state"] = random_state

    marker = f"{data_name}_{method}_num_train-{num_train}_numeric-{include_numeric}_rs-{random_state}"
    results_model_dir = result_save_dir_base + f"/score/{marker}.csv"
    log_dir = result_save_dir_base + f"/log/{marker}_log.csv"

    results_model.to_csv(results_model_dir, index=False)

    if "catboost" not in method:
        cv_results = pd.DataFrame(hyperparameter_search.cv_results_)
        cv_results = cv_results.rename(_shorten_param, axis=1)
        cv_results.to_csv(log_dir, index=False)

    print(marker + " is complete")

    return None


# Load data
def _load_data(data_name, config):
    data_base_dir = config["data_ds_dir"]
    data_pd_dir = data_base_dir + f"raw/{data_name}.parquet"
    data_additional_dir = data_base_dir + f"processed/{data_name}.pickle"
    data_pd = pd.read_parquet(data_pd_dir)
    data_pd.fillna(value=np.nan, inplace=True)
    with open(data_additional_dir, "rb") as pickle_file:
        data_additional = pickle.load(pickle_file)
    if data_additional["mode"] == "A":
        if "name" in data_pd.columns:
            name_col = data_pd["name"]
            name_col = (
                name_col.str.replace("<", "").str.replace(">", "").str.replace("_", " ")
            )
            data_pd["name"] = name_col
    return data_pd, data_additional


# Set train/test split given the random state
def _set_split(data, target_name, num_train, random_state):
    num_data = len(data)
    X = data.drop(columns=target_name)
    y = data[target_name]
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=int(num_data - num_train),
        shuffle=True,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def _prepare_yate_gnn(data_pd, target_name, num_train, random_state):
    from graphlet_construction import Table2GraphTransformer

    data = data_pd.copy()
    X_train, X_test, y_train, y_test = _set_split(
        data,
        target_name,
        num_train,
        random_state=random_state,
    )
    preprocessor = Table2GraphTransformer()
    X_train = preprocessor.fit_transform(X_train, y=y_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test


def _prepare_yate_feature_based(
    data_pd,
    target_name,
    num_train,
    extract_state,
    include_numeric,
    random_state,
    device,
):
    from downstream import YATE_feat_extractor

    data = data_pd.copy()
    yate_feat_extractor = YATE_feat_extractor(n_layers=1, device=device)
    X_train, X_test, y_train, y_test = _set_split(
        data,
        target_name,
        num_train,
        random_state=random_state,
    )
    if extract_state == "initial":
        X_train = yate_feat_extractor.extract(X_train, "initial", include_numeric)
        X_test = yate_feat_extractor.extract(X_test, "initial", include_numeric)
    else:
        X_train = yate_feat_extractor.extract(X_train, "pretrained", include_numeric)
        X_test = yate_feat_extractor.extract(X_test, "pretrained", include_numeric)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def _prepare_tablevectorizer(
    data_pd,
    data_additional,
    include_ken,
    target_name,
    num_train,
    random_state,
):
    from dirty_cat import TableVectorizer

    data = data_pd.copy()
    if include_ken:
        name_col = data_pd["name"]
        name_col = "<" + name_col + ">"
        name_col = name_col.str.replace(" ", "_")
        data["name"] = name_col
        data_aug = data.merge(right=data_additional["ken"], how="inner", on="name")
        data = data_aug.copy()
    X_train, X_test, y_train, y_test = _set_split(
        data,
        target_name,
        num_train,
        random_state=random_state,
    )
    preprocessor = TableVectorizer(auto_cast=True, sparse_threshold=0)
    X_train = preprocessor.fit_transform(X_train, y=y_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test


def _prepare_catboost(data_pd, target_name, cat_col_names, num_train, random_state):
    data = data_pd.copy()
    data_cat = data_pd[cat_col_names]
    data_cat = data_cat.replace(np.nan, "nan", regex=True)
    data[cat_col_names] = data_cat
    for col in cat_col_names:
        data[col] = data[col].astype("category")
    X_train, X_test, y_train, y_test = _set_split(
        data,
        target_name,
        num_train,
        random_state=random_state,
    )
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def _prepare_resnet(data_pd, target_name, cat_col_names, num_train, random_state):
    data = data_pd.copy()
    X_train, X_test, y_train, y_test = _set_split(
        data,
        target_name,
        num_train,
        random_state=random_state,
    )
    numerical_preprocessor = Pipeline([
        ('power_transform', PowerTransformer()),
        ('imputer', SimpleImputer(strategy='mean')),
    ])
    categorical_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    print("cat cols", cat_col_names)
    print("num cols", [col for col in X_train.columns if col not in cat_col_names])
    preprocessor = ColumnTransformer([
        ('numerical', numerical_preprocessor, [col for col in X_train.columns if col not in cat_col_names]),
        ('categorical', categorical_preprocessor, cat_col_names),
    ])
    X_train = preprocessor.fit_transform(X_train, y=y_train)
    X_test = preprocessor.transform(X_test)

    #TODO export categories to the main thing
    # and use them here


    return (np.array(X_train).astype(np.float32), 
            np.array(X_test).astype(np.float32), 
            np.array(y_train).astype(np.float32), 
            np.array(y_test).astype(np.float32))


def _assign_estimator(
    estim_method: str,
    task: str,
    include_numeric,
    device,
    cat_features,
    categories,
):
    if estim_method == "yate-gnn":
        fixed_params = dict()
        fixed_params["include_numeric"] = include_numeric
        fixed_params["device"] = device
        fixed_params["num_model"] = 1
        fixed_params["n_jobs"] = 1
        if task == "regression":
            estimator = YateGNNRegressor(**fixed_params)
        else:
            estimator = YateGNNClassifier(**fixed_params)
    elif estim_method == "catboost":
        fixed_params = dict()
        fixed_params["cat_features"] = cat_features
        fixed_params["verbose"] = False
        fixed_params["allow_writing_files"] = False
        fixed_params["thread_count"] = 5
        # fixed_params["boosting_type"] = "Plain"
        # fixed_params["leaf_estimation_iterations"] = 5
        if task == "regression":
            estimator = CatBoostRegressor(**fixed_params)
        else:
            estimator = CatBoostClassifier(**fixed_params)
    elif estim_method == "histgb":
        if task == "regression":
            estimator = HistGradientBoostingRegressor()
        else:
            estimator = HistGradientBoostingClassifier()
    elif estim_method == "tabpfn":
        estimator = TabpfnClassifier()
    elif estim_method == "resnet":
        if task == "regression":
            estimator = create_resnet_regressor_skorch(
                cat_features=cat_features,
                categories=categories,
            )
        else:
            raise NotImplementedError
    return estimator


def _set_param_distributions(estim_method: str, num_train: int):
    param_distributions = dict()
    if estim_method == "yate-gnn":
        param_distributions["learning_rate"] = [1e-3, 2.5e-3, 5e-3, 7.5e-4, 5e-4]
        param_distributions["batch_size"] = [128, 256]
        if num_train < 129:
            param_distributions["batch_size"] = [num_train]
        elif 128 < num_train < 256:
            param_distributions["batch_size"] = [128]
    elif estim_method == "catboost":
        param_distributions["learning_rate"] = uniform(1e-3, 1e-2)
        param_distributions["iterations"] = randint(400, 1001)
        param_distributions["depth"] = randint(4, 11)
        param_distributions["l2_leaf_reg"] = loguniform(2, 10)
        param_distributions["random_strength"] = uniform(0, 10)
    elif estim_method == "histgb":
        param_distributions["loss"] = ["squared_error", "absolute_error"]
        param_distributions["learning_rate"] = loguniform(1e-2, 10)
        # param_distributions["l2_regularization"] = loguniform(1e-6, 1e3)
        param_distributions["max_depth"] = [None, 2, 3, 4]
        param_distributions["max_leaf_nodes"] = norm_int(31, 5)
        param_distributions["min_samples_leaf"] = norm_int(20, 2)
    elif estim_method == "resnet":
        param_distributions = {
            "module__activation": ["reglu"],
            "module__normalization": ["batchnorm", "layernorm"],
            "module__n_layers": randint(1, 17),  # equivalent to q_uniform(1, 16)
            "module__d": randint(64, 1025),  # equivalent to q_uniform(64, 1024)
            "module__d_hidden_factor": uniform(1, 3),  # uniform distribution between 1 and 4
            "module__hidden_dropout": uniform(0.0, 0.5),  # uniform distribution between 0.0 and 0.5
            "module__residual_dropout": uniform(0.0, 0.5),  # uniform distribution between 0.0 and 0.5
            "lr": loguniform(1e-5, 1e-2),  # log uniform distribution between 1e-5 and 1e-2
            "optimizer__weight_decay": loguniform(1e-8, 1e-3),  # log uniform distribution between 1e-8 and 1e-3
            "module__d_embedding": randint(64, 513),  # equivalent to q_uniform(64, 512)
            #"lr_scheduler": [True, False]  # two possible values
        }
    else:
        param_distributions["n_ensemble_configurations"] = list(range(1, 101))
    return param_distributions


def _set_refit_yate_gnn(best_param, task, include_numeric, device):
    # estimator
    fixed_params = dict()
    fixed_params["include_numeric"] = include_numeric
    fixed_params["device"] = device
    fixed_params["num_model"] = 30
    fixed_params["n_jobs"] = 10
    estimator_params_name = [name for name in best_param.keys() if "estimator" in name]
    estimator_params = {
        name.replace("estimator__", ""): best_param[name]
        for name in estimator_params_name
    }
    if task == "regression":
        estimator = YateGNNRegressor(**fixed_params, **estimator_params)
    else:
        estimator = YateGNNClassifier(**fixed_params, **estimator_params)
    return estimator


# Set scoring method for CV and score criterion in final result
def _set_score_criterion(task):
    if task == "regression":
        scoring = "r2"
        score_criterion = ["r2", "rmse"]
    else:
        scoring = "auc"
        score_criterion = ["auc", "avg_precision"]
    score_criterion += ["run_time"]
    return scoring, score_criterion


# Return score results
def _return_score(y_target: np.array, y_pred: np.array, task: str):
    if task == "regression":
        score_r2 = r2_score(y_target, y_pred)
        score_rmse = mean_squared_error(y_target, y_pred, squared=False)
        return score_r2, score_rmse
    else:
        score_auc = roc_auc_score(y_target, y_pred)
        score_avg_precision = average_precision_score(y_target, y_pred)
        return score_auc, score_avg_precision


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


class norm_int:
    """Integer valued version of the normal distribution"""

    def __init__(self, a, b):
        self._distribution = norm(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        if self._distribution.rvs(*args, **kwargs).astype(int) < 1:
            return 1
        else:
            return self._distribution.rvs(*args, **kwargs).astype(int)


def _shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


# Main
def main(data_name, num_train, include_numeric, method, random_state, device):
    # Load configurations
    config = load_config()

    # Load data
    _, data_additional = _load_data(data_name, config)

    # Setting methods
    if method == "all":
        mode = data_additional["mode"]
        if mode == "A":
            method_list = config["comparing_methods"]
        elif mode == "B":
            method_list = [
                method for method in config["comparing_methods"] if "ken" not in method
            ]
        elif mode == "C":
            method_list = [
                method for method in config["comparing_methods"] if "ken" not in method
            ]
            method_list = [
                method for method in method_list if "_fasttext_" not in method
            ]

        if data_additional["task"] == "regression":
            method_list.remove("tabpfn")
    else:
        method_list = [method]

    for method_name in method_list:
        _run_model(
            config,
            data_name,
            num_train,
            method_name,
            include_numeric,
            random_state,
            device,
        )

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Model")
    parser.add_argument(
        "-ds",
        "--data_name",
        type=str,
        help="Name of data",
    )
    parser.add_argument(
        "-nt",
        "--num_train",
        type=int,
        help="Number of train",
    )
    parser.add_argument(
        "-i",
        "--include_numeric",
        type=str,
        help="Include numerical features or not (T/F)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="Method to evaluate",
    )
    parser.add_argument(
        "-rs",
        "--random_state",
        type=int,
        help="Random_state",
    )
    parser.add_argument(
        "-dv",
        "--device",
        type=str,
        help="Device, cpu or cuda",
    )
    args = parser.parse_args()

    if args.include_numeric == "True":
        include_numeric = True
    else:
        include_numeric = False

    main(
        args.data_name,
        args.num_train,
        include_numeric,
        args.method,
        args.random_state,
        args.device,
    )
