"""
Temporary python script for loading configs
Subject to change later
"""


def load_config():
    config = dict()

    # Directories
    config["fasttext_dir"] = "/data/parietal/store3/work/mkim/gitlab/cc.en.300.bin"
    config[
        "ken_embed_dir"
    ] = "/data/parietal/store3/work/jstojano/gitlab/alexis_cvetkov/KEN/experiments/embedding_visualization/emb_mure_yago3_2022_full.parquet"

    config["data_ds_dir"] = "/data/parietal/store3/work/mkim/gitlab/YATE/data/ds_data/"
    config["data_kg_dir"] = "/data/parietal/store3/work/mkim/gitlab/YATE/data/kg_data/"
    config[
        "data_eval_ds_dir"
    ] = "/data/parietal/store3/work/mkim/gitlab/YATE/data/eval_ds_data/"
    config[
        "data_eval_jl_dir"
    ] = "/data/parietal/store3/work/mkim/gitlab/YATE/data/eval_jl_data/"
    config[
        "data_eval_kg_dir"
    ] = "/data/parietal/store3/work/mkim/gitlab/YATE/data/eval_kg_data/"
    config[
        "hyperopt_log_dir"
    ] = "/data/parietal/store3/work/mkim/gitlab/YATE/data/hyperopt_log/"
    config["results_dir"] = "/data/parietal/store3/work/mkim/gitlab/YATE/results/"

    config[
        "pretrained_model_dir"
    ] = "/data/parietal/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_2007_NB128_NH1_NP1_MN100_CL/ckpt_step190000.pt"
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_pretrained_MS.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_num_pretrained_CS.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_num_pretrained_MS.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_pretrained_CS.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_2705_NB128_NH2_NP1_MN100_CL/ckpt_step86000.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_2705_NB256_NH2_NP1_MN100_CL/ckpt_step15000.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_3105_NB512_NH2_NP1_MN100_CL/ckpt_step7000.pt
    # /data/parietal/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_2007_NB128_NH1_NP1_MN100_CL/ckpt_step190000.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_2107_NB128_NH2_NL1_NP1_MN100_CL/ckpt_step90000.pt
    # /storage/store3/work/mkim/gitlab/YATE/data/saved_model/yago3_2022_2107_NB128_NH1_NL0_NP1_MN100_CL/ckpt_step228000.pt


    config["comparing_methods"] = [
        "yate-gnn",
        "yate-feature-initial_histgb",
        "yate-feature-pretrained_histgb",
        "tablevectorizer_histgb",
        "tablevectorizer_ken_histgb",
        "catboost",
        "lm-roberta_histgb",
        "fasttext_histgb",
        "tabpfn",
    ]

    # "lm-fasttext_histgb",
    # "tablevectorizer_fasttext_histgb",
    # "catboost_ken_catboost",
    # "catboost_fasttext_catboost",
    # "yate-gnn_augment",
    # "yate-initial_xgboost",
    # "tablevectorizer_xgboost",
    # "tablevectorizer_ken_xgboost",
    # "tablevectorizer_fasttext_xgboost",
    # "lm-roberta_ken_histgb",
    # "lm-roberta_xgboost",
    # "lm-roberta_ken_xgboost",
    # "lm-fasttext_ken_histgb",
    # "lm-fasttext_xgboost",
    # "lm-fasttext_ken_xgboost",
    # "yate-pretrained_xgboost",

    # config["comparing_methods_indicator"] = {
    #     method: True for method in config["comparing_methods"]
    # }

    # # Hyperparameter search spaces
    # from hyperopt import hp
    # from hyperopt.pyll import scope

    # config["yate_finetune_hparam"] = {
    #     "learning_rate": hp.uniform("learning_rate", 1e-5, 5e-4),
    #     "batch_size": hp.choice("batch_size", [128, 256, 512]),
    # }
    # config["histgb_hparam"] = {
    #     "learning_rate": hp.lognormal("learning_rate", np.log(0.01), np.log(10.0)),
    #     "max_depth": hp.pchoice(
    #         "max_depth",
    #         [
    #             (0.1, None),
    #             (0.1, 2),
    #             (0.7, 3),
    #             (0.1, 4),
    #         ],
    #     ),
    #     "min_samples_leaf": scope.int(
    #         hp.qloguniform("min_samples_leaf", np.log(1.5), np.log(50.5), 1)
    #     ),
    #     "max_leaf_nodes": hp.pchoice(
    #         "max_leaf_nodes",
    #         [
    #             (0.85, None),
    #             (0.05, 5),
    #             (0.05, 10),
    #             (0.05, 15),
    #         ],
    #     ),
    # }
    # config["gb_hparam"] = {
    #     "learning_rate": hp.lognormal("learning_rate", np.log(0.01), np.log(10.0)),
    #     "subsample": hp.uniform("subsample", 0.5, 1),
    #     "n_estimators": scope.int(
    #         hp.qloguniform("n_estimators", np.log(10.5), np.log(1000.5), 1)
    #     ),
    #     "criterion": hp.choice("criterion", ["friedman_mse", "squared_error"]),
    #     "max_depth": hp.pchoice(
    #         "max_depth",
    #         [
    #             (0.1, None),
    #             (0.1, 2),
    #             (0.6, 3),
    #             (0.1, 4),
    #             (0.1, 5),
    #         ],
    #     ),
    #     "min_samples_split": hp.pchoice(
    #         "min_samples_split",
    #         [
    #             (0.95, 2),
    #             (0.05, 3),
    #         ],
    #     ),
    #     "min_samples_leaf": scope.int(
    #         hp.qloguniform("min_samples_leaf", np.log(1.5), np.log(50.5), 1)
    #     ),
    #     "min_impurity_decrease": hp.pchoice(
    #         "min_impurity_decrease",
    #         [
    #             (0.85, 0.0),
    #             (0.05, 0.01),
    #             (0.05, 0.02),
    #             (0.05, 0.05),
    #         ],
    #     ),
    #     "max_leaf_nodes": hp.pchoice(
    #         "max_leaf_nodes",
    #         [
    #             (0.85, None),
    #             (0.05, 5),
    #             (0.05, 10),
    #             (0.05, 15),
    #         ],
    #     ),
    # }
    # config["xgboost_hparam"] = {
    #     "max_depth": scope.int(hp.uniform("max_depth", 1, 11)),
    #     "n_estimators": scope.int(hp.quniform("n_estimators", 100, 6000, 200)),
    #     "min_child_weight": scope.int(
    #         hp.loguniform("min_child_weight", np.log(1), np.log(100))
    #     ),
    #     "subsample": hp.uniform("subsample", 0.5, 1),
    #     "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.5))
    #     - 0.0001,
    #     "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
    #     "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    #     "gamma": hp.loguniform("gamma", np.log(0.0001), np.log(5)) - 0.0001,
    #     "reg_alpha": hp.loguniform("reg_alpha", np.log(0.0001), np.log(1)) - 0.0001,
    #     "reg_lambda": hp.loguniform("reg_lambda", np.log(1), np.log(4)),
    # }
    # config["catboost_hparam"] = {
    #     # "iterations": scope.int(hp.uniform("iterations", 1, 1000)),
    #     "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1)),
    #     "depth": scope.int(hp.uniform("depth", 1, 16)),
    #     "random_strength": scope.int(hp.uniform("random_strength", 1, 20)),
    #     "l2_leaf_reg": hp.loguniform("bagging_temperature", np.log(1), np.log(10)),
    #     # "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
    #     "leaf_estimation_iterations": scope.int(
    #         hp.uniform("leaf_estimation_iterations", 1, 20)
    #     ),
    # }

    return config
