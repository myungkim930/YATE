### This example runs through an experiment of YATE finetuning

import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from dirty_cat import TableVectorizer

from downstream import Yate_FineTune_Regressor
from graphlet_construction import Table2Graph

# import data
data_pd_dir = "/storage/store3/work/mkim/gitlab/YATE/data/eval_kg_data/raw/company_employees.parquet"
data_pd = pd.read_parquet(data_pd_dir)
data_pd = data_pd.drop(columns=["col_to_embed"])

# data_pd["name"] = (
#     data_pd["name"]
#     .str.replace("<", "")
#     .str.replace(">", "")
#     .str.replace("_", " ")
#     .str.lower()
# )

target_name = "target"
y = data_pd[target_name]
data_pd = pd.concat([data_pd.select_dtypes(include="object"), y], axis=1)

# Train/Test split
num_data = len(data_pd)
num_train = 64  # For small portion of labeled data

data_train, data_test = train_test_split(
    data_pd,
    test_size=int(num_data - num_train),
    shuffle=True,
    random_state=1,
)

# Table2Graph transformer
num_transformer_params = {"n_quantiles": 125}
table2graph_transformer = Table2Graph(
    numerical_cardinality_threshold=1,
    numerical_transformer="quantile",
    num_transformer_params=num_transformer_params,
)
train_dataset = table2graph_transformer.fit_transform(
    data_train, target_name=target_name
)
test_dataset = table2graph_transformer.transform(data_test)

# YATE finetune regressor
yate_ft = Yate_FineTune_Regressor(
    gpu_id=0,
    learning_rate=3e-3,
    max_epoch=200,
    early_stopping_patience=None,
    val_size=0.1,
    batch_size=256,
    num_model=30,
    load_pretrain=True,
    freeze=True,
    num_layers=0,
    include_numeric=False,
    n_jobs=10,
    disable_pbar=False,
)

yate_ft.fit(dataset_train=train_dataset)
y_pred_yate, y_target = yate_ft.predict(test_dataset, format="numpy", return_y=True)
r2_yate = r2_score(y_target, y_pred_yate)
print("{:.4f}".format(r2_yate))

# As a very brief comparison, we compare it with TableVectorizer
X_train = data_train.drop(columns=[target_name])
y_train = data_train[target_name]

X_test = data_test.drop(columns=[target_name])
y_test = data_test[target_name]

table_vec = TableVectorizer()
X_train_enc = table_vec.fit_transform(X_train)
X_test_enc = table_vec.transform(X_test)
reg_TableVec = HistGradientBoostingRegressor(max_iter=18)
reg_TableVec.fit(X_train_enc, y_train)
y_pred_TableVec = reg_TableVec.predict(X_test_enc)
y_pred_TableVec
r2_TableVec = r2_score(y_test, y_pred_TableVec)
print("{:.4f}".format(r2_TableVec))
