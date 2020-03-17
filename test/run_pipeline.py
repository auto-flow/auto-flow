import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

from autopipeline import constants
from autopipeline.pipeline.components.classification.sgd import SGD
from autopipeline.pipeline.components.feature_engineer.encode.one_hot_encode import OneHotEncoder
from autopipeline.pipeline.components.feature_engineer.impute.fill_cat import FillCat
from autopipeline.pipeline.components.feature_engineer.impute.fill_num import FillNum
from autopipeline.pipeline.dataframe import GeneralDataFrame
from autopipeline.pipeline.pipeline import GeneralPipeline

df = pd.read_csv("../examples/classification/train_classification.csv")
y = df.pop("Survived").values
df = df.loc[:, ["Sex", "Cabin", "Age"]]
feat_grp = ["cat_nan", "cat_nan", "num_nan"]
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=10)
df_train = GeneralDataFrame(df_train, feat_grp=feat_grp)
df_test = GeneralDataFrame(df_test, feat_grp=feat_grp)
cv = KFold(n_splits=5, random_state=10, shuffle=True)
train_ix, valid_ix = next(cv.split(df_train))

df_train, df_valid = df_train.split([train_ix, valid_ix])
y_valid = y_train[valid_ix]
y_train = y_train[train_ix]

fill_cat = FillCat()
fill_cat.in_feat_grp = "cat_nan"
fill_cat.out_feat_grp = "cat"
fill_cat.update_hyperparams({"strategy": "<NULL>"})

fill_num = FillNum()
fill_num.in_feat_grp = "num_nan"
fill_num.out_feat_grp = "num"
fill_num.update_hyperparams({"strategy": "median"})

ohe = OneHotEncoder()
ohe.in_feat_grp = "cat"
ohe.out_feat_grp = "num"

sgd = SGD()
sgd.in_feat_grp = "num"
sgd.update_hyperparams({"loss": "log", "random_state": 10})

pipeline = GeneralPipeline([
    ("fill_cat", fill_cat),
    ("fill_num", fill_num),
    ("ohe", ohe),
    ("sgd", sgd),
])

pipeline.fit(df_train, y_train, df_valid, y_valid, df_test, y_test)
pred_train = pipeline.predict(df_train)
pred_test = pipeline.predict(df_test)
pred_valid = pipeline.predict(df_valid)
score_valid = pipeline.predict_proba(df_valid)
print(accuracy_score(y_train, pred_train))
print(accuracy_score(y_valid, pred_valid))
print(accuracy_score(y_test, pred_test))
ret = pipeline.procedure(constants.binary_classification_task, df_train, y_train, df_valid, y_valid, df_test, y_test)
pred_test = ret["pred_test"]
pred_valid = ret["pred_valid"]
print(accuracy_score(y_valid, (pred_valid > .5).astype("int")[:, 1]))
print(accuracy_score(y_test, (pred_test > .5).astype("int")[:, 1]))

pipeline = GeneralPipeline([
    ("fill_cat", fill_cat),
    ("fill_num", fill_num),
    ("ohe", ohe),
])

pipeline.fit(df_train, y_train, df_valid, y_valid, df_test, y_test)
ret1 = pipeline.transform(df_train, df_valid, df_test)
ret2 = pipeline.fit_transform(df_train, y_train, df_valid, y_valid, df_test, y_test)
for key in ["X_train","X_valid","X_test"]:
    assert np.all(ret1[key]==ret2[key])

pipeline = GeneralPipeline([
    ("sgd", sgd),
])

ret = pipeline.procedure(constants.binary_classification_task, ret1["X_train"], y_train, ret1["X_valid"], y_valid,
                         ret1["X_test"], y_test)
pred_test = ret["pred_test"]
pred_valid = ret["pred_valid"]
print(accuracy_score(y_valid, (pred_valid > .5).astype("int")[:, 1]))
print(accuracy_score(y_test, (pred_test > .5).astype("int")[:, 1]))
