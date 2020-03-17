import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

from autopipeline.pipeline.components.classification.sgd import SGD
from autopipeline.pipeline.components.feature_engineer.encode.one_hot import OneHotEncoder
from autopipeline.pipeline.components.feature_engineer.impute.fill_cat import FillCat
from autopipeline.pipeline.components.feature_engineer.impute.fill_num import FillNum
from autopipeline.pipeline.dataframe import GeneralDataFrame

df = pd.read_csv("../examples/classification/train_classification.csv")
y = df.pop("Survived").values
df = df.loc[:, ["Sex", "Cabin", "Age"]]
feat_grp = ["cat_nan", "cat_nan", "num_nan"]
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2,random_state=10)
df_train = GeneralDataFrame(df_train, feat_grp=feat_grp)
df_test = GeneralDataFrame(df_test, feat_grp=feat_grp)
cv = KFold(n_splits=5,random_state=10,shuffle=True)
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
sgd.update_hyperparams({"loss": "log","random_state":10})

ret1 = fill_cat.fit_transform(df_train, y_train, df_valid, y_valid, df_test)
ret2 = fill_num.fit_transform(**ret1)
ret3 = ohe.fit_transform(**ret2)
sgd.fit(**ret3, y_train=y_train)

y_pred = sgd.predict(ret3["X_valid"])
acc_valid = accuracy_score(y_valid, y_pred)
print(acc_valid)

y_pred = sgd.predict(ret3["X_train"])
acc_train = accuracy_score(y_train, y_pred)
print(acc_train)

y_pred = sgd.predict(ret3["X_test"])
acc_test = accuracy_score(y_test, y_pred)
print(acc_test)
