import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from autopipeline.pipeline.components.classification.catboost import CatBoostClassifier
from autopipeline.pipeline.components.classification.k_nearest_neighbors import KNearestNeighborsClassifier
from autopipeline.pipeline.components.classification.libsvm_svc import LibSVM_SVC
from autopipeline.pipeline.components.classification.lightgbm import LGBMClassifier
from autopipeline.pipeline.components.feature_engineer.encode.one_hot import OneHotEncoder
from autopipeline.pipeline.components.feature_engineer.impute.fill_cat import FillCat
from autopipeline.pipeline.components.feature_engineer.impute.fill_num import FillNum
from autopipeline.pipeline.components.feature_engineer.operate.drop import DropAll
from autopipeline.pipeline.components.feature_engineer.operate.split.cat import SplitCat
from autopipeline.pipeline.components.feature_engineer.operate.split.cat_num import SplitCatNum
from autopipeline.pipeline.components.feature_engineer.operate.split.nan import SplitNan
from autopipeline.pipeline.dataframe import GeneralDataFrame

df = pd.read_csv("../examples/classification/train_classification.csv")
y = df.pop("Survived").values
df = df.loc[:, ["Pclass", "Name", "Sex", "Age", "SibSp", "Ticket", "Fare", "Cabin", "Embarked"]]
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=10)
feat_grp = ["num", "cat", "cat", "nan", "num", "cat", "num", "nan", "nan"]
df_train = GeneralDataFrame(df_train, feat_grp=feat_grp)
df_test = GeneralDataFrame(df_test, feat_grp=feat_grp)
cv = KFold(n_splits=5, random_state=10, shuffle=True)
train_ix, valid_ix = next(cv.split(df_train))
df_train, df_valid = df_train.split([train_ix, valid_ix])
y_valid = y_train[valid_ix]
y_train = y_train[train_ix]
# 1. 将nan划分为highR_nan与lowR_nan
split_nan = SplitNan()
split_nan.update_hyperparams({
    "highR": "highR_nan",
    "lowR": "lowR_nan"
})
split_nan.in_feat_grp = "nan"
ret1 = split_nan.fit_transform(X_train=df_train, X_valid=df_valid, X_test=df_test)
# 2. 删除highR_nan
drop1 = DropAll()
drop1.in_feat_grp = "highR_nan"
drop1.out_feat_grp = "drop"
ret2 = drop1.fit_transform(**ret1)
# 3. 将lowR_nan切分成num_nan与cat_nan
split_cat_num = SplitCatNum()
split_cat_num.in_feat_grp = "lowR_nan"
split_cat_num.update_hyperparams({
    "cat_name": "cat_nan",
    "num_name": "num_nan"
})
ret3 = split_cat_num.fit_transform(**ret2)
# 4. 对num_nan进行impute, 填充为 median
fill_num = FillNum()
fill_num.update_hyperparams({"strategy": "median"})
fill_num.in_feat_grp = "num_nan"
fill_num.out_feat_grp = "num"
ret4 = fill_num.fit_transform(**ret3)
# 5. 对cat_nan进行impute, 填充为 <NULL>
fill_cat = FillCat()
fill_cat.update_hyperparams({"strategy": "<NULL>"})
fill_cat.in_feat_grp = "cat_nan"
fill_cat.out_feat_grp = "cat"
ret5 = fill_cat.fit_transform(**ret4)
# 6. 将cat分解为highR_cat与lowR_cat
split_cat = SplitCat()
split_cat.in_feat_grp = "cat"
split_cat.update_hyperparams({
    "highR": "highR_cat",
    "lowR": "lowR_cat"
})
ret6 = split_cat.fit_transform(**ret5)
# 7. 删除highR
drop2 = DropAll()
drop2.in_feat_grp = "highR_cat"
drop2.out_feat_grp = "drop"
ret7 = drop2.fit_transform(**ret6)
# 8. 对lowR_cat做label_encode 变成 num
ohe = OneHotEncoder()
ohe.in_feat_grp = "lowR_cat"
ohe.out_feat_grp = "num"
ret8 = ohe.fit_transform(**ret7)
# 9.1 lightGBM
lightgbm = LGBMClassifier()
lightgbm.in_feat_grp = "num"
ans = lightgbm.fit(**ret8, y_train=y_train, y_valid=y_valid)
y_pred = ans.predict(ret8["X_valid"])
acc = accuracy_score(y_valid, y_pred)
print("acc=", acc)
# 9.2 catboost
catboost = CatBoostClassifier()
catboost.in_feat_grp = "num"
ans = catboost.fit(**ret8, y_train=y_train, y_valid=y_valid)
y_pred = ans.predict(ret8["X_valid"])
acc = accuracy_score(y_valid, y_pred)
print("acc=", acc)
# 9.3 SVC
svc = LibSVM_SVC()
svc.in_feat_grp = "num"
ans = svc.fit(**ret8, y_train=y_train, y_valid=y_valid)
y_pred = ans.predict(ret8["X_valid"])
acc = accuracy_score(y_valid, y_pred)
print("acc=", acc)
# 9.4 knn
knn = KNearestNeighborsClassifier()
knn.in_feat_grp = "num"
ans = knn.fit(**ret8, y_train=y_train, y_valid=y_valid)
y_pred = ans.predict(ret8["X_valid"])
acc = accuracy_score(y_valid, y_pred)
print("acc=", acc)
