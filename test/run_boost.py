import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from autopipeline.pipeline.components.classification.catboost import CatBoostClassifier
from autopipeline.pipeline.components.classification.lightgbm import LGBMClassifier

from autopipeline.pipeline.components.feature_engineer.encode.label import LabelEncoder
from autopipeline.pipeline.components.feature_engineer.impute.fill_abnormal import FillAbnormal
from autopipeline.pipeline.components.feature_engineer.operate.merge import Merge
from autopipeline.pipeline.components.feature_engineer.operate.split.cat import SplitCat
from autopipeline.pipeline.components.feature_engineer.operate.split.cat_num import SplitCatNum
from autopipeline.pipeline.components.feature_engineer.operate.split.nan import SplitNan
from autopipeline.pipeline.dataframe import GenericDataFrame

df = pd.read_csv("../examples/classification/train_classification.csv")
y = df.pop("Survived").values
df = df.loc[:, ["Pclass", "Name", "Sex", "Age", "SibSp", "Ticket", "Fare", "Cabin", "Embarked"]]
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=10)
feat_grp=["num", "cat", "cat", "nan", "num", "cat", "num", "nan", "nan"]

df_train = GenericDataFrame(df_train, feat_grp=feat_grp)
df_test = GenericDataFrame(df_test, feat_grp=feat_grp)
cv = KFold(n_splits=5, random_state=10, shuffle=True)
train_ix, valid_ix = next(cv.split(df_train))
df_train, df_valid = df_train.split([train_ix, valid_ix])
y_valid = y_train[valid_ix]
y_train = y_train[train_ix]
# 1. 将nan划分为highR_nan与lowR_nan
split_nan = SplitNan()
split_nan.update_hyperparams({
    "highR_name": "highR_nan",
    "lowR": "lowR_nan"
})
split_nan.in_feat_grp = "nan"
ret1 = split_nan.fit_transform(X_train=df_train,X_valid=df_valid,X_test=df_test)
# 2. lightgbm对nan不敏感，不需要drop，保留highR_nan为lowR_nan
# fixme: 此操作与所有(非lightgbm模型)互斥
merge1 = Merge()
merge1.in_feat_grp = "highR_nan"
merge1.out_feat_grp = "lowR_nan"
ret2 = merge1.fit_transform(**ret1)
# 3. 将lowR_nan切分成num_nan与cat_nan
split_cat_num = SplitCatNum()
split_cat_num.in_feat_grp = "lowR_nan"
split_cat_num.update_hyperparams({
    "cat_name": "cat_nan",
    "num_name": "num_nan"
})
ret3 = split_cat_num.fit_transform(**ret2)
# 4. 对num_nan进行impute, 填充为-999
# fixme: 此操作与所有(非lightgbm模型)互斥
fill_abnormal_num = FillAbnormal()
fill_abnormal_num.in_feat_grp = "num_nan"
fill_abnormal_num.out_feat_grp = "num"
ret4 = fill_abnormal_num.fit_transform(**ret3)
# 5. 对cat_nan进行impute, 填充为-999
# fixme: 此操作与所有(非lightgbm模型)互斥
fill_abnormal_cat = FillAbnormal()
fill_abnormal_cat.in_feat_grp = "cat_nan"
fill_abnormal_cat.out_feat_grp = "cat"
ret5 = fill_abnormal_cat.fit_transform(**ret4)
# 6. 将cat分解为highR_cat与lowR_cat
split_cat = SplitCat()
split_cat.in_feat_grp = "cat"
split_cat.update_hyperparams({
    "highR": "highR_cat",
    "lowR": "lowR_cat"
})
ret6 = split_cat.fit_transform(**ret5)
# 7. 将highR_cat变成lowR_cat
# fixme: 此操作与所有(非lightgbm模型)互斥
merge2 = Merge()
merge2.in_feat_grp = "highR_cat"
merge2.out_feat_grp = "lowR_cat"
ret7 = merge2.fit_transform(**ret6)
# 8. 对lowR_cat做label_encode 变成 num
# fixme: 此操作与所有(非lightgbm模型)互斥
label_encode = LabelEncoder()
label_encode.in_feat_grp = "lowR_cat"
label_encode.out_feat_grp = "num"
ret8 = label_encode.fit_transform(**ret7)
# 9.1 lightGBM
lightgbm = LGBMClassifier()
lightgbm.in_feat_grp = "num"
ans = lightgbm.fit(**ret8,  y_train=y_train,y_valid=y_valid)
y_pred = ans.predict(ret8["X_test"])
acc = accuracy_score(y_test, y_pred)
print("acc=", acc)
# 9.1 catboost
catboost = CatBoostClassifier()
catboost.in_feat_grp = "num"
ans = catboost.fit(**ret8, y_train=y_train,y_valid=y_valid)
y_pred = ans.predict(ret8["X_test"])
acc = accuracy_score(y_test, y_pred)
print("acc=", acc)
