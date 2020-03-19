import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from autopipeline.estimator.base import AutoPipelineEstimator
from autopipeline.pipeline.components.classification.catboost import CatBoostClassifier
from autopipeline.pipeline.components.classification.lightgbm import LGBMClassifier
from autopipeline.pipeline.components.feature_engineer.encode.label import LabelEncoder
from autopipeline.pipeline.components.feature_engineer.impute.fill_abnormal import FillAbnormal
from autopipeline.pipeline.components.feature_engineer.operate.merge import Merge
from autopipeline.pipeline.components.feature_engineer.operate.split.cat import SplitCat
from autopipeline.pipeline.components.feature_engineer.operate.split.cat_num import SplitCatNum
from autopipeline.pipeline.components.feature_engineer.operate.split.nan import SplitNan
from autopipeline.pipeline.dataframe import GeneralDataFrame
from autopipeline.tuner.smac_tuner import SmacPipelineTuner

df = pd.read_csv("../examples/classification/train_classification.csv")
y = df.pop("Survived").values
df = df.loc[:, ["Pclass", "Name", "Sex", "Age", "SibSp", "Ticket", "Fare", "Cabin", "Embarked"]]
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=10)
feat_grp=["num", "cat", "cat", "nan", "num", "cat", "num", "nan", "nan"]

# df_train = GeneralDataFrame(df_train, feat_grp=feat_grp)
# df_test = GeneralDataFrame(df_test, feat_grp=feat_grp)
# cv = KFold(n_splits=5, random_state=10, shuffle=True)
# train_ix, valid_ix = next(cv.split(df_train))
# df_train, df_valid = df_train.split([train_ix, valid_ix])
# y_valid = y_train[valid_ix]
# y_train = y_train[train_ix]

tuner = SmacPipelineTuner(
    random_state=50,
    initial_runs=12,
    runcount_limit=12,
)
auto_pipeline = AutoPipelineEstimator(tuner)
auto_pipeline.fit(df_train,y_train,df_test,y_test,feat_grp,n_jobs=1)