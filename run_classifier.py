import pandas as pd
import joblib
df=joblib.load("data/2198.bz2")
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.manager.data_container.ndarray import NdArrayContainer
from autoflow.hdl.utils import get_default_hdl_bank
from autoflow.workflow.components.classification.catboost import CatBoostClassifier

from copy import deepcopy

def get_hp_of_cls(cls, hdl_bank, key1, kk=None):
    module = cls.__module__
    key2 = module.split(".")[-1]
    hp = deepcopy(hdl_bank[key1])
    if kk is not None:
        hp=hp[kk]
    hp=hp[key2]
    res={}
    for k, v in hp.items():
        if isinstance(v, dict) and "_default" in v:
            res[k] = v["_default"]
        if not isinstance(v, (dict, list)):
            res[k]=v
    return res

y=df.pop("labels")
df.pop("Smiles")
df.pop("Name")
df.pop("pIC50")

values=df.values

M=5000

selector=SelectFromModel(
    ExtraTreesClassifier(n_estimators=10,max_depth=7,min_samples_split=10,min_samples_leaf=10,n_jobs=12),
    threshold=-np.inf,max_features=M)
selected=selector.fit_transform(values, y)

X_train, X_test, y_train, y_test = train_test_split(
     selected, y, test_size=0.33, random_state=42)

X_train_=DataFrameContainer("TrainSet",dataset_instance=X_train)
X_train_.set_feature_groups(["num"]*M)
y_train_=NdArrayContainer("TrainLabel", dataset_instance=np.array(y_train))
X_test_=DataFrameContainer("TestSet",dataset_instance=X_test)
X_test_.set_feature_groups(["num"]*M)
y_test_=NdArrayContainer("TestLabel", dataset_instance=np.array(y_test))


hdl_bank = get_default_hdl_bank()

reg=CatBoostClassifier(
    **get_hp_of_cls(CatBoostClassifier, hdl_bank, "classification")
)
reg.in_feature_groups="num"

reg.fit(X_train=X_train_, y_train=y_train_, X_valid=X_test_, y_valid=y_test_)
score=reg.component.score(X_test, y_test).mean()
print(score)