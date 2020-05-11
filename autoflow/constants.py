import re

from autoflow.utils.ml_task import MLTask

binary_classification_task = MLTask("classification", "binary", "classifier")
multiclass_classification_task = MLTask("classification", "multiclass", "classifier")
multilabel_classification_task = MLTask("classification", "multilabel", "classifier")
regression_task = MLTask("regression", "regression", "regressor")
PHASE1 = "preprocessing"
PHASE2 = "estimating"
SERIES_CONNECT_LEADER_TOKEN = "#"
SERIES_CONNECT_SEPARATOR_TOKEN = "|"
NATIVE_FEATURE_GROUPS = ("text", "date", "cat", "highR_cat", "num")
AUXILIARY_FEATURE_GROUPS = ("id", "target", "ignore")
UNIQUE_FEATURE_GROUPS = ("id", "target")
NAN_FEATURE_GROUPS = ("nan", "highR_nan")
VARIABLE_PATTERN = re.compile(f"[a-zA-Z_][a-zA-Z_0-9]]*")
