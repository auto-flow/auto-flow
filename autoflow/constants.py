import enum
import re

from autoflow.utils.ml_task import MLTask


class ExperimentType(enum.Enum):
    AUTO = "auto_modeling"
    MANUAL = "manual_modeling"
    ENSEMBLE = "ensemble_modeling"


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
JOBLIB_CACHE = "/tmp/joblib_cache"
ITERATIONS_BUDGET_MODE = "iterations"
SUBSAMPLES_BUDGET_MODE = "subsamples"
RESOURCE_MANAGER_CLOSE_ALL_LOGGER = "ResourceManager.close_all"
CONNECTION_POOL_CLOSE_MSG = "Connection pool in ResourceManger all closed."
START_SAFE_CLOSE_MSG = "Start to safely close connection pool..."
END_SAFE_CLOSE_MSG = "The connection pool has been safely closed."
STACK_X_MSG = "Stack Xs when prepare X to ."
LOGGING_LEVELS = {
    "CRITICAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
}
ERR_LOSS=65535
MINIMAL_COST_FOR_LOG=0.001
