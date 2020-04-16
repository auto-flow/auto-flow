from autoflow.utils.ml_task import MLTask

binary_classification_task = MLTask("classification", "binary", "classifier")
multiclass_classification_task = MLTask("classification", "multiclass", "classifier")
multilabel_classification_task = MLTask("classification", "multilabel", "classifier")
regression_task = MLTask("regression", "regression", "regressor")
PHASE1 = "preprocessing"
PHASE2 = "estimating"
