from collections import namedtuple


class MLTask(namedtuple("Task", ["mainTask", "subTask", "role"])):
    pass

binary_classification_task=MLTask("classification", "binary", "classifier")
multiclass_classification_task=MLTask("classification", "multiclass", "classifier")
multilabel_classification_task=MLTask("classification", "multilabel", "classifier")
regression_task=MLTask("regression", "regression", "regressor")


