from collections import namedtuple


class Task(namedtuple("Task", ["mainTask", "subTask", "role"])):
    pass

binary_classification_task=Task("classification", "binary", "classifier")
multiclass_classification_task=Task("classification", "multiclass", "classifier")
multilabel_classification_task=Task("classification", "multilabel", "classifier")
regression_task=Task("regression", "regression", "regressor")


