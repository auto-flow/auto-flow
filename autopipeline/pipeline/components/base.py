import inspect
from copy import deepcopy
from importlib import import_module
from typing import Dict

from sklearn.base import BaseEstimator

from autopipeline.utils.data import densify


class AutoPLComponent(BaseEstimator):
    cls_hyperparams: dict = {}
    module__ = None
    class__ = None
    classification_only = False
    regression_only = False

    def __init__(self):
        self.estimator = None
        self.hyperparams = deepcopy(self.cls_hyperparams)
        self.set_params(**self.hyperparams)

    # @classmethod
    @property
    def class_(cls):
        if not cls.class__:
            raise NotImplementedError()
        return cls.class__

    # @classmethod
    @property
    def module_(self):
        if not self.module__:
            raise NotImplementedError()
        return self.module__

    def get_estimator_class(self):
        M = import_module(self.module_)
        return getattr(M, self.class_)

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = deepcopy(hyperparams)
        should_pop = []
        updated = {}
        for key, value in hyperparams.items():
            key: str
            if key.startswith("_") and (not key.startswith("__")):
                should_pop.append(key)
                key = key[1:]
                new_key, indicator = key.split("-")
                updated[new_key] = self.do_process(indicator, hyperparams, value)
        for key in should_pop:
            hyperparams.pop(key)
        hyperparams.update(updated)
        return hyperparams

    def after_process_estimator(self, estimator, X, y):
        return estimator

    def before_fit_X(self, X):
        return X

    def before_fit_y(self, y):
        return y

    def filter_invalid(self, cls, hyperparams: Dict) -> Dict:
        validated = {}
        for key, value in hyperparams.items():
            if key in inspect.signature(cls.__init__).parameters.keys():
                validated[key] = value
            else:
                pass
        return validated

    def fit(self, X, y):
        self.shape = X.shape
        cls = self.get_estimator_class()
        self.estimator = cls(
            **self.filter_invalid(
                cls, self.after_process_hyperparams(self.hyperparams)
            )
        )
        X = self.before_fit_X(X)
        y = self.before_fit_y(y)
        self.estimator = self.after_process_estimator(self.estimator, X, y)
        X=densify(X)
        self.estimator.fit(X, y)
        return self

    def set_addition_info(self, dict_: dict):
        for key, value in dict_.items():
            setattr(self, key, value)

    def update_hyperparams(self, hp: dict):
        '''set default hyperparameters in init'''
        self.hyperparams.update(hp)
        # fixme ValueError: Invalid parameter C for estimator LibSVM_SVC(). Check the list of available parameters with `estimator.get_params().keys()`.
        # self.set_params(**self.hyperparams)

    def get_estimator(self):
        return self.estimator

    def do_process(self, indicator, hyperparams, value):
        if indicator == "lr_ratio":
            lr = hyperparams["learning_rate"]
            return int(value * (1 / lr))
        elif indicator == "sp1_ratio":
            if hasattr(self, "shape"):
                n_components = max(
                    int(self.shape[1] * value),
                    1
                )
            else:
                print("warn")
                n_components = 100
            return n_components
        else:
            raise NotImplementedError()
