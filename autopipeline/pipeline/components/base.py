from copy import deepcopy
from importlib import import_module
from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier

from autopipeline.utils.data import get_task_from_y


class AutoPLComponent(BaseEstimator):
    cls_hyperparams: dict = {}
    name: dict = None
    module__ = None
    class__ = None
    classification_only=False
    regression_only=False

    def __init__(self):
        self.estimator=None
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

    def after_process_hyperparams(self,hyperparams) -> Dict:
        return hyperparams

    def after_process_estimator(self, estimator, X, y):
        return estimator

    def before_fit_X(self, X):
        return X

    def before_fit_y(self, y):
        return y

    def fit(self, X, y):
        self.shape = X.shape
        cls = self.get_estimator_class()
        self.estimator = cls(**self.after_process_hyperparams(self.hyperparams))
        X=self.before_fit_X(X)
        y=self.before_fit_y(y)
        self.estimator = self.after_process_estimator(self.estimator, X, y)
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


class AutoPLRegressionAlgorithm(AutoPLComponent):
    """Provide an abstract interface for regression algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/pipeline/components/regression` to make it available."""


    def after_process_pred_y(self,y):
        return y

    def predict(self, X):
        if not self.estimator:
            raise NotImplementedError()
        pred_y= self.estimator.predict(X)
        return self.after_process_pred_y(pred_y)

class AutoPLClassificationAlgorithm(AutoPLComponent):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    OVR__: bool = False

    def isOVR(self):
        return self.OVR__


    def after_process_estimator(self, estimator, X, y):
        if self.isOVR() and get_task_from_y(y).subTask != "binary":
            estimator = OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    def after_process_pred_y(self,y):
        return y

    def predict(self, X):
        if not self.estimator:
            raise NotImplementedError()
        pred_y= self.estimator.predict(X)
        return self.after_process_pred_y(pred_y)

    def predict_proba(self, X):
        if not self.estimator or (not hasattr(self.estimator, "predict_proba")):
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

class AutoPLPreprocessingAlgorithm(AutoPLComponent):
    """Provide an abstract interface for preprocessing algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""
    def before_trans_X(self, X):
        return X

    def transform(self, X):
        if not self.estimator or (not hasattr(self.estimator, "transform")):
            raise NotImplementedError()
        X=self.before_trans_X(X)
        return self.estimator.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def after_process_hyperparams(self,hyperparams)->Dict:
        hyperparams = deepcopy(hyperparams)
        pop_name = "_n_components_ratio"
        if pop_name in hyperparams:
            n_components_ratio = hyperparams[pop_name]
            hyperparams.pop(pop_name)
            if hasattr(self, "shape"):
                n_components = max(
                    int(self.shape[1] * n_components_ratio),
                    1
                )
            else:
                n_components = 100
            hyperparams["n_components"] = n_components
        return hyperparams
