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

    def __init__(self):
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

    def after_process_hyperparams(self) -> Dict:
        return self.hyperparams

    def after_process_estimator(self, estimator, X, y):
        return estimator

    def fit(self, X, y):
        self.shape = X.shape
        cls = self.get_estimator_class()
        self.estimator = cls(**self.after_process_hyperparams())
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

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of the underlying algorithm.

        Find more information at :ref:`get_properties`

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        dict
        """
        raise NotImplementedError()


class AutoPLRegressionAlgorithm(AutoPLComponent):
    """Provide an abstract interface for regression algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/pipeline/components/regression` to make it available."""

    def __init__(self):
        super(AutoPLRegressionAlgorithm, self).__init__()
        self.estimator = None
        self.properties = None

    def predict(self, X):
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator


class AutoPLClassificationAlgorithm(AutoPLComponent):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    OVR__: bool = False

    def isOVR(self):
        return self.OVR__

    def __init__(self):
        super(AutoPLClassificationAlgorithm, self).__init__()
        self.estimator = None
        self.properties = None

    def after_process_estimator(self, estimator, X, y):
        if self.isOVR() and get_task_from_y(y).subTask != "binary":
            estimator = OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    def predict(self, X):
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,) or shape = (n_samples, n_labels)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """Predict probabilities.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        if not self.estimator or (not hasattr(self.estimator, "predict_proba")):
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator


class AutoPLPreprocessingAlgorithm(AutoPLComponent):
    """Provide an abstract interface for preprocessing algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    def __init__(self):
        super(AutoPLPreprocessingAlgorithm, self).__init__()
        self.preprocessor = None

    def transform(self, X):
        """The transform function calls the transform function of the
        underlying scikit-learn model and returns the transformed array.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        X : array
            Return the transformed training data

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        if not self.preprocessor or (not hasattr(self.preprocessor, "transform")):
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_preprocessor(self):
        """Return the underlying preprocessor object.

        Returns
        -------
        preprocessor : the underlying preprocessor object
        """
        return self.preprocessor

    def after_process_hyperparams(self):
        hyperparams = deepcopy(self.hyperparams)
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
