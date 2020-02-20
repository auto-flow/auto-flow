from copy import deepcopy
from importlib import import_module
from sklearn.base import BaseEstimator


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
    def module_(cls):
        if not cls.module__:
            raise NotImplementedError()
        return cls.module__

    def get_estimator_class(self):
        M=import_module(self.module_)
        return getattr(M,self.class_)

    def fit(self,X,y):
        cls=self.get_estimator_class()
        self.estimator=cls(**self.hyperparams)
        self.estimator.fit(X,y)
        return self


    @classmethod
    def set_params_from_dict(cls, dict_: dict):
        for key, value in dict_.items():
            setattr(cls, key, value)

    @classmethod
    def _get_param_names(cls):
        # in base class, return several parameters from __init__ function
        return list(cls.cls_hyperparams.keys())

    @classmethod
    def set_cls_hyperparams(cls, hp: dict):
        '''set default hyperparameters in init'''
        cls.cls_hyperparams = hp

    @classmethod
    def update_cls_hyperparams(cls, hp: dict):
        '''set default hyperparameters in init'''
        cls.cls_hyperparams.update(hp)

    def update_hyperparams(self, hp: dict):
        '''set default hyperparameters in init'''
        self.hyperparams.update(hp)
        self.set_params(**self.hyperparams)

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

    def __init__(self):
        super(AutoPLClassificationAlgorithm, self).__init__()
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

    def fit(self,X,y=None):
        cls=self.get_estimator_class()
        self.preprocessor=cls(**self.rebuild_hyperparameters())
        self.preprocessor.fit(X,y)
        return self

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

    def rebuild_hyperparameters(self):
        hp = self.hyperparams
        __n_components_ratio = "n_components_ratio"
        if __n_components_ratio in hp:
            n_components_ratio = hp[__n_components_ratio]
            hp.pop(__n_components_ratio)
            if hasattr(self, "shape"):
                n_components = int(self.shape[1] * n_components_ratio)
            else:
                n_components = 100
            hp["n_components"] = n_components
        return hp
