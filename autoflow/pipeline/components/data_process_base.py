import pandas as pd

from autoflow.manager.data_container.dataframe import DataFrameContainer
from autoflow.manager.data_container.base import copy_data_container_structure
from autoflow.pipeline.components.base import AutoFlowComponent


class AutoFlowDataProcessAlgorithm(AutoFlowComponent):
    need_y = True

    def fit(self, X_train, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        self.build_proxy_estimator()
        return self

    def fit_transform(self, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None):
        return self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test).transform(X_train, X_valid, X_test, y_train)

    def transform(self, X_train: DataFrameContainer = None, X_valid=None, X_test=None, y_train=None):
        sample_X_test = self.hyperparams.get("sample_X_test", False)
        if y_train is not None:
            X_train, y_train = self._transform(X_train, y_train)
        if (not self.need_y) and sample_X_test:
            X_valid, _ = self._transform(X_valid, None)
            X_test, _ = self._transform(X_test, None)
        return {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "y_train": y_train
        }

    def _transform(self, X: DataFrameContainer, y):
        X_data, y_ = self._transform_proc(X.data, y)
        X = copy_data_container_structure(X)
        X_data = pd.DataFrame(X_data, columns=X.columns, index=X.index)
        X.data = X_data
        return X, y_

    def _transform_proc(self, X_train, y_train):
        return self.estimator.fit_sample(X_train, y_train)
