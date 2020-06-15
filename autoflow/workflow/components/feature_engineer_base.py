from sklearn.base import TransformerMixin

from autoflow.constants import STACK_X_MSG
from autoflow.data_container import DataFrameContainer
from autoflow.workflow.components.base import AutoFlowComponent
from autoflow.workflow.components.utils import stack_Xs


class AutoFlowFeatureEngineerAlgorithm(AutoFlowComponent, TransformerMixin):
    need_y = False

    def fit_transform(self, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None,
                      ):
        return self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test).transform(X_train, X_valid, X_test, y_train)

    def transform(self, X_train=None, X_valid=None, X_test=None, y_train=None):
        if not self.is_fit:
            self.logger.warning(
                f"Component: {self.__class__.__name__} is not fitted. Maybe it fit a empty data.\nReturn origin X defaultly.")
        else:
            X_train = self._transform(X_train)
            X_valid = self._transform(X_valid)
            X_test = self._transform(X_test)
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "X_test": X_test,
        }

    def _transform_procedure(self, X):
        if X is None:
            return None
        else:
            return self.component.transform(X)

    def _transform(self, X: DataFrameContainer):
        if X is None:
            return None
        X_ = X.filter_feature_groups(self.in_feature_groups, True)
        X_data = self.before_trans_X(X_)
        X_trans = self._transform_procedure(X_data)
        return X.replace_feature_groups(self.in_feature_groups, X_trans, self.out_feature_groups)

    def before_trans_X(self, X):
        return X.data

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None, **kwargs):
        X_train = self.before_fit_X(X_train)
        X_valid = self.before_fit_X(X_valid)
        X_test = self.before_fit_X(X_test)
        if not self.need_y:
            self.logger.debug(STACK_X_MSG)
            return stack_Xs(X_train, X_valid, X_test)
        else:
            return X_train
