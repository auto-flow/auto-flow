from sklearn.base import TransformerMixin

from autoflow.constants import STACK_X_MSG
from autoflow.data_container import DataFrameContainer
from autoflow.data_container.base import get_container_data
from autoflow.workflow.components.base import AutoFlowComponent
from autoflow.workflow.components.utils import stack_Xs
import pandas as pd
import numpy as np


class AutoFlowFeatureEngineerAlgorithm(AutoFlowComponent, TransformerMixin):
    need_y = False

    def fit_transform(self, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None,
                      ):
        return self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test).transform(X_train, X_valid, X_test, y_train)

    def assemble_result(self, X_stack, X_origin):
        if X_origin is None:
            return None
        return X_stack.sub_sample(X_origin.index)

    def assemble_all_result(self, X_stack, X_trans, X_train, X_valid, X_test, y_train):
        X_stack_trans = X_stack.replace_feature_groups(self.in_feature_groups, X_trans, self.out_feature_groups)
        X_train = self.assemble_result(X_stack_trans, X_train)
        X_valid = self.assemble_result(X_stack_trans, X_valid)
        X_test = self.assemble_result(X_stack_trans, X_test)
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "X_test": X_test,
        }

    def get_X_stack(self, X_train, X_valid, X_test):
        X_stack = X_train.copy()
        X_stack.data = stack_Xs(
            get_container_data(X_train),
            get_container_data(X_valid),
            get_container_data(X_test),
        )
        return X_stack

    def transform(self, X_train=None, X_valid=None, X_test=None, y_train=None, return_stack_trans=False):
        X_stack = self.get_X_stack(X_train, X_valid, X_test)
        if not self.is_fit:
            if return_stack_trans:
                X_trans=pd.DataFrame(np.zeros([X_stack.shape[0],0]))
                return X_stack, X_trans
            self.logger.warning(
                f"Component: {self.__class__.__name__} is not fitted. Maybe it fit a empty data.\nReturn origin X defaultly.")
            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_valid": X_valid,
                "X_test": X_test,
            }
        # fixme return_stack_trans
        # 1. stack_Xs(values)
        # 2. only activated feature_groups
        X_ = X_stack.filter_feature_groups(self.in_feature_groups, copy=True)
        # 3. get core data before transform
        X_data = self.before_trans_X(X_)
        # 4. transform by component.transform()
        X_trans = self._transform_procedure(X_data)
        if return_stack_trans:
            return X_stack, X_trans
        # replace core data to 'in_feature_groups', and rename feature_groups to 'out_feature_groups'
        return self.assemble_all_result(X_stack, X_trans, X_train, X_valid, X_test, y_train)

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
