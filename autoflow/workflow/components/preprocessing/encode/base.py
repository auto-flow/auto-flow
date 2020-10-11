from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm


class BaseEncoder(AutoFlowFeatureEngineerAlgorithm):

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        # todo 用cols代替 astype str
        estimator.cols = list(X_train.columns)
        return estimator

class BaseCategoryEncoders(BaseEncoder):


    def _transform_procedure(self, X):
        if X is None:
            return None
        else:
            X_ = X.astype(str)
            trans = self.component.transform(X_)
            return trans

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, columns_metadata=None, **kwargs):
        X_ = X.astype(str)
        return estimator.fit(X_,y)

