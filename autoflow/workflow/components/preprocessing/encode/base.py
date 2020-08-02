from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm


class BaseEncoder(AutoFlowFeatureEngineerAlgorithm):

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        # todo 用cols代替 astype str
        estimator.cols = list(X_train.columns)
        return estimator
