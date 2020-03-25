from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm
from autopipeline.pipeline.components.utils import stack_Xs


class BaseEncoder(AutoPLPreprocessingAlgorithm):
    need_y = False

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None):
        if not self.need_y:
            return stack_Xs(X_train, X_valid, X_test)
        else:
            return X_train

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            trans[X == -999] = -999
            return trans
