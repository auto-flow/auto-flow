from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm
from autopipeline.pipeline.components.utils import stack_Xs

__all__ = ["LabelEncoder"]


class LabelEncoder(AutoPLPreprocessingAlgorithm):
    class__ = "LabelEncoder"
    module__ = "autopipeline.pipeline.proxy.label_encode"

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None):
        return stack_Xs(X_train, X_valid, X_test)
