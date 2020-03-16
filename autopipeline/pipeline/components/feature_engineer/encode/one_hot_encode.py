from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm
from autopipeline.pipeline.components.utils import stack_Xs

__all__ = ["OneHotEncoder"]


class OneHotEncoder(AutoPLPreprocessingAlgorithm):
    class__ = "OneHotEncoder"
    module__ = "category_encoders"

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None):
        return stack_Xs(X_train, X_valid, X_test)
