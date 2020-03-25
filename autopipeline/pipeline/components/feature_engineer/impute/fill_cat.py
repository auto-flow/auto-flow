from typing import Dict

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm
from autopipeline.pipeline.components.utils import stack_Xs

__all__ = ["FillCat"]


class FillCat(AutoPLPreprocessingAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(FillCat, self).after_process_hyperparams(hyperparams)
        if hyperparams.get("strategy") == "<NULL>":
            hyperparams["fill_value"]="<NULL>"
            hyperparams["strategy"]="constant"
        return hyperparams

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None):
        return stack_Xs(X_train,X_valid,X_test)
