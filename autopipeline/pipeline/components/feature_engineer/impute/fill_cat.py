from typing import Dict

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["FillCat"]


class FillCat(AutoPLPreprocessingAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(FillCat, self).after_process_hyperparams(hyperparams)
        if hyperparams.get("strategy") == "<NULL>":
            hyperparams["fill_value"] = "<NULL>"
            hyperparams["strategy"] = "constant"
        return hyperparams
