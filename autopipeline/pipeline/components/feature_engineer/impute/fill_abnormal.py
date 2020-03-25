from typing import Dict

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["FillAbnormal"]


class FillAbnormal(AutoPLPreprocessingAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(FillAbnormal, self).after_process_hyperparams(hyperparams)
        hyperparams["fill_value"] = -999
        hyperparams["strategy"] = "constant"
        return hyperparams
