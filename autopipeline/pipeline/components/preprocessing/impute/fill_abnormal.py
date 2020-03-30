from typing import Dict

from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__ = ["FillAbnormal"]


class FillAbnormal(AutoPLFeatureEngineerAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(FillAbnormal, self).after_process_hyperparams(hyperparams)
        hyperparams["fill_value"] = -999
        hyperparams["strategy"] = "constant"
        return hyperparams
