from typing import Dict

from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["FillCat"]


class FillCat(AutoFlowFeatureEngineerAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(FillCat, self).after_process_hyperparams(hyperparams)
        if hyperparams.get("strategy") == "<NULL>":
            hyperparams["fill_value"] = "<NULL>"
            hyperparams["strategy"] = "constant"
        return hyperparams
