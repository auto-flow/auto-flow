from typing import Dict

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["RobustScalerComponent"]

class RobustScalerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "RobustScaler"
    module__ = "sklearn.preprocessing"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(RobustScalerComponent, self).after_process_hyperparams(hyperparams)
        q_min=hyperparams.pop("q_min")
        q_max=hyperparams.pop("q_max")
        hyperparams["quantile_range"]=(q_min,q_max)
        return hyperparams
