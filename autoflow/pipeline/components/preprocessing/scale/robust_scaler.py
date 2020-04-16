from typing import Dict

from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["RobustScalerComponent"]

class RobustScalerComponent(AutoFlowFeatureEngineerAlgorithm):
    class__ = "RobustScaler"
    module__ = "sklearn.preprocessing"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(RobustScalerComponent, self).after_process_hyperparams(hyperparams)
        q_min=hyperparams.pop("q_min")
        q_max=hyperparams.pop("q_max")
        hyperparams["quantile_range"]=(q_min,q_max)
        return hyperparams
