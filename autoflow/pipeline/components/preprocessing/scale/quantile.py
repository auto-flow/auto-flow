from typing import Dict

from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["QuantileTransformerComponent"]

class QuantileTransformerComponent(AutoFlowFeatureEngineerAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(QuantileTransformerComponent, self).after_process_hyperparams(hyperparams)
        hyperparams["n_quantiles"]=min(self.shape[1],hyperparams["n_quantiles"])
        return hyperparams

