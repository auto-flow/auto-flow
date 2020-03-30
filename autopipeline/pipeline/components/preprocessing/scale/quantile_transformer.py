from typing import Dict

from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["QuantileTransformerComponent"]

class QuantileTransformerComponent(AutoPLFeatureEngineerAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(QuantileTransformerComponent, self).after_process_hyperparams(hyperparams)
        hyperparams["n_quantiles"]=min(self.shape[1],hyperparams["n_quantiles"])
        return hyperparams

