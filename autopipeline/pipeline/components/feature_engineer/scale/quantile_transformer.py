from typing import Dict

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["QuantileTransformerComponent"]

class QuantileTransformerComponent(AutoPLPreprocessingAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(QuantileTransformerComponent, self).after_process_hyperparams(hyperparams)
        hyperparams["n_quantiles"]=min(self.shape[1],hyperparams["n_quantiles"])
        return hyperparams

