from typing import Dict
from importlib import import_module
from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["FeatureAgglomeration"]

class FeatureAgglomeration(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.cluster"
    class__ = "FeatureAgglomeration"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(FeatureAgglomeration, self).after_process_hyperparams(hyperparams)
        pooling_func=hyperparams["pooling_func"]
        module,func=pooling_func.split(".")
        M=import_module(module)
        func=getattr(M,func)
        hyperparams["pooling_func"]=func
        return hyperparams
