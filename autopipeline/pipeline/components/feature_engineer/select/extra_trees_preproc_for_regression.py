from copy import deepcopy

from sklearn.ensemble import ExtraTreesRegressor

from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["ExtraTreesPreprocessorRegression"]

class ExtraTreesPreprocessorRegression(AutoPLPreprocessingAlgorithm):
    class__ = "SelectFromModel"
    module__ = "sklearn.feature_selection"


    def after_process_hyperparams(self,hyperparams):
        hyperparams=deepcopy(self.hyperparams)
        self.base_estimator=ExtraTreesRegressor()
        hyperparams.update({
            "estimator":self.base_estimator
        })
        return hyperparams
