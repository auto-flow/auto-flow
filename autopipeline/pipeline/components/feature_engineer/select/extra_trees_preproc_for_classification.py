from copy import deepcopy

from sklearn.ensemble import ExtraTreesClassifier

from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["ExtraTreesPreprocessorClassification"]

class ExtraTreesPreprocessorClassification(AutoPLPreprocessingAlgorithm):
    class__ = "SelectFromModel"
    module__ = "sklearn.feature_selection"

    def after_process_hyperparams(self,hyperparams):
        hyperparams=deepcopy(self.hyperparams)
        self.base_estimator=ExtraTreesClassifier()
        hyperparams.update({
            "estimator":self.base_estimator
        })
        return hyperparams
