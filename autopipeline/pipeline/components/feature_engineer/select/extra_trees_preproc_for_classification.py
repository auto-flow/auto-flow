from sklearn.ensemble import ExtraTreesClassifier

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["ExtraTreesPreprocessorClassification"]


class ExtraTreesPreprocessorClassification(AutoPLPreprocessingAlgorithm):
    class__ = "SelectFromModel"
    module__ = "sklearn.feature_selection"
    classification_only = True

    def after_process_hyperparams(self, hyperparams):
        hyperparams = super(ExtraTreesPreprocessorClassification, self).after_process_hyperparams(hyperparams)
        self.base_estimator = ExtraTreesClassifier()
        hyperparams.update({
            "estimator": self.base_estimator
        })
        return hyperparams
