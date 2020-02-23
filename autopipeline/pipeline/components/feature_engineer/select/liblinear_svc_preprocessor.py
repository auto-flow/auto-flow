from copy import deepcopy

from sklearn.svm import LinearSVC

from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class LibLinear_Preprocessor(AutoPLPreprocessingAlgorithm):
    class__ = "SelectFromModel"
    module__ = "sklearn.feature_selection"


    def after_process_hyperparams(self,hyperparams):
        hyperparams=deepcopy(self.hyperparams)
        self.base_estimator=LinearSVC()
        hyperparams.update({
            "estimator":self.base_estimator
        })
        return hyperparams