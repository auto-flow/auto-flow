from copy import deepcopy

from sklearn.svm import LinearSVC

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["LibLinear_Preprocessor"]

class LibLinear_Preprocessor(AutoPLPreprocessingAlgorithm):
    class__ = "SelectFromModel"
    module__ = "sklearn.feature_selection"
    classification_only = True

    def after_process_hyperparams(self,hyperparams):
        hyperparams=super(LibLinear_Preprocessor, self).after_process_hyperparams(hyperparams)
        self.base_estimator=LinearSVC()
        hyperparams.update({
            "estimator":self.base_estimator
        })
        return hyperparams