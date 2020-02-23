from copy import deepcopy

from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm

class GradientBoostingClassifier(AutoPLClassificationAlgorithm):
    module__ =  "sklearn.ensemble"
    class__ = "GradientBoostingClassifier"

    def after_process_hyperparams(self,hyperparams):
        hyperparams=deepcopy(hyperparams)
        _estimators_lr_ratio=hyperparams.pop("_estimators_lr_ratio")
        hyperparams.update({
            "n_estimators":int((1/hyperparams["learning_rate"])*_estimators_lr_ratio)
        })
        return hyperparams
