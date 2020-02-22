from copy import deepcopy

from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class AdaboostRegressor(AutoPLRegressionAlgorithm):
    class__ = "AdaBoostRegressor"
    module__ = "sklearn.ensemble"

    def after_process_hyperparams(self):
        import sklearn.tree
        hyperparams = deepcopy(self.hyperparams)
        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=hyperparams.pop("max_depth"))
        hyperparams.update({
            "base_estimator":base_estimator
        })
        return hyperparams