from copy import deepcopy

from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class AdaboostRegressor(AutoPLRegressionAlgorithm):
    class__ = "AdaBoostRegressor"
    module__ = "sklearn.ensemble"

    def after_process_hyperparams(self,hyperparams):
        import sklearn.tree
        hyperparams = deepcopy(hyperparams)
        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=hyperparams.pop("max_depth"))
        hyperparams.update({
            "base_estimator":base_estimator
        })
        return hyperparams