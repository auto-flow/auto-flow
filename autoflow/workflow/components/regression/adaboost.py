from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["AdaboostRegressor"]


class AdaboostRegressor(AutoFlowRegressionAlgorithm):
    class__ = "AdaBoostRegressor"
    module__ = "sklearn.ensemble"

    def after_process_hyperparams(self, hyperparams):
        import sklearn.tree
        hyperparams = super(AdaboostRegressor, self).after_process_hyperparams(hyperparams)
        base_estimator = sklearn.tree.DecisionTreeRegressor(max_depth=hyperparams.pop("max_depth"),
                                                            random_state=hyperparams.get("random_state", 42))
        hyperparams.update({
            "base_estimator": base_estimator
        })
        return hyperparams
