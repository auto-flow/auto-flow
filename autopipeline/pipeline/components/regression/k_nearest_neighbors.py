from typing import Dict

from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class KNearestNeighborsRegressor(AutoPLRegressionAlgorithm):
    class__ = "KNeighborsRegressor"
    module__ = "sklearn.neighbors"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(KNearestNeighborsRegressor, self).after_process_hyperparams(hyperparams)
        if "n_neighbors" in self.hyperparams:
            hyperparams["n_neighbors"] = min(self.shape[0] - 1, hyperparams["n_neighbors"])
        return hyperparams