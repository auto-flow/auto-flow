from typing import Dict

from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm

__all__=["FeatureAgglomeration"]

class FeatureAgglomeration(AutoPLPreprocessingAlgorithm):
    module__ = "sklearn.cluster"
    class__ = "FeatureAgglomeration"

    def after_process_hyperparams(self,hyperparams) ->Dict:
        hyperparams=super(FeatureAgglomeration, self).after_process_hyperparams(hyperparams)
        pop_name = "_n_clusters_ratio"
        if pop_name in hyperparams:
            n_clusters_ratio = hyperparams[pop_name]
            hyperparams.pop(pop_name)
            if hasattr(self, "shape"):
                n_clusters = max(
                    int(self.shape[1] * n_clusters_ratio),
                    1
                )
            else:
                n_clusters = 100
            hyperparams["n_clusters"] = n_clusters
        return hyperparams