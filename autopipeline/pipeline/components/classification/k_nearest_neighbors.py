from typing import Dict

from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__ = ["KNearestNeighborsClassifier"]


class KNearestNeighborsClassifier(AutoPLClassificationAlgorithm):
    class__ = "KNeighborsClassifier"
    module__ = "sklearn.neighbors"

    # fixme ovr

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(KNearestNeighborsClassifier, self).after_process_hyperparams(hyperparams)
        if "n_neighbors" in self.hyperparams:
            hyperparams["n_neighbors"] = min(self.shape[0] - 1, hyperparams["n_neighbors"])
        return hyperparams
