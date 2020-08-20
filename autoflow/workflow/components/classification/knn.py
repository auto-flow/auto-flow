from typing import Dict

from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["KNeighborsClassifier"]


class KNeighborsClassifier(AutoFlowClassificationAlgorithm):
    class__ = "KNeighborsClassifier"
    module__ = "sklearn.neighbors"

    # fixme ovr

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(KNeighborsClassifier, self).after_process_hyperparams(hyperparams)
        if "n_neighbors" in self.hyperparams:
            hyperparams["n_neighbors"] = min(self.shape[0] - 1, hyperparams["n_neighbors"])
        return hyperparams
