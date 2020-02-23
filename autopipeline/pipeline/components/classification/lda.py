from copy import deepcopy
from typing import Dict

from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class LDA(AutoPLClassificationAlgorithm):
    class__ = "LinearDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True

    def after_process_hyperparams(self,hyperparams) -> Dict:
        hyperparams = deepcopy(hyperparams)
        pop_name = "_n_components_ratio"
        if pop_name in hyperparams:
            n_components_ratio = hyperparams[pop_name]
            hyperparams.pop(pop_name)
            if hasattr(self, "shape"):
                n_components = max(int(self.shape[1] * n_components_ratio), 1)
            else:

                n_components = 100
            hyperparams["n_components"] = n_components
        return hyperparams
