from copy import deepcopy
from typing import Dict

import sklearn.feature_selection

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

excludeToken = True


class SelectPercentileBase(AutoPLPreprocessingAlgorithm):
    class__ = "SelectPercentile"
    module__ = "sklearn.feature_selection"

    def get_name2func(self):
        raise NotImplementedError()

    def get_default_name(self):
        raise NotImplementedError()

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = super(SelectPercentileBase, self).after_process_hyperparams(hyperparams)
        name2func = self.get_name2func()
        default_name = self.get_default_name()
        name = hyperparams.get("score_func", default_name)
        self.score_func = name2func[name]
        hyperparams.update({
            "score_func": self.score_func
        })
        return hyperparams

    def before_fit_X(self, X):
        X = deepcopy(X)
        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0.0
        return X

    def before_pred_X(self, X):
        X = deepcopy(X)
        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0.0
        return X
