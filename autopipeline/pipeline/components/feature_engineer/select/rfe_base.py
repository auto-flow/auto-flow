from importlib import import_module

from autopipeline.pipeline.components.feature_engineer.select.sklearn_select_mixin import SklearnSelectMixin
from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm


class REF_Base(AutoPLPreprocessingAlgorithm, SklearnSelectMixin):
    class__ = "RFE"
    module__ = "sklearn.feature_selection"

    def after_process_hyperparams(self, hyperparams):
        hyperparams = super(REF_Base, self).after_process_hyperparams(hyperparams)
        estimator_ = hyperparams["estimator"]
        splitted = estimator_.split(".")
        class_ = splitted[-1]
        module_ = ".".join(splitted[:-1])
        M = import_module(module_)
        cls = getattr(M, class_)
        base_estimator_hp = self.filter_invalid(cls, hyperparams)
        if "max_features" in base_estimator_hp:
            base_estimator_hp.pop("max_features")
        base_estimator = cls(**base_estimator_hp)
        hyperparams["estimator"] = base_estimator
        return hyperparams
