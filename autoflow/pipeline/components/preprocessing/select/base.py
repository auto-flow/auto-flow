from copy import deepcopy
from importlib import import_module
from typing import Dict

import numpy as np
import pandas as pd
import sklearn.feature_selection

from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm
from autoflow.utils.dataframe import DataFrameValuesWrapper


class SklearnSelectMixin():
    max_feature_name = "_max_features-sp1_percent"

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            mask = self.estimator.get_support()
            columns = X.columns[mask]
            return pd.DataFrame(trans, columns=columns)

    def before_parse_escape_hyperparameters(self, hyperparams):
        hyperparams = deepcopy(hyperparams)
        if "_select_percent" in hyperparams:
            _select_percent = hyperparams.pop("_select_percent")
            hyperparams[self.max_feature_name] = _select_percent

        return hyperparams


# class SelectFromModelBase(AutoPLFeatureEngineerAlgorithm, SklearnSelectMixin):
class SelectFromModelBase(SklearnSelectMixin, AutoFlowFeatureEngineerAlgorithm):
    class__ = "SelectFromModel"
    module__ = "sklearn.feature_selection"
    need_y = True
    max_feature_name = "_max_features-sp1_percent"

    def after_process_hyperparams(self, hyperparams):
        hyperparams = super(SelectFromModelBase, self).after_process_hyperparams(hyperparams)
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
        hyperparams["threshold"] = -np.inf
        return hyperparams


class REF_Base(SklearnSelectMixin, AutoFlowFeatureEngineerAlgorithm):
    class__ = "RFE"
    module__ = "sklearn.feature_selection"
    need_y = True
    max_feature_name = "_n_features_to_select-sp1_percent"

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


class SelectPercentileBase(SklearnSelectMixin, AutoFlowFeatureEngineerAlgorithm):
    class__ = "GenericUnivariateSelect"
    module__ = "sklearn.feature_selection"
    need_y = True
    max_feature_name = "param"

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
        wrapper = DataFrameValuesWrapper(X)
        X = wrapper.array
        if X is None:
            return None
        X = deepcopy(X)
        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0.0
        return wrapper.wrap_to_dataframe(X)

    def before_trans_X(self, X):
        wrapper = DataFrameValuesWrapper(X)
        X = wrapper.array
        X = deepcopy(X)
        if self.score_func == sklearn.feature_selection.chi2:
            X[X < 0] = 0.0
        return wrapper.wrap_to_dataframe(X)
