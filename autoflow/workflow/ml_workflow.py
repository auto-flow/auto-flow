import hashlib
from copy import deepcopy
from time import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.metaestimators import if_delegate_has_method

from autoflow.data_container.base import DataContainer
from autoflow.utils.hash import get_hash_of_dict, get_hash_of_str
from autoflow.utils.logging_ import get_logger
from autoflow.utils.ml_task import MLTask
from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.workflow.components.regression_base import AutoFlowRegressionAlgorithm

__all__ = ["ML_Workflow"]


class ML_Workflow(Pipeline):
    # todo: 现在是用类似拓扑排序的方式实现，但是计算是线性的，希望以后能用更科学的方法！
    # 可以当做Transformer，又可以当做estimator！
    def __init__(self, steps, should_store_intermediate_result=False, resource_manager=None):
        self.config_id = None
        self.config = None
        self.logger = get_logger(self)
        if resource_manager is None:
            from autoflow import ResourceManager
            self.logger.warning(
                "In ML_Workflow __init__, resource_manager is None, create a default local resource_manager.")
            resource_manager = ResourceManager()
        self.resource_manager = resource_manager
        self.should_store_intermediate_result = should_store_intermediate_result
        self.steps = steps
        self.memory = None
        self.verbose = False
        self._validate_steps()
        self.intermediate_result = {}
        self.fitted = False
        self.budget = 0

    def set_budget(self, budget):
        self.budget = budget

    @property
    def is_estimator(self):
        is_estimator = False
        if isinstance(self[-1], (AutoFlowClassificationAlgorithm, AutoFlowRegressionAlgorithm)):
            is_estimator = True
        return is_estimator

    def update_data_container_to_dataset_id(self, step_name, name: str, data_container: DataContainer,
                                            dict_: Dict[str, str]):
        if data_container is not None:
            data_copied = data_container.copy()
            data_copied.dataset_source = "IntermediateResult"
            task_id = getattr(self.resource_manager, "task_id", "")
            experiment_id = getattr(self.resource_manager, "experiment_id", "")
            data_copied.dataset_metadata.update(
                {"task_id": task_id, "experiment_id": experiment_id, "step_name": step_name})
            data_copied.upload("fs")
            dataset_id = data_copied.dataset_id
            dict_[name] = dataset_id

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, fit_final_estimator=True,
            resouce_manager=None):
        # set default `self.last_data` to prevent exception in only classifier cases
        self.last_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "X_valid": X_valid,
        }
        time_cost_list = []
        # if `is_estimator` or `fit_final_estimator` is True, `with_final` is False
        for (step_idx, step_name, transformer) in self._iter(
                with_final=(not self.is_estimator),
                filter_passthrough=False):
            # todo : 做中间结果的存储
            cache_intermediate = False
            hyperparams = {}
            hit_cache = False
            dataset_id = None
            cache_key = None
            start_time = time()
            if getattr(transformer, "cache_intermediate", False):
                if self.resource_manager is None:
                    self.logger.warning(
                        f"In ML Workflow step '{step_name}', 'cache_intermediate' is set to True, but resource_manager is None.")
                else:
                    hyperparams = getattr(transformer, "hyperparams")
                    if not isinstance(hyperparams, dict):
                        self.logger.warning(f"In ML Workflow step '{step_name}', transformer haven't 'hyperparams'.")
                    else:
                        cache_intermediate = True
            if cache_intermediate:
                if hasattr(transformer, "prepare_X_to_fit"):
                    def stack_X_before_fit(X_train, X_valid, X_test, **kwargs):
                        t = transformer
                        X_train_f = t.filter_feature_groups(X_train)
                        X_stack_ = t.prepare_X_to_fit(
                            X_train_f,
                            t.filter_feature_groups(X_valid),
                            t.filter_feature_groups(X_test),
                        )
                        X_stack_pre = X_train_f.copy()
                        X_stack_pre.data = X_stack_
                        return X_stack_pre

                else:
                    def stack_X_before_fit(X_train, X_valid, X_test, **kwargs):
                        self.logger.warning(
                            f"In ML Workflow step '{step_name}', transformer haven't attribute 'prepare_X_to_fit'. ")
                        return X_train

                X_stack_pre = stack_X_before_fit(X_train, X_valid, X_test)
                dataset_id = X_stack_pre.get_hash()
                component_name = transformer.__class__.__name__
                m = hashlib.md5()
                get_hash_of_str(component_name, m)
                get_hash_of_str(str(transformer.in_feature_groups), m)
                get_hash_of_str(str(transformer.out_feature_groups), m)
                component_hash = get_hash_of_dict(hyperparams, m)
                cache_key = f"workflow-{component_hash}-{dataset_id}"
                cache_results = self.resource_manager.cache.get(cache_key)
                if X_stack_pre.shape[1] > 0 and \
                        cache_results is not None and isinstance(cache_results, dict) \
                        and "X_trans" in cache_results and "component" in cache_results:
                    # hit cache
                    self.logger.debug(f"workflow cache hit, component_name = {component_name},"
                                      f" dataset_id = {dataset_id}, cache_key = '{cache_key}'")
                    X_trans = cache_results["X_trans"]
                    fitted_transformer = cache_results["component"]  # set the variable in later
                    X_stack = fitted_transformer.get_X_stack(X_train, X_valid, X_test)
                    result = fitted_transformer.assemble_all_result(
                        X_stack, X_trans, X_train, X_valid, X_test, y_train)
                    hit_cache = True
                else:
                    # not hit cache
                    self.logger.debug(f"workflow cache miss, component_name = {component_name},"
                                      f" dataset_id = {dataset_id}, cache_key = '{cache_key}'")
                    fitted_transformer = transformer.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
                    X_stack, X_trans = transformer.transform(X_train, X_valid, X_test, y_train, return_stack_trans=True)
                    result = transformer.assemble_all_result(X_stack, X_trans, X_train, X_valid, X_test, y_train)
                    if X_stack_pre.shape[1] > 0:
                        self.resource_manager.cache.set(
                            cache_key, {
                                "X_trans": X_trans,
                                "component": fitted_transformer
                            }
                        )
                    # todo: 增加一些元信息
            else:
                fitted_transformer = transformer.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
                result = transformer.transform(X_train, X_valid, X_test, y_train)
            if self.resource_manager.should_record_workflow_step:
                self.resource_manager.insert_workflow_step_record(
                    config_id=self.config_id,
                    experiment_id=self.resource_manager.experiment_id,
                    config=self.config,
                    step_idx=step_idx,
                    step_name=step_name,
                    component_name=transformer.__class__.__name__,
                    hyperparams=hyperparams,
                    dataset_id=dataset_id,
                    hit_cache=hit_cache,
                    cache_key=cache_key
                )
            X_train = result["X_train"]
            X_valid = result.get("X_valid")
            X_test = result.get("X_test")
            y_train = result.get("y_train")
            if self.should_store_intermediate_result:
                current_dict = {}
                self.update_data_container_to_dataset_id(step_name, "X_train", X_train, current_dict)
                self.update_data_container_to_dataset_id(step_name, "X_valid", X_valid, current_dict)
                self.update_data_container_to_dataset_id(step_name, "X_test", X_test, current_dict)
                self.update_data_container_to_dataset_id(step_name, "y_train", y_train, current_dict)
                self.intermediate_result.update({step_name: current_dict})
            self.last_data = result
            self.steps[step_idx] = (step_name, fitted_transformer)
            cost_time = time() - start_time
            time_cost_list.append([
                step_idx,
                step_name,
                transformer.__class__.__name__,
                hit_cache,
                cost_time,
            ])

        if (fit_final_estimator and self.is_estimator):
            # self._final_estimator.resource_manager = self.resource_manager
            start_time = time()
            self._final_estimator.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
            cost_time = time() - start_time
            time_cost_list.append([
                len(self.steps),
                self.steps[-1][0],
                self._final_estimator.__class__.__name__,
                False,
                cost_time
            ])
            # self._final_estimator.resource_manager = None
        self.fitted = True
        self.time_cost_list = time_cost_list
        return self

    def fit_transform(self, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None):
        return self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test). \
            transform(X_train, X_valid, X_test, y_train)

    def procedure(
            self, ml_task: MLTask, X_train, y_train, X_valid=None, y_valid=None,
            X_test=None, y_test=None, max_iter=-1, budget=0, should_finish_evaluation=False
    ):
        if max_iter > 0:
            # set final model' max_iter param
            self[-1].set_max_iter(max_iter)
        # 高budget的模型不能在低budget时刻被加载
        if self.fitted and self.budget <= budget and max_iter > 0:
            # todo: transform 的过程中使用缓存（如，gbt_imputer）
            if self.last_data is None:
                self.logger.debug(
                    f"fitted is True, (self.budget = {self.budget}) <= (budget = {budget}), self.last_data is None ")
                self.last_data = self.transform(X_train, X_valid, X_test, y_train)
            start_time = time()
            self[-1].fit(
                self.last_data.get("X_train"), self.last_data.get("y_train"),
                self.last_data.get("X_valid"), y_valid,
                self.last_data.get("X_test"), y_test
            )
            cost_time = time() - start_time
            self.time_cost_list.append([
                len(self.steps),
                self.steps[-1][0],
                self._final_estimator.__class__.__name__,
                False,
                cost_time
            ])
        else:
            self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
        self.set_budget(budget)
        if max_iter > 0 and should_finish_evaluation:
            self[-1].finish_evaluation()
        X_train = self.last_data["X_train"]
        y_train = self.last_data["y_train"]
        X_valid = self.last_data.get("X_valid")
        X_test = self.last_data.get("X_test")
        self.last_data = None  # GC
        try:
            check_array(X_test.data) if X_test is not None else None
            check_array(X_valid.data) if X_valid is not None else None
            if ml_task.mainTask == "classification":
                pred_valid = self._final_estimator.predict_proba(X_valid)
                pred_test = self._final_estimator.predict_proba(X_test) if X_test is not None else None
            else:
                pred_valid = self._final_estimator.predict(X_valid)
                pred_test = self._final_estimator.predict(X_test) if X_test is not None else None
        except Exception as e:
            # self.logger.warning(f"INF: {np.count_nonzero(~np.isfinite(X_test.data), axis=0)}")
            # self.logger.warning(f"NAN: {np.count_nonzero(pd.isna(X_test.data), axis=0)}")
            self.logger.error(e)
            pred_test = -65535
            pred_valid = -65535
        self.resource_manager = None  # 避免触发 resource_manager 的__reduce__导致连接池消失

        if (max_iter > 0 and should_finish_evaluation) or (max_iter <= 0):
            self.last_data = None
        return (
            {
                "pred_valid": pred_valid,
                "pred_test": pred_test,
                "y_train": y_train  # todo: evaluator 中做相应的改变
            }
        )

    def transform(self, X_train, X_valid=None, X_test=None, y_train=None):
        for _, _, transformer in self._iter(with_final=(not self.is_estimator)):
            result = transformer.transform(X_train, X_valid, X_test, y_train)  # predict procedure
            X_train = result["X_train"]
            X_valid = result.get("X_valid")
            X_test = result.get("X_test")
            y_train = result.get("y_train")
        return {"X_train": X_train, "X_valid": X_valid, "X_test": X_test, "y_train": y_train}

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X):
        result = self.transform(X)
        X = result["X_train"]
        return self.steps[-1][-1].predict(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        result = self.transform(X)
        X = result["X_train"]
        return self.steps[-1][-1].predict_proba(X)

    def copy(self):
        tmp = self.resource_manager
        self.resource_manager = None
        res = deepcopy(self)
        self.resource_manager = tmp
        res.resource_manager = tmp
        return res
