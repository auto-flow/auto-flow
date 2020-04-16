import re
import sys
from collections import defaultdict
from contextlib import redirect_stderr
from io import StringIO
from time import time
from typing import Dict, Optional

import numpy as np
from ConfigSpace import Configuration

from dsmac.runhistory.utils import get_id_of_config
from autoflow.constants import PHASE2, PHASE1
from autoflow.ensemble.utils import vote_predicts, mean_predicts
from autoflow.evaluation.base import BaseEvaluator
from autoflow.manager.data_manager import DataManager
from autoflow.manager.resource_manager import ResourceManager
from autoflow.metrics import Scorer, calculate_score
from autoflow.pipeline.dataframe import GenericDataFrame
from autoflow.pipeline.pipeline import GenericPipeline
from autoflow.shp2dhp.shp2dhp import SHP2DHP
from autoflow.utils.dict import group_dict_items_before_first_dot
from autoflow.utils.logging import get_logger
from autoflow.utils.ml_task import MLTask
from autoflow.utils.packages import get_class_object_in_pipeline_components
from autoflow.utils.pipeline import concat_pipeline
from autoflow.utils.sys import get_trance_back_msg


class TrainEvaluator(BaseEvaluator):
    def __init__(self):
        # ---member variable----
        self.debug = False

    def init_data(
            self,
            random_state,
            data_manager: DataManager,
            metric: Scorer,
            should_calc_all_metric: bool,
            splitter,
            should_store_intermediate_result: bool,
            resource_manager: ResourceManager
    ):
        self.random_state = random_state
        if hasattr(splitter, "random_state"):
            setattr(splitter, "random_state", self.random_state)
        self.splitter = splitter
        self.data_manager = data_manager
        self.X_train = self.data_manager.X_train
        self.y_train = self.data_manager.y_train
        self.X_test = self.data_manager.X_test
        self.y_test = self.data_manager.y_test
        self.should_store_intermediate_result = should_store_intermediate_result
        self.metric = metric
        self.ml_task: MLTask = self.data_manager.ml_task

        self.should_calc_all_metric = should_calc_all_metric

        if self.ml_task.mainTask == "regression":
            self.predict_function = self._predict_regression
        else:
            self.predict_function = self._predict_proba

        self.logger = get_logger(self)
        self.resource_manager = resource_manager

    def loss(self, y_true, y_hat):
        score = calculate_score(
            y_true, y_hat, self.ml_task, self.metric,
            should_calc_all_metric=self.should_calc_all_metric)

        if isinstance(score, dict):
            err = self.metric._optimum - score[self.metric.name]
            all_score = score
        elif isinstance(score, (int, float)):
            err = self.metric._optimum - score
            all_score = None
        else:
            raise TypeError

        return err, all_score

    def set_resource_manager(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager

    def _predict_proba(self, X, model):
        y_pred = model.predict_proba(X)
        return y_pred

    def _predict_regression(self, X, model):
        y_pred = model.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((-1, 1))
        return y_pred

    def get_Xy(self):
        # fixme: 会出现结果被改变的情况！  目前这个bug在autoflow.pipeline.components.preprocessing.operate.merge.Merge 出现过
        return (self.X_train), (self.y_train), (self.X_test), (self.y_test)
        # return deepcopy(self.X_train), deepcopy(self.y_train), deepcopy(self.X_test), deepcopy(self.y_test)

    def evaluate(self, model: GenericPipeline, X, y, X_test, y_test):
        assert self.resource_manager is not None
        warning_info = StringIO()
        with redirect_stderr(warning_info):
            # splitter 必须存在
            losses = []
            models = []
            y_true_indexes = []
            y_preds = []
            y_test_preds = []
            all_scores = []
            status = "SUCCESS"
            failed_info = ""
            for train_index, valid_index in self.splitter.split(X, y):
                X: GenericDataFrame
                X_train, X_valid = X.split([train_index, valid_index])
                y_train, y_valid = y[train_index], y[valid_index]
                if self.should_store_intermediate_result:
                    intermediate_result = []
                else:
                    intermediate_result = None
                try:
                    procedure_result = model.procedure(self.ml_task, X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                       self.resource_manager, intermediate_result)
                except Exception as e:
                    failed_info = get_trance_back_msg()
                    status = "FAILED"
                    if self.debug:
                        self.logger.error("re-raise exception")
                        raise sys.exc_info()[1]
                    break
                models.append(model)
                y_true_indexes.append(valid_index)
                y_pred = procedure_result["pred_valid"]
                y_test_pred = procedure_result["pred_test"]
                y_preds.append(y_pred)
                y_test_preds.append(y_test_pred)
                loss, all_score = self.loss(y_valid, y_pred)
                losses.append(float(loss))
                all_scores.append(all_score)
            if len(losses) > 0:
                final_loss = float(np.array(losses).mean())
            else:
                final_loss = 65535
            if len(all_scores) > 0 and all_scores[0]:
                all_score = defaultdict(list)
                for cur_all_score in all_scores:
                    if isinstance(cur_all_score, dict):
                        for key, value in cur_all_score.items():
                            all_score[key].append(value)
                    else:
                        self.logger.warning(f"TypeError: cur_all_score is not dict.\ncur_all_score = {cur_all_score}")
                for key in all_score.keys():
                    all_score[key] = float(np.mean(all_score[key]))
            else:
                all_score = {}
                all_scores = []
            info = {
                "loss": final_loss,
                "losses": losses,
                "all_score": all_score,
                "all_scores": all_scores,
                "models": models,
                "y_true_indexes": y_true_indexes,
                "y_preds": y_preds,
                "intermediate_result": intermediate_result,
                "status": status,
                "failed_info": failed_info
            }
            # todo
            if y_test is not None:
                # 验证集训练模型的组合去预测测试集的数据
                if self.ml_task.mainTask == "classification":
                    y_test_pred = vote_predicts(y_test_preds)
                else:
                    y_test_pred = mean_predicts(y_test_preds)
                test_loss, test_all_score = self.loss(y_test, y_test_pred)
                info.update({
                    "test_loss": test_loss,
                    "test_all_score": test_all_score,
                    "y_test_true": y_test,
                    "y_test_pred": y_test_pred
                })
        info["warning_info"] = warning_info.getvalue()
        return info

    def __call__(self, shp: Configuration):
        # 1. 将php变成model
        config_id = get_id_of_config(shp)
        start = time()
        dhp, model = self.shp2model(shp)
        # 2. 获取数据
        X_train, y_train, X_test, y_test = self.get_Xy()
        # 3. 进行评价
        info = self.evaluate(model, X_train, y_train, X_test, y_test)  # todo : 考虑失败的情况
        # 4. 持久化
        cost_time = time() - start
        info["config_id"] = config_id
        info["program_hyper_param"] = shp
        info["dict_hyper_param"] = dhp
        estimator = list(dhp.get(PHASE2, {"unk": ""}).keys())[0]
        info["estimator"] = estimator
        info["cost_time"] = cost_time
        self.resource_manager.insert_to_trials_table(info)
        return {
            "loss": info["loss"],
            "status": info["status"]
        }

    def shp2model(self, shp):
        shp2dhp = SHP2DHP()
        dhp = shp2dhp(shp)
        preprocessor = self.create_preprocessor(dhp)
        estimator = self.create_estimator(dhp)
        pipeline = concat_pipeline(preprocessor, estimator)
        self.logger.debug(str(pipeline))
        return dhp, pipeline

    def parse_key(self, key: str):
        cnt = ""
        ix = 0
        for i, c in enumerate(key):
            if c.isdigit():
                cnt += c
            else:
                ix = i
                break
        cnt = int(cnt)
        key = key[ix:]
        pattern = re.compile(r"(\{.*\})")
        match = pattern.search(key)
        additional_info = {}
        if match:
            braces_content = match.group(1)
            _to = braces_content[1:-1]
            param_kvs = _to.split(",")
            for param_kv in param_kvs:
                k, v = param_kv.split("=")
                additional_info[k] = v
            key = pattern.sub("", key)
        if "->" in key:
            _from, _to = key.split("->")
            in_feature_groups = _from
            out_feature_groups = _to
        else:
            in_feature_groups, out_feature_groups = None, None
        if not in_feature_groups:
            in_feature_groups = None
        if not out_feature_groups:
            out_feature_groups = None
        return in_feature_groups, out_feature_groups, additional_info

    def create_preprocessor(self, dhp: Dict) -> Optional[GenericPipeline]:
        preprocessing_dict: dict = dhp[PHASE1]
        pipeline_list = []
        for key, value in preprocessing_dict.items():
            name = key  # like: "cat->num"
            in_feature_groups, out_feature_groups, outsideEdge_info = self.parse_key(key)
            sub_dict = preprocessing_dict[name]
            if sub_dict is None:
                continue
            preprocessor = self.create_component(sub_dict, PHASE1, name, in_feature_groups, out_feature_groups,
                                                 outsideEdge_info)
            pipeline_list.extend(preprocessor)
        if pipeline_list:
            return GenericPipeline(pipeline_list)
        else:
            return None

    def create_estimator(self, dhp: Dict) -> GenericPipeline:
        # 根据超参构造一个估计器
        return GenericPipeline(self.create_component(dhp[PHASE2], PHASE2, self.ml_task.role))

    def _create_component(self, key1, key2, params):
        cls = get_class_object_in_pipeline_components(key1, key2)
        component = cls()
        # component.set_addition_info(self.addition_info)
        component.update_hyperparams(params)
        return component

    def create_component(self, sub_dhp: Dict, phase: str, step_name, in_feature_groups="all", out_feature_groups="all",
                         outsideEdge_info=None):
        pipeline_list = []
        assert phase in (PHASE1, PHASE2)
        packages = list(sub_dhp.keys())[0]
        params = sub_dhp[packages]
        packages = packages.split("|")
        grouped_params = group_dict_items_before_first_dot(params)
        if len(packages) == 1:
            if bool(grouped_params):
                grouped_params[packages[0]] = grouped_params.pop("single")
            else:
                grouped_params[packages[0]] = {}
        for package in packages[:-1]:
            preprocessor = self._create_component(PHASE1, package, grouped_params[package])
            preprocessor.in_feature_groups = in_feature_groups
            preprocessor.out_feature_groups = in_feature_groups
            pipeline_list.append([
                package,
                preprocessor
            ])
        key1 = PHASE1 if phase == PHASE1 else self.ml_task.mainTask
        component = self._create_component(key1, packages[-1], grouped_params[packages[-1]])
        component.in_feature_groups = in_feature_groups
        component.out_feature_groups = out_feature_groups
        if outsideEdge_info:
            component.update_hyperparams(outsideEdge_info)
        pipeline_list.append([
            step_name,
            component
        ])
        return pipeline_list
