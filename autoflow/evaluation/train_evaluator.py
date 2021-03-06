import datetime
from collections import OrderedDict
from collections import defaultdict
from contextlib import redirect_stderr
from io import StringIO
from time import time
from typing import Dict, Optional, List

import numpy as np
from frozendict import frozendict

from autoflow.constants import PHASE2, PHASE1, SERIES_CONNECT_LEADER_TOKEN, SERIES_CONNECT_SEPARATOR_TOKEN, \
    SUBSAMPLES_BUDGET_MODE, ITERATIONS_BUDGET_MODE, ERR_LOSS
from autoflow.data_container import DataFrameContainer, NdArrayContainer
from autoflow.data_manager import DataManager
from autoflow.ensemble.utils import vote_predicts, mean_predicts
from autoflow.evaluation.budget import implement_subsample_budget
from autoflow.hdl.shp2dhp import SHP2DHP
from autoflow.metrics import Scorer, calculate_score, calculate_confusion_matrix
from autoflow.opt.core.worker import Worker
from autoflow.resource_manager.base import ResourceManager
from autoflow.utils.dict_ import group_dict_items_before_first_token
from autoflow.utils.hash import get_hash_of_config
from autoflow.utils.klass import StrSignatureMixin
from autoflow.utils.ml_task import MLTask
from autoflow.utils.packages import get_class_object_in_pipeline_components
from autoflow.utils.pipeline import concat_pipeline
from autoflow.utils.sys_ import get_trance_back_msg
from autoflow.workflow.ml_workflow import ML_Workflow


class TrainEvaluator(Worker, StrSignatureMixin):
    def __init__(
            self,
            run_id,
            data_manager: DataManager,
            resource_manager: ResourceManager,
            random_state: int,
            metric: Scorer,
            groups: List[int],
            should_calc_all_metric: bool,
            splitter,
            should_store_intermediate_result: bool,
            should_stack_X: bool,
            should_finally_fit: bool,  # todo: rename to refit
            model_registry: dict,
            budget2kfold: Optional[Dict[float, int]] = None,
            algo2budget_mode: Optional[Dict[str, str]] = None,
            algo2weight_mode: Optional[Dict[str, str]] = None,
            algo2iter: Optional[Dict[str, int]] = None,
            specific_out_feature_groups_mapper: Dict[str, str] = frozendict(),
            max_budget: float = np.inf,
            nameserver=None,
            nameserver_port=None,
            host=None,
            worker_id=None,
            timeout=None,
            debug: bool = False,
    ):
        super(TrainEvaluator, self).__init__(
            run_id, nameserver, nameserver_port, host, worker_id, timeout, debug
        )
        self.specific_out_feature_groups_mapper = dict(specific_out_feature_groups_mapper)
        self.algo2weight_mode = algo2weight_mode
        self.max_budget = max_budget
        self.algo2iter = algo2iter
        self.algo2budget_mode = algo2budget_mode
        self.budget2kfold = budget2kfold
        self.timeout = timeout
        self.id = worker_id
        self.host = host
        self.nameserver_port = nameserver_port
        self.nameserver = nameserver
        self.model_registry = model_registry
        self.should_finally_fit = should_finally_fit
        self.resource_manager = resource_manager
        self.should_stack_X = should_stack_X
        self.should_store_intermediate_result = should_store_intermediate_result
        self.splitter = splitter
        self.should_calc_all_metric = should_calc_all_metric
        self.groups = groups
        self.metric = metric
        self.data_manager = data_manager
        self.random_state = random_state
        self.run_id = run_id
        # ---member variable----
        self.ml_task: MLTask = self.data_manager.ml_task
        if self.ml_task.mainTask == "regression":
            self.predict_function = self._predict_regression
        else:
            self.predict_function = self._predict_proba
        self.X_train = self.data_manager.X_train
        self.y_train = self.data_manager.y_train
        self.X_test = self.data_manager.X_test
        self.y_test = self.data_manager.y_test
        self.rng = np.random.RandomState(self.random_state)

    def loss(self, y_true, y_hat):
        score, true_score = calculate_score(
            y_true, y_hat, self.ml_task, self.metric,
            should_calc_all_metric=self.should_calc_all_metric)

        if isinstance(score, dict):
            err = self.metric._optimum - score[self.metric.name]
            all_score = true_score
        elif isinstance(score, (int, float)):
            err = self.metric._optimum - score
            all_score = None
        else:
            raise TypeError

        return err, all_score

    def _predict_proba(self, X, model):
        y_pred = model.predict_proba(X)
        return y_pred

    def _predict_regression(self, X, model):
        y_pred = model.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((-1, 1))
        return y_pred

    def calc_balanced_sample_weight(self, y_train: np.ndarray):

        unique, counts = np.unique(y_train, return_counts=True)
        # This will result in an average weight of 1!
        cw = 1 / (counts / np.sum(counts)) / len(unique)

        sample_weights = np.ones(y_train.shape)

        for i, ue in enumerate(unique):
            mask = y_train == ue
            sample_weights[mask] *= cw[i]
        return sample_weights

    def get_Xy(self):
        # fixme: 会出现结果被改变的情况！
        #  目前这个bug在autoflow.workflow.components.preprocessing.operate.keep_going.KeepGoing 出现过
        # fixme: autoflow.manager.data_container.dataframe.DataFrameContainer#sub_sample 函数采用deepcopy，
        #  应该能从源头上解决X_train数据集的问题，但是要注意X_test
        return (self.X_train), (self.y_train), (self.X_test), (self.y_test)
        # return deepcopy(self.X_train), deepcopy(self.y_train), deepcopy(self.X_test), deepcopy(self.y_test)

    def evaluate(self, config_id, model: ML_Workflow, X, y, X_test, y_test, budget, dhp, config):
        warning_info = StringIO()
        additional_info = {}
        final_model = model[-1]
        final_model_name = final_model.__class__.__name__
        support_early_stopping = getattr(final_model, "support_early_stopping", False)
        budget_mode = self.algo2budget_mode[final_model_name]
        is_iter_algo = self.algo2iter.get(final_model_name) is not None
        max_iter = -1
        # if final model is iterative algorithm, max_iter should be specified
        if is_iter_algo:
            if budget_mode == ITERATIONS_BUDGET_MODE:
                fraction = min(1, budget)
            else:
                fraction = 1
            max_iter = max(round(self.algo2iter[final_model_name] * fraction), 1)
        balance_strategy: Optional[dict] = dhp.get("strategies", {}).get("balance")
        if isinstance(balance_strategy, dict):
            balance_strategy_name: str = list(balance_strategy.keys())[0]
            balance_strategy_params: dict = balance_strategy[balance_strategy_name]
        else:
            balance_strategy_name: str = "None"
            balance_strategy_params: dict = {}
        if balance_strategy_name == "weight":
            algo_name = model[-1].__class__.__name__
            if algo_name not in self.algo2weight_mode:
                self.logger.warning(f"Algorithm '{algo_name}' not in self.algo2weight_mode !")
            weight_mode = self.algo2weight_mode.get(algo_name)
        else:
            weight_mode = None
        if weight_mode == "class_weight":
            model[-1].update_hyperparams({"class_weight": "balanced"})
        with redirect_stderr(warning_info):
            losses = []
            models = []
            y_true_indexes = []
            y_preds = []
            y_test_preds = []
            all_scores = []
            status = "SUCCESS"
            failed_info = ""
            intermediate_results = []
            start_time = datetime.datetime.now()
            confusion_matrices = []
            best_iterations = []
            component_infos = []
            cost_times = []
            for fold_ix, (train_index, valid_index) in enumerate(self.splitter.split(X.data, y.data, self.groups)):
                cloned_model = model.copy()
                X: DataFrameContainer
                X_train = X.sub_sample(train_index)
                X_valid = X.sub_sample(valid_index)
                y_train = y.sub_sample(train_index)
                y_valid = y.sub_sample(valid_index)
                # subsamples budget_mode.
                if fold_ix == 0 and budget_mode == SUBSAMPLES_BUDGET_MODE and budget < 1:
                    X_train, y_train, (X_valid, X_test) = implement_subsample_budget(
                        X_train, y_train, [X_valid, X_test],
                        budget, self.random_state
                    )
                cache_key = self.get_cache_key(config_id, X_train, y_train)
                cached_model = self.resource_manager.cache.get(cache_key)
                if cached_model is not None:
                    cloned_model = cached_model
                # 如果是iterations budget mode, 采用一个统一的接口调整 max_iter
                # 未来争取做到能缓存ML_Workflow, 只训练最后的拟合器
                if weight_mode == "sample_weight":  # sample_weight balance
                    cloned_model[-1].set_inside_dict(
                        {"sample_weight": self.calc_balanced_sample_weight(y_train.data)})
                if self.debug:
                    procedure_result = cloned_model.procedure(
                        self.ml_task, X_train, y_train, X_valid, y_valid,
                        X_test, y_test, max_iter, budget, (budget == self.max_budget)
                    )
                else:
                    try:
                        procedure_result = cloned_model.procedure(
                            self.ml_task, X_train, y_train, X_valid, y_valid,
                            X_test, y_test, max_iter, budget, (budget == self.max_budget)
                        )
                    except Exception as e:
                        self.logger.error(str(e))
                        self.logger.error(str(config))
                        failed_info = get_trance_back_msg()
                        status = "FAILED"  # todo: 实现 timeout， memory out
                        self.logger.error("re-raise exception")
                        break
                # save model as cache
                if (budget_mode == ITERATIONS_BUDGET_MODE and budget <= 1) or \
                        (budget == 1):  # and isinstance(final_model, AutoFlowIterComponent)
                    self.resource_manager.cache.set(cache_key, cloned_model)
                intermediate_results.append(cloned_model.intermediate_result)
                models.append(cloned_model)
                y_true_indexes.append(valid_index)
                y_pred = procedure_result["pred_valid"]
                y_test_pred = procedure_result["pred_test"]
                if self.ml_task.mainTask == "classification":
                    confusion_matrices.append(calculate_confusion_matrix(y_valid.data, y_pred))
                if support_early_stopping:
                    estimator = cloned_model.steps[-1][1]
                    # todo: 重构 LGBM等
                    best_iterations.append(getattr(
                        estimator, "best_iteration_", -1))
                cost_times.append(cloned_model.time_cost_list)
                component_info = {}
                for step_name, component in cloned_model.steps:
                    component_name = component.__class__.__name__
                    component_additional_info = component.additional_info
                    if bool(component_additional_info):
                        component_info[component_name] = component.additional_info
                component_infos.append(component_info)
                y_preds.append(y_pred)
                if y_test_pred is not None:
                    y_test_preds.append(y_test_pred)
                loss, all_score = self.loss(y_valid.data, y_pred)  # todo: 非1d-array情况下的用户自定义评估器
                losses.append(float(loss))
                all_scores.append(all_score)
                # when  budget  <= 1 , hold out validation
                if fold_ix == 0 and budget <= 1:
                    break
                # when  budget  > 1 , budget will be interpreted as kfolds num by 'budget2kfold'
                # for example, budget = 4 , budget2kfold = {4: 10}, we only do 10 times cross-validation,
                # so we break when fold_ix == 10 - 1 == 9
                if isinstance(self.budget2kfold, dict) and budget in self.budget2kfold:
                    if budget > 1 and fold_ix == self.budget2kfold[budget] - 1:
                        break
            if self.ml_task.mainTask == "classification":
                additional_info["confusion_matrices"] = confusion_matrices
            if support_early_stopping:
                additional_info["best_iterations"] = best_iterations
            additional_info["cost_times"] = cost_times
            additional_info["component_infos"] = component_infos
            end_time = datetime.datetime.now()
            # finally fit
            if status == "SUCCESS" and self.should_finally_fit:
                # make sure have resource_manager to do things like connect redis
                model.resource_manager = self.resource_manager
                finally_fit_model = model.fit(X, y, X_test=X_test, y_test=y_test)
                if self.ml_task.mainTask == "classification":
                    y_test_pred_by_finally_fit_model = model.predict_proba(X_test)
                else:
                    y_test_pred_by_finally_fit_model = model.predict(X_test)
                model.resource_manager = None
            else:
                finally_fit_model = None
                y_test_pred_by_finally_fit_model = None

            if len(losses) > 0:
                final_loss = float(np.array(losses).mean())
            else:
                # train and validation is failed.
                final_loss = ERR_LOSS
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
                "finally_fit_model": finally_fit_model,
                "y_true_indexes": y_true_indexes,
                "y_preds": y_preds,
                "intermediate_results": intermediate_results,
                "status": status,
                "failed_info": failed_info,
                "start_time": start_time,
                "end_time": end_time,
                "additional_info": additional_info
            }
            # todo
            if y_test is not None:
                # 验证集训练模型的组合去预测测试集的数据
                if self.should_finally_fit:
                    y_test_pred = y_test_pred_by_finally_fit_model
                else:
                    if self.ml_task.mainTask == "classification":
                        y_test_pred = vote_predicts(y_test_preds)
                    else:
                        y_test_pred = mean_predicts(y_test_preds)
                test_loss, test_all_score = self.loss(y_test.data, y_test_pred)
                # todo: 非1d-array情况下的用户自定义评估器
                info.update({
                    "test_loss": test_loss,
                    "test_all_score": test_all_score,
                    # "y_test_true": y_test,
                    "y_test_pred": y_test_pred
                })
        info["warning_info"] = warning_info.getvalue()
        return info

    def compute(self, config: dict, config_info: dict, budget: float, **kwargs):
        # 1. 将php变成model
        config_id = get_hash_of_config(config)
        start = time()
        dhp, model = self.shp2model(config, config, config_id)
        # 2. 获取数据
        X_train, y_train, X_test, y_test = self.get_Xy()
        # todo: iter budget类型的支持
        # 3. 进行评价
        info = self.evaluate(config_id, model, X_train, y_train, X_test, y_test, budget, dhp, config)
        # 4. 持久化
        cost_time = time() - start
        info["config_id"] = config_id
        info["config"] = config
        info["config_info"] = config_info
        info["budget"] = budget
        # info["instance_id"] = self.run_id
        # info["run_id"] = self.run_id
        # info["program_hyper_param"] = shp
        info["dict_hyper_param"] = dhp
        estimator = list(dhp.get(PHASE2, {"unk": ""}).keys())[0]
        info["estimator"] = estimator
        info["cost_time"] = cost_time
        # info["additional_info"].update({
        #     "config_origin": getattr(shp, "origin", "unk")
        # })
        # fixme: 改到result_logger
        trial_id = self.resource_manager.insert_trial_record(info)
        return {
            "loss": info["loss"],
            "info": {
                "config_id": config_id,
                "trial_id": trial_id
            },
        }

    def shp2model(self, shp, config, config_id):
        shp2dhp = SHP2DHP()
        dhp = shp2dhp(shp)
        # todo : 引入一个参数，描述运行模式。一共有3种模式：普通，深度学习，大数据。对以下三个翻译的步骤进行重构
        preprocessing = dhp.pop("preprocessing")
        sorted_preprocessing = OrderedDict()
        process_sequence = dhp["process_sequence"].split(";")
        for key in process_sequence:
            sorted_preprocessing[key] = preprocessing[key]
        dhp["preprocessing"] = dict(sorted_preprocessing)
        preprocessor = self.create_preprocessor(dhp)
        estimator = self.create_estimator(dhp)
        pipeline = concat_pipeline(preprocessor, estimator)
        pipeline.config = config
        pipeline.config_id = config_id
        return dhp, pipeline

    def parse_key(self, key: str):
        # todo: 支持多结点的输入输出，与dataframe.py耦合
        if "->" in key:
            _from, _to = key.split("->")
            in_feature_groups = _from.split(",")[0]
            out_feature_groups = _to.split(",")[0]
        else:
            in_feature_groups, out_feature_groups = None, None
        if not in_feature_groups:
            in_feature_groups = None
        if not out_feature_groups:
            out_feature_groups = None
        return in_feature_groups, out_feature_groups

    def create_preprocessor(self, dhp: Dict) -> Optional[ML_Workflow]:
        preprocessing_dict: dict = dhp[PHASE1]
        pipeline_list = []
        for key, value in preprocessing_dict.items():
            name = key  # like: "cat->num"
            in_feature_groups, out_feature_groups = self.parse_key(key)
            sub_dict = preprocessing_dict[name]
            component_name = list(sub_dict.keys())[0]
            # todo: 根据Component name 调整 out_feature_groups
            specific_out_feature_groups = self.specific_out_feature_groups_mapper.get(component_name)
            if specific_out_feature_groups is not None:
                out_feature_groups = specific_out_feature_groups
            if sub_dict is None:
                continue
            preprocessor = self.create_component(sub_dict, PHASE1, name, in_feature_groups, out_feature_groups,
                                                 )
            pipeline_list.extend(preprocessor)
        if pipeline_list:
            return ML_Workflow(pipeline_list, self.should_store_intermediate_result, self.resource_manager)
        else:
            return None

    def create_estimator(self, dhp: Dict) -> ML_Workflow:
        # 根据超参构造一个估计器
        return ML_Workflow(self.create_component(dhp[PHASE2], PHASE2, self.ml_task.role),
                           self.should_store_intermediate_result, self.resource_manager)

    def _create_component(self, key1, key2, params):
        cls = get_class_object_in_pipeline_components(key1, key2, self.model_registry)
        component = cls(**params)
        # component.set_inside_dict(self.addition_info)
        return component

    def create_component(self, sub_dhp: Dict, phase: str, step_name, in_feature_groups="all", out_feature_groups="all",
                         outsideEdge_info=None):
        pipeline_list = []
        assert phase in (PHASE1, PHASE2)
        # packages = list(sub_dhp.keys())[0]
        # params = sub_dhp[packages]
        packages = None
        params = {}
        for k, v in sub_dhp.items():
            if isinstance(v, dict):
                if packages is not None:
                    self.logger.warning(f"Duplicated packages: {k} , {packages}")
                packages = k
                params.update(v)
            else:
                self.logger.debug(f"Detect {k} = {v} out of core dict maybe params, type of v is {type(v).__name__}")
                params.update({k: v})
        assert packages is not None, ValueError
        packages = packages.split(SERIES_CONNECT_SEPARATOR_TOKEN)
        grouped_params = group_dict_items_before_first_token(params, SERIES_CONNECT_LEADER_TOKEN)
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
        hyperparams = grouped_params[packages[-1]]
        if outsideEdge_info:
            hyperparams.update(outsideEdge_info)
        component = self._create_component(key1, packages[-1], hyperparams)
        component.in_feature_groups = in_feature_groups
        component.out_feature_groups = out_feature_groups
        if not self.should_stack_X:
            setattr(component, "need_y", True)
        pipeline_list.append([
            step_name,
            component
        ])
        return pipeline_list

    def get_cache_key(self, config_id, X_train: DataFrameContainer, y_train: NdArrayContainer):
        experiment_id = str(self.resource_manager.experiment_id)
        return "-".join([experiment_id, config_id, X_train.get_hash(), y_train.get_hash()])
