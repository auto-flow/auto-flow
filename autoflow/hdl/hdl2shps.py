import re
from collections import defaultdict, Counter
from copy import deepcopy
from importlib import import_module
from typing import Dict, List

import numpy as np
from ConfigSpace import CategoricalHyperparameter, Constant
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace import ForbiddenInClause, ForbiddenEqualsClause, ForbiddenAndConjunction
from ConfigSpace import InCondition, EqualsCondition
from hyperopt import fmin, tpe, hp

import autoflow.hdl.smac as smac_hdl
from autoflow.constants import PHASE2, SERIES_CONNECT_LEADER_TOKEN
from autoflow.hdl.utils import is_hdl_bottom, get_origin_models, purify_keys, purify_key, add_leader_model
from autoflow.utils.dict_ import filter_item_by_key_condition
from autoflow.utils.klass import StrSignatureMixin
from autoflow.utils.logging_ import get_logger
from autoflow.utils.ml_task import MLTask
from autoflow.utils.packages import get_class_name_of_module


class RelyModels:
    info = []


class HDL2SHPS(StrSignatureMixin):
    def __init__(self):
        self.ml_task = None
        self.logger = get_logger(__name__)

    def set_task(self, ml_task: MLTask):
        self.ml_task = ml_task

    def get_forbid_hit_in_models_by_rely(self, models, rely_model="boost_model"):
        forbid_in_value = []
        hit = []
        models = get_origin_models(models)
        for model in models:
            module_path = f"autoflow.workflow.components.{self.ml_task.mainTask}.{model}"
            _class = get_class_name_of_module(module_path)
            if _class is not None:
                M = import_module(module_path)
                cls = getattr(M, _class)
                is_hit = getattr(cls, rely_model, False)
            else:
                is_hit = False
            if not is_hit:
                forbid_in_value.append(model)
            else:
                hit.append(model)
        return forbid_in_value, hit

    def set_probabilities_in_cs(
            self, cs: ConfigurationSpace,
            relied2models: Dict[str, List[str]],
            relied2AllModels: Dict[str, List[str]],
            all_models: List[str],
            **kwargs
    ):
        estimator = cs.get_hyperparameter(f"{PHASE2}:__choice__")
        probabilities = []
        model2prob = {}
        L = 0
        for rely_model in relied2models:
            cur_models = relied2models[rely_model]
            L += len(cur_models)
            for model in cur_models:
                model2prob[model] = kwargs[rely_model] / len(cur_models)
        p_rest = (1 - sum(model2prob.values())) / (len(all_models) - L)
        for model in estimator.choices:
            probabilities.append(model2prob.get(model, p_rest))
        estimator.probabilities = probabilities
        default_estimator_choice = None
        for models in relied2models.values():
            if models:
                default_estimator_choice = models[0]
        estimator.default_value = default_estimator_choice
        for rely_model, path in RelyModels.info:
            forbid_eq_value = path[-1]
            path = path[:-1]
            forbid_eq_key = ":".join(path + ["__choice__"])
            forbid_eq_key_hp = cs.get_hyperparameter(forbid_eq_key)
            forbid_in_key = f"{PHASE2}:__choice__"
            hit = relied2AllModels.get(rely_model)
            if not hit:
                choices = list(forbid_eq_key_hp.choices)
                choices.remove(forbid_eq_value)
                forbid_eq_key_hp.choices = tuple(choices)
                forbid_eq_key_hp.default_value = choices[0]
                forbid_eq_key_hp.probabilities = [1 / len(choices)] * len(choices)
                # fixme  最后我放弃了在这上面进行修改，在hdl部分就做了预处理
                continue
            forbid_in_value = list(set(all_models) - set(hit))
            # 只选择了boost模型
            if not forbid_in_value:
                continue
            choices = forbid_eq_key_hp.choices
            probabilities = []
            p: float = kwargs[rely_model]
            p_rest = (1 - p) * (len(choices) - 1)
            for choice in choices:
                if choice == forbid_eq_value:
                    probabilities.append(p)
                else:
                    probabilities.append(p_rest)
            forbid_eq_key_hp.probabilities = probabilities
            cs.add_forbidden_clause(ForbiddenAndConjunction(
                ForbiddenEqualsClause(forbid_eq_key_hp, forbid_eq_value),
                ForbiddenInClause(cs.get_hyperparameter(forbid_in_key), forbid_in_value),
            ))

    def __rely_model(self, cs: ConfigurationSpace):
        if not RelyModels.info:
            return
        all_models = list(cs.get_hyperparameter(f"{PHASE2}:__choice__").choices)
        rely_model_counter = Counter([x[0] for x in RelyModels.info])
        # 依赖模式->所有相应模型
        relied2AllModels = {}
        # 依赖模式->无交集相应模型
        relied2models = {}
        for rely_model in rely_model_counter.keys():
            _, hit = self.get_forbid_hit_in_models_by_rely(all_models, rely_model)
            relied2AllModels[rely_model] = hit
        # 如果某依赖模式不对应任何模型，删除
        for k, v in list(relied2AllModels.items()):
            if not v:
                relied2AllModels.pop(k)
                rely_model_counter.pop(k)
        has_any_hit = any(relied2AllModels.values())
        if not has_any_hit:
            return
        # 按照规则计算  relied2models  ：  无交集相应模型
        relied_cnts_tuples = [(k, v) for k, v in rely_model_counter.items()]
        relied_cnts_tuples.sort(key=lambda x: x[-1])
        visited = set()
        for rely_model, _ in relied_cnts_tuples:
            models = relied2AllModels[rely_model]
            for other in set(rely_model_counter.keys()) - {rely_model}:
                if (rely_model, other) in visited:
                    continue
                other_models = relied2AllModels[other]
                if len(other_models) <= len(models):
                    models = list(set(models) - set(other_models))
                    visited.add((rely_model, other))
                    visited.add((other, rely_model))
            relied2models[rely_model] = models

        # 键的顺序遵循rely_model_counter.keys()
        def objective(relyModel2prob, debug=False):
            # relyModel2prob = {rely_model: prob for rely_model, prob in zip(list(rely_model_counter.keys()), args)}
            cur_cs = deepcopy(cs)
            self.set_probabilities_in_cs(cur_cs, relied2models, relied2AllModels, all_models, **relyModel2prob)

            cur_cs.seed(42)
            try:
                sample_times = len(all_models) * 15
                counter = Counter([_hp.get(f"{PHASE2}:__choice__") for _hp in
                                   cur_cs.sample_configuration(sample_times)])

                if debug:
                    self.logger.info(f"Finally, sample {sample_times} times in estimator list's frequency: \n{counter}")
            except Exception:
                return np.inf
            vl = list(counter.values())
            return np.var(vl) + 100 * (len(models) - len(vl))

        space = {}
        eps = 0.001
        N_rely_model = len(rely_model_counter.keys())
        for rely_model in rely_model_counter.keys():
            space[rely_model] = hp.uniform(rely_model, eps, (1 / N_rely_model) - eps)

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            rstate=np.random.RandomState(42),
            show_progressbar=False,

        )
        self.logger.info(f"The best probability is {best}")
        objective(best, debug=True)
        self.set_probabilities_in_cs(cs, relied2models, relied2AllModels, all_models, **best)
        # todo: 将计算的概率缓存

    def purify_isolate_rely_in_hdl(self, hdl: Dict, models: List[str]):
        # 为了应对estimator中全为boost的情况
        # 做法： 删除 __rely_model键
        for key, value in hdl.items():
            if isinstance(value, dict):
                ok = False
                for name, sub_dict in value.items():
                    if isinstance(sub_dict, dict) and "__rely_model" in purify_keys(sub_dict):
                        ok = True
                        endswith_rely_model_dicts = filter_item_by_key_condition(
                            sub_dict,
                            lambda x: x.endswith("__rely_model"))
                        rely_models = list(endswith_rely_model_dicts.values())
                        rely_model = rely_models[0]
                        _, hit = self.get_forbid_hit_in_models_by_rely(models, rely_model)
                        if set(hit) == set(models):
                            for key in endswith_rely_model_dicts:
                                sub_dict.pop(key)
                if not ok:
                    self.purify_isolate_rely_in_hdl(value, models)

    def drop_invalid_rely_in_hdl(self, hdl: Dict, models: List[str]):
        # 为了应对estimator中没有boost，但是特征工程序列中却有依赖boost的特征工程的情况
        # 做法：将这样的特征工程删除
        for key, value in hdl.items():
            if isinstance(value, dict):
                ok = False
                deleted_keys = []
                for name, sub_dict in value.items():
                    if isinstance(sub_dict, dict) and "__rely_model" in purify_keys(sub_dict):
                        ok = True
                        rely_models = filter_item_by_key_condition(
                            sub_dict,
                            lambda x: x.endswith("__rely_model")).values()
                        rely_models = list(rely_models)
                        rely_model = rely_models[0]
                        _, hit = self.get_forbid_hit_in_models_by_rely(models, rely_model)
                        if not hit:
                            deleted_keys.append(name)
                for deleted_key in deleted_keys:
                    value.pop(deleted_key)
                if not ok:
                    self.drop_invalid_rely_in_hdl(value, models)

    def __call__(self, hdl: Dict):
        # 对HDL进行处理
        models = hdl[f"{PHASE2}(choice)"]
        self.drop_invalid_rely_in_hdl(hdl, models)
        self.purify_isolate_rely_in_hdl(hdl, models)
        RelyModels.info = []
        cs = self.recursion(hdl)
        self.__rely_model(cs)
        return cs

    def __condition(self, item: Dict, store: Dict, leader_model):
        child = add_leader_model(item["_child"], leader_model, SERIES_CONNECT_LEADER_TOKEN)
        child = store[child]
        parent = add_leader_model(item["_parent"], leader_model, SERIES_CONNECT_LEADER_TOKEN)
        parent = store[parent]
        value = (item["_values"])
        if (isinstance(value, list) and len(value) == 1):
            value = value[0]
        if isinstance(value, list):
            cond = InCondition(child, parent, list(map(smac_hdl._encode, value)))
        else:
            cond = EqualsCondition(child, parent, smac_hdl._encode(value))
        return cond

    def __forbidden(self, value: List, store: Dict, cs: ConfigurationSpace, leader_model):
        assert isinstance(value, list)
        for item in value:
            assert isinstance(item, dict)
            clauses = []
            for name, forbidden_values in item.items():
                true_name = add_leader_model(name, leader_model, SERIES_CONNECT_LEADER_TOKEN)
                if isinstance(forbidden_values, list) and len(forbidden_values) == 1:
                    forbidden_values = forbidden_values[0]
                if isinstance(forbidden_values, list):
                    clauses.append(ForbiddenInClause(store[true_name], list(map(smac_hdl._encode, forbidden_values))))
                else:
                    clauses.append(ForbiddenEqualsClause(store[true_name], smac_hdl._encode(forbidden_values)))
            cs.add_forbidden_clause(ForbiddenAndConjunction(*clauses))

    # def activate_helper(self,value):
    def reverse_dict(self, dict_: Dict):
        reversed_dict = defaultdict(list)
        for key, value in dict_.items():
            if isinstance(value, list):
                for v in value:
                    reversed_dict[v].append(key)
            else:
                reversed_dict[value].append(key)
        reversed_dict = dict(reversed_dict)
        for key, value in reversed_dict.items():
            reversed_dict[key] = list(set(value))
        return reversed_dict

    def pop_covered_item(self, dict_: Dict, length: int):
        dict_ = deepcopy(dict_)
        should_pop = []
        for key, value in dict_.items():
            assert isinstance(value, list)
            if len(value) > length:
                self.logger.warning("len(value) > length")
                should_pop.append(key)
            elif len(value) == length:
                should_pop.append(key)
        for key in should_pop:
            dict_.pop(key)
        return dict_

    def __activate(self, value: Dict, store: Dict, cs: ConfigurationSpace, leader_model):
        assert isinstance(value, dict)
        for k, v in value.items():
            assert isinstance(v, dict)
            reversed_dict = self.reverse_dict(v)
            reversed_dict = self.pop_covered_item(reversed_dict, len(v))
            for sk, sv in reversed_dict.items():
                cond = self.__condition(
                    {
                        "_child": sk,
                        "_values": sv,
                        "_parent": k
                    },
                    store,
                    leader_model
                )
                cs.add_condition(cond)

    def eliminate_suffix(self, key: str):
        s = "(choice)"
        if key.endswith(s):
            key = key[:-len(s)]
        return key

    def add_configuration_space(
            self, cs: ConfigurationSpace, cs_name: str, hdl_value: dict, is_choice: bool,
            option_hp: Configuration, children_is_choice=False):
        if is_choice:
            cs.add_configuration_space(cs_name, self.recursion(hdl_value, children_is_choice),
                                       parent_hyperparameter={"parent": option_hp, "value": cs_name})
        else:
            cs.add_configuration_space(cs_name, self.recursion(hdl_value, children_is_choice))

    def recursion(self, hdl, is_choice=False):
        ############ Declare ConfigurationSpace variables ###################
        cs = ConfigurationSpace()
        ####### Fill placeholder to empty ConfigurationSpace ################
        key_list = list(hdl.keys())
        if len(key_list) == 0:
            cs.add_hyperparameter(Constant("placeholder", "placeholder"))
            return cs
        ###################### Declare common variables #####################
        option_hp = None
        pattern = re.compile(r"(.*)\((.*)\)")
        store = {}
        conditions_dict = {}
        ########### If parent is choice configuration_space #################
        if is_choice:
            choices = []
            choice2proba = {}
            not_specific_proba_choices = []
            sum_proba = 0
            for k, v in hdl.items():
                if not is_hdl_bottom(k, v) and isinstance(v,dict):
                    k=self.eliminate_suffix(k)
                    if "__proba" in v:
                        proba = v.pop("__proba")
                        choice2proba[k] = proba
                        sum_proba += proba
                    else:
                        not_specific_proba_choices.append(k)
                    choices.append(self.eliminate_suffix(k))
            if sum_proba <= 1:
                if len(not_specific_proba_choices) > 0:
                    p_rest = (1 - sum_proba) / len(not_specific_proba_choices)
                    for not_specific_proba_choice in not_specific_proba_choices:
                        choice2proba[not_specific_proba_choice] = p_rest
            else:
                choice2proba = {k: 1 / len(choices) for k in choices}
            proba_list = [choice2proba[k] for k in choices]
            # fixme: choices = list(map(smac_hdl._encode, choices))
            option_hp = CategoricalHyperparameter('__choice__', choices, weights=proba_list)
            cs.add_hyperparameter(option_hp)  # todo: weight
        #### Travel key,value in hdl items, if value is dict(hdl), do recursion ######
        for hdl_key, hdl_value in hdl.items():
            mat = pattern.match(hdl_key)
            # add_configuration_space (choice)
            if mat and isinstance(hdl_value, dict):
                groups = mat.groups()
                assert len(groups) == 2, ValueError(f"Invalid hdl_key {hdl_key}")
                cs_name, method = groups
                assert method == "choice", ValueError(f"Invalid suffix {method}")
                self.add_configuration_space(cs, cs_name, hdl_value, is_choice, option_hp, True)
            elif is_hdl_bottom(hdl_key, hdl_value):
                if purify_key(hdl_key).startswith("__"):
                    conditions_dict[hdl_key] = hdl_value
                else:
                    hp = self.__parse_dict_to_config(hdl_key, hdl_value)
                    cs.add_hyperparameter(hp)
                    store[hdl_key] = hp
            # add_configuration_space
            elif isinstance(hdl_value, dict):
                cs_name = hdl_key
                self.add_configuration_space(cs, cs_name, hdl_value, is_choice, option_hp)
            else:
                raise NotImplementedError
        ########### Processing conditional hyperparameters #################
        for key, value in conditions_dict.items():
            if SERIES_CONNECT_LEADER_TOKEN in key:
                leader_model, condition_indicator = key.split(SERIES_CONNECT_LEADER_TOKEN)
            else:
                leader_model, condition_indicator = None, key
            if condition_indicator == "__condition":
                assert isinstance(value, list)
                for item in value:
                    cond = self.__condition(item, store, leader_model)
                    cs.add_condition(cond)
            elif condition_indicator == "__activate":
                self.__activate(value, store, cs, leader_model)
            elif condition_indicator == "__forbidden":
                self.__forbidden(value, store, cs, leader_model)
            else:
                self.logger.warning(f"Invalid condition_indicator: {condition_indicator}")
            # fixme: remove 'rely_model'
        return cs
        # add_hyperparameter

    def __parse_dict_to_config(self, key, value):
        if isinstance(value, dict):
            _type = value.get("_type")
            _value = value.get("_value")
            _default = value.get("_default")
            assert _value is not None
            if _type == "choice":
                return smac_hdl.choice(key, _value, _default)
            else:
                return eval(f'''smac_hdl.{_type}("{key}",*_value,default=_default)''')
        else:
            return Constant(key, smac_hdl._encode(value))
