import re
from collections import defaultdict, Counter
from copy import deepcopy
from importlib import import_module
from typing import Dict, List

import numpy as np
from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenInClause, ForbiddenEqualsClause, ForbiddenAndConjunction
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from hyperopt import fmin, tpe, hp

import autopipeline.hdl.smac as smac_hdl
from autopipeline.constants import Task
from autopipeline.hdl.utils import is_hdl_bottom
from autopipeline.hdl2phps.base import HDL2PHPS
from autopipeline.utils.packages import get_class_of_module


class RelyModels:
    info = []


class SmacHDL2PHPS(HDL2PHPS):
    @property
    def task(self):
        return self._task

    def set_task(self, task: Task):
        self._task = task

    def get(self, models, rely_model="boost_model"):
        forbid_in_value = []
        hitted = []
        for model in models:
            module_path = f"autopipeline.pipeline.components.{self.task.mainTask}.{model}"
            _class = get_class_of_module(module_path)
            M = import_module(module_path)
            cls = getattr(M, _class)
            hit = getattr(cls, rely_model, False)
            if not hit:
                forbid_in_value.append(model)
            else:
                hitted.append(model)
        return forbid_in_value, hitted

    def get_p(self, len_hitted, len_forbid, len_choices_list: list):
        def objective(p, debug=False):
            cs = ConfigurationSpace()
            for i, len_choices in enumerate(len_choices_list):
                cs.add_hyperparameter(CategoricalHyperparameter(f"P{i}", list(map(str, range(len_choices))),
                                                                weights=[p] + [(1 - p) / (len_choices - 1)] * (
                                                                        len_choices - 1)),
                                      )
            cs.add_hyperparameter(CategoricalHyperparameter(
                "E",
                [f"H{i}" for i in range(len_hitted)] + [f"F{i}" for i in
                                                        range(len_forbid)],
                weights=[p / len_hitted] * len_hitted + [(1 - p) / len_forbid] * len_forbid))
            for i, len_choices in enumerate(len_choices_list):
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter(f"P{i}"), "0"),
                    ForbiddenInClause(cs.get_hyperparameter("E"), [f"F{i}" for i in range(len_forbid)])
                ))
            cs.seed(42)
            try:
                counter = Counter([hp.get("E") for hp in cs.sample_configuration((len_hitted + len_forbid) * 15)])

                if debug:
                    print(counter)
            except Exception:
                return np.inf
            vl = list(counter.values())
            return np.var(vl) + 100 * (len_hitted + len_forbid - len(vl))

        best = fmin(
            fn=objective,
            space=hp.uniform('p', 0.001, 0.999),
            algo=tpe.suggest,
            max_evals=100,
            rstate=np.random.RandomState(42),
            show_progressbar=False,

        )
        print("best =",best)
        objective(best["p"], debug=True)
        return best["p"]

    def __call__(self, hdl: Dict,p=None):
        RelyModels.info = []
        cs = self.recursion(hdl)
        models = list(hdl["MHP(choice)"].keys())
        # 在进入主循环之前，就计算好概率
        len_choices_list = []
        for rely_model, path in RelyModels.info:
            path = path[:-1]
            forbid_eq_key = ":".join(path + ["__choice__"])
            forbid_eq_key_hp = cs.get_hyperparameter(forbid_eq_key)
            choices = forbid_eq_key_hp.choices
            len_choices_list.append(len(choices))
        forbid_in_value, hitted = self.get(models)
        len_hitted=len(hitted)
        len_forbid=len(forbid_in_value)
        if p is None:
            p = self.get_p(len_hitted, len_forbid, len_choices_list)
        # fixme : 复杂情况
        for rely_model, path in RelyModels.info:
            forbid_eq_value = path[-1]
            path = path[:-1]
            forbid_eq_key = ":".join(path + ["__choice__"])
            forbid_in_key = "MHP:__choice__"
            forbid_in_value, hitted = self.get(models, rely_model)
            forbid_eq_key_hp = cs.get_hyperparameter(forbid_eq_key)
            choices = forbid_eq_key_hp.choices
            probabilities = []
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
        MHP = cs.get_hyperparameter("MHP:__choice__")
        p_hitted = p / len_hitted
        p_forbid=(1 - p) / len_forbid
        probabilities = []
        for model in MHP.choices:
            if model in hitted:
                probabilities.append(p_hitted)
            else:
                probabilities.append(p_forbid)
        MHP.probabilities = probabilities
        # todo: 将MLP的默认模型设为boost
        # todo: 超参空间中没有boost的情况
        return cs
        # return {
        #     "phps":cs,
        #     "p":p
        # }

    def __condition(self, item: Dict, store: Dict):
        child = item["_child"]
        child = store[child]
        parent = item["_parent"]
        parent = store[parent]
        value = (item["_values"])
        if (isinstance(value, list) and len(value) == 1):
            value = value[0]
        if isinstance(value, list):
            cond = InCondition(child, parent, list(map(smac_hdl._encode, value)))
        else:
            cond = EqualsCondition(child, parent, smac_hdl._encode(value))
        return cond

    def __forbidden(self, value: List, store: Dict, cs: ConfigurationSpace):
        assert isinstance(value, list)
        for item in value:
            assert isinstance(item, dict)
            clauses = []
            for k, v in item.items():
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                if isinstance(v, list):
                    clauses.append(ForbiddenInClause(store[k], list(map(smac_hdl._encode, v))))
                else:
                    clauses.append(ForbiddenEqualsClause(store[k], smac_hdl._encode(v)))
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
                print("warn")
                should_pop.append(key)
            elif len(value) == length:
                should_pop.append(key)
        for key in should_pop:
            dict_.pop(key)
        return dict_

    def __activate(self, value: Dict, store: Dict, cs: ConfigurationSpace):
        assert isinstance(value, dict)
        for k, v in value.items():
            assert isinstance(v, dict)
            reversed_dict = self.reverse_dict(v)
            reversed_dict = self.pop_covered_item(reversed_dict, len(v))
            for sk, sv in reversed_dict.items():
                cond = self.__condition({
                    "_child": sk,
                    "_values": sv,
                    "_parent": k
                }, store)
                cs.add_condition(cond)

    def recursion(self, hdl: Dict, path=()) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        # 检测一下这个dict是否在直接描述超参
        key_list = list(hdl.keys())
        if len(key_list) == 0:
            cs.add_hyperparameter(Constant("placeholder", "placeholder"))
            return cs
        else:
            sample_key = key_list[0]
            sample_value = hdl[sample_key]
            if is_hdl_bottom(sample_key, sample_value):
                store = {}
                conditions_dict = {}
                for key, value in hdl.items():
                    if key.startswith("__"):
                        conditions_dict[key] = value
                    else:
                        # assert isinstance(value, dict)  # fixme ： 可以对常量进行编码
                        hp = self.__parse_dict_to_config(key, value)
                        # hp.name = key
                        cs.add_hyperparameter(hp)
                        store[key] = hp
                for key, value in conditions_dict.items():
                    if key == "__condition":
                        assert isinstance(value, list)
                        for item in value:
                            cond = self.__condition(item, store)
                            cs.add_condition(cond)
                    elif key == "__activate":
                        self.__activate(value, store, cs)
                    elif key == "__forbidden":
                        self.__forbidden(value, store, cs)
                    elif key == "__rely_model":
                        RelyModels.info.append([
                            value,
                            deepcopy(path)
                        ])

                return cs
        pattern = re.compile(r"(.*)\((.*)\)")
        for key, value in hdl.items():
            mat = pattern.match(key)
            if mat:
                groups = mat.groups()
                assert len(groups) == 2
                prefix_name, method = groups
                value_list = list(value.keys())
                assert len(value_list) >= 1
                if method == "choice":
                    pass
                elif method == "optional-choice":
                    value_list.append(None)
                else:
                    raise NotImplementedError()
                cur_cs = ConfigurationSpace()
                assert isinstance(value, dict)
                # 不能用constant，会报错
                value_list = list(map(smac_hdl._encode, value_list))
                option_param = CategoricalHyperparameter('__choice__', value_list)  # todo : default
                cur_cs.add_hyperparameter(option_param)
                for sub_key, sub_value in value.items():
                    assert isinstance(sub_value, dict)
                    sub_cs = self.recursion(sub_value, path=list(path) + [prefix_name, sub_key])
                    parent_hyperparameter = {'parent': option_param, 'value': sub_key}
                    cur_cs.add_configuration_space(sub_key, sub_cs, parent_hyperparameter=parent_hyperparameter)
                cs.add_configuration_space(prefix_name, cur_cs)
            elif isinstance(value, dict):
                sub_cs = self.recursion(value, path=list(path) + [key])
                cs.add_configuration_space(key, sub_cs)
            else:
                raise NotImplementedError()

        return cs

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
