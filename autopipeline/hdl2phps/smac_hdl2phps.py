from collections import defaultdict
from copy import deepcopy
from typing import Dict

from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, Constant

import autopipeline.hdl.smac as smac_hdl
from autopipeline.hdl2phps.base import HDL2PHPS


def param_dict_to_ConfigurationSpace(param_dict: dict):
    cs = ConfigurationSpace()
    for key, value in param_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, Hyperparameter)
        value.name = key
        cs.add_hyperparameter(value)


def is_bottom(key, value):
    if isinstance(key, str) and key.startswith("__"):
        return True
    if isinstance(value, dict) and "_type" in value:
        return True
    return False


class SmacHDL2PHPS(HDL2PHPS):

    def __call__(self, hdl: Dict):
        return self.recursion(hdl)

    def __condition(self, item: Dict, store: Dict):
        child = item["_child"]
        child = store[child]
        parent = item["_parent"]
        parent = store[parent]
        value = item["_values"]
        if (isinstance(value, list) and len(value) == 1):
            value = value[0]
        if isinstance(value, list):
            cond = InCondition(child, parent, value)
        else:
            cond = EqualsCondition(child, parent, value)
        return cond

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

    def __activate(self, value: Dict,store:Dict,cs:ConfigurationSpace):
        assert isinstance(value, dict)
        for k, v in value.items():
            assert isinstance(v, dict)
            reversed_dict = self.reverse_dict(v)
            reversed_dict = self.pop_covered_item(reversed_dict, len(v))
            for sk, sv in reversed_dict.items():
                cond=self.__condition({
                    "_child":sk,
                    "_values":sv,
                    "_parent":k
                },store)
                cs.add_condition(cond)

    def recursion(self, hdl: Dict) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        # 检测一下这个dict是否在直接描述超参
        key_list = list(hdl.keys())
        if len(key_list) == 0:
            cs.add_hyperparameter(Constant("placeholder", "placeholder"))
            return cs
        else:
            sample_key = key_list[0]
            sample_value = hdl[sample_key]
            if is_bottom(sample_key, sample_value):
                store = {}
                conditions_dict = {}
                for key, value in hdl.items():
                    if key.startswith("__"):
                        conditions_dict[key] = value
                    else:
                        assert isinstance(value, dict)
                        hp = self.__parse_dict_to_config(value)
                        hp.name = key
                        cs.add_hyperparameter(hp)
                        store[key] = hp
                for key, value in conditions_dict.items():
                    if key == "__condition":
                        assert isinstance(value, list)
                        for item in value:
                            cond = self.__condition(item, store)
                            cs.add_condition(cond)
                    elif key == "__activate":
                        self.__activate(value,store,cs)

                return cs
        for key, value in hdl.items():
            if key.endswith("(choice)"):
                prefix_name = key.split("(choice)")[0]
                cur_cs = ConfigurationSpace()
                assert isinstance(value, dict)
                option_param = CategoricalHyperparameter('__choice__',
                                                         list(value.keys()))  # todo : default
                cur_cs.add_hyperparameter(option_param)
                for sub_key, sub_value in value.items():
                    assert isinstance(sub_value, dict)
                    sub_cs = self.recursion(sub_value)
                    parent_hyperparameter = {'parent': option_param, 'value': sub_key}
                    cur_cs.add_configuration_space(sub_key, sub_cs, parent_hyperparameter=parent_hyperparameter)
                cs.add_configuration_space(prefix_name, cur_cs)
            elif isinstance(value, dict):
                sub_cs = self.recursion(value)
                cs.add_configuration_space(key, sub_cs)
            else:
                raise NotImplementedError()

        return cs

    def __parse_dict_to_config(self, dict_: dict):
        _type = dict_.get("_type")
        _value = dict_.get("_value")
        _default = dict_.get("_default")
        assert _value is not None
        if _type == "choice":
            return smac_hdl.choice("", _value, _default)
        else:
            return eval(f'''smac_hdl.{_type}("",*_value,_default)''')


if __name__ == '__main__':
    pass
