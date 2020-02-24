import re
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple, Dict

from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter

import autopipeline.hdl.smac as smac_hdl
from autopipeline.hdl2phps.base import HDL2PHPS


def param_dict_to_ConfigurationSpace(param_dict: dict):
    cs = ConfigurationSpace()
    for key, value in param_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, Hyperparameter)
        value.name = key
        cs.add_hyperparameter(value)


class SmacHDL2PHPS(HDL2PHPS):
    def eval_str_to_dict(self, hdl):
        hdl_ = deepcopy(hdl)
        self.__recursion_eval(hdl_)
        return hdl_

    def after_process_dict(self, dict_: dict):
        cs = ConfigurationSpace()
        conditions = defaultdict(list)
        ret = self.__recursion_after_process(dict_,conditions)
        for name, hyperparam in ret.items():
            cs.add_hyperparameter(hyperparam)
        # ----conditions----
        InCondition_bins=defaultdict(list)
        for item in conditions["conditions"]:
            child=item["_child"]
            child=ret[child]
            parent=item["_parent"]
            parent=ret[parent]
            value=item["_values"]
            _type=item.get("_type","InCondition")
            if _type=="InCondition":
                InCondition_bins[child].append([child,parent,value])
            else:
                raise NotImplementedError()
        for members in InCondition_bins.values():
            if len(members)==1:

            cs.add_condition(
                cond
            )
            cs.add_condition(cond)
        return cs

    def __recursion_eval(self, dict_: dict):
        # 用含"_type"的dict描述超参。这里将这个值翻译出来
        for key, value in dict_.items():
            if isinstance(value, dict):
                if ("_type" in value):
                    dict_[key] = self.__parse_dict_to_config(value)
                else:
                    self.__recursion_eval(value)
            # else:
            #     raise AttributeError()

    def __parse_dict_to_config(self, dict_: dict):
        _type = dict_.get("_type")
        _value = dict_.get("_value")
        _default = dict_.get("_default")
        assert _value is not None
        if _type == "choice":
            return smac_hdl.choice("", _value, _default)
        else:
            return eval(f'''smac_hdl.{_type}("",*_value,_default)''')

    def __concat_prefix(self, prefix, name):
        if not prefix:
            return name
        else:
            return f'{prefix}/{name}'

    def __get_rely(self, key) -> Optional[Tuple]:
        pattern_str = r'^\[(.*)\]$'
        pattern = re.compile(pattern_str)
        m = pattern.match(key)
        if m:
            rely_tuple: tuple = m.groups()
            if len(rely_tuple) == 1:
                return rely_tuple[0]
        return None

    def add_inside_conditions_prefix(self, prefix, inside_conditions):
        keys=["_child","_parent"]
        for key in keys:
            value=inside_conditions[key]
            value=self.__concat_prefix(prefix,value)
            inside_conditions[key]=value

    def __recursion_after_process(
            self, dict_: dict,conditions:Dict, prefix: str = None, rely_parent=None, rely_value=None,

    ) -> Dict:
        ret = {}
        for key, value in dict_.items():
            if isinstance(value, Hyperparameter):
                cur_prefix = self.__concat_prefix(prefix, key)
                value.name = cur_prefix
                ret[cur_prefix] = value
                if rely_parent and rely_value:
                    conditions["conditions"].append(
                        {"_child":cur_prefix,"_parent":rely_parent,"_values":rely_value}
                    )
            elif isinstance(value, dict) and (not  key.startswith("__")):
                rely_on = self.__get_rely(key)
                cur_prefix = self.__concat_prefix(prefix, key)
                if rely_on:
                    sub_ret= self.__recursion_after_process(value, conditions,cur_prefix, rely_on)
                    ret.update(sub_ret)
                else:
                    sub_ret = self.__recursion_after_process(value, conditions,cur_prefix, rely_parent,
                                                                               rely_value=key)
                    ret.update(sub_ret)
            elif key=="__condition":
                assert isinstance(value,list)
                for item in value:
                    assert isinstance(item, dict)
                    self.add_inside_conditions_prefix(prefix,item)
                    conditions["conditions"].append(item)
            else:
                raise Exception()

        return ret


if __name__ == '__main__':
    pass
