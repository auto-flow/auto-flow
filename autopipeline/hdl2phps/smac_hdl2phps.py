import re
from copy import deepcopy
from typing import Optional, Tuple, Dict

from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter
from importlib import import_module
from autopipeline.hdl2phps.base import HDL2PHPS
import autopipeline.hdl.smac as smac_hdl


def param_dict_to_ConfigurationSpace(param_dict: dict):
    cs = ConfigurationSpace()
    for key, value in param_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, Hyperparameter)
        value.name = key
        cs.add_hyperparameter(value)


class SmacHDL2PHPS(HDL2PHPS):
    def eval_str_to_dict(self, hdl):
        hdl_=deepcopy(hdl)
        self.__recursion_eval(hdl_)
        return hdl_


    def after_process_dict(self, dict_: dict):
        cs=ConfigurationSpace()
        ret,rely_on_dict=self.__recursion_after_process(dict_)
        for name,hyperparam in ret.items():
            cs.add_hyperparameter(hyperparam)
        for name, (rely_parent, rely_value) in rely_on_dict.items():
            cs.add_condition(
                InCondition(
                    child=ret[name], parent=ret[rely_parent], values=[rely_value]
                )
            )
        return cs

    def __recursion_eval(self,dict_:dict):
        for key, value in dict_.items():
            if isinstance(value,dict):
                if ("_type" in  value):
                    dict_[key]=self.__parse_dict_to_config(value)
                else:
                    self.__recursion_eval(value)
            else:
                raise AttributeError()


    def __parse_dict_to_config(self,dict_:dict):
        _type=dict_.get("_type")
        _value=dict_.get("_value")
        _default=dict_.get("_default")
        assert _value is not None
        if _type=="choice":
            return smac_hdl.choice("",_value,_default)
        else:
            return eval(f'''smac_hdl.{_type}("",*_value,_default)''')



    def __concat_prefix(self, prefix, name):
        if not prefix:
            return name
        else:
            return f'{prefix}/{name}'

    def __get_rely(self, key) -> Optional[Tuple]:
        pattern_str = f'^\[(.*)\]$'
        pattern = re.compile(pattern_str)
        m = pattern.match(key)
        if m:
            rely_tuple: tuple = m.groups()
            if len(rely_tuple) == 1:
                return rely_tuple[0]
        return None

    def __recursion_after_process(self, dict_: dict, prefix: str = None, rely_parent=None, rely_value=None) -> Tuple[Dict, Dict]:
        ret = {}
        rely_on_dict = {}
        for key, value in dict_.items():
            if isinstance(value, Hyperparameter):
                cur_prefix = self.__concat_prefix(prefix, key)
                value.name = cur_prefix
                ret[cur_prefix] = value
                if rely_parent and rely_value:
                    rely_on_dict.update({cur_prefix: (rely_parent, rely_value)})
            elif isinstance(value, dict):
                rely_on = self.__get_rely(key)
                cur_prefix = self.__concat_prefix(prefix, key)
                if rely_on:
                    sub_ret, sub_rely_on_dict = self.__recursion_after_process(value, cur_prefix, rely_on)
                    ret.update(sub_ret)
                    rely_on_dict.update(sub_rely_on_dict)
                else:
                    sub_ret, sub_rely_on_dict = self.__recursion_after_process(value, cur_prefix, rely_parent, rely_value=key)
                    ret.update(sub_ret)
                    rely_on_dict.update(sub_rely_on_dict)
            else:
                raise Exception()

        return ret, rely_on_dict


if __name__ == '__main__':
    pass