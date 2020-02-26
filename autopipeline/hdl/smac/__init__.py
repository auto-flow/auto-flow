from typing import Any, Iterable, List

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Hyperparameter, Constant


def _encode(value: Any) -> str:
    if isinstance(value,str):
        return value
    return f'{value}:{(value).__class__.__name__}'


def _decode(str_value: str) -> Any:
    lst = str_value.split(':')
    if len(lst) == 2:
        value_, type_ = lst
        if type_ in ('NoneType',):
            return eval(value_)
        cls = eval(type_)
        return cls(value_)
    elif len(lst)==1:
        return lst[0]
    else:
        raise Exception()


def choice(label: str, options: List, default=None):
    if len(options)==1:
        return Constant(label,_encode(options[0]))
    kwargs = {}
    if default:
        kwargs.update({'default_value': _encode(default)})
    return CategoricalHyperparameter(label, [_encode(option) for option in options], **kwargs)


def int_uniform(label: str, low: int, high: int, default=None):
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, **kwargs)

def uniform(label: str, low: float, high: float, default=None):
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, **kwargs)

def loguniform(label: str, low: float, high: float, default=None):
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, **kwargs)



