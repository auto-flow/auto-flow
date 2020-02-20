from typing import Any, Iterable

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Hyperparameter


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


def choice(label: str, options: Iterable, default=None):
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
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


def param_dict_to_ConfigurationSpace(param_dict: dict):
    cs = ConfigurationSpace()
    for key, value in param_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, Hyperparameter)
        value.name = key
        cs.add_hyperparameter(value)
    return cs


def Configuration_to_param_dict(cfg: Configuration):
    cfg = cfg.get_dictionary()  # fixme : None value ?
    cfg_ = {}
    for key, value in cfg.items():
        if value is None:
            continue
        if isinstance(value, str) and len(value.split(':')) == 2:
            value = _decode(value)
        cfg_[key] = value
    return cfg_


