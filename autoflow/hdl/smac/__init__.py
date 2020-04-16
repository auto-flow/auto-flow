import math
from typing import Any, List

from ConfigSpace import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from autoflow.utils.math import float_gcd


def _encode(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f'{value}:{(value).__class__.__name__}'


def _decode(str_value: str) -> Any:
    if str_value == "None":
        return None
    ix = str_value.rfind(":")
    if ix < 0:
        return str_value
    else:
        value_ = str_value[:ix]
        type_ = str_value[ix + 1:]
        return eval(value_)


def choice(label: str, options: List, default=None):
    if len(options) == 1:
        return Constant(label, _encode(options[0]))  # fixme: if declare probability in here?
    # fixme: copy from autoflow/hdl2shps/hdl2shps.py:354
    choice2proba = {}
    not_specific_proba_choices = []
    sum_proba = 0
    choices = []
    raw_choices = []
    for option in options:
        if isinstance(option, (tuple, list)) and len(option) == 2:
            choice = None
            proba = None
            for item in option:
                if isinstance(item, (float, int)) and 0 <= item <= 1:
                    proba = item
                else:
                    choice = item
            assert choice is not None and proba is not None
            choice2proba[choice] = proba
            sum_proba += proba
        else:
            choice = option
            not_specific_proba_choices.append(choice)
        choices.append(_encode(choice))
        raw_choices.append(choice)
    if sum_proba <= 1:
        if len(not_specific_proba_choices) > 0:
            p_rest = (1 - sum_proba) / len(not_specific_proba_choices)
            for not_specific_proba_choice in not_specific_proba_choices:
                choice2proba[not_specific_proba_choice] = p_rest
    else:
        choice2proba = {k: 1 / len(options) for k in choices}
    proba_list = [choice2proba[k] for k in raw_choices]
    kwargs = {}
    if default:
        kwargs.update({'default_value': _encode(default)})
    hp=CategoricalHyperparameter(label, choices, weights=proba_list, **kwargs)
    hp.probabilities=proba_list  # fixme: don't make sense
    return hp


def int_quniform(label: str, low: int, high: int, q: int = None, default=None):
    if not q:
        q = math.gcd(low, high)
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, q=q, **kwargs)


def int_uniform(label: str, low: int, high: int, default=None):
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, **kwargs)


def quniform(label: str, low: float, high: float, q: float = None, default=None):
    if not q:
        q = float_gcd(low, high)
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, q=q, **kwargs)


def uniform(label: str, low: float, high: float, default=None):
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, **kwargs)


# fixme: have some bug in practice
def qloguniform(label: str, low: float, high: float, q: float = None, default=None):
    if not q:
        q = float_gcd(low, high)
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, q=q, **kwargs)


def loguniform(label: str, low: float, high: float, default=None):
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, **kwargs)


def int_qloguniform(label: str, low: int, high: int, q: int = None, default=None):
    if not q:
        q = min(low, 1)
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, q=q, **kwargs)


def int_loguniform(label: str, low: int, high: int, default=None):
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, **kwargs)
