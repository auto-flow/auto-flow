from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Iterator

import json5 as json
from joblib import Memory

from autoflow import hdl
from autoflow.constants import SERIES_CONNECT_LEADER_TOKEN, SERIES_CONNECT_SEPARATOR_TOKEN, JOBLIB_CACHE


def is_hdl_bottom(key, value):
    if isinstance(key, str) and purify_key(key).startswith("__"):
        return True
    if isinstance(value, dict) and "_type" in value:
        return True
    if not isinstance(value, dict):
        return True
    return False


def get_hdl_bank(path: str, logger=None) -> Dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    else:
        if logger is not None:
            logger.warning(f"Specific hdl_bank: {path} is not exists, using get_default_hdl_bank() default.")
        return get_default_hdl_bank()


def _get_default_hdl_bank() -> Dict:
    return json.loads((Path(hdl.__file__).parent / f"hdl_bank.json").read_text())


memory = Memory(JOBLIB_CACHE, verbose=1)
get_default_hdl_bank = memory.cache(_get_default_hdl_bank)


def get_origin_models(raw_models: List[str]):
    result = []
    for raw_model in raw_models:
        last = raw_model.split(SERIES_CONNECT_SEPARATOR_TOKEN)[-1]
        if last not in result:
            result.append(last)
    return result


def purify_key(key: str):
    if not isinstance(key, str):
        return key
    if SERIES_CONNECT_LEADER_TOKEN in key:
        ix = key.find(SERIES_CONNECT_LEADER_TOKEN) + 1
        return key[ix:]
    return key


def add_leader_model(key, leader_model, SERIES_CONNECT_LEADER_TOKEN):
    if leader_model is None:
        return key
    else:
        return leader_model + SERIES_CONNECT_LEADER_TOKEN + key


def purify_keys(dict_: dict) -> Iterator[str]:
    for key in dict_.keys():
        yield purify_key(key)


def get_default_hp_of_cls(cls):
    module = cls.__module__
    module = module.replace("autoflow.workflow.components.", "")
    hdl_bank = get_default_hdl_bank()
    hp = deepcopy(hdl_bank)
    for x in module.split("."):
        hp = hp.get(x, {})
    res={}
    for k, v in hp.items():
        if isinstance(v, dict) and "_default" in v:
            res[k] = v["_default"]
    print(f"default hyperparams of {cls.__name__} :")
    pprint(res)
    return res
