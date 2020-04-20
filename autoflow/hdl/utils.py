from pathlib import Path
from typing import Dict, List, Iterator

import json5 as json

from autoflow import hdl
from autoflow.constants import SERIES_CONNECT_LEADER_TOKEN, SERIES_CONNECT_SEPARATOR_TOKEN


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


def get_default_hdl_bank() -> Dict:
    return json.loads((Path(hdl.__file__).parent / f"hdl_bank.json").read_text())


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
