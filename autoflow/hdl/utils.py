from pathlib import Path
from typing import Dict, List

import json5 as json

from autoflow import hdl


def is_hdl_bottom(key, value):
    if isinstance(key, str) and key.startswith("__"):
        return True
    if isinstance(value, dict) and "_type" in value:
        return True
    if not isinstance(value, dict):
        return True
    return False


def get_hdl_bank(path: str,logger=None) -> Dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    else:
        if logger is not None:
            logger.warning(f"Specific hdl_bank: {path} is not exists, using get_default_hdl_bank() default.")
        return get_default_hdl_bank()


def get_default_hdl_bank() -> Dict:
    return json.loads((Path(hdl.__file__).parent / f"hdl_bank.json").read_text())

def get_origin_models(raw_models:List[str]):
    result=[]
    for raw_model in raw_models:
        last=raw_model.split("|")[-1]
        if last not in result:
            result.append(last)
    return result

