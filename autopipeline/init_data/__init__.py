import os
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional

try:
    import json5 as json
except:
    import json


def init_all(
        specific_init_dict=None,
        additional:Optional[Dict]=None,
        public_hp:Optional[Dict]=None
):
    if specific_init_dict:
        dict_ = specific_init_dict
    else:
        _file = (Path(__file__).parent / "init_data.json").as_posix()
        assert os.path.exists(_file)
        dict_: dict = json.loads(Path(_file).read_text())
    __init_all_recursion(dict_, additional=additional,public_hp=public_hp)


def __init_all_recursion(dict_: dict, path=(), additional=None,public_hp=None):
    for key, value in dict_.items():
        splited = key.split(".")
        L = len(splited)
        if L == 2:
            module = import_module(f"autopipeline.pipeline.components.{'.'.join(path)}.{splited[0]}")
            cls = getattr(module, splited[1])
            assert isinstance(value, dict)
            cls.set_cls_hyperparams(value)
            if additional and isinstance(additional,dict):
                cls.set_params_from_dict(additional)
            if public_hp and isinstance(public_hp,dict):
                cls.update_cls_hyperparams(public_hp)
        elif L == 1 and isinstance(value, dict):
            __init_all_recursion(value, list(path) + [key], additional,public_hp)
        else:
            raise Exception()
