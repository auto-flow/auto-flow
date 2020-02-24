import inspect
from pathlib import Path
from typing import Dict
from importlib import import_module
from autopipeline import init_data
import json5 as json

def get_class_of_module(input_module):
    if isinstance(input_module,str):
        _module=import_module(input_module)
    else:
        _module=input_module
    if hasattr(_module,"__all__"):
        return _module.__all__[0]
    else:
        return inspect.getmembers(_module,inspect.isclass)[0][0]

def get_default_hdl_db()->Dict:
    return json.loads((Path(init_data.__file__).parent / f"hdl_db.json").read_text())

def get_hdl_db(path:str)->Dict:
    path=Path(path)
    if path.exists():
        return json.loads(path.read_text())
    else:
        print("warn")
        return get_default_hdl_db()
