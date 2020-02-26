import inspect
from importlib import import_module


def get_class_of_module(input_module):
    if isinstance(input_module,str):
        _module=import_module(input_module)
    else:
        _module=input_module
    if hasattr(_module,"__all__"):
        return _module.__all__[0]
    else:
        return inspect.getmembers(_module,inspect.isclass)[0][0]

