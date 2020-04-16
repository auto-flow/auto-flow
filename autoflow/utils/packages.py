import importlib
import inspect
import pkgutil
import sys
from collections import OrderedDict
from importlib import import_module


def get_class_name_of_module(input_module):
    if isinstance(input_module,str):
        _module=import_module(input_module)
    else:
        _module=input_module
    if hasattr(_module,"__all__"):
        return _module.__all__[0]
    else:
        return inspect.getmembers(_module,inspect.isclass)[-1][0]

def get_class_object_in_pipeline_components(key1, key2):
    module_path = f"autoflow.pipeline.components.{key1}.{key2}"
    _class = get_class_name_of_module(module_path)
    M = import_module(
        module_path
    )
    assert hasattr(M, _class)
    cls = getattr(M, _class)
    return cls


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        # if full_module_name not in sys.modules and not ispkg:
        module = importlib.import_module(full_module_name)
        if hasattr(module,"excludeToken"):
            continue
        for member_name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, base_class) and \
                    obj != base_class:
                # TODO test if the obj implements the interface
                # Keep in mind that this only instantiates the ensemble_wrapper,
                # but not the real target classifier
                classifier = obj
                components[module_name] = classifier

    return components


