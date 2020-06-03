import importlib
import inspect
import pkgutil
from collections import OrderedDict
from importlib import import_module


def get_class_name_of_module(input_module):
    if isinstance(input_module, str):
        try:
            _module = import_module(input_module)
        except:
            return None
    else:
        _module = input_module
    if hasattr(_module, "__all__"):
        return _module.__all__[0]
    else:
        return inspect.getmembers(_module, inspect.isclass)[-1][0]


def get_class_object_in_pipeline_components(key1, key2):
    try:
        module_path = f"autoflow.workflow.components.{key1}.{key2}"
        _class = get_class_name_of_module(module_path)
        M = import_module(
            module_path
        )
        assert hasattr(M, _class)
        cls = getattr(M, _class)
        return cls
    except Exception:
        return None


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        # if full_module_name not in sys.modules and not ispkg:
        module = importlib.import_module(full_module_name)
        if hasattr(module, "excludeToken"):
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

def import_by_package_url(package_url:str):
    assert "." in package_url
    ix=package_url.rfind(".")
    module=package_url[:ix]
    class_name=package_url[ix+1:]
    return getattr(import_module(module),class_name)
