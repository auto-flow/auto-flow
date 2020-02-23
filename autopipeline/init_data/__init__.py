from copy import deepcopy
from typing import Dict


def __recursion(dict_: Dict):
    should_pop = []
    should_recursion = []
    for key, value in dict_.items():
        if isinstance(value, dict):
            if "_type" in value:
                should_pop.append(key)
            else:
                should_recursion.append(value)
    for key in should_pop:
        dict_.pop(key)
    for sub_dict_ in should_recursion:
        __recursion(sub_dict_)


def extract_default_hp_from_hdl_db(hdl_db: Dict) -> Dict:
    dict_ = deepcopy(hdl_db)
    __recursion(dict_)
    return dict_


def is_bottom_dict(value):
    if isinstance(value, dict) :
        value_list=list(value.values())
        if len(value_list)==0:
            return True
        if not isinstance(value_list[0],dict):
            return True
    return False


def add_public_info(default_hp: Dict, public_info: Dict):
    for key, value in default_hp.items():
        if is_bottom_dict(value):
            for p_k, p_v in public_info.items():
                value[p_k] = p_v
        elif isinstance(value, dict):
            add_public_info(default_hp[key], public_info)

# def init_all(
#         specific_init_dict=None,
#         additional:Optional[Dict]=None,
#         public_hp:Optional[Dict]=None
# ):
#     if specific_init_dict:
#         dict_ = specific_init_dict
#     else:
#         _file = (Path(__file__).parent / "init_hp_space.json").as_posix()
#         assert os.path.exists(_file)
#         dict_: dict = json.loads(Path(_file).read_text())
#     __init_all_recursion(dict_, additional=additional,public_hp=public_hp)
#
#
# def __init_all_recursion(dict_: dict, path=(), additional=None,public_hp=None):
#     for key, value in dict_.items():
#         splited = key.split(".")
#         L = len(splited)
#         if L == 2:
#             module = import_module(f"autopipeline.pipeline.components.{'.'.join(path)}.{splited[0]}")
#             cls = getattr(module, splited[1])
#             assert isinstance(value, dict)
#             cls.set_cls_hyperparams(value)
#             if additional and isinstance(additional,dict):
#                 cls.set_params_from_dict(additional)
#             if public_hp and isinstance(public_hp,dict):
#                 cls.update_cls_hyperparams(public_hp)
#         elif L == 1 and isinstance(value, dict):
#             __init_all_recursion(value, list(path) + [key], additional,public_hp)
#         else:
#             raise Exception()
