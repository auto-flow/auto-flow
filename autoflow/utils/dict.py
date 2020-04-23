from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any, List, Callable

from autoflow.utils.list import remove_suffix_in_list
from autoflow.utils.logging_ import get_logger

logger = get_logger(__name__)


def add_prefix_in_dict_keys(dict_: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    result = {}
    for key, value in dict_.items():
        result[f"{prefix}{key}"] = value
    return result


def group_dict_items_before_first_token(dict_: Dict[str, Any], token: str) -> Dict[str, Dict[str, Any]]:
    result = defaultdict(dict)
    for packages, value in dict_.items():
        if token in packages:
            split_list = packages.split(token)
            key1 = token.join(split_list[:-1])
            key2 = split_list[-1]
        else:
            key1 = "single"
            key2 = packages
        result[key1][key2] = value
    return result


def replace_kv(dict_: Dict, rk, rv):
    for k, v in dict_.items():
        if isinstance(v, dict):
            replace_kv(v, rk, rv)
        elif k == rk:
            dict_[k] = rv


def sort_dict(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = sort_dict(v)
        return dict(sorted(obj.items(), key=lambda x: str(x[0])))
    elif isinstance(obj, list):
        for i, elem in enumerate(obj):
            obj[i] = sort_dict(elem)
        return list(sorted(obj, key=str))
    else:
        return obj


def remove_None_value(dict_: dict) -> dict:
    result = {}
    for k, v in dict_.items():
        if v is not None:
            result[k] = v
    return result


class GlobalList:
    data = []


def _find_token_in_dict(dict_: dict, token, path=[]):  # -> Iterator[Tuple[List[str], str]]:
    for key, value in dict_.items():
        if key == token:
            # yield list(path), "key"
            GlobalList.data.append([list(path), "key"])
        elif isinstance(value, str) and value == token:
            # yield list(path) + [key], "value"
            GlobalList.data.append([list(path) + [key], "value"])
        elif isinstance(value, dict):
            _find_token_in_dict(value, token, list(path) + [key])


def find_token_in_dict(dict_: dict, token):
    GlobalList.data = []
    _find_token_in_dict(dict_, token)
    return GlobalList.data


def read_dict_in_path_mode(dict_: dict, path: List[str]):
    for i in path:
        dict_ = dict_[i]
    return dict_


def write_dict_in_path_mode(dict_: dict, path: List[str], value):
    for i in path[:-1]:
        dict_ = dict_[i]
    dict_[path[-1]] = value


def update_mask_from_other_dict(should_update: Dict, template: Dict, ignored_suffix: str = "(choice)"):
    token = "<mask>"
    updated = deepcopy(should_update)
    for path, type_ in find_token_in_dict(updated, token):
        origin_path = path
        template_path = remove_suffix_in_list(path, ignored_suffix)
        if type_ == "key":
            # origin_value = updated[origin_path][token]
            origin_value = read_dict_in_path_mode(updated, origin_path)[token]
            template_item = read_dict_in_path_mode(template, template_path)
            assert isinstance(template_item, dict)
            template_keys = list(template_item.keys())
            assert len(template_keys) == 1
            template_key = template_keys[0]
            template_value: dict = template_item[template_key]
            template_value.update(origin_value)
            write_dict_in_path_mode(updated, origin_path, {template_key: template_value})
        elif type_ == "value":
            write_dict_in_path_mode(updated, origin_path, read_dict_in_path_mode(template, template_path))
        else:
            raise NotImplementedError
    return updated


def filter_item_by_key_condition(dict_: dict, func: Callable) -> dict:
    available_keys = [key for key in dict_ if func(key)]
    result = {key: dict_[key] for key in available_keys}
    return result


def update_data_structure(old_dict: dict, additional_dict: dict) -> dict:
    '''
    >>> update_data_structure(
    ...     {"key1": "a", "key2": [1, True, {"a", (2, )}], "key3": {"a":1, "b":2}},
    ...     {"key1": "b", "key2": [1, False], "key3": {"a":{"b":2}}, "key4": "new_value"}
    ... )
    {"key1": "b", "key2": [1, True, {"a", (2, )}, 1, False], "key3": {"a":{"b":2, "b":2}}, "key4": "new_value"}

    Returns
    -------
    updated_dict: dict
    '''
    updated_dict = deepcopy(old_dict)
    for k, new_v in additional_dict.items():
        if k in updated_dict:
            old_v = deepcopy(updated_dict[k])
            new_v = deepcopy(new_v)
            if isinstance(old_v, dict):
                if isinstance(new_v, dict):
                    old_v.update(new_v)
                else:
                    logger.warning(f"In 'update_data_structure', old_v = {old_v}, but new_v = "
                                   f"{new_v} is not dict, cannot update.")
            elif isinstance(old_v, list):
                if not isinstance(new_v, list):
                    new_v = [new_v]
                old_v += new_v
            else:
                old_v = new_v
            updated_dict[k] = old_v
        else:
            updated_dict[k] = additional_dict[k]
    return updated_dict


if __name__ == '__main__':
    updated_dict = update_data_structure(
        {"key1": "a", "key2": [1, True, {"a", (2,)}], "key3": {"a": 1, "b": 2}},
        {"key1": "b", "key2": [1, False], "key3": {"a": {"b": 2}}, "key4": "new_value"}
    )
    wanted = {"key1": "b", "key2": [1, True, {"a", (2,)}, 1, False], "key3": {"a": {"b": 2}, "b": 2},
              "key4": "new_value"}
    print(updated_dict)
    print(wanted)
    print(updated_dict == wanted)
