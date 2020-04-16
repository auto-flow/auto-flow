from collections import defaultdict
from copy import deepcopy
from typing import Dict, Any, List

from autoflow.utils.list import remove_suffix_in_list


def add_prefix_in_dict_keys(dict_: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    result = {}
    for key, value in dict_.items():
        result[f"{prefix}{key}"] = value
    return result


def group_dict_items_before_first_dot(dict_: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = defaultdict(dict)
    for packages, value in dict_.items():
        if "." in packages:
            split_list = packages.split(".")
            key1 = ".".join(split_list[:-1])
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


def update_placeholder_from_other_dict(should_update: Dict, template: Dict, ignored_suffix: str = "(choice)"):
    token = "<placeholder>"
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


if __name__ == '__main__':
    from pprint import pprint

    hdl = {
        'preprocessing': {'0nan->{highR=highR_nan,lowR=lowR_nan}(choice)': {'operate.split.nan': {'random_state': 42}},
                          '1lowR_nan->nan(choice)': {'impute.fill_abnormal': {'random_state': 42}},
                          '2highR_nan->nan(choice)': {'operate.drop': {'random_state': 42}},
                          '3all->{cat_name=cat,num_name=num}(choice)': {'operate.split.cat_num': {'random_state': 42}},
                          '4cat->num(choice)': {'encode.label': {'random_state': 42}},
                          '5num->num(choice)': {'<placeholder>': {'_select_percent': {'_type': 'quniform',
                                                                                      '_value': [1, 100, 0.5],
                                                                                      '_default': 80},
                                                                  'random_state': 42}}},
        'estimator(choice)': {'lightgbm': {"boosting_type": "<placeholder>"}}}
    last_best_dhp = {'estimator': {'lightgbm': {"boosting_type": "gbdt"}},
                     'preprocessing': {
                         '0nan->{highR=highR_nan,lowR=lowR_nan}': {'operate.split.nan': {'random_state': 42}},
                         '1lowR_nan->nan': {'impute.fill_abnormal': {'random_state': 42}},
                         '2highR_nan->nan': {'operate.drop': {'random_state': 42}},
                         '3all->{cat_name=cat,num_name=num}': {'operate.split.cat_num': {'random_state': 42}},
                         '4cat->num': {'encode.label': {'random_state': 42}},
                         '5num->num': {'select.from_model_clf': {'_select_percent': 80,
                                                                 'estimator': 'sklearn.svm.LinearSVC',
                                                                 'random_state': 42,
                                                                 'C': 1,
                                                                 'dual': False,
                                                                 'multi_class': 'ovr',
                                                                 'penalty': 'l1'}}}}
    updated_hdl = update_placeholder_from_other_dict(hdl, last_best_dhp)
    target = {'estimator(choice)': {'lightgbm': {'boosting_type': 'gbdt'}},
              'preprocessing': {
                  '0nan->{highR=highR_nan,lowR=lowR_nan}(choice)': {'operate.split.nan': {'random_state': 42}},
                  '1lowR_nan->nan(choice)': {'impute.fill_abnormal': {'random_state': 42}},
                  '2highR_nan->nan(choice)': {'operate.drop': {'random_state': 42}},
                  '3all->{cat_name=cat,num_name=num}(choice)': {'operate.split.cat_num': {'random_state': 42}},
                  '4cat->num(choice)': {'encode.label': {'random_state': 42}},
                  '5num->num(choice)': {'select.from_model_clf': {'C': 1,
                                                                  '_select_percent': {'_default': 80,
                                                                                      '_type': 'quniform',
                                                                                      '_value': [1,
                                                                                                 100,
                                                                                                 0.5]},
                                                                  'dual': False,
                                                                  'estimator': 'sklearn.svm.LinearSVC',
                                                                  'multi_class': 'ovr',
                                                                  'penalty': 'l1',
                                                                  'random_state': 42}}}}
    pprint(updated_hdl)
