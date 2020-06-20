from copy import deepcopy
from typing import List


def remove_suffix_in_list(list_: List[str], suffix: str):
    result = deepcopy(list_)
    for i, elem in enumerate(result):
        if elem.endswith(suffix):
            ix = elem.rfind(suffix)
            elem = elem[:ix]
        result[i] = elem
    return result


def multiply_to_list(item, n) -> list:
    if not isinstance(item, list):
        return [item] * n
    else:
        assert len(item) == n
        return item
