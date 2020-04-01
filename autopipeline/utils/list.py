from copy import deepcopy
from typing import List


def remove_suffix_in_list(list_:List[str],suffix:str):
    result=deepcopy(list_)
    for i,elem in enumerate(result):
        if elem.endswith(suffix):
            ix=elem.rfind(suffix)
            elem=elem[:ix]
        result[i]=elem
    return result