from typing import List

from ConfigSpace.configuration_space import Configuration

from autopipeline.hdl.smac import _decode
from autopipeline.php2dhp.base import PHP2DHP


class SmacPHP2DHP(PHP2DHP):
    def set_kv(self, dict_: dict, key_path: list, value):
        tmp = dict_
        for i, key in enumerate(key_path):
            if i != len(key_path) - 1:
                if key not in tmp:
                    tmp[key] = {}
                tmp = tmp[key]
        key = key_path[-1]
        if (key == "placeholder" and value == "placeholder"):
            pass
        else:
            tmp[key] = value

    def split_key(self, key, token=":", ignore=("[", "]")) -> List[str]:
        L = len(key)
        stack = 0
        res = []
        cursor = ""
        for i, e in enumerate(key):
            if e == ignore[0]:
                stack += 1
            if e == token and stack == 0:
                res.append(cursor)
                cursor = ""
            else:
                cursor = cursor + e
            if e == ignore[1]:
                stack -= 1
        res.append(cursor)
        return res

    def convert(self, php: Configuration):
        dict_ = php.get_dictionary()
        ret = {}
        for k, v in dict_.items():
            if isinstance(v, str):
                v = _decode(v)
            key_path = k.split(":")
            if key_path[-1] == "__choice__":
                # fixme
                key_path = key_path[:-1]
                if v is not None:
                    key_path+=[v]
                    v={}

            self.set_kv(ret, key_path, v)  # self.split_key(k)
        return ret

