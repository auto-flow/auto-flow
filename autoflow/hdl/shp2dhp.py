from typing import List

from autoflow.hdl.smac import _decode
from autoflow.utils.klass import StrSignatureMixin


class SHP2DHP(StrSignatureMixin):
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
        result = []
        cursor = ""
        for i, e in enumerate(key):
            if e == ignore[0]:
                stack += 1
            if e == token and stack == 0:
                result.append(cursor)
                cursor = ""
            else:
                cursor = cursor + e
            if e == ignore[1]:
                stack -= 1
        result.append(cursor)
        return result

    def __call__(self, shp):
        dict_ = shp.get_dictionary()
        result = {}
        for k, v in dict_.items():
            if isinstance(v, str):
                v = _decode(v)
            key_path = k.split(":")
            if key_path[-1] == "__choice__":
                key_path = key_path[:-1]
                if v is not None:
                    key_path += [v]
                    v = {}
            if "None" in key_path:
                continue
            self.set_kv(result, key_path, v)  # self.split_key(k)
        return result
