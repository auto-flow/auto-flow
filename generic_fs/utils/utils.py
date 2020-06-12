import datetime
import json
import pickle
from uuid import uuid4

import numpy
import numpy as np
import pandas as pd
from frozendict import frozendict


def get_id():
    return str(uuid4().hex)


def dumps_pickle(data):
    return pickle.dumps(data)


def loads_pickle(bits):
    return pickle.loads(bits)


def remove_None_value(dict_: dict) -> dict:
    result = {}
    for k, v in dict_.items():
        if v is not None:
            result[k] = v
    return result


# todo: 增加系统性的测试

class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if (obj is None) or isinstance(obj, (str, int, float, bool, list, tuple)):
            return json.JSONEncoder.default(self, obj)
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return str(obj)
        elif isinstance(obj, bytes):
            # todo: base64
            return obj.decode(encoding="utf-8")
        elif isinstance(obj, frozendict):
            return dict(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.bool):
            return bool(obj)
        elif isinstance(obj, numpy.float):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        else:
            return str(obj)
