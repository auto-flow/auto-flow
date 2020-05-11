import pickle
from uuid import uuid4


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


