import hashlib

import numpy as np
from ConfigSpace import Configuration


def get_id_of_config(config: Configuration):
    # todo:, instance="", seed=0
    X: np.ndarray = config.get_array()
    m = hashlib.md5()
    if X.flags['C_CONTIGUOUS']:
        m.update(X.data)
        m.update(str(X.shape).encode('utf8'))
    else:
        X_tmp = np.ascontiguousarray(X.T)
        m.update(X_tmp.data)
        m.update(str(X_tmp.shape).encode('utf8'))
    # m.update(instance.encode())
    # m.update(str(seed).encode())
    hash_value = m.hexdigest()
    return hash_value
