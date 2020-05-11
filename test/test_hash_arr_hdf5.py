#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

import numpy as np
import h5py
from autoflow.utils.hash import get_hash_of_array

def test_hash_arr_hdf5():
    np.random.seed(0)
    data_to_write = np.random.random(size=(100, 100))
    hv1=get_hash_of_array(data_to_write)
    print(hv1)
    with h5py.File('name-of-file.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset", data=data_to_write)

    with h5py.File('name-of-file.h5', 'r') as hf:
        data = hf['name-of-dataset'][:]

    hv2=get_hash_of_array(data)
    print(hv2)
    assert hv1==hv2
