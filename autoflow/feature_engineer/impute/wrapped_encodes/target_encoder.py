#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import numpy as np
from category_encoders import TargetEncoder as OriginTargetEncoder

from skimpute.utils import process_dataframe


class TargetEncoder(OriginTargetEncoder):

    def get_ordinal_map(self, ordinal_map_list, column):
        for item in ordinal_map_list:
            if item["col"] == column:
                return item["mapping"]
        return None

    def inverse_transform(self, X):
        X_ = process_dataframe(X)
        result_list = []
        for i, column in enumerate(X_.columns):
            if column in self.mapping:
                map_val = deepcopy(self.mapping[column].values[:-2])
                map_val = map_val.reshape(1, -1)
                sub = np.abs(X_.values[:, i].reshape(-1, 1) - map_val)
                indexes = np.argmin(sub, axis=1)
                enc_indexes = self.mapping[column].index[indexes]
                ordinal_map_list = self.ordinal_encoder.mapping
                ordinal_map = self.get_ordinal_map(ordinal_map_list, column)
                dict_ = ordinal_map.to_dict()
                inv_dict = {v: k for k, v in dict_.items()}
                mapped = enc_indexes.map(inv_dict)
                result = mapped.values.reshape(-1, 1)
            else:
                result = X[:, i].reshape(-1, 1)
            result_list.append(result)
        # todo : 重构为DataFrame的形式？
        return np.concatenate(result_list, axis=1)


if __name__ == '__main__':
    X_train = np.array([[1, 1, 2],
                        [3, 2, 3],
                        [3, 2, 1]])
    y = [0, 1, 0]
    enc = TargetEncoder(cols=[i for i in range(3)], return_df=False).fit(X_train, y)
    X_test = np.array([[3, 1, 3],
                       [3, 1, 1],
                       [3, 1, 2]])
    processed = enc.transform(X_test)
    inversed = enc.inverse_transform(processed)
    print(inversed)
    print(inversed==X_test)
