#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from pprint import pprint

from autoflow.utils.dict import update_mask_from_other_dict

hdl = {
    'preprocessing': {'0nan->{highR=highR_nan,lowR=lowR_nan}(choice)': {'operate.split.nan': {'random_state': 42}},
                      '1lowR_nan->nan(choice)': {'impute.fill_abnormal': {'random_state': 42}},
                      '2highR_nan->nan(choice)': {'operate.drop': {'random_state': 42}},
                      '3all->{cat_name=cat,num_name=num}(choice)': {'operate.split.cat_num': {'random_state': 42}},
                      '4cat->num(choice)': {'encode.label': {'random_state': 42}},
                      '5num->num(choice)': {'<mask>': {'_select_percent': {'_type': 'quniform',
                                                                           '_value': [1, 100, 0.5],
                                                                           '_default': 80},
                                                       'random_state': 42}}},
    'estimator(choice)': {'lightgbm': {"boosting_type": "<mask>"}}}
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
updated_hdl = update_mask_from_other_dict(hdl, last_best_dhp)
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
