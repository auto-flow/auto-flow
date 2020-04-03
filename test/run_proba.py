from collections import Counter
from pprint import pprint

from hyperflow import constants
from hyperflow.hdl2shps.hdl2shps import HDL2SHPS
from hyperflow.shp2dhp.shp2dhp import SHP2DHP

HDL = {'preprocessing': {'0nan->{highR=highR_nan,lowR=lowR_nan}(choice)': {'operate.split.nan': {}},
              '1highR_nan->lowR_nan(choice)': {'operate.drop': {},
                                               'operate.merge': {'__rely_model': 'boost_model'}},
              '2lowR_nan->{cat_name=cat_nan,num_name=num_nan}(choice)': {'operate.split.cat_num': {}},
              '3num_nan->num(choice)': {'impute.fill_abnormal': {'__rely_model': 'boost_model'},
                                        'impute.fill_num': {'strategy': {'_default': 'mean',
                                                                         '_type': 'choice',
                                                                         '_value': ['median',
                                                                                    'mean']}}},
              '4cat_nan->cat(choice)': {'impute.fill_abnormal': {'__rely_model': 'boost_model'},
                                        'impute.fill_cat': {'strategy': {'_default': '<NULL>',
                                                                         '_type': 'choice',
                                                                         '_value': ['<NULL>',
                                                                                    'most_frequent']}}},
              '5cat->{highR=highR_cat,lowR=lowR_cat}(choice)': {'operate.split.cat': {}},
              '6highR_cat->num(choice)': {'encode.label': {},
                                          'operate.drop': {}},
              '7lowR_cat->num(choice)': {'encode.label': {},
                                         'encode.one_hot': {}}},
       'estimator(choice)': {

           'catboost': {},
           "adaboost":{},
           "sgd": {},
           "bernoulli_nb": {},
           "extra_trees": {},
           # 'decision_tree': {'criterion': {'_default': 'gini',
           #                                 '_type': 'choice',
           #                                 '_value': ['gini', 'entropy']},
           #                   'max_features': None,
           #                   'max_leaf_nodes': None,
           #                   'min_impurity_decrease': 0,
           #                   'min_samples_split': {'_default': 2,
           #                                         '_type': 'int_uniform',
           #                                         '_value': [2, 10]},
           #                   'min_weight_fraction_leaf': 0},
           'k_nearest_neighbors': {'n_neighbors': {'_default': 3,
                                                   '_type': 'int_loguniform',
                                                   '_value': [1, 100]},
                                   'p': {'_default': 2,
                                         '_type': 'choice',
                                         '_value': [1, 2]},
                                   'weights': {'_default': 'uniform',
                                               '_type': 'choice',
                                               '_value': ['uniform',
                                                          'distance']}},
           'libsvm_svc': {'C': {'_default': 1.0,
                                '_type': 'loguniform',
                                '_value': [0.01, 10000]},
                          '__activate': {'kernel': {'poly': ['degree',
                                                             'gamma',
                                                             'coef0'],
                                                    'rbf': ['gamma'],
                                                    'sigmoid': ['gamma',
                                                                'coef0']}},
                          'class_weight': {'_default': None,
                                           '_type': 'choice',
                                           '_value': ['balanced', None]},
                          'coef0': {'_default': 0,
                                    '_type': 'quniform',
                                    '_value': [-1, 1]},
                          'decision_function_shape': 'ovr',
                          'degree': {'_default': 3,
                                     '_type': 'int_uniform',
                                     '_value': [2, 5]},
                          'gamma': {'_default': 0.1,
                                    '_type': 'loguniform',
                                    '_value': [1e-05, 8]},
                          'kernel': {'_default': 'rbf',
                                     '_type': 'choice',
                                     '_value': ['rbf',
                                                'poly',
                                                'sigmoid']},
                          'probability': True,
                          'shrinking': {'_default': True,
                                        '_type': 'choice',
                                        '_value': [True, False]}},
           'lightgbm': {}
       }
    }

i = 0
estimators = []

hdl2phps = HDL2SHPS()
hdl2phps.set_task(constants.binary_classification_task)
shps = hdl2phps(HDL)
for i in range(300):
    # print(shps)
    shp = shps.sample_configuration()
    # print(shp)
    php2dhp = SHP2DHP()
    dhp = php2dhp(shp)
    # i+=1
    estimators.append(list(dhp["estimator"].keys())[0])
    pprint(dhp)
    break
print(Counter(estimators))
