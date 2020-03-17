from pprint import pprint

from autopipeline import constants
from autopipeline.hdl2phps.smac_hdl2phps import SmacHDL2PHPS
from autopipeline.php2dhp.smac_php2dhp import SmacPHP2DHP

HDL = {'FE': {'cat->{highR=highR_cat,lowR=lowR_cat}(choice)': {'operate.split.cat': {}},
              'cat_nan->cat(choice)': {'impute.fill_abnormal': {'__rely_model': 'boost_model'},
                                       'impute.fill_cat': {}},
              'highR_cat->num(choice)': {'encode.label': {'__rely_model': 'tree_model'},
                                         'operate.drop': {}},
              'highR_nan->lowR_nan(choice)': {'operate.drop': {},
                                              'operate.merge': {'__rely_model': 'boost_model'}},
              'lowR_cat->num(choice)': {'encode.label': {'__rely_model': 'tree_model'},
                                        'encode.one_hot': {}},
              'lowR_nan->{cat_name=cat_nan,num_name=num_nan}(choice)': {'operate.split.cat_num': {}},
              'nan->{highR=highR_nan,lowR=lowR_nan}(choice)': {'operate.split.nan': {}},
              'num_nan->num(choice)': {'impute.fill_abnormal': {'__rely_model': 'boost_model'},
                                       'impute.fill_num': {}}},
       'MHP(choice)': {'catboost': {},
                       'decision_tree': {'criterion': {'_default': 'gini',
                                                       '_type': 'choice',
                                                       '_value': ['gini', 'entropy']},
                                         'max_features': None,
                                         'max_leaf_nodes': None,
                                         'min_impurity_decrease': 0,
                                         'min_samples_split': {'_default': 2,
                                                               '_type': 'int_uniform',
                                                               '_value': [2, 10]},
                                         'min_weight_fraction_leaf': 0},
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
                       'lightgbm': {}}}
i=0
while True:
    hdl2phps = SmacHDL2PHPS()
    hdl2phps.set_task(constants.binary_classification_task)
    phps = hdl2phps(HDL)
    # print(phps)
    php=phps.sample_configuration()
    # print(php)
    php2dhp=SmacPHP2DHP()
    dhp=php2dhp(php)
    i+=1
    if list(dhp["MHP"].keys())[0] not in ["lightgbm","catboost"]:
        pprint(dhp)
        print(i)
        break
