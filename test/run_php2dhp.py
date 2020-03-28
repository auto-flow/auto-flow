from pprint import pprint

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from autopipeline import constants
from autopipeline.hdl2phps.smac_hdl2phps import SmacHDL2PHPS
from autopipeline.php2dhp.smac_php2dhp import SmacPHP2DHP
from autopipeline.pipeline.dataframe import GenericDataFrame
from autopipeline.tuner.smac_tuner import SmacPipelineTuner
from autopipeline.utils.pipeline import concat_pipeline

HDL = {'feature_engineer': {'0nan->{highR=highR_nan,lowR=lowR_nan}(choice)': {'operate.split.nan': {}},
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
              '6highR_cat->num(choice)': {'encode.label': {'__rely_model': 'tree_model'},
                                          'operate.drop': {}},
              '7lowR_cat->num(choice)': {'encode.label': {'__rely_model': 'tree_model'},
                                         'encode.one_hot': {}}},
       'estimator(choice)': {'catboost': {},
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

i = 0
estimators = []
for i in range(100):
    hdl2phps = SmacHDL2PHPS()
    hdl2phps.set_task(constants.binary_classification_task)
    phps = hdl2phps(HDL)
    # print(phps)
    php = phps.sample_configuration()
    # print(php)
    php2dhp = SmacPHP2DHP()
    dhp = php2dhp(php)
    # i+=1
    estimators.append(list(dhp["estimator"].keys())[0])
    pprint(dhp)
    break
# print(Counter(estimators))
tuner = SmacPipelineTuner()
tuner.set_task(constants.binary_classification_task)
preprocessor = tuner.create_preprocessor(dhp)
estimators = tuner.create_estimator(dhp)
pipeline = concat_pipeline(preprocessor, estimators)
print(pipeline)
df = pd.read_csv("../examples/classification/train_classification.csv")
y = df.pop("Survived").values
df = df.loc[:, ["Pclass", "Name", "Sex", "Age", "SibSp", "Ticket", "Fare", "Cabin", "Embarked"]]
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=10)
feat_grp = ["num", "cat", "cat", "nan", "num", "cat", "num", "nan", "nan"]
df_train = GenericDataFrame(df_train, feat_grp=feat_grp)
df_test = GenericDataFrame(df_test, feat_grp=feat_grp)
cv = KFold(n_splits=5, random_state=10, shuffle=True)
train_ix, valid_ix = next(cv.split(df_train))
df_train, df_valid = df_train.split([train_ix, valid_ix])
y_valid = y_train[valid_ix]
y_train = y_train[train_ix]
ans = pipeline.procedure(constants.binary_classification_task, df_train, y_train, df_valid, y_valid, df_test)
print(ans)
