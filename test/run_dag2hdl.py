from copy import deepcopy
from pprint import pprint

from autopipeline.hdl.utils import get_default_hdl_db

DAG = {
    "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
    "highR_nan->lowR_nan": [
        "operate.drop",
        {"_name": "operate.merge", "__rely_model": "boost_model"}
    ],
    "lowR_nan->{cat_name=cat_nan,num_name=num_nan}": "operate.split.cat_num",
    "num_nan->num": [
        "impute.fill_num",
        {"_name": "impute.fill_abnormal", "__rely_model": "boost_model"}
    ],
    "cat_nan->cat": [
        "impute.fill_cat",
        {"_name": "impute.fill_abnormal", "__rely_model": "boost_model"}
    ],
    "cat->{highR=highR_cat,lowR=lowR_cat}": "operate.split.cat",
    "highR_cat->num": [
        "operate.drop",
        {"_name": "encode.label", "__rely_model": "tree_model"}  # 迁移到hdl_db
    ],
    "lowR_cat->num": [
        "encode.one_hot",
        {"_name": "encode.label", "__rely_model": "tree_model"}
    ],
    "num->target": [
        "decision_tree", "libsvm_svc", "k_nearest_neighbors", "catboost", "lightgbm"
    ]
}

HDL = {
    "FE": {
        "nan->{highR=highR_nan,lowR=lowR_nan}(choice)": {"operate.split.nan": {}},
        "highR_nan->lowR_nan(choice)": {
            "operate.drop": {},
            "operate.merge": {"__rely_model": "boost_model"}
        }
    },
    "MHP(choice)": {
        "decision_tree": {},
        "libsvm_svc": {}
    }
}

target_key = ""

for key in DAG.keys():
    if key.split("->")[-1] == "target":
        target_key = key
MHP_values = DAG.pop(target_key)
FE_dict = {}
mainTask = "classification"
FE_package = "autopipeline.pipeline.components.feature_engineer"
hdl_db = get_default_hdl_db()
FE_hdl_db = hdl_db["feature_engineer"]
MHP_hdl_db = hdl_db[mainTask]


def get_params_in_dict(dict_, package):
    ans = deepcopy(dict_)
    for path in package.split("."):
        ans = ans.get(path, {})
    return ans


for key, values in DAG.items():
    if not isinstance(values, (list, tuple)):
        values = [values]
    if None in values:
        used_key = key + "(optional-choice)"
    else:
        used_key = key + "(choice)"
    sub_dict = {}
    for value in values:
        if isinstance(value, dict):
            name = value.pop("_name")
            addition_dict = value
        elif isinstance(value, str):
            name = value
            addition_dict = {}
        else:
            raise TypeError
        sub_dict[name] = get_params_in_dict(FE_hdl_db, name)
        sub_dict[name].update(addition_dict)
    FE_dict[used_key] = sub_dict

MHP_dict = {}

for MHP_value in MHP_values:
    name = MHP_value
    MHP_dict[name] = get_params_in_dict(MHP_hdl_db, name)
final_dict = {
    "FE": FE_dict,
    "MHP(choice)": MHP_dict
}

pprint(final_dict)

