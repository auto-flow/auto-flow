from copy import deepcopy

from autopipeline.constants import Task
from autopipeline.hdl.utils import get_hdl_db, get_default_hdl_db
from autopipeline.manager.xy_data_manager import XYDataManager


class HDL_Constructor():
    def __init__(
            self,
            hdl_db_path=None,
            DAG_descriptions=None
    ):
        if DAG_descriptions is None:
            DAG_descriptions = {
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
                    "encode.label"
                ],
                "lowR_cat->num": [
                    "encode.one_hot",
                    "encode.label"
                ],
                "num->target": [
                    # "decision_tree", "libsvm_svc",
                    "k_nearest_neighbors",
                    "catboost",
                    "lightgbm"
                ]
            }
        self.DAG_describe = DAG_descriptions
        if hdl_db_path:
            hdl_db = get_hdl_db(hdl_db_path)
        else:
            hdl_db = get_default_hdl_db()
        self.hdl_db = hdl_db
        self.params = {
            "hdl_db_path": hdl_db_path,
            "DAG_describe": DAG_descriptions,
        }

    def set_task(self, task: Task):
        self._task = task

    def set_random_state(self, random_state):
        self._random_state = random_state

    @property
    def random_state(self):
        if not hasattr(self, "_random_state"):
            return 10
        else:
            return self._random_state

    @property
    def task(self):
        if not hasattr(self, "_task"):
            raise NotImplementedError()
        return self._task

    def set_data_manager(self, data_manager: XYDataManager):
        self._data_manager = data_manager
        self._task = data_manager.task

    @property
    def data_manager(self):
        if not hasattr(self, "_data_manager"):
            raise NotImplementedError()
        return self._data_manager

    def run(self):
        # make sure:
        # set_task
        target_key = ""

        for key in self.DAG_describe.keys():
            if key.split("->")[-1] == "target":
                target_key = key
        MHP_values = self.DAG_describe.pop(target_key)
        FE_dict = {}
        mainTask = self.task.mainTask
        FE_package = "autopipeline.pipeline.components.feature_engineer"
        hdl_db = get_default_hdl_db()
        FE_hdl_db = hdl_db["feature_engineer"]
        MHP_hdl_db = hdl_db[mainTask]

        def get_params_in_dict(dict_, package):
            ans = deepcopy(dict_)
            for path in package.split("."):
                ans = ans.get(path, {})
            return ans

        for i, (key, values) in enumerate(self.DAG_describe.items()):
            if not isinstance(values, (list, tuple)):
                values = [values]
            if None in values:
                formed_key = f"{i}{key}(optional-choice)"
                values.remove(None)
            else:
                formed_key = f"{i}{key}(choice)"
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
                addition_dict.update({"random_state": self.random_state})  # fixme
                sub_dict[name] = get_params_in_dict(FE_hdl_db, name)
                sub_dict[name].update(addition_dict)
            FE_dict[formed_key] = sub_dict

        MHP_dict = {}

        for MHP_value in MHP_values:
            name = MHP_value
            MHP_dict[name] = get_params_in_dict(MHP_hdl_db, name)
        final_dict = {
            "FE": FE_dict,
            "MHP(choice)": MHP_dict
        }
        self.hdl = final_dict

    def get_hdl(self):
        # 获取hdl
        return self.hdl
