from importlib import import_module
from typing import List, Dict

from autopipeline.constants import Task
from autopipeline.manager.xy_data_manager import XYDataManager
from autopipeline.hdl.default_hp import extract_pure_hdl_db_from_hdl_db, extract_default_hp_from_hdl_db
from autopipeline.hdl.utils import get_hdl_db, get_default_hdl_db
from autopipeline.utils.packages import get_class_of_module


class HDL_Constructor():
    def __init__(
            self,
            hdl_db_path=None,
            include_estimators=None,
            exclude_estimators=None,
            **kwargs
    ):
        if include_estimators is not None and exclude_estimators is not None:
            print("warn")
            exclude_estimators = None

        self.exclude_estimators = exclude_estimators
        self.include_estimators = include_estimators
        if hdl_db_path:
            hdl_db = get_hdl_db(hdl_db_path)
        else:
            hdl_db = get_default_hdl_db()
        self.hdl_db = hdl_db
        self.FE_seq = {}
        self.kwargs = kwargs
        self.params={
            "hdl_db_path":hdl_db_path,
            "include_estimators":include_estimators,
            "exclude_estimators":exclude_estimators,
            "kwargs":kwargs
        }

    def parse_kwarg(self, kwargs: Dict):
        for key, value in kwargs.items():
            if key.startswith("FE"):
                if key == "FE":
                    self.FE_seq["FE"] = value
                else:
                    assert key.startswith("FE_")
                    selected_group = key.replace("FE_", "")
                    if selected_group in self.data_manager.unique_feature_groups:
                        self.FE_seq[f"FE-{selected_group}"] = value
                    else:
                        print("warn")

    def set_task(self, task: Task):
        self._task = task

    @property
    def task(self):
        if not hasattr(self, "_task"):
            raise NotImplementedError()
        return self._task

    def set_data_manager(self, data_manager: XYDataManager):
        self._data_manager = data_manager
        self._task=data_manager.task

    @property
    def data_manager(self):
        if not hasattr(self, "_data_manager"):
            raise NotImplementedError()
        return self._data_manager

    def purify(self, dict_: Dict, module_path, authorized_methods=None) -> Dict:
        ret = {}
        for key, value in dict_.items():
            cur_module_path = module_path + "." + key
            # 把类取出来看是否符合规范
            class_ = get_class_of_module(cur_module_path)
            cls = getattr(import_module(cur_module_path), class_)
            if cls.classification_only and self.task.mainTask == "regression":
                continue
            if cls.regression_only and self.task.mainTask == "classification":
                continue
            if authorized_methods and key not in authorized_methods:
                continue
            ret[key] = value
        return ret

    def parse_fe_seq(self, seq):
        ret = {}
        feature_engineer_db = self.pure_hdl_db["feature_engineer"]
        module_path = "autopipeline.pipeline.components.feature_engineer"
        for item in seq:
            if isinstance(item, str):
                if item in feature_engineer_db:
                    ret[f"{item}(optional-choice)"] = \
                        self.purify(feature_engineer_db[item], module_path + "." + item)
                else:
                    print("warn")
            elif isinstance(item, dict):  # {"scale":["minmax","normalize"]}
                keys_list = list(item.keys())
                assert len(keys_list) == 1
                k = keys_list[0]
                authorized_methods = item[k]

                value = self.purify(feature_engineer_db[k], module_path + "." + k, authorized_methods)
                if None in authorized_methods:
                    key = f"{k}(optional-choice)"
                else:
                    key = f"{k}(choice)"
                ret[key] = value
            else:
                raise NotImplementedError()
        return ret

    def run(self):
        # make sure:
        # set_task
        # set_feature_groups
        self.pure_hdl_db = extract_pure_hdl_db_from_hdl_db(self.hdl_db)
        self.default_hp = extract_default_hp_from_hdl_db(self.hdl_db)
        self.parse_kwarg(self.kwargs)
        if self.data_manager.feature_groups:
            for selected_group in self.data_manager.unique_feature_groups:
                FE_seq_key = f"FE-{selected_group}"
                if FE_seq_key not in self.FE_seq:
                    seq = []
                    if selected_group == "categorical":  # 有特定意义的feature group
                        seq = [{"encode": ["one_hot_encode"]}]
                    self.FE_seq[FE_seq_key] = seq
        if "FE" not in self.FE_seq:
            self.FE_seq["FE"] = []  # 默认不做特征工程
        # 对于普通用户，如果不增加配置，只会做算法选择和算法调参（like ATM）
        # 对于高级用户，用户可以自定义Pipeline搜索空间
        hdl = {}
        for fe_name, fe_seq in self.FE_seq.items():
            hdl[fe_name] = self.parse_fe_seq(fe_seq)
        # ---开始解析MHP----------------
        mainTask = self.task.mainTask
        correspond_hdl_db: dict = self.pure_hdl_db[mainTask]
        estimators = self.fetch_estimators(correspond_hdl_db)
        MHP_key = "MHP(choice)"
        hdl[MHP_key] = {}
        for estimator in estimators:
            hdl[MHP_key][estimator] = correspond_hdl_db[estimator]
        self.hdl = hdl

    def get_hdl(self):
        # 获取hdl
        return self.hdl

    def get_default_hp(self):
        return self.default_hp

    def check_include_estimators(self, correspond_hdl_db):
        ans = []
        for name in self.include_estimators:
            if name not in correspond_hdl_db:
                print("warn")
                continue
            ans.append(name)
        return ans

    def check_exclude_estimators(self, correspond_hdl_db):
        ans = set(correspond_hdl_db.keys())
        for name in self.exclude_estimators:
            if name not in correspond_hdl_db:
                print("warn")
                continue
            ans.remove(name)
        return list(ans)

    def fetch_estimators(self, correspond_hdl_db: Dict) -> List:
        if self.include_estimators:
            estimators = self.check_include_estimators(correspond_hdl_db)
        elif self.exclude_estimators:
            estimators = self.check_exclude_estimators(correspond_hdl_db)
        else:
            estimators = list(correspond_hdl_db.keys())
        return estimators


if __name__ == '__main__':
    from autopipeline.constants import multiclass_classification_task

    hdl_constructor = HDL_Constructor(
        include_estimators=["gradient_boosting"],
        FE=["scale", "select"], FE_categorical=[{"scale": ["minmax", "normalize"]}, "threshold"])
    hdl_constructor.set_task(multiclass_classification_task)
    # todo 完善测试用例
    hdl_constructor.set_data_manager(
        ["numerical", "categorical", "numerical", "categorical", "numerical", "categorical"])
    hdl_constructor.run()
    hdl = hdl_constructor.get_hdl()
    print(hdl)
    # -----------
    from autopipeline.hdl2phps.smac_hdl2phps import SmacHDL2PHPS
    from autopipeline.php2dhp.smac_php2dhp import SmacPHP2DHP
    from autopipeline.tuner.smac import SmacPipelineTuner

    hdl2phps = SmacHDL2PHPS()
    phps = hdl2phps(hdl)
    print(phps)
    php = phps.sample_configuration()
    print(php)
    php2dhp = SmacPHP2DHP()
    dhp = php2dhp(php)
    print(dhp)
    from sklearn.datasets import load_iris
    from autopipeline import constants

    iris = load_iris()
    X = iris.data
    y = iris.target
    tuner = SmacPipelineTuner()
    tuner.set_task(constants.multiclass_classification_task)
    tuner.set_default_hp({})
    tuner.set_addition_info({"shape": X.shape})
    tuner.set_feature_groups(["numerical", "numerical", "categorical", "categorical"])
    tuner.set_hdl(hdl)
    model = tuner.php2model(php)
    print(model)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)
