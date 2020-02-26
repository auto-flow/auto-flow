from typing import Union, List, Dict

from autopipeline.constants import Task
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
        if hdl_db_path:
            hdl_db = get_hdl_db(hdl_db_path)
        else:
            hdl_db = get_default_hdl_db()
        self.hdl_db = hdl_db
        self.FE_seq = {}
        self.kwargs = kwargs

    def parse_kwarg(self, kwargs: Dict):
        for key, value in kwargs.items():
            if key.startswith("FE"):
                if key == "FE":
                    self.FE_seq["FE"] = value
                else:
                    assert key.startswith("FE_")
                    selected_group = key.replace("FE_", "")
                    if selected_group in self.feature_groups:
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

    def set_feature_groups(self, feature_groups: Union[None, List, str]):
        # feature_group:
        #    auto: 自动搜索 numerical categorical
        #    list
        if isinstance(feature_groups, str):
            if feature_groups == "auto":
                pass
                # todo
        elif isinstance(feature_groups, list):
            self._feature_groups = feature_groups
        else:
            self._feature_groups = None
        # ----
        if self.feature_groups:
            self.unique_feature_groups = set(self.feature_groups)
        else:
            self.unique_feature_groups = None

    @property
    def feature_groups(self):
        if not hasattr(self, "_feature_groups"):
            raise NotImplementedError()
        return self._feature_groups

    def purify(self, dict_: Dict, module_path,authorized_methods=None) -> Dict:
        ret = {}
        for key, value in dict_.items():
            cls = get_class_of_module(module_path + "." + key)
            if cls.classification_only and self.task.mainTask == "regression":
                continue
            if cls.regression_only and self.task.mainTask == "classification":
                continue
            if authorized_methods and key in authorized_methods:
                continue
            ret[key] = value
        return ret

    def parse_value(self, seq):
        ret = {}
        feature_engineer_db = self.pure_hdl_db["feature_engineer"]
        module_path = "autopipeline.pipeline.components.feature_engineer"
        for item in seq:
            if isinstance(item, str):
                if item in feature_engineer_db:
                    ret[f"{item}(choice)"] = self.purify(feature_engineer_db[item], module_path + "." + item)
                else:
                    print("warn")
            elif isinstance(item, dict):# {"scale":["minmax","normalize"]}
                keys_list=list(item.keys())
                assert len(keys_list)==1
                key=keys_list[0]
                authorized_methods = item[key]
                ret[f"{key}(choice)"] = self.purify(feature_engineer_db[key], module_path + "." + key)
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
        if self.unique_feature_groups:
            for selected_group in self.unique_feature_groups:
                FE_seq_key = f"FE-{selected_group}"
                if FE_seq_key not in self.FE_seq:
                    seq = []
                    if selected_group == "categorical":  # 有特定意义的feature group
                        seq = [{"encode": "one_hot_encode"}]
                    self.FE_seq[FE_seq_key] = seq
        if "FE" not in self.FE_seq:
            self.FE_seq=[]   # 默认不做特征工程
        # 对于普通用户，如果不增加配置，只会做算法选择和算法调参（like ATM）
        # 对于高级用户，用户可以自定义Pipeline搜索空间
        for key, value in self.FE_seq.items():
            pass

    def get_hdl(self):
        # 获取hdl
        return self

    def get_default_hp(self):
        return self.default_hp


if __name__ == '__main__':
    hdl_constructor = HDL_Constructor()
    hdl_constructor.run()
    print()
