from typing import Union, List

from autopipeline.constants import Task
from autopipeline.utils.packages import get_default_hdl_db, get_hdl_db


class HDL_Constructor():
    def __init__(
            self,
            hdl_db_path=None,
            include_estimators=None,
            exclude_estimators=None,
    ):
        if hdl_db_path:
            hdl_db = get_hdl_db(hdl_db_path)
        else:
            hdl_db = get_default_hdl_db()
        self.hdl_db=hdl_db
        # hdl_db 包含 默认值 与 超参空间， 需要将两个部分分离开来


    def set_task(self, task: Task):
        self._task = task

    @property
    def task(self):
        if not hasattr(self, "_task"):
            raise NotImplementedError()
        return self._task

    def set_feature_group(self,feature_group:Union[None,List,str]):
        # feature_group:
        #    auto: 自动搜索 numerical categorical
        #    list
        self._feature_group=None

    @property
    def feature_group(self):
        if not hasattr(self, "_feature_group"):
            raise NotImplementedError()
        return self._feature_group


    def run(self):
        pass

    def get_hdl(self):
        # 获取hdl
        pass

    def get_default_hp(self):
        pass
