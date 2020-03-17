from importlib import import_module
from typing import Dict
import re

from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.pipeline import Pipeline

from autopipeline.constants import Task
from autopipeline.manager.xy_data_manager import XYDataManager
from autopipeline.evaluation.train_evaluator import TrainEvaluator
from autopipeline.pipeline.components.feature_engineer.feature_group import FeatureGroup
from autopipeline.utils.packages import get_class_of_module
from autopipeline.utils.pipeline import union_pipeline, concat_pipeline
from autopipeline.manager.resource_manager import ResourceManager


class PipelineTuner():

    def __init__(
            self,
            runcount_limit: int,
            initial_runs: int,
            random_state: int,
            evaluator: TrainEvaluator,
    ):
        self.evaluator = evaluator
        self.initial_runs = initial_runs
        self.runcount_limit = runcount_limit
        self.random_state = random_state
        if not self.evaluator:
            self.evaluator = TrainEvaluator()

    def set_default_hp(self, default_hp):
        self._default_hp = default_hp

    @property
    def default_hp(self):
        if not hasattr(self, "_default_hp"):
            raise NotImplementedError()
        return self._default_hp

    def set_addition_info(self, addition_info):
        self._addition_info = addition_info

    @property
    def addition_info(self):
        if not hasattr(self, "_addition_info"):
            raise NotImplementedError()
        return self._addition_info

    def set_hdl(self, hdl: Dict):
        self.hdl = hdl
        # todo: 泛化ML管线后，可能存在多个FE
        self.phps:ConfigurationSpace = self.hdl2phps(hdl)
        self.phps.seed(self.random_state)

    def get_rely_param_in_dhp(self, dhp, key, module_class) -> Dict:
        wrap_key = f"[{key}]"
        if wrap_key not in dhp:
            return {}
        relied_value = dhp[wrap_key]
        if module_class not in relied_value:
            return {}
        ret = relied_value[module_class]
        assert isinstance(ret, dict)
        return ret

    def set_resource_manager(self,resource_manager:ResourceManager):
        self._resource_manager=resource_manager

    @property
    def resource_manager(self):
        return self._resource_manager

    def set_data_manager(self, data_manager:XYDataManager):
        self._data_manager = data_manager

    @property
    def data_manager(self):
        if not hasattr(self, "_data_manager"):
            raise NotImplementedError()
        return self._data_manager

    def __create_preprocessor(self, dhp, selected_group=None, feature_groups=None):
        if selected_group is None:
            key = "FE"
        else:
            key = f"FE-{selected_group}"
        if key not in self.hdl:
            return None
        # 删除括号
        sequences = list(map(lambda x: re.sub(r"\(.*\)","",x), self.hdl[key].keys()))
        pipeline_list = []
        # todo
        if feature_groups and selected_group:
            pipeline_list.append((
                f"{selected_group}-split",
                FeatureGroup(selected_group, feature_groups)
            ))
        for phase in sequences:
            if dhp[key][phase] is None:
                continue
            _module = list(dhp[key][phase].keys())[0]
            if _module is None:  # optional-choice
                continue
            module_path = f"autopipeline.pipeline.components.feature_engineer.{phase}.{_module}"
            _class = get_class_of_module(module_path)
            M = import_module(
                module_path
            )
            assert hasattr(M, _class)
            cls = getattr(M, _class)
            param = dhp[key][phase][_module]
            preprocessor = cls()
            default_hp = self.default_hp.get("feature_engineer", {}) \
                .get(phase, {}).get(_module, {})
            default_hp.update(param)
            preprocessor.update_hyperparams(default_hp)
            preprocessor.set_addition_info(self.addition_info)
            pipeline_list.append(
                (phase, preprocessor)
            )
        if pipeline_list:
            return Pipeline(pipeline_list)
        else:
            return None

    def create_preprocessor(self, dhp: Dict) -> Pipeline:

        if self.data_manager.feature_groups:
            feature_groups_set = set(self.data_manager.feature_groups)
            preprocessors = {}
            for selected_group in feature_groups_set:
                preprocessors[selected_group] = self.__create_preprocessor(dhp, selected_group, self.feature_groups)
            # 将估计器之前的步骤都整合成一个Pipeline，返回
            union_feature_pipeline = union_pipeline(preprocessors)
        else:
            union_feature_pipeline=None
        union_preprocessor = self.__create_preprocessor(dhp)
        return concat_pipeline(union_feature_pipeline, union_preprocessor)

    def create_estimator(self, dhp: Dict) -> Pipeline:
        # 根据超参构造一个估计器
        _module = list(dhp["MHP"].keys())[0]
        param = dhp["MHP"][_module]
        module_path = f"autopipeline.pipeline.components.{self.task.mainTask}.{_module}"
        _class = get_class_of_module(module_path)
        M = import_module(
            module_path
        )
        assert hasattr(M, _class)
        cls = getattr(M, _class)
        default_hp = self.default_hp.get(self.task.mainTask, {}) \
            .get(f"{_module}", {})
        default_hp.update(param)
        estimator = cls()
        estimator.set_addition_info(self.addition_info)
        estimator.update_hyperparams(default_hp)
        return Pipeline([(
            self.task.role,
            estimator
        )])

    def run(self, *args):
        raise NotImplementedError()

    def php2model(self, php):
        raise NotImplementedError()

    def hdl2phps(self, hdl: Dict):
        raise NotImplementedError()

    def set_task(self, task: Task):
        self._task = task

    @property
    def task(self):
        if not hasattr(self, "_task"):
            raise NotImplementedError()
        return self._task


