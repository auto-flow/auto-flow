from importlib import import_module
from typing import Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.pipeline import Pipeline

from autopipeline.constants import Task
from autopipeline.evaluation.train_evaluator import TrainEvaluator
from autopipeline.manager.resource_manager import ResourceManager
from autopipeline.manager.xy_data_manager import XYDataManager
from autopipeline.pipeline.pipeline import GeneralPipeline
from autopipeline.utils.packages import get_class_of_module


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
            # raise NotImplementedError()
            return {}
        return self._addition_info

    def set_hdl(self, hdl: Dict):
        self.hdl = hdl
        # todo: 泛化ML管线后，可能存在多个FE
        self.phps: ConfigurationSpace = self.hdl2phps(hdl)
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

    def set_resource_manager(self, resource_manager: ResourceManager):
        self._resource_manager = resource_manager

    @property
    def resource_manager(self):
        return self._resource_manager

    def set_data_manager(self, data_manager: XYDataManager):
        self._data_manager = data_manager

    @property
    def data_manager(self):
        if not hasattr(self, "_data_manager"):
            raise NotImplementedError()
        return self._data_manager

    def parse(self, key: str):
        cnt = ""
        ix = 0
        for i, c in enumerate(key):
            if c.isdigit():
                cnt += c
            else:
                ix = i
                break
        cnt = int(cnt)
        key = key[ix:]
        _from, _to = key.split("->")
        outside_edge_info = {}
        in_feat_grp = _from
        out_feat_grp = None
        if _to.startswith("{") and _to.endswith("}"):
            _to = _to[1:-1]
            param_kvs = _to.split(",")
            for param_kv in param_kvs:
                k, v = param_kv.split("=")
                outside_edge_info[k] = v
        else:
            out_feat_grp=_to
        return in_feat_grp, out_feat_grp, outside_edge_info

    def create_preprocessor(self, dhp: Dict) -> Optional[GeneralPipeline]:
        FE_dict: dict = dhp["FE"]
        pipeline_list = []
        for key, value in FE_dict.items():
            name = key
            in_feat_grp, out_feat_grp, outside_edge_info = self.parse(key)
            _module = list(FE_dict[name].keys())[0]
            if _module is None:  # optional-choice
                continue
            module_path = f"autopipeline.pipeline.components.feature_engineer.{_module}"
            _class = get_class_of_module(module_path)
            M = import_module(
                module_path
            )
            assert hasattr(M, _class)
            cls = getattr(M, _class)
            param = FE_dict[name][_module]
            preprocessor = cls()
            preprocessor.in_feat_grp = in_feat_grp
            preprocessor.out_feat_grp = out_feat_grp

            # todo: default_hp
            # default_hp = self.default_hp.get("feature_engineer", {}) \
            #     .get(phase, {}).get(_module, {})
            # default_hp.update(param)
            param.update(outside_edge_info)
            preprocessor.update_hyperparams(param)  # param
            preprocessor.set_addition_info(self.addition_info)
            pipeline_list.append(
                (name, preprocessor)
            )
        if pipeline_list:
            return GeneralPipeline(pipeline_list)
        else:
            return None

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
        # default_hp = self.default_hp.get(self.task.mainTask, {}) \
        #     .get(f"{_module}", {})
        default_hp = {}
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
