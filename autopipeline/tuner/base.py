from typing import Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from autopipeline.constants import Task
from autopipeline.evaluation.train_evaluator import TrainEvaluator
from autopipeline.manager.resource_manager import ResourceManager
from autopipeline.manager.xy_data_manager import XYDataManager
from autopipeline.pipeline.pipeline import GenericPipeline
from autopipeline.utils.data import group_dict_items_before_first_dot
from autopipeline.utils.packages import get_class_object_in_pipeline_components


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
        outsideEdge_info = {}
        in_feat_grp = _from
        out_feat_grp = None
        if _to.startswith("{") and _to.endswith("}"):
            _to = _to[1:-1]
            param_kvs = _to.split(",")
            for param_kv in param_kvs:
                k, v = param_kv.split("=")
                outsideEdge_info[k] = v
        else:
            out_feat_grp = _to
        return in_feat_grp, out_feat_grp, outsideEdge_info

    def create_preprocessor(self, dhp: Dict) -> Optional[GenericPipeline]:
        FE_dict: dict = dhp["FE"]
        pipeline_list = []
        for key, value in FE_dict.items():
            name = key  # like: "cat->num"
            in_feat_grp, out_feat_grp, outsideEdge_info = self.parse(key)
            sub_dict = FE_dict[name]
            if sub_dict is None:
                continue
            pipeline_list.extend(self.create_component(sub_dict, "FE", name, in_feat_grp, out_feat_grp))
        if pipeline_list:
            return GenericPipeline(pipeline_list)
        else:
            return None

    def create_estimator(self, dhp: Dict) -> GenericPipeline:
        # 根据超参构造一个估计器
        return GenericPipeline(self.create_component(dhp["MHP"], "MHP", self.task.role))

    def _create_component(self, key1, key2, params):
        cls = get_class_object_in_pipeline_components(key1, key2)
        component = cls()
        component.set_addition_info(self.addition_info)
        component.update_hyperparams(params)
        return component

    def create_component(self, sub_dhp: Dict, phase: str, step_name, in_feat_grp="all", out_feat_grp="all"):
        pipeline_list = []
        assert phase in ("FE", "MHP")
        packages = list(sub_dhp.keys())[0]
        params = sub_dhp[packages]
        packages = packages.split("|")
        grouped_params = group_dict_items_before_first_dot(params)
        if len(packages) == 1:
            grouped_params[packages[0]] = grouped_params.pop("single")
        for package in packages[:-1]:
            preprocessor = self._create_component("feature_engineer", package, grouped_params[package])
            preprocessor.in_feat_grp = in_feat_grp
            preprocessor.out_feat_grp = in_feat_grp
            pipeline_list.append([
                package,
                preprocessor
            ])
        key1 = "feature_engineer" if phase == "FE" else self.task.mainTask
        component = self._create_component(key1, packages[-1], grouped_params[packages[-1]])
        component.in_feat_grp = in_feat_grp
        component.out_feat_grp = out_feat_grp
        pipeline_list.append([
            step_name,
            component
        ])
        return pipeline_list

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
