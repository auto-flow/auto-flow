import re
from typing import Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from hyperflow.constants import Task
from hyperflow.evaluation.train_evaluator import TrainEvaluator
from hyperflow.manager.resource_manager import ResourceManager
from hyperflow.manager.xy_data_manager import XYDataManager
from hyperflow.pipeline.pipeline import GenericPipeline
from hyperflow.utils.dict import group_dict_items_before_first_dot
from hyperflow.utils.packages import get_class_object_in_pipeline_components


class PipelineTuner():



    def set_default_hp(self, default_hp):
        self._default_hp = default_hp

    @property
    def default_hp(self):
        if not hasattr(self, "_default_hp"):
            raise NotImplementedError()
        return self._default_hp





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





    def run(self, *args):
        raise NotImplementedError()

    def php2model(self, php):
        raise NotImplementedError()

    def hdl2phps(self, hdl: Dict):
        raise NotImplementedError()




