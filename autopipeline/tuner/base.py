from importlib import import_module
from typing import Dict

from sklearn.pipeline import Pipeline

from autopipeline.constants import Task
from autopipeline.evaluation.train_evaluator import TrainEvaluator
from autopipeline.pipeline.components.feature_engineer.feature_group import FeatureGroup
from autopipeline.utils.packages import get_class_of_module
from autopipeline.utils.pipeline import union_pipeline, concat_pipeline


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

    def set_phps(self, hdl: Dict):
        self.hdl = hdl
        # todo: 泛化ML管线后，可能存在多个FE
        self._FE_keys = list(self.hdl["FE"].keys())
        self.phps = self.hdl2phps(hdl)

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

    def set_feature_groups(self, feature_groups):
        self._feature_groups = feature_groups

    @property
    def feature_groups(self):
        if not hasattr(self, "_feature_groups"):
            raise NotImplementedError()
        return self.feature_groups

    def __create_preprocessor(self, dhp, selected_group=None, feature_groups=None):
        if selected_group is None:
            key = "FE"
        else:
            key = f"FE-{selected_group}"
        if key not in self.hdl:
            return None
        sequences = list(self.hdl[key].keys)
        pipeline_list = []
        # todo
        if feature_groups and selected_group:
            pipeline_list.append((
                f"{selected_group}-split",
                FeatureGroup(selected_group, feature_groups)
            ))
        for phase in sequences:
            _module = list(dhp[key][phase].keys())[0]
            if _module is None:
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
        feature_groups_set = set(self.feature_groups)
        preprocessors = {}
        for selected_group in feature_groups_set:
            preprocessors[selected_group] = self.__create_preprocessor(dhp, selected_group, self.feature_groups)
        # 将估计器之前的步骤都整合成一个Pipeline，返回
        union_feature_pipeline = union_pipeline(preprocessors)
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

    def init_task(self, task: Task):
        self._task = task

    @property
    def task(self):
        if not hasattr(self, "_task"):
            raise NotImplementedError()
        return self._task


if __name__ == '__main__':
    from autopipeline.init_data import init_all
    import numpy as np

    init_all(additional={"shape": (1000, 200)}, public_hp={'random_state': 42})
    sample = \
        {'FE': {'reduce': 'kernel_pca.KernelPCA', 'scale': None},
         'MHP': 'libsvm_svc.LibSVM_SVC',
         '[FE/reduce]': {
             'kernel_pca.KernelPCA':
                 {
                     'kernel': 'rbf',
                     'n_components_ratio': 0.1
                 }
         },
         '[MHP]': {'libsvm_svc.LibSVM_SVC': {'C': 4, 'kernel': 'rbf'}}}
    ans = PipelineTuner().create_preprocessor(sample)
    X_ = ans.fit_transform(np.random.rand(1000, 200))
    print(X_.shape)
    ans = PipelineTuner().create_estimator(sample)
    print(ans)
