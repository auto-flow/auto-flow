from importlib import import_module
from typing import Dict

from sklearn.pipeline import Pipeline

from autopipeline.constants import Task
from autopipeline.evaluation.train_evaluator import TrainEvaluator
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

    def set_FE_keys(self, FE_keys):
        self._FE_keys = FE_keys

    @property
    def FE_keys(self):
        if not hasattr(self, "_FE_keys"):
            raise NotImplementedError()
        return self._FE_keys

    def create_preprocessor(self, dhp: Dict):
        # 将估计器之前的步骤都整合成一个Pipeline，返回
        FE_dict: dict = dhp["FE"]
        pipeline_list = []
        # for name, module_class in FE_dict.items():
        for name in self.FE_keys:
            _module = FE_dict[name]
            if _module is None:
                continue
            module_path=f"autopipeline.pipeline.components.feature_engineer.{name}.{_module}"
            _class = get_class_of_module(module_path)
            M = import_module(
                module_path
            )
            assert hasattr(M, _class)
            cls = getattr(M, _class)
            param = self.get_rely_param_in_dhp(dhp, f"FE/{name}", _module)
            preprocessor = cls()
            default_hp = self.default_hp.get("feature_engineer", {}) \
                .get(name, {}).get(_module, {})
            default_hp.update(param)
            preprocessor.update_hyperparams(default_hp)
            preprocessor.set_addition_info(self.addition_info)
            pipeline_list.append(
                (name, preprocessor)
            )
        if not pipeline_list:
            # pipeline_list=[('no_preprocessing',NoPreprocessing())]
            return None
        else:
            return Pipeline(pipeline_list)

    def create_estimator(self, dhp: Dict):
        # 根据超参构造一个估计器
        _module = dhp["MHP"]
        param = dhp["[MHP]"][_module]
        module_path=f"autopipeline.pipeline.components.{self.task.mainTask}.{_module}"
        _class=get_class_of_module(module_path)
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
