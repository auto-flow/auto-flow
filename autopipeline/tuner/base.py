from importlib import import_module
from typing import Dict

from sklearn.pipeline import Pipeline

from autopipeline.constants import Task
from autopipeline.evaluation.train_evaluator import TrainEvaluator
from autopipeline.pipeline.components.feature_engineer.no_preprocessing import NoPreprocessing


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
            self.evaluator=TrainEvaluator()

    def set_phps(self,hdl:Dict):
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

    def create_preprocessor(self, dhp: Dict):
        # 将估计器之前的步骤都整合成一个Pipeline，返回
        FE_dict: dict = dhp["FE"]
        pipeline_list = []
        for name, module_class in FE_dict.items():
            if module_class is None:
                continue
            splited = module_class.split(".")
            assert len(splited) == 2
            _module, _class = splited
            M = import_module(
                f"autopipeline.pipeline.components.feature_engineer.{name}.{_module}"
            )
            assert hasattr(M, _class)
            cls = getattr(M, _class)
            param = self.get_rely_param_in_dhp(dhp, f"FE/{name}", module_class)
            preprocessor = cls()
            preprocessor.update_hyperparams(param)
            pipeline_list.append(
                (name, preprocessor)
            )
        # todo: fix if pipeline is empty
        if not pipeline_list:
            pipeline_list=[('no_preprocessing',NoPreprocessing())]
        return Pipeline(pipeline_list)

    def create_estimator(self, dhp: Dict):
        # 根据超参构造一个估计器
        selected_model = dhp["MHP"]
        param = dhp["[MHP]"][selected_model]
        splited = selected_model.split(".")
        assert len(splited) == 2
        _module, _class = splited
        M = import_module(
            f"autopipeline.pipeline.components.{self.task.mainTask}.{_module}"
        )
        assert hasattr(M, _class)
        cls = getattr(M, _class)
        estimator = cls()
        estimator.update_hyperparams(param)
        return Pipeline([(
            self.task.role,
            estimator
        )])

    def run(self,*args):
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
