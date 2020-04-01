from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["SVMSMOTE"]


class SVMSMOTE(HyperFlowDataProcessAlgorithm):
    class__ = "SVMSMOTE"
    module__ = "imblearn.over_sampling"
