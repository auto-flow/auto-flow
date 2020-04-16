from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["SVMSMOTE"]


class SVMSMOTE(AutoFlowDataProcessAlgorithm):
    class__ = "SVMSMOTE"
    module__ = "imblearn.over_sampling"
