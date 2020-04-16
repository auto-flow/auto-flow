from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["ADASYN"]


class ADASYN(AutoFlowDataProcessAlgorithm):
    class__ = "ADASYN"
    module__ = "imblearn.over_sampling"
