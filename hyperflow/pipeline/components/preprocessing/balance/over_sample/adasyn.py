from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["ADASYN"]


class ADASYN(HyperFlowDataProcessAlgorithm):
    class__ = "ADASYN"
    module__ = "imblearn.over_sampling"
