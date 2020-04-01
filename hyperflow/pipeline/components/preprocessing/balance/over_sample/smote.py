from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["SMOTE"]


class SMOTE(HyperFlowDataProcessAlgorithm):
    class__ = "SMOTE"
    module__ = "imblearn.over_sampling"
