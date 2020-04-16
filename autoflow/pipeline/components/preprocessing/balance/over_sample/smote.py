from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["SMOTE"]


class SMOTE(AutoFlowDataProcessAlgorithm):
    class__ = "SMOTE"
    module__ = "imblearn.over_sampling"
